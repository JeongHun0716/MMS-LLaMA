import sys,logging
import contextlib
from argparse import Namespace

import os
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from fairseq import checkpoint_utils, tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import FairseqEncoder, FairseqEncoderModel, register_model, BaseFairseqModel, FairseqEncoderDecoderModel
from typing import Any, Optional
from fairseq import utils
import math

from fairseq.dataclass import FairseqDataclass
from omegaconf import II, MISSING

from pathlib import Path
from transformers import  AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model


import torch.nn.functional as F
from avhubert.hubert_asr import AVHubertAsrConfig, HubertEncoderWrapper
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperModel
from .sub_model.modules import WhisperEncoderWrapper, Projector, Multimodal_Attention, Speech_Rate_Predictor
from .sub_model.Qformer import BertConfig, BertLMHeadModel
from torch.nn.utils.rnn import pad_sequence


logger = logging.getLogger(__name__)


@dataclass
class MMS_LLaMA_Config(AVHubertAsrConfig):
    llm_path: str = field(
        default='meta-llama/Llama-3.2-3B'
    )
    target_modules: str = field(
        default='q_proj.v_proj.k_proj.o_proj'
    )
    whisper_embed_dim: int = field(
        default=1024, metadata={"help": "whisper embedding dimension"}
    )
    avhubert_embed_dim: int = field(
        default=1024, metadata={"help": "avhubert embedding dimension"}
    )
    llama_embed_dim: int = field(
        default=3072, metadata={"help": "llama embedding dimension"}
    )
    lora_rank: int = field(
        default=16, metadata={"help": "lora_rank"}
    )
    lora_alpha: int = field(
        default=32, metadata={"help": "lora_alpha"}
    )
    modality_fuse: str = field(
        default='concat', metadata={'help': 'fusing two modalities: concat, add, cross-att'}
    )
    ### Speech Q-Former Config ###
    use_qformer: bool = field(
        default=True
    )
    window_level: bool = field(
        default=False
    )
    queries_per_sec: int = field(
        default=4, metadata={"help": "queries_per_sec"}
    )
    qformer_layers: int = field(
        default=2, metadata={"help": "number of qformer layers"}
    )
    qformer_dim: int = field(
        default=1024, metadata={"help": "qformer dim"}
    )
    ### Speech Rate Predictor Config ###
    use_sr_predictor: bool = field(
        default=False
    )
    sr_predictor_layers: int = field(
        default=2, metadata={"help": "number of sr predictor layers"}
    )
              
@register_model("MMS-LLaMA", dataclass=MMS_LLaMA_Config)
class MMS_LLaMA(BaseFairseqModel):    
    def __init__(self, avhubert, whisper, llm, tokenizer, cfg):
        super().__init__() 
        self.cfg = cfg
        self.avhubert = avhubert
        self.whisper = whisper
        self.llama = llm   
        
        self.tokenizer = tokenizer
        
        for param in self.avhubert.parameters():
            param.requires_grad = False
            
        for param in self.whisper.parameters():
            param.requires_grad = False
    
        self.modality_fuse = cfg.modality_fuse
        if self.modality_fuse == 'concat':
            self.embed = cfg.whisper_embed_dim + cfg.avhubert_embed_dim
        elif self.modality_fuse == 'add':
            self.embed = cfg.whisper_embed_dim
        elif self.modality_fuse == 'cross-att':
            self.multimodal_attention_layer = Multimodal_Attention(embed_dim=cfg.whisper_embed_dim, num_heads=8)
            self.embed = cfg.whisper_embed_dim
        
        #### Qformer ####
        if cfg.use_qformer:
            if cfg.window_level:
                cfg.max_queries = 1
            self.afeat_1d_conv = nn.Conv1d(in_channels=cfg.whisper_embed_dim, out_channels=cfg.whisper_embed_dim, kernel_size=2, stride=2, padding=0) # 50Hz -> 25Hz
            if cfg.use_sr_predictor:
                max_queries = int(cfg.queries_per_sec * 20 * 2)
            else:
                max_queries = int(cfg.queries_per_sec * 20)
                
            qformer_config = BertConfig.from_pretrained("bert-large-uncased")
            qformer_config.num_hidden_layers = cfg.qformer_layers
            qformer_config.encoder_width = self.embed
            qformer_config.hidden_size = cfg.qformer_dim 
            qformer_config.add_cross_attention = True
            qformer_config.cross_attention_freq = 1
            qformer_config.query_length = max_queries
            self.Qformer = BertLMHeadModel(config=qformer_config)
            self.query_tokens = nn.Parameter(
                torch.zeros(1, max_queries, qformer_config.hidden_size)
            )
            self.query_tokens.data.normal_(mean=0.0, std=qformer_config.initializer_range)
            

            if cfg.use_sr_predictor:
                max_queries = int(cfg.queries_per_sec * 20 * 2)
                self.sr_predictor = Speech_Rate_Predictor(num_layers=cfg.sr_predictor_layers)
                root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                sr_ckpt_path = f'{root_dir}/pretrained_models/sr_predictor/checkpoint.pt'
                sr_state = torch.load(sr_ckpt_path)['model']
                sr_state_ = {}
                for k, v in sr_state.items():
                    sr_state_[k[13:]] = v
                self.sr_predictor.load_state_dict(sr_state_)
                for param in self.sr_predictor.parameters():
                    param.requires_grad = False

            self.avfeat_to_llm = Projector(input_dim=qformer_config.hidden_size,
                                        hidden_dim=math.floor((qformer_config.hidden_size + self.cfg.llama_embed_dim)/2),
                                        output_dim=self.cfg.llama_embed_dim) 
        else:
            self.afeat_1d_conv = nn.Conv1d(in_channels=cfg.whisper_embed_dim, out_channels=cfg.whisper_embed_dim, kernel_size=4, stride=4, padding=0) # 50Hz -> 12.5Hz
            self.vfeat_1d_conv = nn.Conv1d(in_channels=cfg.whisper_embed_dim, out_channels=cfg.whisper_embed_dim, kernel_size=2, stride=2, padding=0) # 25Hz -> 12.5Hz
            self.avfeat_to_llm = Projector(input_dim=self.embed,
                                        hidden_dim=math.floor((self.embed + self.cfg.llama_embed_dim)/2),
                                        output_dim=self.cfg.llama_embed_dim) 
            
            
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.freeze_params = [n for n,p in self.named_parameters() if p.requires_grad == False]
        
    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""
        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        w2v_path = f'{root_dir}/pretrained_models/avhubert/large_vox_iter5.pt'

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                w2v_path, arg_overrides
            )
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                    w2v_args
                )

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )

        w2v_args.task.data = cfg.data

        task_pretrain = tasks.setup_task(w2v_args.task)
        if state is not None:
            task_pretrain.load_state_dict(state['task_state'])

        encoder_ = task_pretrain.build_model(w2v_args.model)

        avhubert = HubertEncoderWrapper(encoder_)
        if state is not None and not cfg.no_pretrained_weights:
            # set strict=False because we omit some modules
            del state['model']['mask_emb']
            avhubert.w2v_model.load_state_dict(state["model"], strict=False)

        avhubert.w2v_model.remove_pretraining_modules()

        whisper_ = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium.en").model.encoder
        whisper = WhisperEncoderWrapper(whisper_)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
        llm = AutoModelForCausalLM.from_pretrained(cfg.llm_path, quantization_config=bnb_config)
        
        target_modules = cfg.target_modules.split('.')
        
        config = LoraConfig(
            r=cfg.lora_rank, 
            lora_alpha=cfg.lora_alpha, 
            target_modules=target_modules, 
            lora_dropout=0.05, 
            bias="none", 
            task_type="CAUSAL_LM" 
        )

        llm = get_peft_model(llm, config)
        llm.print_trainable_parameters()
        
        tokenizer = AutoTokenizer.from_pretrained(cfg.llm_path)
        
        return cls(avhubert, whisper, llm.base_model.model, tokenizer, cfg)
    

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates
        
    def state_dict(self):
        old_state = super().state_dict()
        state = {k:v for k,v in old_state.items() if k not in self.freeze_params}
        return state
    
    def load_state_dict(self,state,**kwargs):
        super().load_state_dict(state, strict=False)   

    def forward(self, **kwargs):
        # ============================
        # 1. Whisper & AVHubert feature extraction (no grad)
        # ============================
        with torch.no_grad():
            # Whisper encoder: B x T x D
            whisper_enc_out = self.whisper(kwargs['source'])

            # Prepare input for AVHubert (audio is None)
            avhubert_source = {'audio': None, 'video': kwargs['source']['video']}
            avhubert_output = self.avhubert(source=avhubert_source, padding_mask=kwargs['padding_mask'])
            # Transpose from (T x B x D) to (B x T x D)
            avhubert_output['encoder_out'] = avhubert_output['encoder_out'].transpose(0, 1)

        video_lengths = torch.sum(~avhubert_output['padding_mask'], dim=1).tolist()
        max_vid_len = max(video_lengths)
        
        # ============================
        # 2. Speech rate predictor and query length calculation
        # ============================
        if self.cfg.use_sr_predictor:
            len_queries, resized_len_list = self.query_length_calculation(whisper_enc_out, video_lengths, max_vid_len)
        else:
            len_queries = [max(int(vid_len / 25 * self.cfg.queries_per_sec), 1) for vid_len in video_lengths]

        # ============================
        # 3. Feature processing and modality fusion
        # ============================
        whisper_enc_out = self.afeat_1d_conv(whisper_enc_out.transpose(1, 2)).transpose(1, 2) # Process Whisper features with 1D conv: (B x T x D) -> (B x T x D')

        if self.cfg.use_qformer:
            padding_mask = (~avhubert_output['padding_mask']).long()
            len_feat = video_lengths 
        else:
            # Without Qformer: downsample the visual feature and corresponding padding mask (e.g., 25Hz -> 12.5Hz)
            padding_mask = avhubert_output['padding_mask'][:, 1::2]
            padding_mask = (~padding_mask).long()
            len_feat = torch.sum(padding_mask, dim=1).tolist()
            avhubert_output['encoder_out'] = self.vfeat_1d_conv(
                avhubert_output['encoder_out'].transpose(1, 2)
            ).transpose(1, 2)

        B, T_v, _ = avhubert_output['encoder_out'].size()
        whisper_enc_out = whisper_enc_out[:, :T_v, :]

        # Fuse modalities based on configuration
        if self.modality_fuse == 'concat':
            av_feat = torch.cat([whisper_enc_out, avhubert_output['encoder_out']], dim=2)
        elif self.modality_fuse == 'add':
            av_feat = whisper_enc_out + avhubert_output['encoder_out']
        elif self.modality_fuse == 'cross-att':
            av_feat = self.multimodal_attention_layer(
                audio_feature=whisper_enc_out,
                visual_feature=avhubert_output['encoder_out']
            )
        else:
            raise ValueError(f"Unknown modality fusion type: {self.modality_fuse}")

        # ============================
        # 4. Prepare inputs for LLM (using Qformer or not)
        # ============================
        instructions = kwargs['source']['instruction']  # List[torch.Tensor] of length B
        labels = kwargs['target_list']                   # List[torch.Tensor] of length B

        if self.cfg.use_qformer:
            query_output = self.compression_using_qformer(len_queries, resized_len_list, len_feat, av_feat)
            # Map Qformer output to LLM embedding space
            query_output = self.avfeat_to_llm(query_output)
            llm_inputs, attention_mask, llm_labels = self.prepare_inputs_labels_for_queries(
                instructions, query_output, len_queries, labels
            )
        else:
            # Directly map fused AV features to LLM embedding space
            av_feat = self.avfeat_to_llm(av_feat)
            llm_inputs, attention_mask, llm_labels = self.prepare_inputs_labels_for_queries(
                instructions, av_feat, len_feat, labels
            )

        # ============================
        # 5. Forward through the LLM and return outputs
        # ============================
        llm_out = self.llama(
            inputs_embeds=llm_inputs,
            attention_mask=attention_mask,
            labels=llm_labels,
            return_dict=True,
            use_cache=False
        )

        loss = llm_out.loss
        logits = llm_out.logits

        return loss, logits, llm_labels


    @torch.no_grad()
    def generate(self,
                num_beams=5,
                temperature=0.3,
                max_length=100,
                min_length=1,
                **kwargs):

        # --------------------------------------
        # 1. Whisper & AVHubert Feature Extraction
        # --------------------------------------

        whisper_enc_out = self.whisper(kwargs['source'])
        avhubert_source = {'audio': None, 'video': kwargs['source']['video']}
        avhubert_output = self.avhubert(source=avhubert_source, padding_mask=kwargs['padding_mask'])
        # AVHubert encoder output: (T x B x D) -> (B x T x D)
        avhubert_output['encoder_out'] = avhubert_output['encoder_out'].transpose(0, 1)

        video_lengths = torch.sum(~avhubert_output['padding_mask'], dim=1).tolist()
        max_vid_len = max(video_lengths)

        if self.cfg.use_sr_predictor:
            len_queries, resized_len_list = self.query_length_calculation(whisper_enc_out, video_lengths, max_vid_len)
        else:
            len_queries = [max(int(x / 25 * self.cfg.queries_per_sec), 1) for x in video_lengths]

        # --------------------------------------
        # 2. Whisper Feature Processing
        # --------------------------------------

        whisper_enc_out = self.afeat_1d_conv(whisper_enc_out.transpose(1, 2)).transpose(1, 2)

        # --------------------------------------
        # 3. AVHubert Feature & Padding Mask Preparation
        # --------------------------------------
        if self.cfg.use_qformer:
            padding_mask = (~avhubert_output['padding_mask']).long()
            len_feat = video_lengths  
        else:
            padding_mask = avhubert_output['padding_mask'][:, 1::2]
            padding_mask = (~padding_mask).long()
            len_feat = torch.sum(padding_mask, dim=1).tolist()
            avhubert_output['encoder_out'] = self.vfeat_1d_conv(
                avhubert_output['encoder_out'].transpose(1, 2)
            ).transpose(1, 2)

        # --------------------------------------
        # 4. Temporal Alignment & Modality Fusion
        # --------------------------------------
        B, T_v, _ = avhubert_output['encoder_out'].size()
        whisper_enc_out = whisper_enc_out[:, :T_v, :]

        if self.modality_fuse == 'concat':
            av_feat = torch.cat([whisper_enc_out, avhubert_output['encoder_out']], dim=2)
        elif self.modality_fuse == 'add':
            av_feat = whisper_enc_out + avhubert_output['encoder_out']
        elif self.modality_fuse == 'cross-att':
            av_feat = self.multimodal_attention_layer(
                audio_feature=whisper_enc_out,
                visual_feature=avhubert_output['encoder_out']
            )
        else:
            raise ValueError(f"Unknown modality fusion type: {self.modality_fuse}")

        # --------------------------------------
        # 5. Prepare inputs for LLM (using Qformer or not)
        # --------------------------------------
        instructions = kwargs['source']['instruction']  # List[torch.Tensor], B

        if self.cfg.use_qformer:
            query_output = self.compression_using_qformer(len_queries, resized_len_list, len_feat, av_feat)
            query_output = self.avfeat_to_llm(query_output)
            llm_inputs, attention_mask, _ = self.prepare_inputs_labels_for_queries(
                instructions, query_output, len_queries
            )
        else:
            # Directly map fused AV features to LLM embedding space
            av_feat = self.avfeat_to_llm(av_feat)
            llm_inputs, attention_mask, _ = self.prepare_inputs_labels_for_queries(
                instructions, av_feat, len_feat
            )

        # --------------------------------------
        # 6. LLM Generation
        # --------------------------------------
        self.llama.generation_config.pad_token_id = self.tokenizer("<|finetune_right_pad_id|>").input_ids[1]

        outputs = self.llama.generate(
            inputs_embeds=llm_inputs,
            attention_mask=attention_mask,
            num_beams=num_beams,
            temperature=temperature,
            max_new_tokens=max_length,
            min_length=min_length,
        )

        return outputs
        
        
    def prepare_inputs_labels_for_queries(self, instructions, queries, len_queries, labels=None):
        llm_input_list = []
        llm_labels_list = []
        lengths = []  

        for i in range(len(instructions)):
            instruction = instructions[i]
            len_query = len_queries[i]
            query = queries[i][:len_query, :]

            inst_emb = self.llama.model.embed_tokens(instruction.unsqueeze(0)).squeeze(0)
            if labels is not None:
                label = labels[i]
                label_emb = self.llama.model.embed_tokens(label.unsqueeze(0)).squeeze(0)
                combined = torch.cat([inst_emb, query, label_emb], dim=0)
            else:
                combined = torch.cat([inst_emb, query], dim=0)

            llm_input_list.append(combined)
            lengths.append(combined.size(0)) 

            if labels is not None:
                label_mask = torch.full((combined.size(0),), -100, dtype=instruction.dtype, device=instruction.device)
                offset = inst_emb.size(0) + query.size(0)
                label_mask[offset:] = label
                llm_labels_list.append(label_mask)

        # Determine the maximum sequence length across the batch
        max_seq_len = max(lengths)
        batch_size = len(llm_input_list)
        embedding_dim = llm_input_list[0].size(1)

        # Prepare the pad embedding (using the provided pad token)
        pad_token_id = self.tokenizer("<|finetune_right_pad_id|>").input_ids[1]
        pad_token_tensor = torch.tensor([pad_token_id], device=instruction.device)
        pad_embedding = self.llama.model.embed_tokens(pad_token_tensor).squeeze(0)

        # Initialize the left-padded inputs tensor with the pad embedding.
        # Each sequence will occupy the rightmost positions.
        llm_inputs = pad_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, max_seq_len, embedding_dim).clone()
        attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=instruction.device)

        for i, seq in enumerate(llm_input_list):
            seq_len = seq.size(0)
            # Place the sequence at the right end, leaving pad tokens on the left
            llm_inputs[i, max_seq_len - seq_len:] = seq
            attention_mask[i, max_seq_len - seq_len:] = 1

        if labels is not None:
            llm_labels = torch.full((batch_size, max_seq_len), -100, dtype=instruction.dtype, device=instruction.device)
            for i, lab in enumerate(llm_labels_list):
                lab_len = lab.size(0)
                llm_labels[i, max_seq_len - lab_len:] = lab
        else:
            llm_labels = None

        return llm_inputs, attention_mask, llm_labels
    
    def query_length_calculation(self, whisper_enc_out, video_lengths, max_vid_len):
        with torch.no_grad():
            sr_predictions = self.sr_predictor(whisper_enc_out[:,:2*max_vid_len,:][:,::4,:])
        len_queries = []
        resized_len_list = []
        for i, vid_len in enumerate(video_lengths):
            base_queries = vid_len / 25 * self.cfg.queries_per_sec
            factor = sr_predictions[i].item()
            # If predicted speech rate is out of acceptable range, use factor 1.0
            if factor < 1: 
                factor = 1
            elif factor > 2:
                factor = 2
            adjusted_queries = int(base_queries * factor)
            query_count = max(adjusted_queries, self.cfg.queries_per_sec)
            len_queries.append(query_count)
            resized_len_list.append(factor*vid_len) # resized av feat

        return len_queries, resized_len_list

    def compression_using_qformer(self, len_queries, resized_len_list, len_feat, av_feat):
        max_length = max(len_queries)
        B = len(len_queries)
        # Create attention mask for query tokens: (B x max_length)
        query_attn_mask = torch.zeros(B, max_length, dtype=torch.long, device=av_feat.device)
        for i, qlen in enumerate(len_queries):
            query_attn_mask[i, :qlen] = 1

        # Expand and slice query tokens: (B x max_length x token_dim)
        query_tokens = self.query_tokens.expand(B, -1, -1)[:, :max_length, :]


        resized_av_feats = torch.zeros(B,int(max(resized_len_list)),av_feat.size(2)).to(av_feat.device).to(av_feat.dtype)
        resized_padding_masks=torch.zeros(B,int(max(resized_len_list))).to(av_feat.device).to(av_feat.dtype)
        # Resize av_feat depend on the factor_list

        for bs,len_feat_bs in enumerate(len_feat): 
            new_av_feat=av_feat[bs][:len_feat_bs].transpose(0, 1).unsqueeze(0) # 1 x D x T
            resized_av_feat = F.interpolate(new_av_feat, size=int(resized_len_list[bs]), mode='linear')
            resized_av_feat=resized_av_feat.squeeze(0).transpose(0,1)
            resized_av_feats[bs,:resized_av_feat.size(0)]=resized_av_feat
            resized_padding_masks[bs,:int(resized_len_list[bs])]=1
            
        av_feat = resized_av_feats
        padding_mask = resized_padding_masks.long()

        # Run Qformer (using its BERT) with cross attention to AV features
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            attention_mask=query_attn_mask,
            encoder_hidden_states=av_feat,
            encoder_attention_mask=padding_mask,
            return_dict=True
        )['last_hidden_state']
    
        return query_output
        
        
def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
