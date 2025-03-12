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

from .Qformer import BertConfig, BertLMHeadModel
from fairseq.models.wav2vec.wav2vec2 import TransformerEncoder
from types import SimpleNamespace

class WhisperEncoderWrapper(FairseqEncoder):
    def __init__(self, whisper):
        super().__init__(None)
        self.whisper = whisper
    
    def forward(self, source):
        whisper_input = source['audio']
        whisper_enc_out = self.whisper(whisper_input).last_hidden_state # B, T, D


        return whisper_enc_out
         
    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out[
                "encoder_out"
            ].index_select(1, new_order)
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out[
                "padding_mask"
            ].index_select(0, new_order)
        return encoder_out
     
class Projector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Projector, self).__init__()
        # create a list of layers
        self.layers = nn.ModuleList([
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.Linear(in_features=hidden_dim, out_features=output_dim)
        ])
    
    def forward(self, x):
        # iterate through all the layers
        for layer in self.layers:
            x = layer(x)
        return x         
          
class Multimodal_Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Multimodal_Attention, self).__init__()
        # create a list of layers
        self.mha0 = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.mha1 = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
    
    def forward(self, audio_feature, visual_feature):
        # iterate through all the layers
        x, _ = self.mha0(query=visual_feature, key=audio_feature, value=audio_feature) # T B D
        x = x + audio_feature
        x = self.layer_norm(x)
        x2, _ = self.mha1(query=visual_feature, key=audio_feature, value=audio_feature) # T B D
        x2 = x + x2

        return x2       
           
           
def get_sinusoidal_positional_encoding(seq_len, d_model, device=None):
    """
    seq_len:
    d_model: 
    device: 
    """
    device = device or torch.device("cpu")
    pe = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # shape: (seq_len, d_model)
           
           
class Speech_Rate_Predictor(nn.Module):
    def __init__(self, num_layers):
        super(Speech_Rate_Predictor, self).__init__()
        args_dict = {
            'dropout': 0.0,
            'encoder_embed_dim': 256,
            'conv_pos': 128,
            'conv_pos_groups': 16,
            'encoder_ffn_embed_dim': 1024,
            'encoder_attention_heads': 4,
            'attention_dropout': 0.0,
            'activation_dropout': 0.1,
            'activation_fn': 'gelu',
            'layer_norm_first': True,
            'encoder_layers': num_layers,
            'encoder_layerdrop': 0.1,
        }

        args = SimpleNamespace(**args_dict)
        self.sr_token = nn.Parameter(torch.zeros(1, 1, 256))
        nn.init.xavier_uniform_(self.sr_token)
        self.linear=nn.Linear(1024,256)
        self.encoder = TransformerEncoder(args)
        self.sr_predictor = nn.Linear(256, 1) 
        self.activation = nn.ReLU()  

    def forward(self, x):
        x = self.linear(x)
        batch_size = x.size(0)
        sr_token_expanded = self.sr_token.expand(batch_size, -1, -1)
        x = torch.cat([sr_token_expanded, x], dim=1) 
        x, _ = self.encoder(x)
        sr_prediction = self.activation(self.sr_predictor(x[:, 0, :]))
        
        return sr_prediction