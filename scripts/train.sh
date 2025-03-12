#! /bin/bash
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

ROOT=$(pwd)
SRC_PTH=$ROOT/src
LLM_PATH=meta-llama/Llama-3.2-3B
NGPUS=8
QPS=3 
OUT_PATH=$ROOT/exp/mms-llama/433h

export TOKENIZERS_PARALLELISM=false
PYTHONPATH=$ROOT/fairseq \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fairseq-hydra-train \
    --config-dir ${SRC_PTH}/conf/ \
    --config-name mms-llama.yaml \
    task.data=$ROOT/manifest/433h \
    task.label_dir=$ROOT/manifest/433h \
    task.llm_path=${LLM_PATH} \
    task.noise_prob=0.75 \
    task.noise_wav=$ROOT/noise/babble_noise.wav \
    hydra.run.dir=${OUT_PATH} \
    common.user_dir=${SRC_PTH} \
    common.seed=1 \
    common.empty_cache_freq=1000 \
    model.w2v_path=$ROOT/pretrained_models/avhubert/large_vox_iter5.pt \
    model.llm_path=${LLM_PATH} \
    model.llama_embed_dim=3072 \
    model.queries_per_sec=$QPS \
    model.target_modules=q_proj.k_proj.v_proj.o_proj \
    model.modality_fuse=concat \
    model.lora_rank=16 \
    model.lora_alpha=32 \
    model.use_qformer=true \
    model.use_sr_predictor=true \
    optimization.update_freq=[1] \
    optimization.lr=[1e-4] \
    optimization.max_update=30000 \
    lr_scheduler._name=cosine \
    lr_scheduler.warmup_updates=500 \
    distributed_training.distributed_world_size=${NGPUS} \
    distributed_training.nprocs_per_node=${NGPUS} \
    distributed_training.ddp_backend=legacy_ddp \
    distributed_training.find_unused_parameters=true