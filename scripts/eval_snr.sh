#! /bin/bash
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


ROOT=$(pwd)
SRC_PTH=$ROOT/src
MODEL_PATH=$ROOT/pretrained_models/mms_llama/1759h/checkpoint_best.pt

LLM_PATH=meta-llama/Llama-3.2-3B
SNR=0

OUT_PATH=$ROOT/results/

PYTHONPATH=$ROOT/fairseq \
CUDA_VISIBLE_DEVICES=0 python -B $SRC_PTH/eval.py --config-dir ${SRC_PTH}/conf --config-name s2s_decode \
    dataset.gen_subset=test \
    common.user_dir=${SRC_PTH} \
    generation.beam=5 \
    generation.temperature=0.3 \
    override.llm_path=${LLM_PATH} \
    override.modalities=['video','audio'] \
    common_eval.path=${MODEL_PATH} \
    common_eval.results_path=${OUT_PATH} \
    override.label_dir=$ROOT/manifest/433h \
    override.data=$ROOT/manifest/433h \
    override.noise_wav=$ROOT/noise/babble_noise.wav \
    override.noise_prob=1 \
    override.noise_snr=${SNR}

