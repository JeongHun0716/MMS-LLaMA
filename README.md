# MMS-LLaMA: Efficient LLM-based Audio-Visual Speech Recognition with Minimal Multimodal Speech Tokens

This repository contains the PyTorch implementation of the following paper:
> **MMS-LLaMA: Efficient LLM-based Audio-Visual Speech Recognition with Minimal Multimodal Speech Tokens**<be>
><br>
> Authors: Jeong Hun Yeo*, Hyeongseop Rha*, Se Jin Park, Yong Man Ro (*Equal contribution)<br>
> **Paper Link**: [Soon Available]

## Introduction
MMS-LLaMA is an efficient multimodal speech LLM framework, for AVSR that minimizes the length of multimodal speech tokens while preserving their linguistic content.




## Environment Setup
```bash
conda create -n mms-llama python=3.9 -y
conda activate mms-llama
git clone https://github.com/JeongHun0716/MMS-LLaMA
cd MMS-LLaMA
```
```bash
# PyTorch and related packages
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.23.5 scipy opencv-python
pip install editdistance python_speech_features einops soundfile sentencepiece tqdm tensorboard unidecode librosa
pip install omegaconf==2.0.6 hydra-core==1.0.7 (If your pip version > 24.1, please run "python3 -m pip install --upgrade pip==24.0")
pip install transformers peft bitsandbytes
cd fairseq
pip install --editable ./
```

## Preparation
We propose Mulitlingual Audio-Visual Romanized Corpus (MARC), the Roman transcription labels for 2,916 hours of audiovisual speech data across 82 languages.

All manifests files for training and evaluation are available for download from [this link](https://www.dropbox.com/scl/fi/11ol40wf9p03vedni2zmh/manifests.tar.gz?rlkey=37rz25mklrkoeqfcszyvpkf12&st=jqkjfeg1&dl=0).

Download the manifests.tar.gz file into the marc folder and extract it, and then please run: 
```tar -xzvf manifests.tar.gz```

This will result in the following directory structure:

```
marc/
├── manifest/              
│   ├── stage1/            # Files for av-romanizer training
│   │   ├── all/           # All training/test files for stage1
│   │   └── zero_shot/     # Zero-shot experiment files for stage1
│   └── stage2/            # Files for zero-avsr training and evaluation
└── update_dataset_paths.py   # Script to update placeholders to absolute paths in tsv files
└── avspeech_train_segments.txt   # Metadata file for AVSpeech training segments
```
More detailed information is provided in [marc](https://github.com/JeongHun0716/zero-avsr/tree/main/marc)





## Training
### Train a new MMS-LLaMA

```bash
bash scripts/train.sh
```

### Evaluation of the MMS-LLaMA
To evaluate the performance of MMS-LLaMA, execute the evaluation script by running:

```bash
bash scripts/eval.sh
```

### Evaluation of the MMS-LLaMA, under noisy environment
To evaluate the performance of MMS-LLaMA in a noisy environment, run the evaluation script using:

```bash
bash scripts/eval_snr.sh
```


## Pretrained Models
1. Download the ```AV-HuBERT Large model``` from this [link](https://github.com/facebookresearch/av_hubert) 
2. Download the ```Whisper-medium.en``` from this [link](https://huggingface.co/openai/whisper-medium.en) 
3. Download the ```LLaMA-3.2 3B model``` from this [link](https://huggingface.co/meta-llama/Llama-3.2-3B)

After downloading, make sure to place the models in the correct directories:
- The `large_vox_iter5.pt(AV-HuBERT)` model should be placed in the `pretrained_models/avhubert` folder.

> ```MMS-LLaMA```

| Model         | Used Datasets  | Training data (# hours)   | WER(\%), Clean  | WER(\%), Noisy | 
|--------------|:----------:|:------------------:|:------------------:|:------------------:|
| [ckpt.pt](https://www.dropbox.com/scl/fi/mph3q2rtgwcb0sn9na47g/checkpoint_best.pt?rlkey=r6t9l2l27cmtgj10yt1uio5mi&st=erysg487&dl=0) |       LRS3       |       433       |       0.92       |      2.8       |
| [ckpt.pt](https://www.dropbox.com/scl/fi/mph3q2rtgwcb0sn9na47g/checkpoint_best.pt?rlkey=r6t9l2l27cmtgj10yt1uio5mi&st=erysg487&dl=0) |       LRS3, VoxCeleb2       |       1759       |       0.74       | 1.9   |

> ```Speech Rate Predictor```

| Model         | Used Datasets  | Training data (# hours)   |
|--------------|:----------:|:------------------:|
| [ckpt.pt](https://www.dropbox.com/scl/fi/mph3q2rtgwcb0sn9na47g/checkpoint_best.pt?rlkey=r6t9l2l27cmtgj10yt1uio5mi&st=erysg487&dl=0) |       LRS3       |       433       |



