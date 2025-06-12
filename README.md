# MMS-LLaMA: Efficient LLM-based Audio-Visual Speech Recognition with Minimal Multimodal Speech Tokens

This repository contains the PyTorch implementation of the following paper:
> **MMS-LLaMA: Efficient LLM-based Audio-Visual Speech Recognition with Minimal Multimodal Speech Tokens**<be>
><br>
> Authors: Jeong Hun Yeo*, Hyeongseop Rha*, Se Jin Park, Yong Man Ro (*Equal contribution)<br>
> **Paper Link**: [http://arxiv.org/abs/2503.11315](http://arxiv.org/abs/2503.11315)

## Introduction
MMS-LLaMA is an efficient multimodal speech LLM framework, for AVSR that minimizes the length of multimodal speech tokens while preserving their linguistic content.
<div align="center"><img width="90%" src="image.png?raw=true" /></div>



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
pip install transformers==4.47.1 peft bitsandbytes
cd fairseq
pip install --editable ./
```

## Preparation
Before running any training or evaluation, you must update the dataset file paths in the ```tsv files```. These ```tsv files``` contain placeholders (e.g., ```{LRS3_ROOT}```) that need to be replaced with the absolute paths to your local copies of the datasets. The provided script (```update_dataset_paths.py```) automates this process, ensuring that all references in the ```tsv files``` point to the correct locations on your system.

The required datasets are:

* **VoxCeleb2**  
  Download from the [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) website.

* **LRS3**  
  Download from the [LRS3-TED](https://mmai.io/datasets/lip_reading/) dataset page.

Once you have downloaded these datasets, you should pre-process every video clip to crop the mouth regions. You can follow the pre-processing instructions provided in [Auto-AVSR](https://github.com/mpc001/auto_avsr/tree/main/preparation).

Note that for the LRS3 and VoxCeleb2 datasets, the facial landmarks are already provided in the Auto-AVSR repository.

After the pre-processing, update the ```tsv files``` with the absolute paths to the dataset directories using the provided script. This ensures that all dataset references point to the correct locations on your system.


```bash
python update_dataset_paths.py --input_dir ./ --vox2 'path for the VoxCeleb2 dataset' --lrs3 'path for the LRS3 dataset' 
```

For example:
```bash
python update_dataset_paths.py --input_dir ./ --vox2 /Dataset/vox2 --lrs3 /Dataset/lrs3
```

The above command updates the placeholder paths in the ```tsv files``` to your absolute dataset paths.

Each ```tsv files``` contains one line per data sample, with the following fields separated by a tab ```(\t)```:

* **used dataset**
* **video_path**
* **audio_path**
* **num_video_frames**
* **num_audio_frames**    
* **speech_rate**

Below are the expected directory structures for LRS3 dataset:

### LRS3
```
lrs3/
├── lrs3_video_seg24s/              
│   ├── pretrain/
│   ├── test/
│   ├── trainval/            
└── lrs3_text_seg24s/
    ├── pretrain/
    ├── test/
    └── trainval/    
```


## Training
### Train a new MMS-LLaMA

```bash
bash scripts/train.sh
```

**Note**: For training on an 8 GPU RTX 3090 setup, the `433h model` can be trained in approximately `6 hours`, while the `1759h model` requires about `20 hours`.


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
- Place the `433h checkpoint` in the `pretrained_models/mms-llama/433h` folder.
- Place the `1759h checkpoint` in the `pretrained_models/mms-llama/1759h` folder.
- Place the `speech rate predictor` checkpoint in the `pretrained_models/sr_predictor` folder.

> ```MMS-LLaMA```

| Model         | Used Datasets  | Training data (# hours)   | WER(\%), Clean  | WER(\%), Noisy | 
|--------------|:----------:|:------------------:|:------------------:|:------------------:|
| [ckpt.pt](https://www.dropbox.com/scl/fi/uiaxa2lgjze4mt7tdi5wu/checkpoint_best.pt?rlkey=o62sc6ann8xm3gpkyj4yk3rwe&st=s5q385op&dl=1) |       LRS3       |       433       |       0.90       |      2.4       |
| [ckpt.pt](https://www.dropbox.com/scl/fi/ou28xe2k9ampxsf4ihoft/checkpoint_best.pt?rlkey=a4q1qgigodhrgwqi9lgsalj7f&st=ga8z79vc&dl=1) |       LRS3, VoxCeleb2       |       1759       |       0.72       | 1.9   |

You can download the pre-trained models using wget with the following command:

```bash
# 433h model
wget -O ckpt.pt "https://www.dropbox.com/scl/fi/uiaxa2lgjze4mt7tdi5wu/checkpoint_best.pt?rlkey=o62sc6ann8xm3gpkyj4yk3rwe&st=s5q385op&dl=1"
```

```bash
# 1759h model
wget -O ckpt.pt "https://www.dropbox.com/scl/fi/ou28xe2k9ampxsf4ihoft/checkpoint_best.pt?rlkey=a4q1qgigodhrgwqi9lgsalj7f&st=ga8z79vc&dl=1"
```



> ```Speech Rate Predictor```

| Model         | Used Datasets  | Training data (# hours)   |
|--------------|:----------:|:------------------:|
| [ckpt.pt](https://www.dropbox.com/scl/fi/rc6jmbzdvj8afn84z47qt/checkpoint.pt?rlkey=aoa0ifkdydgm9gjmt2ljwpgrc&st=we9qoqtb&dl=1) |       LRS3       |       433       |

```bash
wget -O ckpt.pt "https://www.dropbox.com/scl/fi/rc6jmbzdvj8afn84z47qt/checkpoint.pt?rlkey=aoa0ifkdydgm9gjmt2ljwpgrc&st=we9qoqtb&dl=1"
```

## Citation
If you find this work useful in your research, please cite the paper:


```bibtex
@article{yeo2025mms,
  title={MMS-LLaMA: Efficient LLM-based Audio-Visual Speech Recognition with Minimal Multimodal Speech Tokens},
  author={Yeo, Jeong Hun and Rha, Hyeongseop and Park, Se Jin and Ro, Yong Man},
  journal={arXiv preprint arXiv:2503.11315},
  year={2025}
}
```


## Acknowledgement
This project is based on the [avhubert](https://github.com/facebookresearch/av_hubert), [auto-avsr](https://github.com/mpc001/auto_avsr), and [fairseq](https://github.com/facebookresearch/fairseq) code. We would like to thank the developers of these projects for their contributions and the open-source community for making this work possible.

