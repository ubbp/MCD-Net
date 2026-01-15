# MCD-Net
<<<<<<< HEAD
This repository is the offical implementation for "A Multiscale Vision-Text Collaborative Dual-Encoder for Referring RS Image Segmentation."[[IEEE TGRS](https://ieeexplore.ieee.org/document/10816052)] 

## Setting Up
### Preliminaries
The code has been verified to work with PyTorch v1.12.1 and Python 3.7.
1. Clone this repository.
2. Change directory to root of this repository.
### Package Dependencies
1. Create a new Conda environment with Python 3.7 then activate it:
```shell
conda create -n MCDNet python==3.7
conda activate MCDNet
```

2. Install PyTorch v1.12.1 with a CUDA version that works on your cluster/machine (CUDA 10.2 is used in this example):
```shell
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
```

3. Install the packages in `requirements.txt` via `pip`:
```shell
pip install -r requirements.txt
```
### The Initialization Weights for Training
1. Create the `./pretrained_weights` directory where we will be storing the weights.
```shell
mkdir ./pretrained_weights
```
2. Download [pre-trained classification weights of
the Swin Transformer](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth),
and put the `pth` file in `./weights`.
These weights are needed for training to initialize the visual encoder.
3. Download [BERT weights from HuggingFace’s Transformer library](https://huggingface.co/google-bert/bert-base-uncased), 
and put it in the root directory. 
4. Download [efficient SAM], and put the `pt` file in `./weights`.

## Datasets
We perform the experiments on two dataset including [RefSegRS](https://github.com/zhu-xlab/rrsis) and [RRSIS-D](https://github.com/Lsan2401/RMSIN). 

## Training
We use one GPU to train our model. 
For training:
```shell
sh train.sh
```


## Testing
For testing:
```shell
sh test.sh
```


## Citation
If you find this code useful for your research, please cite our paper:

