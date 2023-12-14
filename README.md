# FRN

The official implementation for "Embracing Events and Frames with Hierarchical Feature Refinement Network for Robust Object Detection".

![output](https://github.com/HuCaoFighting/FRN/assets/66437581/63188281-6f24-4944-869f-029e4ac26bed)
![output_evt](https://github.com/HuCaoFighting/FRN/assets/66437581/f8e54dda-c623-4fda-91af-012fe24c22fe)

## Setup
- Setup python environment

This code has been tested with Python 3.8, Pytorch 2.0.1, and on Ubuntu 20.04

We recommend you to use Anaconda to create a conda environment:

```
conda create -n env python=3.8
conda activate env
```
- Install pytorch

```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

## Usage 
### Training

```
python train_dsec.py
```
### Evaluation
You can download our pretrained weights below and modify the path of checkpoint in test.py file.
```
python test.py
```

## Pre-trained Weights

Our pre-trained weights can be downloaded [here](https://drive.google.com/file/d/1g_AwWsOJHljpQYIpaeAN8YvYWh0pouaV/view?usp=sharing)

