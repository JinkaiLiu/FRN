# FRN

The official implementation for "Embracing Events and Frames with Hierarchical Feature Refinement Network for Robust Object Detection".

![output](https://github.com/HuCaoFighting/FRN/assets/66437581/63188281-6f24-4944-869f-029e4ac26bed)
![output_evt](https://github.com/HuCaoFighting/FRN/assets/66437581/f8e54dda-c623-4fda-91af-012fe24c22fe)

# Abstract
This work addresses the major challenges in object detection for autonomous driving, particularly under demanding conditions such as motion blur, adverse weather, and image noise. Recognizing the limitations of traditional camera systems in these scenarios, this work focuses on leveraging the unique attributes of event cameras, such as their low latency and high dynamic range. These attributes offer promising solutions to complement and augment the capabilities of standard RGB cameras. To leverage these benefits, this work introduces a novel RGB-Event network architecture with a unique fusion module. This module effectively utilizes information from both RGB and event modalities, integrating attention mechanisms and AdaIN (Adaptive Instance Normalization) for enhanced performance. The effectiveness of this approach is validated using two datasets: DSEC and PKU-DDD17-Car, with additional image corruption tests to assess robustness. Results demonstrate that the proposed method significantly outperforms existing state-of-the-art RGB-Event fusion alternatives in both datasets and shows remarkable stability under various image corruption scenarios.

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

## Dataset 
Labelled DSEC Dataset: [dsec](https://github.com/abhishek1411/event-rgb-fusion)

Download dataset: [PKU-DDD17-CAR](https://www.pkuml.org/resources/pku-ddd17-car.html)



## Usage 
### Training with DSEC or PKU-DDD17 Dataset

```
python train_dsec.py
python train_ddd17.py
```
### Evaluation
You can download our pretrained weights below and modify the path of checkpoint in test_dsec.py file.
```
python test_dsec.py
```
For PKU-DDD17 Dataset
```
python test_ddd17.py
```

## Pre-trained Weights

Our pre-trained weights can be downloaded [dsec](https://drive.google.com/file/d/1g_AwWsOJHljpQYIpaeAN8YvYWh0pouaV/view?usp=sharing) and [ddd17](https://drive.google.com/file/d/1DvmZNCQeHjORzoplOYho7GifkpeOdNZ2/view?usp=sharing)

## Results COCO mAP@0.50.95 on DSEC 

| Method             | AP(car) | AP(person) | AP(largevehicle) | mAP@0.50.95 |
| ------------------ | ------- | ---------- | ---------------- | ----------- |
| FPN-fusion(RetinaNet) | 0.375   | 0.109      | 0.249            | 0.244       |
| DCF                | 0.363   | 0.127      | 0.280            | 0.257       |
| SAGate             | 0.325   | 0.104      | 0.16            | 0.196       |
| Self-Attention     | 0.386   | 0.151      | 0.306            | 0.281       |
| ECANet             | 0.367    | 0.128      | 0.275            | 0.257       |
| EFNet              | 0.411   | 0.158      | 0.326            | 0.3       |
| SPNet              | 0.392   | 0.178      | 0.262            | 0.277       |
| SENet              | 0.384   | 0.149      | 0.26            | 0.262       |
| CBAM               | 0.377   | 0.135      | 0.270            | 0.261       |
| CMX               | 0.416   | 0.164      | 0.294            | 0.291       |
| RAM               | 0.244   | 0.108      | 0.176            | 0.176       |
| FAGC               | 0.398  | 0.144      | 0.336           | 0.293      |
| BDC               | 0.405   | 0.172      | 0.306            | 0.294       |
| Ours               | **0.499**   | **0.258**      | **0.382**            | **0.380**       |

## Results COCO mAP@0.50.95 on DDD17
| Method                    | Test(all) mAP@0.50.95 | Test(day) mAP@0.50.95 | Test(night) mAP@0.50.95 | Test(all) mAP@0.50 | Test(day) mAP@0.50 | Test(night) mAP@0.50 |
| ------------------------- | ------------ | --------- | ----------- |----------------------- | ------------------ | ------------------ | 
| OnlyRGB                   | 0.427          | 0.433      | 0.406          | 0.827         | 0.829        | 0.825          |
| OnlyEvent                 | 0.215        | 0.214     | 0.243       |0.465        | 0.436     | 0.600       |
| FPN-fusion(RetinaNet) | 0.416        | 0.432     | 0.357       |0.819       | 0.828    | 0.789      |
| DCF                   | 0.425        | 0.434     | 0.39        |0.834        | 0.842     | 0.804       |
| SAGate                | 0.434        | 0.449     | 0.38        |0.820        | 0.825     | 0.804       |
| Self-Attention        | 0.424        | 0.433     | 0.388       |0.826        | 0.834    | 0.811       |
| ECANet                | 0.408        | 0.422     | 0.361       |0.822        | 0.831     | 0.790      |
| EFNet                 | 0.416        | 0.434     | 0.351       |0.830        | 0.844     | 0.787       |
| SPNet                 | 0.433        | 0.449     | 0.371       |0.847        | 0.861     | 0.789      |
| CBAM                  | 0.428        | 0.442     | 0.38        |0.819        | 0.823     | 0.810    |
| SENet                 | 0.424        | 0.437     | 0.370       |0.816      | 0.827     | 0.774       |
| CMX                 | 0.390        | 0.402     | 0.354       |0.804      | 0.807     | 0.796       |
| RAM                 | 0.388        | 0.392     | 0.369       |0.796      | 0.799     | 0.782       |
| FAGC                   | 0.436        | 0.448    | 0.395     |0.852     | 0.859     | 0.826      |
| BDC                 | 0.439        | 0.454     | 0.391       |0.814      | 0.819     | 0.804       |
| Ours                | **0.460**        | **0.469**     | **0.421**       | **0.867**  | **0.869**     | **0.861**       |


## Acknowledgements
The retinanet based sensor fusion model presented here builds upon this [implementation](https://github.com/abhishek1411/event-rgb-fusion/)
