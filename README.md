# NCTU_VRDL_HW1
NCTU Visual Recognition using Deep Learning HW1

## Hardware
The following specs were used to create the original solution.

Ubuntu 18.04.3 LTS
Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz
3x GeForce RTX 2080 TI

## Reproducing Submission
To reproduct my submission without retrainig, do the following steps:
1. [Installation](#installation)
2. [Dataset Preparation](#Dataset Preparation)
3. [Training](#Training) detail.
4. [Testing](#Testing) detail.
5. [Reference](#Reference)

## Installation

this code was trained and tested with Python 3.6.10 and Pytorch 1.2.0 (Torchvision 0.4.0) on Ubuntu 18.04

```
conda create -n hpa python=3.6
conda activate hpa
pip install -r requirements.txt
```
and for the optimizer, I used [Ranger](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer). The installation detail is in the link.

## Dataset Preparation
All required files except images are already in cs-t0828-2020-hw1 directory.
```
cs-t0828-2020-hw1
  +- training_data
  |  +-training_data
  +- testing_data
  |  +- testing_data
  +- training_labels.csv
```
I seperate the original training data (11185 images) into two part. One for training (7485 images) and one for evaluating(3700 images). 
The order of the training and evaluating data is correspond to the training_labels csv file.

## Training
To train models, run following commands.
```
$ python main.py
```
After that the terminal will request you to input which mode do you want to choose.
Just type "train".
you should also type which network you want to choose.
0 for ResNext50
1 for ResNet50
```
Current mode: train
Net: 0
```
The expected training times are:
Model | GPUs | Image size | Training Epochs | Training Time
------------ | ------------- | ------------- | ------------- | -------------
ResNext50 | 3x RTX 2080Ti | 400 x 400 | 10 | 30 minutes
ResNet50 | 3x RTX 2080Ti | 400 x 400 | 10 | 30 minutes

## Testing
To train models, run following commands.
```
$ python main.py
```
After that the terminal will request you to input which mode do you want to choose.
Just type "test".
you should also type which network you want to choose.
0 for ResNext50
1 for ResNet50
Finally input the file path of the [checkpoint](https://drive.google.com/drive/u/1/folders/1CpQYyLGR_bD8CZfEU9ch3Z7ZL8IlMAO7). The checkpoint of ResNext50 is in the link.
```
Current mode: test
Net: 0
weight: ./checkpoint/best_result.pkl
```
After testing the result csv file will be generate in the data folder.

## Reference
1. [Ranger](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer).
2. [Car Model Classification](https://github.com/kamwoh/Car-Model-Classification)


