<h1 align="center"> I3CL: Intra- and Inter-Instance Collaborative Learning for Arbitrary-shaped Scene Text Detection </h1> 

<p align="center">
<a href="http://arxiv.org"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>
</p>

<p align="center">
  <a href="#introduction">Introduction</a> |
  <a href="#updates">Updates</a> |
  <a href="#todo">TODO</a> |
  <a href="#results">Results</a> |
  <a href="#usage">Usage</a> |
  <a href="#statement">Statement</a> |
</p >

## Introduction

This is the repo for the paper "I3CL: Intra- and Inter-Instance Collaborative Learning for Arbitrary-shaped Scene Text Detection". 

//structure img

*Note that I3CL model is equiped with the ViTAE-v2 backbone in this repo.*

## Updates

***09/04/2022***

The code is uploaded. Private github repo temporarily.

## TODO

- [ ] Update the SSL result with ViTAE backbone on ArT.
- [ ] Fix the bug that fails to train using more than one image per GPU.

## Results

//to be updated

The results are tested in single scale setting. The shorter side is resized to 1185. I3CL with ViTAE backbone gets promising result without pretraining compared to ResNet-50 counterpart.

## Usage

### Install

//to be updated

### Prepare Data

//to be updated

### Training

//to be updated

### Inference

Get inference results on ICDAR2019 ArT test set with visualization images, txt format records and the json file for testing submission.

```
python demo/art_demo.py --config configs/i3cl_vitae_fpn/i3cl_vitae_fpn_ms_train.py \
--checkpoint pretrained_model/I3CL/epoch_12.pth --score-thr 0.45 --json_file art_test.json
```

*Note: Upload the saved json file to [Link](https://rrc.cvc.uab.es/?ch=14&com=mymethods&task=1) for Recall, Precision and F1 evaluation results. Change the path for saving visualizations and txt files if needed.*

## Statement

This project is for research purpose only.

If you are interested in our work, please consider citing the following:

//citation to be updated

For further questions, please contact ##

## References

Thanks for [mmdetection](https://github.com/open-mmlab/mmdetection).
