<h1 align="center"> I3CL: Intra- and Inter-Instance Collaborative Learning for Arbitrary-shaped Scene Text Detection </h1> 

<p align="center">
<a href="http://arxiv.org"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>
</p>

<p align="center">
  <a href="#updates">Updates</a> |
  <a href="#introduction">Introduction</a> |
  <a href="#results">Results</a> |
  <a href="#usage">Usage</a> |
  <a href="#statement">Statement</a> |
</p >

## Updates

***09/04/2022***

The code will be uploaded.

## Introduction

This is the repo for the paper "I3CL: Intra- and Inter-Instance Collaborative Learning for Arbitrary-shaped Scene Text Detection". 

//structure img

Note that I3CL model is equiped with the ViTAE-v2 backbone and implemented in mmdetection codebase in this repo.

## Results

The result below is tested in single scale setting. The shorter side is resized to 1185. I3CL with ViTAE backbone gets promising result without pretraining compared to ResNet-50 counterpart.

//to be updated

## Usage

### Install

//

### Training

//

### Inference

Get inference results on ICDAR2019 ArT test set with visualization images, txt format records and the json file for testing submission.

```
python demo/art_demo.py --config configs/i3cl_vitae_fpn/i3cl_vitae_fpn_ms_train.py --checkpoint out_dir/epoch_12.pth --score-thr 0.45 --json_file art_1.json
```

*Note: Change the path for saving visualizations and txt files if needed.

## Statement

This project is for research purpose only.

If you are interested in our work, please consider citing the following:

//citation to be updated

For further questions, please contact ##
