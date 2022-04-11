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

*This is the repo for the paper "I3CL: Intra- and Inter-Instance Collaborative Learning for Arbitrary-shaped Scene Text Detection". 
Note that I3CL is equiped with the ViTAE(20M params.) backbone in this repo.*

## Introduction

Existing methods for arbitrary-shaped text detection in natural scenes face two critical issues, i.e., 1) 
fracture detections at the gaps in a text instance; and 2) inaccurate detections of arbitrary-shaped text
instances with diverse background context. To address these issues, we propose a novel method named 
Intra- and Inter-Instance Collaborative Learning (I3CL). Specifically, to address the first issue, we design 
an effective convolutional module with multiple receptive fields, which is able to collaboratively learn 
better character and gap feature representations at local and long ranges inside a text instance. To address 
the second issue, we devise an instance-based transformer module to exploit the dependencies between different 
text instances and a global context module to exploit the semantic context from the shared background, which 
are able to collaboratively learn more discriminative text feature representation. In this way, I3CL can 
effectively exploit the intra- and inter-instance dependencies together in a unified end-to-end trainable 
framework. Besides, to make full use of the unlabeled data, we design an effective semi-supervised learning 
method to leverage the pseudo labels via an ensemble strategy. Without bells and whistles, experimental results 
show that the proposed I3CL sets new state-of-the-art results on three challenging public benchmarks, i.e., 
an F-measure of 77.5% on ArT, 86.9% on Total-Text, and 86.4% on CTW-1500. Notably, our I3CL with the ResNeSt-101 
backbone ranked the 1st place on the ArT leaderboard.

![image](./I3CL.jpg)

## Updates

> ***[2022/04/11]*** Add SSL part for this implementation.
>
>***[2022/04/09]*** The training code for ICDAR2019 ArT dataset is uploaded. Private github repo temporarily.

## Results

Example results from paper.

![image](./demo.jpg)

Evaluation results of I3CL with different backbone on ArT.

This implementation:

|Backbone|Model Link|Training Data|Recall|Precision|F-measure|
|:------:|:------:|:------:|:------:|:------:|:------:|
|<p>ViTAE<br>[this implementation]</p>|[Link(to be updated)]()|LSVT+MLT19+ArT|75.42|82.82|78.95|
|<p>ViTAE+SSL<br>[this implementation]</p>|[Link(to be updated)]()|LSVT+MLT19+ArT+SSL data|-|-|to be updated|

ResNet-series in paper:

|Backbone|Training Data|Recall|Precision|F-measure|
|:------:|:------:|:------:|:------:|:------:|
|<p>ResNet-50<br>[paper]</p>|SynthText+ReCTS+RCTW+LSVT+MLT19+ArT|71.3|82.7|76.6|
|<p>ResNet-50+SSL<br>[paper]</p>|SynthText+ReCTS+RCTW+LSVT+MLT19+ArT+SSL data|72.2|83.6|77.5|
|<p>ResNeXt-101+SSL<br>[paper]</p>|SynthText+ReCTS+RCTW+LSVT+MLT19+ArT+SSL data|74.1|85.5|79.4|
|<p>ResNeSt-101+SSL<br>[paper]</p>|SynthText+ReCTS+RCTW+LSVT+MLT19+ArT+SSL data|75.1|86.3|80.3|
|<p>ResNeXt-151+SSL<br>[paper]</p>|SynthText+ReCTS+RCTW+LSVT+MLT19+ArT+SSL data|74.9|86.0|80.1|

The result of ViTAE is tested in single scale setting without any testing trick. The shorter side is resized to 1185. The submission json file to evaluation website can be downloaded
here ([Link_to_be_updated]()). 

## Usage

### Install

>**Prerequisites：**
>
>- Linux (macOS and Windows are not tested)
>- Python >= 3.6
>- Pytorch >= 1.8.1 (For ViTAE implementation). Please make sure your compilation CUDA version and runtime CUDA version match.
>- GCC >= 5
>- MMCV (We use mmcv-full==1.4.3)

1. Create a conda virtual environment and activate it. Note that This implementation is based on mmdetection 2.20.0.

2. Install Pytorch and torchvision following [official instructions](https://pytorch.org).

3. Install mmcv-full and timm. Please refer to [mmcv](https://github.com/open-mmlab/mmcv) to install the proper version. For example:
    
    ```
    pip install mmcv-full==1.4.3 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
    pip install timm
    ```

4. Clone this repository and them install it:
    
    ```
    git clone https://github.com/ViTAE-Transformer/ViTAE-Transformer-Scene-Text-Detection.git
    cd ViTAE-Transformer-Scene-Text-Detection
    pip install -r requirements/build.txt
    pip install -r requirements/runtime.txt
    pip install -v -e .

### Preparation

**Model:** 

- To train I3CL model yourself, please download the pretrained ViTAE(20M) used in this implementation from here ([Link to be updated]()) 
for backbone initialization. Please put it in [pretrained_model/ViTAE](./pretrained_model/ViTAE).
- Full I3CL model can be downloaded here ([Link to be updated]()). You can put it in [pretrained_model/I3CL](./pretrained_model/I3CL).

**Data**

- We use coco format training datasets. Some offline augmented ArT training datasets are used. You can download 
correspoding datasets in config file from here and put them in [data/](./data):

    |Dataset|<p>Link<br>(Google Drive)</p>|<p>Link<br>(Baidu Wangpan百度网盘)</p>|
    |:------:|:------:|:------:|
    |art|[Link]()|[Link]() (pw: )|
    |art_light|[Link]()|[Link]() (pw: )|
    |art_noise|[Link]()|[Link]() (pw: )|
    |art_sig|[Link]()|[Link]() (pw: )|
    |lsvt|[Link]()|[Link]() (pw: )|
    |lsvt_test|[Link]()|[Link]() (pw: )|
    |icdar2019_mlt|[Link]()|[Link]() (pw: )|

    The file structure should look like:
    ```
    |-- data
        |-- art
        |   |-- train_images
                |-- *.jpg
        |   |-- test_images
        |   |-- train.json
        |   |-- train_lossweight.json
        |-- art_light
        |   |-- train_images
        |   |-- train.json
        |   |-- train_lossweight.json
        ......
        |-- lsvt
        |   |-- train_images1
        |   |-- train_images2
        |   |-- train1.json
        |   |-- train1_lossweight.json
        |   |-- train2.json
        |   |-- train2_lossweight.json
        |-- lsvt_test
        |   |-- train_images
        |   |-- train_lossweight.json
        ......

### Training

Distributed training with 4GPUs:
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 tools/train.py \
configs/i3cl_vitae_fpn_ms_train.py --launcher pytorch --work-dir ./out_dir/${your_dir}
```
*Note:*

- *If you want to train the model with SSL, please: (1) set `with_ssl=True` in [configuration file(line9)](./configs/i3cl_vitae_fpn/coco_instance.py). 
(2) use the provided SSL data below annotation in [configuration file(line107)](./configs/i3cl_vitae_fpn/coco_instance.py), such as `art_lossweight` in 
`*_lossweight` format. Or you can generate data yourself according to the Pseudo-label Generation algorithm in paper.*

- *Due to Transformer-based backbone, if the GPU memory is limited, please adjust `img_scale` in 
[configuration file](./configs/i3cl_vitae_fpn/coco_instance.py). The maximum scale set to (800, 1333) is proper for V100(16G) while 
there is little effect on the performance actually. Please change the training scale according to your condition.*

### Inference

For example, use our trained I3CL model to get inference results on ICDAR2019 ArT test set with visualization images, 
txt format records and the json file for testing submission, please run:

```
python demo/art_demo.py --checkpoint pretrained_model/I3CL/epoch_12.pth --score-thr 0.45 --json_file art_submission.json
```

*Note:*
 - Upload the saved json file to [ICDAR2019-ArT evaluation website](https://rrc.cvc.uab.es/?ch=14) for Recall, Precision and F1 evaluation results. 
 Change the path for saving visualizations and txt files if needed.

## Citation

This project is for research purpose only.

If you are interested in our work, please consider citing the following:

//citation to be updated

For further questions, please contact ##

## Acknowledgement

Thanks for [mmdetection](https://github.com/open-mmlab/mmdetection).
