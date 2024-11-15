<h1 align="center">Scene Text Detection, Spotting, and Hierarchical Segmentation</h1>


This repo contains a comprehensive list of our research works related to **scene text detection, spotting, and hierarchical segmentation**. For any related questions, please contact <strong><a href="https://github.com/ymy-k/">Maoyuan Ye</a></strong> at [yemaoyuan@whu.edu.cn](mailto:yemaoyuan@whu.edu.cn) . Note that the official repo for "I3CL: Intra- and Inter-Instance Collaborative Learning for Arbitrary-shaped Scene Text Detection" has been moved to https://github.com/ViTAE-Transformer/I3CL. 


## Overview

[1. Hi-SAM: Marrying Segment Anything Model for Hierarchical Text Segmentation [TPAMI'24]](#hisam)

<a href="https://arxiv.org/abs/2401.17904"><img  src="https://img.shields.io/badge/arxiv-Paper-brightgreen" ></a> <a href="https://github.com/ymy-k/Hi-SAM"><img src="https://img.shields.io/github/stars/ymy-k/Hi-SAM.svg?logo=github&label=Stars"></a>

[2. GoMatching: A Simple Baseline for Video Text Spotting via Long and Short Term Matching [NeurIPS'24]](#gomatching)

<a href="https://arxiv.org/abs/2401.07080"><img  src="https://img.shields.io/badge/arxiv-Paper-brightgreen" ></a> <a href="https://github.com/Hxyz-123/GoMatching"><img src="https://img.shields.io/github/stars/Hxyz-123/GoMatching.svg?logo=github&label=Stars"></a>

[3. DeepSolo++: Let Transformer Decoder with Explicit Points Solo for Text Spotting [ArXiv'23]](#deepsolo++)

<a href="https://arxiv.org/abs/2305.19957"><img  src="https://img.shields.io/badge/arxiv-Paper-brightgreen" ></a> <a href="https://github.com/ViTAE-Transformer/DeepSolo"><img src="https://img.shields.io/github/stars/ViTAE-Transformer/DeepSolo.svg?logo=github&label=Stars"></a>

[4. DeepSolo: Let Transformer Decoder with Explicit Points Solo for Text Spotting [CVPR'23]](#deepsolo)

<a href="https://arxiv.org/abs/2211.10772"><img  src="https://img.shields.io/badge/arxiv-Paper-brightgreen" ></a> <a href="https://github.com/ViTAE-Transformer/DeepSolo"><img src="https://img.shields.io/github/stars/ViTAE-Transformer/DeepSolo.svg?logo=github&label=Stars"></a>

[5. DPText-DETR: Towards Better Scene Text Detection with Dynamic Points in Transformer [AAAI'23]](#dptext)

<a href="https://arxiv.org/abs/2207.04491"><img  src="https://img.shields.io/badge/arxiv-Paper-brightgreen" ></a> <a href="https://github.com/ymy-k/DPText-DETR"><img src="https://img.shields.io/github/stars/ymy-k/DPText-DETR.svg?logo=github&label=Stars"></a>

[6. I3CL: Intra- and Inter-Instance Collaborative Learning for Arbitrary-shaped Scene Text Detection [IJCV'22]](#i3cl)

<a href="https://arxiv.org/abs/2108.01343"><img  src="https://img.shields.io/badge/arxiv-Paper-brightgreen" ></a> <a href="https://link.springer.com/article/10.1007/s11263-022-01616-6"><img src="https://img.shields.io/badge/publication-Paper-brightgreen"></a> <a href="https://github.com/ViTAE-Transformer/I3CL"><img src="https://img.shields.io/github/stars/ViTAE-Transformer/I3CL.svg?logo=github&label=Stars"></a>

## Projects

### <span id="hisam">📘 Hi-SAM: Marrying Segment Anything Model for Hierarchical Text Segmentation [TPAMI'24]</span>

<em>Maoyuan Ye, Jing Zhang, Juhua Liu, Chenyu Liu, Baocai Yin, Cong Liu, Bo Du, and Dacheng Tao</em>.

[Paper](https://arxiv.org/abs/2401.17904) | [Github Code](https://github.com/ymy-k/Hi-SAM) | [BibTex](./assets/TPAMI_2024_Hi-SAM/Hi-SAM.bib)

We propose the first unified model, Hi-SAM, for segmenting text across **stroke, word, text-line, and paragraph hierarchies**. Hi-SAM also achieves **layout analysis** as a by-product. Hi-SAM supports both **automatic mask generation and interactive promptable mode**. Given a single-point prompt, Hi-SAM provides word, text-line, and paragraph masks.

<img src="./assets/TPAMI_2024_Hi-SAM/overview.jpg" width="100%">

***

### <span id="gomatching">📘 GoMatching: A Simple Baseline for Video Text Spotting via Long and Short Term Matching [NeurIPS'24]</span>

<em>Haibin He, Maoyuan Ye, Jing Zhang, Juhua Liu, and Dacheng Tao</em>.

[Paper](https://arxiv.org/abs/2401.07080) | [Github Code](https://github.com/Hxyz-123/GoMatching) | [BibTex](./assets/NeurIPS_2024_GoMatching/GoMatching.bib)

We identify a main bottleneck in the state-of-the-art video text spotter: the limited recognition capability. In response to this issue, we propose to efficiently turn an off-the-shelf query-based image text spotter into a specialist on video and present a simple baseline termed GoMatching. We establish the ArTVideo test set for addressing the absence of curved texts in current video text spotting datasets and evaluating the performance of video text spotters on videos with arbitrary-shaped text. ArTVideo contains 20 video clips, featuring 30% curved text approximately.

<img src="./assets/NeurIPS_2024_GoMatching/framework.png" width="100%">

***

### <span id="deepsolo++">📘 DeepSolo++: Let Transformer Decoder with Explicit Points Solo for Multilingual Text Spotting [ArXiv'23]</span>

<em>Maoyuan Ye<sup>&#8727;</sup>, Jing Zhang<sup>&#8727;</sup>, Shanshan Zhao, Juhua Liu, Tongliang Liu, Bo Du, and Dacheng Tao</em>.

[Paper](https://arxiv.org/abs/2305.19957) | [Github Code](https://github.com/ViTAE-Transformer/DeepSolo) | [BibTex](./assets/ArXiv_2023_DeepSolo++/deepsolo++.bib)

We propose simple yet effective baseline models named DeepSolo and DeepSolo++ for monolingual and multilingual text spotting, taking the merit of a novel explicit point query form that provides **a unified and pivotal representation for different tasks**. With a single decoder and several simple prediction heads (a linear layer or MLP), we present a much simpler method compared with previous state-of-the-art (SOTA) text spotters. Our method shows several good properties, including **1) simplicity of model structure and training pipeline, 2) efficiency of training and inference, and 3) extensibility of character class, language, and task**. Extensive experiments demonstrate that our method has achieved SOTA performance while enjoying some other distinctions, such as the effectiveness on dense and long text, and compatibility to line annotations.

<img src="./assets/ArXiv_2023_DeepSolo++/deepsolo++.png" width="100%">

***

### <span id="deepsolo">📘 DeepSolo: Let Transformer Decoder with Explicit Points Solo for Text Spotting [CVPR'23]</span>

<em>Maoyuan Ye<sup>&#8727;</sup>, Jing Zhang<sup>&#8727;</sup>, Shanshan Zhao, Juhua Liu, Tongliang Liu, Bo Du, and Dacheng Tao</em>.

[Paper](https://arxiv.org/abs/2211.10772) |  [Github Code](https://github.com/ViTAE-Transformer/DeepSolo) | [BibTex](./assets/CVPR_2023_DeepSolo/deepsolo.bib)

End-to-end text spotting aims to integrate scene text detection and recognition into a unified framework. Dealing with the relationship between the two sub-tasks plays a pivotal role in designing effective spotters. In this paper, we present DeepSolo, a succinct DETR-like baseline that lets a single decoder with explicit points solo for text detection and recognition simultaneously and efficiently. Quantitative experiments on public benchmarks demonstrate that DeepSolo outperforms previous state-of-the-art methods and achieves better training efficiency. In addition, DeepSolo is also compatible with line annotations.

<img src="./assets/CVPR_2023_DeepSolo/deepsolo.jpg" width="100%">

***

### <span id="dptext">📘 DPText-DETR: Towards Better Scene Text Detection with Dynamic Points in Transformer [AAAI'23]</span>

<em>Maoyuan Ye, Jing Zhang, Shanshan Zhao, Juhua Liu, Bo Du, and Dacheng Tao</em>.

[Paper](https://arxiv.org/abs/2207.04491) |  [Github Code](https://github.com/ymy-k/DPText-DETR) | [BibTex](./assets/AAAI_2023_DPText-DETR/dptext.bib)

Recently, Transformer-based methods, which predict polygon points or Bezier curve control points for localizing texts, are popular in scene text detection. However, these methods built upon detection transformer framework might achieve sub-optimal training efficiency and performance due to coarse positional query modeling. In addition, the point label form exploited in previous works implies the reading order of humans, which impedes the detection robustness from our observation. To address these challenges, this paper proposes DPText-DETR, which directly leverages explicit point coordinates to generate position queries and dynamically updates them in a progressive way. Moreover, we present an Enhanced Factorized Self-Attention module to improve the spatial inductive bias of non-local self-attention in Transformer. Furthermore, we design a simple yet effective positional label form to tackle the side effect of the previous form. To further evaluate the impact of different label forms on the detection robustness in real-world scenario, we establish an Inverse-Text test set containing 500 manually labeled images. Extensive experiments prove the high training efficiency, robustness, and state-of-the-art performance of our method on popular benchmarks

<img src="./assets/AAAI_2023_DPText-DETR/dptext_detr.jpg" width="100%">

***

### <span id="i3cl">📘 I3CL: Intra- and Inter-Instance Collaborative Learning for Arbitrary-shaped Scene Text Detection [IJCV'22]</span>

<em>Bo Du, Jian Ye,  Jing Zhang, Juhua Liu, and Dacheng Tao</em>.

[Paper](https://arxiv.org/abs/2108.01343) |  [Github Code](https://github.com/ViTAE-Transformer/I3CL) | [BibTex](./assets/IJCV_2022_I3CL/i3cl.bib)

Existing methods for arbitrary-shaped text detection in natural scenes face two critical issues, i.e., (1) fracture detections at the gaps in a text instance; and (2) inaccurate detections of arbitrary-shaped text instances with diverse background context. To address these issues, we propose a novel method named Intra- and Inter-Instance Collaborative Learning (I3CL). Specifically, to address the first issue, we design an effective convolutional module with multiple receptive fields, which is able to collaboratively learn better character and gap feature representations at local and long ranges inside a text instance. To address the second issue, we devise an instance-based transformer module to exploit the dependencies between different text instances and a global context module to exploit the semantic context from the shared background, which are able to collaboratively learn more discriminative text feature representation. In this way, I3CL can effectively exploit the intra- and inter-instance dependencies together in a unified end-to-end trainable framework. Besides, to make full use of the unlabeled data, we design an effective semi-supervised learning method to leverage the pseudo labels via an ensemble strategy. Without bells and whistles, experimental results show that the proposed I3CL sets new state-of-the-art results on three challenging public benchmarks, i.e., an F-measure of 77.5% on ArT, 86.9% on Total-Text, and 86.4% on CTW-1500. Notably, our I3CL with the ResNeSt-101 backbone ranked the 1st place on the ArT leaderboard.

<img src="./assets/IJCV_2022_I3CL/i3cl.jpg" width="100%">
