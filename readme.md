# MutiNeck-KD

The code repository contains the model and experimental code used in the paper “Central and Directional Muti-Neck Knowledge Distillation”

---

![MutiNeck.png](MutiNeck-KD%20b09ac095799f412e9a2e3dcc476a8ab2/MutiNeck.png)

## Introduction

---

The aim of this project is to implement the MultiNeck multi-teacher knowledge distillation model proposed in the paper and evaluate it on the COCO and PASCAL VOC datasets. The main purpose of this model is to investigate the effectiveness of center response and direction-aware feature clustering learning for multi-teacher knowledge distillation in instance object detection. It utilizes multiple proposals from the teacher models to overlay stronger center response masks on the feature maps, and employs feature clustering to help the student model locate the centers of category features, facilitating the student model in learning more accurate knowledge.

Method:

- developed a novel distillation learning framework, MutiNeck-KD, which integrates adaptive learning of multi-level knowledge from multiple teachers using an intermediate teacher.
- proposed a new learning method that learns the differences in target center features, boundary features, and background features of proposals for the same target by stacking teacher models.
- analyzed the features of teacher models for different target categories, adjusted them to the same size through ROI pooling, and clustered them to maintain the consistency of student model learning through center distance.

## ****Installation****

---

We utilized the mmdetection(version=2.19.0) framework to implement our model, which relies on PyTorch and MMCV. We recommend using this framework to run our code. Below are the quick installation steps. For more detailed instructions, please refer to the installation documentation.

[MMdetection](https://github.com/open-mmlab/mmdetection)

```bash
conda create -n openmmlab python=3.7 -y
conda activate openmmlab
conda install pytorch torchvision -c pytorch
# conda install pytorch cudatoolkit=10.1 torchvision -c pytorch  # No CUDA
conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch # Has CUDA
pip install openmim
mim install mmdet
# pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html #Choose your version 
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

## ****Data Preparation****

---

- We utilized the original COCO dataset, which consists of 120k training images and 5k validation images.
- We performed a redivision of the PASCAL VOC dataset and converted the annotations to COCO annotation format, splitting it into training, validation, and test sets in an 8:1:1 ratio.

## ****Getting Started****

---

This code repository is based on the basic usage of [MMdetection](https://github.com/open-mmlab/mmdetection). Please refer to [get_started.](https://mmdetection.readthedocs.io/zh_CN/latest/get_started.html) for more information

## Train

---

Use the prescribed instructions of MMdetection for training

```python
# 1 .Obtain the ROI features of all instances' targets in the dataset,we can get the 'txt' files
python tools/train.py configs\distillers\merge_kd\new_config\fgd_fcos_r101_distill_ROI_features.py
# 2 .Cluster all the features and identify the centroids of the category features.
python other_code/kmeans_find_center.py
# 3 .Training the MultiNeck-KD model with multi-teacher knowledge distillation.
python tools/train.py configs\distillers\merge_kd/kd_ed_fasterrcnn_maskrcnn_resnet101_merge_resnet50_coco80_similarity05.py
```

The URL of the weight file.[MutiNeck-KD.pth](https://pan.baidu.com/s/17T6V1die7YxNu0YUGFQTLw?pwd=1110)

## ****Acknowledgements****

---

We used practical program functions from other wonderful open source projects.

Special thanks to the following authors:

[FGD](https://github.com/yzd-v/FGD)

Thanks to the third-party libraries:

[PyTorch](https://pytorch.org/)

[MMdetection](https://github.com/open-mmlab/mmdetection)