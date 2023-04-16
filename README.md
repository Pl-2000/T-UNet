## T-UNet: Triplet UNet for Change Detection in High-Resolution Remote Sensing Images
In this paper, a novel triple-branch encoder-based network framework (Triplet UNet, T-UNet) is proposed for remote sensing image change detection.

## Overview
Remote sensing image change detection aims to identify the differences between images acquired at different times in the same area. It is widely used in land management, environmental monitoring, disaster assessment and other fields. Currently, most change detection methods are based on Siamese network structure or early fusion structure. Siamese structure focuses on extracting object features at different times but lacks attention to change information, which leads to false alarms and missed detections. Early fusion (EF) structure focuses on extracting features after the fusion of images of different phases but ignores the significance of object features at different times for detecting change details, making it difficult to accurately discern the edges of changed objects.

## Methodology
We propose a novel network, Triplet UNet(T-UNet), based on a three-branch encoder, which is capable to simultaneously extract the object features and the change features between the pre- and post-time-phase images through triplet encoder. To effectively interact and fuse the features extracted from the three branches of triplet encoder, we propose a multi-branch spatial-spectral cross-attention module (MBSSCA). In the decoder stage, we introduce the channel attention mechanism (CAM) and spatial attention mechanism (SAM) to fully mine and integrate detailed textures information at the shallow layer and semantic localization information at the deep layer.

## How to use the code
###  Environment configuration 
Deep learning framework: Pytorch1.8

### Code configuration
1. TUNet.py is the main code of the proposed network T-UNet.
2. VGG16.py is used for construction of triplet encoder.
3. Decoder.py is used for construction of decoder.

## Dataset
Three publicly available CD datasets are employed in this paper for the evaluation of the model performance, namely the LEVIR-CD dataset, the WHU-CD dataset and the DSIFN-CD dataset.
