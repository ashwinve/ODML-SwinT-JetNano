# ODML-SwinT-JetNano

# Getting started
Follow get_started.md from ./ODML-Swin-Transformer/get_started.md

[TODO] [Add env.yaml] Use env.yaml file to create repository with required dependencies

RESCIS45 dataset is too big to store on git.
Download from: https://1drv.ms/u/s!AmgKYzARBl5ca3HNaHIlzp_IXjs

Create top-level data storage folder as ./RESCIS45/ locally

# Fine-tuning Swin Transformer from Imagenet 1K on RESCIS45
Improving Classification of Remotely Sensed
Images with the Swin Transformer
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9764016&tag=1

# Implementation Details
Starter model: Pre-trained Swin transformer trained on ImageNet-1k

Fine-tuning hyperparameters:
NUM_EPOCHS: 25
BATCHSIZE: 4
LEARNINGRATE: 1e-3
OPTIMIZER: SGD
LOSS: CrossEntropy Loss
SCHEDULER: Cosine AnnealingLR
PREPROCESSING: Images are resized into 224x224
