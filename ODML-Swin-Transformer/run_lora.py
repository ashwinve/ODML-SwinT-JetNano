import numpy as np
from matplotlib import pyplot as plt
from scipy.special import softmax
import torch
import argparse
from logger import create_logger
import os


from utils import load_checkpoint, load_pretrained
from config import get_config
from data import build_loader
from models import build_model

from main import train_one_epoch, validate, throughput

from config import get_only_config
import json
import copy
import math
import time
import sys

import datetime
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.utils import accuracy, AverageMeter




if __name__ == '__main__':

    config_name = sys.argv[1]



    config_path = 'configs/swin/swin_tiny_patch4_window7_224_resisc45.yaml'
    config = get_only_config(config_path)


    config.defrost()
    config.OUTPUT = "/afs/ece.cmu.edu/usr/bmarimut/Private/output"
    # config.MODEL.PRETRAINED = "/afs/ece.cmu.edu/usr/ashwinve/Public/ckpt_epoch_29_6.pth"
    config.MODEL.PRETRAINED = "/afs/ece.cmu.edu/usr/ashwinve/Public/golden_resisc45.pth"
    config.MODEL.RESUME = "/afs/ece.cmu.edu/usr/ashwinve/Public/golden_resisc45.pth"
    config.DATA.CACHE_MODE = 'no'
    config.DATA.DATA_PATH = './data/RESISC45/'
    config.DATA.ZIP_MODE = True
    config.PRINT_FREQ = 120
    config.DATA.BATCH_SIZE = 8
    config.freeze()
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")
    
    my_model = None

    if config_name == 'baseline':
        checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
        config_name = checkpoint['model']
        lora_selector = 0
        lora_layer = 4
        keep_qkv = True
        my_model = build_model(config, lora_selector, lora_layer, keep_qkv)
        my_model.load_state_dict(config_name, strict=False)

    else:
        config_name = "config"+config_name+"_chkpt.pth"
        lora_selector = int(config_name)
        lora_layer = 0
        keep_qkv = False
        my_model = build_model(config, lora_selector, lora_layer, keep_qkv)
        my_model.load_state_dict(torch.load(config_name, map_location='cpu'), strict=False)



    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    # acc1, acc5, loss = teacher_validate(config, data_loader_val, super_model)
    throughput(data_loader_val, my_model, logger)
    batch_latency(data_loader_val, my_model, logger)

