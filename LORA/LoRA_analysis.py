#%%
%matplotlib inline
%cd ODML-Swin-Transfomer
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

from config import get_only_config
import json

import copy

# %%
def svd(m):
    return np.linalg.svd(m, compute_uv=False)

# %%
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    return windows

# %%
class attn_model(torch.nn.Module):
    def __init__(self, i, H, W, B, L, C, num_heads, window_size):
        super(attn_model, self).__init__()
        
        self.i = i
        self.H = H
        self.W = W
        self.B = B
        self.L = L
        self.C = C
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv_layer = torch.nn.Linear(C, C * 3, bias=True)
        self.relative_position_bias_table = torch.nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.softmax = torch.nn.Softmax(dim=-1)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)


    def load_pretrained_weights(self, params_names_list, super_model):
        new_sd = copy.deepcopy(self.state_dict())
        for name in params_names_list:
            new_sd['qkv_layer.weight'] = super_model.state_dict()[name+'.qkv.weight']
            new_sd['qkv_layer.bias'] = super_model.state_dict()[name+'.qkv.bias']
        
        new_sd['relative_position_bias_table'] = super_model.state_dict()[name+'.relative_position_bias_table']
        
        self.load_state_dict(new_sd)

    def forward(self, x):
        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, self.C)
        B_, N_, C_ = x_windows.shape
        print(" ")
        qkv = self.qkv_layer(x_windows)
        qkv = qkv.reshape(B_, N_, 3, num_heads, C_ // num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        print(" q : ", q.shape)
        print(" k : ", k.shape)
        print(" v : ", v.shape)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        print("attn : ", attn.shape)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        return self.softmax(attn), q, k, v


#%%
config_path = 'configs/swin/swin_tiny_patch4_window7_224_resisc45.yaml'
config = get_only_config(config_path)

config.defrost()
config.OUTPUT = "/afs/ece.cmu.edu/usr/ashwinve/Private/output"
config.MODEL.PRETRAINED = "/afs/ece.cmu.edu/usr/ashwinve/Public/ckpt_epoch_29_6.pth"
config.freeze()
os.makedirs(config.OUTPUT, exist_ok=True)
logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

#%%
print(config)

# %%
model = build_model(config)
load_pretrained(config, model, logger)

# %%
NUM_HEADS = [ 3, 6, 12, 24 ]
i = 1
H = 224
W = 224
B = 1
L = 224 * 224
C = 96 * 2**i
num_heads = NUM_HEADS[i]
dim = C
window_size = 7

# %%
x = torch.tensor(np.random.randint(0, 256, size=(B, H, W, C)))

x_windows = window_partition(x, window_size)

my_model = attn_model(i, H, W, B, L, C, num_heads, window_size)

params_names = ['layers.1.blocks.0.attn']
my_model.load_pretrained_weights(params_names, model)


x = torch.tensor(np.random.randint(0, 256, size=(B, H, W, C)), dtype=torch.float)
attn_op, q, k, v = my_model(x) 

q_ = attn_op.detach().numpy()
q_ = q_.transpose(1, 0, 2, 3)
q_.shape

# %%
plt.plot(np.cumsum(svd(q_[0][1022])))
# %%
