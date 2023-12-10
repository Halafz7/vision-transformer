import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Optional, Union

import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch import Tensor

import brevitas.nn as qnn
from brevitas.nn import QuantIdentity
from brevitas.quant import TruncTo8bit
from brevitas.quant_tensor import QuantTensor
from brevitas.quant.scaled_int import IntBias
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat

__all__ = ["VisionTransformer"]


def make_quant_conv2d(
        in_channels, out_channels, kernel_size, weight_bit_width, stride=1, padding=0, bias=False, input_quant=None, bias_quant=Int32Bias):
    return qnn.QuantConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
        weight_bit_width=weight_bit_width,
        weight_scaling_per_output_channel=True,
        input_quant=input_quant,
        weight_quant=Int8WeightPerChannelFloat,
        bias_quant=bias_quant)

class QuantMultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.q_weights = [nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_heads)]
        self.k_weights = [nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_heads)]
        self.v_weights = [nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_heads)]
        self.softmax = nn.Softmax(dim=-1)
        self.linear = nn.Linear(num_heads*hidden_dim, hidden_dim)
        
    def forward(self, X):
        #B, N, D = X.shape 
        result = []
        for x in X:
            x_result = [] # H, N, D
            for head in range(self.num_heads):
                q = self.q_weights[head](x)
                k = self.k_weights[head](x)
                v = self.v_weights[head](x)
                h = self.softmax(q @ k.T / self.hidden_dim**2) @ v # N, D
                x_result.append(h)
            result.append(torch.hstack(x_result)) # B, H, N, D
        H = torch.cat([torch.unsqueeze(r, dim=0) for r in result]) 
        out = self.linear(H)
        return out # N, D
    
class VisionTransformer(nn.Module):
    def __init__(self, img_shape, patch_size, hidden_dim, num_heads, out_dim, num_encoder_blocks=6):
        super().__init__()
        
        self.img_shape = img_shape
        self.patch_size = img_shape[0]*patch_size[0]*patch_size[1]
        self.num_patches = int(img_shape[0]*img_shape[1]/patch_size[0]) ** 2
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.num_encoder_blocks = num_encoder_blocks
        
        # Linear patching
        self.linear_patching = nn.Linear(self.patch_size, self.hidden_dim)
        
        # CLS embedding
        self.cls_embedding = nn.Parameter(torch.rand(1, self.hidden_dim))
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.rand(1+self.num_patches, self.hidden_dim))
        
        # Transformer
        self.transformer_1 = nn.Sequential(
                                nn.LayerNorm((1+self.num_patches, self.hidden_dim)),
                                QuantMultiHeadSelfAttention(self.hidden_dim, self.num_heads)
                            )
        self.transformer_2 = nn.Sequential(
                                nn.LayerNorm((1+self.num_patches, self.hidden_dim)),
                                nn.Linear(self.hidden_dim, self.hidden_dim),
                            )
        
        # MLP head
        self.mlp_head = nn.Sequential(
                            nn.Linear(self.hidden_dim, self.out_dim),
                            nn.Tanh(),
                        )
    
    def forward(self, X):
        N, C, H, W = X.shape
        patches = X.reshape(N, self.num_patches, self.patch_size)
        E = self.linear_patching(patches)
        cls_embedding = nn.Parameter(self.cls_embedding.repeat(N, 1, 1))
        E = torch.cat([cls_embedding, E], dim=1)
        Epos = nn.Parameter(self.pos_embedding.repeat(N, 1, 1))
        Z = E + Epos
        for _ in range(self.num_encoder_blocks):
            res1 = self.transformer_1(Z)
            Z = self.transformer_2(res1 + Z)
        C = self.mlp_head(Z[:, 0])
        return C

def get_transformer(**kwargs: Any) -> VisionTransformer:
    model = VisionTransformer(**kwargs)

    return model