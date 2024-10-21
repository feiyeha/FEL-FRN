from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# 文字处理模型bert的构造
#--------------------------------------#
#   Gelu激活函数的实现
#   利用近似的数学公式，就一个数学公式计算一下返回值就行
# GELU激活函数具有以下优点：
# 1.
# 与RELU相比，函数不会对所有小于等于0的x一视同仁全取为0，全取为0后会导致导数恒等于0，从而导致梯度消失，从而GELU激活函数消除了梯度消失的问题。
# 2.
# 在x = 0处，RELU激活函数不可导，而GELU激活函数在x = 0处是光滑的曲线，是可导的。
# 3.
# GELU函数在激活函数的非线性变换中引入了类似于sigmoid函数的变换，这使得GELU函数的输出可以落在一个更广的范围内，有助于加速模型的收敛速度。
#--------------------------------------#
class GELU(nn.Module):
    def __init__(self):
        # 它调用父类nn.Module的初始化函数super(GELU, self).__init__()，
        super(GELU, self).__init__()

    def forward(self, x):
        # 这个函数定义了当数据通过GELU模块时应该进行的操作。
        # https://blog.csdn.net/weixin_71719718/article/details/132241290
        return 0.5 * x * (1 + F.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x,3))))
    
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", GELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    # Transformer模块通过堆叠多个ResidualAttentionBlock层来创建一个具有深度和宽度可配置的神经网络
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)