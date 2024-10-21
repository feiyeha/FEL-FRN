import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBlock(nn.Module):
    # ConvBlock 是一个自定义的卷积块，它继承自 nn.Module
    def __init__(self,input_channel,output_channel):
        super().__init__()
        # 接收两个参数：input_channel（输入通道数）和 output_channel（输出通道数）。
        self.layers = nn.Sequential(
            nn.Conv2d(input_channel,output_channel,kernel_size=3,padding=1),
            nn.BatchNorm2d(output_channel))
    # 使用 nn.Conv2d 创建一个卷积层，卷积核大小为 3x3，步长为 1（默认），填充为 1（以保持输入输出的空间尺寸不变）。
    # 使用 nn.BatchNorm2d 创建一个批量归一化层，其通道数与输出通道数相同。两者封装，前向运行这个卷积块
    def forward(self,inp):
        return self.layers(inp)


class BackBone(nn.Module):

    def __init__(self,num_channel=64):
        super().__init__()
        
        self.layers = nn.Sequential(
            ConvBlock(3,num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ConvBlock(num_channel,num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ConvBlock(num_channel,num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ConvBlock(num_channel,num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

    def forward(self,inp):

        return self.layers(inp)