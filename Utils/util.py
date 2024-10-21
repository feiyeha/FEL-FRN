from PIL import Image
import torch
import os
import numpy as np
import sys
import argparse
import shutil
from tqdm import tqdm
import torchvision.transforms as transforms

def mkdir(path):
    # 您的函数 mkdir 旨在创建一个新的目录（文件夹），但如果目录已经存在，则打印一条消息表示目录已存在
    if os.path.exists(path): 
        print("---  the folder already exists  ---")
    else:
        os.makedirs(path)


# get pre-resized 84x84 images for validation and test
def get_pre_folder(image_folder,transform_type):
    split = ['val','test']

    if transform_type == 0:
        transform = transforms.Compose([transforms.Resize(92),
                                    transforms.CenterCrop(84)])
    elif transform_type == 1:
        transform = transforms.Compose([transforms.Resize([92,92]),
                                    transforms.CenterCrop(84)])

    cat_list = []

    for i in split:
        
        cls_list = os.listdir(os.path.join(image_folder,i))

        folder_name = i+'_pre'

        mkdir(os.path.join(image_folder,folder_name))

        for j in tqdm(cls_list):

            mkdir(os.path.join(image_folder,folder_name,j))

            img_list = os.listdir(os.path.join(image_folder,i,j))

            for img_name in img_list:
        
                img = Image.open(os.path.join(image_folder,i,j,img_name))
                img = img.convert('RGB')
                img = transform(img)
                img.save(os.path.join(image_folder,folder_name,j,img_name[:-3]+'png'))
# 对验证（val）和测试（test）数据集中的图像进行预处理，具体是调整它们的大小并裁剪到84x84像素。

def get_device_map(gpu):
    cuda = lambda x: 'cuda:%d'%x
    temp = {}
    for i in range(4):
        temp[cuda(i)]=cuda(gpu)
    return temp
# 该函数的功能是创建一个从多个 CUDA 设备索引（在这个例子中是 'cuda:0' 到 'cuda:3'）映射到单个 GPU 设备索引（通过 gpu 参数指定）的字典。