import os
import math
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Sampler  
import numpy as np
from copy import deepcopy
from PIL import Image
from . import samplers,transform_manager
import torchvision.transforms as transforms



def get_dataset(data_path,is_training,transform_type,pre):

    dataset = datasets.ImageFolder(
        data_path,
        loader = lambda x: image_loader(path=x,is_training=is_training,transform_type=transform_type,pre=pre))

    return dataset

def get_dataset2(data_path,is_training,transform_type,pre):

    dataset = datasets.ImageFolder(
        data_path,
        loader = lambda x: image_loader2(path=x,is_training=is_training,transform_type=transform_type,pre=pre))

    return dataset

train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])


def meta_train_dataloader(data_path,way,shots,transform_type):

    dataset = get_dataset(data_path=data_path,is_training=True,transform_type=transform_type,pre=None)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler = samplers.meta_batchsampler(data_source=dataset,way=way,shots=shots),
        num_workers = 3,
        pin_memory = False)

    return loader
# 您正在创建一个用于元训练的数据加载器。这个函数接收数据路径、类别数（way）、每个类别的样本数（shots）、以及转换类型（transform_type）作为输入，并返回一个配置好的 DataLoader 对象。


def meta_test_dataloader(data_path1,data_path2,way,shot,pre,transform_type=None,query_shot=16,trial=1000):

    dataset1 = get_dataset(data_path=data_path1,is_training=False,transform_type=transform_type,pre=pre)#加载pre的84*84
    dataset2 = get_dataset2(data_path=data_path2,is_training=False,transform_type=train_tranform,pre=pre)#加载test224*224
# batch_sampler 返回一个批次索引的列表的迭代器
    loader1 = torch.utils.data.DataLoader(
        dataset1,
        batch_sampler = samplers.random_sampler(data_source=dataset1,way=way,shot=shot,query_shot=query_shot,trial=trial),
        num_workers = 3,
        pin_memory = False)

    sampler=samplers.CLIP_sampler(data_source=dataset2,way=way,shot=shot,query_shot=query_shot,trial=trial)
    loader2 = DataLoader(  
        dataset2,  
        batch_sampler=sampler, 
        num_workers=3,  
        pin_memory=False  
    )  
    

    return loader1, dataset1, loader2, dataset2,sampler

def meta_test_dataloader2(data_path,way,shot,pre,transform_type=None,query_shot=16,trial=1000):

    dataset = get_dataset(data_path=data_path,is_training=False,transform_type=transform_type,pre=pre)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler = samplers.random_sampler(data_source=dataset,way=way,shot=shot,query_shot=query_shot,trial=trial),
        num_workers = 3,
        pin_memory = False)

    return loader,dataset



def normal_train_dataloader(data_path,batch_size,transform_type):

    dataset = get_dataset(data_path=data_path,is_training=True,transform_type=transform_type,pre=None)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 3,
        pin_memory = False,
        drop_last=True)

    return loader


def image_loader(path,is_training,transform_type,pre):

    p = Image.open(path)
    p = p.convert('RGB')

    final_transform = transform_manager.get_transform(is_training=is_training,transform_type=transform_type,pre=pre)
#     pre为True代表，用的84*84也就是已经处理过的图像，transform_type如果给了赋值，就是要对图像进行处理，不给默认为None，也就是要对图像处理
# 他的意思就是两个不能同时存在，又告诉系统这个已经处理过的图像，又让系统去处理图像，他的意思是，如果 pre为True代表，用的84*84也就是已经处理过的图像，transform_type就为None,不要处理图像，要么pre为False，用的是未处理过的图像，transform_type就给定处理的代码，去给图像进行处理。

    p = final_transform(p)

    return p


def image_loader2(path,is_training,transform_type,pre):

    p = Image.open(path)
    p = p.convert('RGB')

    final_transform = transform_manager.get_transform2(transform_type=transform_type)
    p = final_transform(p)

    return p

