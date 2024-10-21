import argparse
import json
import os

import torch
import yaml
import random
import clip
from PIL import Image
import numpy as np
from datasets.imagenet import ImageNet
from datasets.utils import build_data_loader
import torchvision.transforms as transforms
from utils import *
from Utils import util
from trainers.eval import meta_test
from torch.utils.data import Dataset
from datasets import build_dataset
from clip import longclip
from clip.FRN import FRN
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()

    return args


def load_hyperparameters(filename='best_hyperparameters.txt'):
    best_hyperparams = {}
    with open(filename, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            best_hyperparams[key.strip()] = float(value.strip())  # 假设值是浮点数
    return best_hyperparams.get('Best beta'), best_hyperparams.get('Best alpha')

def load_hyperparameters_F(filename='best_hyperparameters_F.txt'):
    best_hyperparams = {}
    with open(filename, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            best_hyperparams[key.strip()] = float(value.strip())  # 假设值是浮点数
    return best_hyperparams.get('Best beta'), best_hyperparams.get('Best alpha')


def run_FRN_test(args):
    with open('config.yml', 'r') as f:
        temp = yaml.safe_load(f)
    data_path = os.path.abspath(temp['data_path'])

    test_path = os.path.join(data_path,'CUB_fewshot_raw/test_pre')
#     测试集所用图片的路径
# 使用的模型
    # model_path = './model_ResNet-12.pth'
    model_path = 'trained_model_weights/CUB_fewshot_raw/FRN/ResNet-12/model.pth'

    gpu = 0
    torch.cuda.set_device(gpu)

    model = FRN(resnet=True)
    model.cuda()
    model.load_state_dict(torch.load(model_path,map_location=util.get_device_map(gpu)),strict=True)
    model.eval()

    with torch.no_grad():
        way = 5
        for shot in [1,5]:
            mean,interval = meta_test(data_path=test_path,
                                model=model,
                                way=way,
                                shot=shot,
                                pre=True,
                                transform_type=None,
                                trial=10000)
            print('%d-way-%d-shot acc: %.3f\t%.3f'%(way,shot,mean,interval))

def run_tip_adapter(test_features, clip_weights,cache_keys, cache_values,folder_path,categories_with_descriptions):
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_names = []
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in extensions):
            image_names.append(filename)
    # print("\n-------- Searching hyperparameters on the val set. --------")
    #
    # # 是提取好的， val_features验证集所有图像的特征clip_weights所有文本的特征
    # # Zero-shot CLIP
    # clip_logits = 100. * val_features @ clip_weights
    # # print(clip_logits)
    # acc = cls_acc(clip_logits, val_labels)
    # print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(acc))

    # # Tip-Adapter
    # # 初始权重或偏置
    # beta, alpha = cfg['init_beta'], cfg['init_alpha']
    #
    # # 我们计算这个测试图像的特征与cache model中所有keys（即few-shot训练图像的特征）的相似度,使用余弦相似度点乘来计算衡量。
    # affinity = test_features @ cache_keys
    # cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    #
    # # 总的分数,有clip和adapter的结果残差连接（加权求和），进行融合
    # tip_logits = clip_logits + cache_logits * alpha
    # acc = cls_acc(tip_logits, val_labels)


        # 调用函数加载超参数

    best_beta, best_alpha = load_hyperparameters()


    print("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights#3*512    512*40
    print(clip_weights.shape)
    cls_result(clip_logits,image_names,"result/CLIP",'CLIP',categories_with_descriptions)
    # acc = cls_acc(clip_logits,)
    # print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    affinity = test_features @ cache_keys#3*512    512*1632   =3*1632
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values#3*102
    # cache_values 1632*102，每张图片对应所有类别，训练集16*102=1632张
    print(cache_values.shape)
    tip_logits = clip_logits + cache_logits * best_alpha
    cls_result(tip_logits, image_names,"result/Tip-Adapter",'Tip',categories_with_descriptions)

    # acc = cls_acc(tip_logits, test_labels)准确率就不计算了，测试集哪来的标签
    # print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))


def run_tip_adapter_F(cfg,test_features, clip_weights,cache_keys, cache_values,folder_path,clip_model,categories_with_descriptions):
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_names = []
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in extensions):
            image_names.append(filename)

    # Enable the cached keys to be learnable,让keys可以被训练，所以创建一个线性层（即“adapter”）
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    # Search Hyperparameters
    best_beta, best_alpha = load_hyperparameters_F()
    # 找到好的超参数
    print("\n-------- Evaluating on the test set. --------")
    clip_logits = 100. * test_features @ clip_weights
    affinity = adapter(test_features)
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    # 这边使用最高的超参数
    tip_logits = clip_logits + cache_logits * best_alpha
    cls_result(tip_logits, image_names, "result/Tip-Adapter-F", 'Tip-F',categories_with_descriptions)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = get_arguments()
    assert (os.path.exists(args.config))

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cache_dir = os.path.join('caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")
    # CLIP
    clip_model, preprocess = longclip.load("model/longclip-B.pt")
    # clip_model, preprocess = longclip.load_from_clip(cfg['backbone'])
    clip_model.eval()
    # 预训练模型，那索性我直接跑，跑完了之后用它跑完的pt文件导入
    # ImageNet dataset
    random.seed(1)
    torch.manual_seed(1)
    run_FRN_test(cfg)
#     with open('bird/cat_to_name.json', 'r', encoding='utf-8') as f:
#         categories_with_descriptions = json.load(f)
#     # 提取所有的文本描述并放入列表中
#     text_inputs = list(categories_with_descriptions.values())
#     template = ['a photo of a {}, a type of bird.']
#     # 加载test测试集
#     print("Preparing dataset.")
#     # dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])
      test_dir = r'test/jpg'
      test_loader = build_data_loader(data_source=test_dir, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)

#     # Textual features
#     # 这部分代码使用clip_classifier函数来获取CLIP模型的文本特征，用于后面获得图像特征后就可以直接乘了。
#     # dataset.classnames可能是数据集中所有类别的名称，
#     # dataset.template可能是用于将类别名称转换为CLIP模型可以理解的文本描述的模板，而clip_model是已经加载的CLIP模型。
#     # 这边我通过json文件获取类别并且获取文本特征
#     print("\nGetting textual features as CLIP's classifier.")
#     clip_weights = clip_classifier(text_inputs, template, clip_model)
#     # Construct the cache model by few-shot training set，你测试的时候，直接用之前训练时训练好的那些pt文件构建cache就行了
#     print("\nConstructing cache model by few-shot visual features and labels.")
#     cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
#     cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")
#     print(cache_keys.shape)
#     print(cache_values.shape)

#     # Pre-load test features，训练好的话，直接导入,图像是自己新加的，所以是要提取特征的
#     # 与验证集类似，但这次是从测试集中加载。测试集通常用于评估模型的性能，但在某些场景下（如少样本学习），它也可能被用于模型训练或微调。
#     print("\nLoading visual features and labels from test set.")
#     test_features= pre_load_features_test(cfg, "test", clip_model, test_loader)

#     run_tip_adapter(test_features,clip_weights,cache_keys, cache_values,test_dir,categories_with_descriptions)
#     run_tip_adapter_F(cfg,test_features, clip_weights, cache_keys, cache_values, test_dir,clip_model,categories_with_descriptions)


if __name__ == '__main__':
    main()

