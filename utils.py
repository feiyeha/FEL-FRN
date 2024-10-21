import os

from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
import torch.nn as nn
import json
import clip
from clipmain import CLIP
# from transformers import CLIPTokenizer, CLIPModel
from torchvision import transforms  
from tqdm import tqdm  
# train_image_names = [item[0] for item in data['train']]
# val_image_names = [item[0] for item in data['val']]
# test_image_names = [item[0] for item in data['test']]
# image_names =val_image_names
# def cls_acc(output, target, topk=1):


def cls_result(output,image_names,result_dir,encoder,categories_with_descriptions,topk=1):
        pred = output.topk(topk, 1, True, True)[1].t()
        # print(pred.shape)
        prob, _ = output.softmax(dim=1).topk(topk, 1, True, True)

        # 打印或返回预测的类别序号（如果需要）
        # for pred_id in pred.cpu().squeeze(0):
        #     print(pred_id.item())  # 使用.item()将tensor中的单个值转换为Python数值类型（如int或float）
        # print("Predicted classes:", pred.cpu().numpy())
        # 获取每个图像预测第一的类别
        first_pred_indices = pred.cpu().squeeze(0)
        first_pred_indices=first_pred_indices.numpy().astype(int)
        print(first_pred_indices)
        first_pred_classes = [categories_with_descriptions[idx] for idx in first_pred_indices]
        if encoder == 'Tip-F':
            first_pred_prob = prob.squeeze(1).detach().cpu().numpy()
        else:
            first_pred_prob = prob.squeeze(1).detach().cpu().numpy()
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        for pred_class,image_name,prob_val in zip(first_pred_classes,image_names,first_pred_prob):
            print(f"Predicted class_name = {pred_class}: Probability = {prob_val:.4f}")
#             img_path = f'test/jpg/'
#             img = Image.open(img_path)
#             draw = ImageDraw.Draw(img)
#             font = ImageFont.truetype("SimHei.ttf", 21)  # 你可以选择一个你有的.ttf字体文件
#             # 将预测结果和概率打印到图片上
#             # text = f"{encoder}，Predicted class: {pred_class}, Probability: {prob_val:.4f}"
#             text = f"c: {pred_class}, P: {prob_val:.4f}"
#             text_width, text_height = draw.textsize(text, font)
#             draw.text((10, 10), text, font=font, fill=(255, 255, 255))  # 红色字体
            
#             output_path = os.path.join(result_dir, os.path.basename(img_name))
#             # 保存图片
#             img.save(output_path)
            

def train_clip(test_dir,captions):
    clip = CLIP()
    trainclip_logits=[ ]
    
    
    # 遍历文件夹中的文件
    for filename in os.listdir(test_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # 检查是否是图像文件
            image_path = os.path.join(test_dir, filename)

            # 读取图像
            with Image.open(image_path) as image:
                # 调用detect_image函数并获取概率
                # image=image.cuda()
                print(image)
                logits,probs = clip.detect_image(image, captions)
                #logits是没有softmax,probs是softmax后的，probs直接输出结果，logits之后用cls_result算出来
                values, indices = probs[0].topk(1)
                trainclip_logits.append(logits)  
                # print(probs)
                # print(f"\n{filename}:")  
                # for value, index in zip(values, indices):
                #     print(f"{captions[index]:>16s}: {100 * value.item():.2f}%")
    trainclip_logits = torch.cat(trainclip_logits)
    trainclip_logits = trainclip_logits.to('cuda')
    return trainclip_logits


# def pre_load_features_trained(cfg, split,loader):
#     clip = CLIP()
#     # 当你设置 cfg['load_pre_feat'] = False 时，
#     # 这边准备用我自己的模型去提取图像验证集和测试集特征
#     # 函数会遍历整个数据集，计算每张图像的特征，并将这些特征和对应的标签保存为 PyTorch 张量（.pt 文件）。
#     if cfg['load_pre_feat'] == False:
#         features = []
        
#         with torch.no_grad():
#             for i, (images) in enumerate(tqdm(loader)):
#                 images = images.cuda()
#                 image_features = clip.encode_image(images)
#                 image_features /= image_features.norm(dim=-1, keepdim=True)
#                 features.append(image_features)
#         features = torch.cat(features)
#         torch.save(features, cfg['cache_dir'] + "/" + split + "_f_trained.pt")
#     else:
#         features = torch.load(cfg['cache_dir'] + "/" + split + "_f_trained.pt")
#     return features

def train_clip_features(cfg, split,loader,captions):
    clip = CLIP()
    trainclip_logits=[ ]
    to_pil_image = transforms.ToPILImage()  
    
    with torch.no_grad():
        for i, (images_tensor,_) in enumerate(tqdm(loader)): 
#             images_tensor形状是[64, 3, 224, 224]，说明pre_load_features，zero shot也是enumerate(tqdm(loader))这样得到所以也是224*224，这个没有关系应该
            batch_size = images_tensor.size(0)  
            # 创建一个空列表来保存这个批次的所有PIL图像  
            pil_images = []  
              
            # 将tensor转换为PIL图像，并保存在列表中  
            for j in range(batch_size):  
                single_image_tensor = images_tensor[j]  
                # if single_image_tensor.is_cuda:  
                #     single_image_tensor = single_image_tensor.cpu()  
                pil_image = to_pil_image(single_image_tensor)  
                pil_images.append(pil_image)
            for pil_image in pil_images: 
                # print(pil_image)
                logits,probs = clip.detect_image(pil_image, captions)
                #logits是没有softmax,probs是softmax后的，probs直接输出结果，logits之后用cls_result算出来
                values, indices = probs[0].topk(1)
                trainclip_logits.append(logits)
                
        # for i, (images,_) in enumerate(tqdm(loader)):
        #     images=to_pil_image(images)
        #     logits,probs = clip.detect_image(images, captions)
        #     #logits是没有softmax,probs是softmax后的，probs直接输出结果，logits之后用cls_result算出来
        #     values, indices = probs[0].topk(1)
        #     trainclip_logits.append(logits)
            
    trainclip_logits = torch.cat(trainclip_logits)
    trainclip_logits = trainclip_logits.to('cuda')
    return trainclip_logits
        
def cls_acc(output, target, topk=1):#target真实类别id
#     类别id，取决于您的模型输出output张量的结构
    pred = output.topk(topk, 1, True, True)[1].t()
#    每个样本预测的类别索引id.
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # 比较预测类别和真实标签
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    # 计算准确率：
    acc = 100 * acc / target.shape[0]
    # 计算并返回百分比形式的准确率：
    return pred,acc



def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def clip_research(captions, clip_model):
    with torch.no_grad():
        clip_weights = []

        for caption in captions:
            # Tokenize the prompts
            texts = caption
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights

def build_cache_model(cfg, clip_model, train_loader_cache):
    # 当你设置 cfg['load_cache'] = False 时，就
    # 构建并保存一个“缓存”模型，keys就是图像的特征，通过定义好的augment_epoch，对于每次遍历（即“增强周期”）
    #   它都会计算所有图像的特征，并将这些特征存储在 train_features 列表中。
    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        # 是把对应的标签，也就是id存进 cache_values
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")
    # 当你设置 cfg['load_cache'] = True 时，直接从磁盘上加载之前保存好的cache_model。
    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values

def build_cache_model_map(cfg, model, train_loader_cache):
    # 对于每个类别（共c个类别），将其K个支持样本的特征图进行池化（如平均池化），就是每个类别的所有特征图进行池化，形成一个支持特征矩阵S_c，大小为kr × d。对于每个支持样本的特征图，独立地应用池化操作（如平均池化）。经过池化操作后，每个支持样本的特征图都会被转换为一个较小尺寸的特征图，通常会将每个支持样本的池化后特征图“展平”成一个向量，然后将这些向量堆叠起来形成S_c，成了一个类别向量。
    if cfg['load_cache'] == False:
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_map = model.get_feature_map(images)
                    
                    train_features_map.append(image_map)
                    if augment_idx == 0:
                        # 是把对应的标签，也就是id存进 cache_values
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")
    # 当你设置 cfg['load_cache'] = True 时，直接从磁盘上加载之前保存好的cache_model。
    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values


def pre_load_features(cfg, split, clip_model, loader):
    # 当你设置 cfg['load_pre_feat'] = False 时，
    # 函数会遍历整个数据集，计算每张图像的特征，并将这些特征和对应的标签保存为 PyTorch 张量（.pt 文件）。
    if cfg['load_pre_feat'] == False:
        features, labels = [], []
        # target实际是类别的那个数字id.由DatasetWrapper(TorchDataset)的def __getitem__(self, idx):获得
        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
    # 如果你设置 cfg['load_pre_feat'] = True，函数则会跳过特征计算步骤，直接从磁盘上加载之前保存的特征和标签。
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        print(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")
    
    return features, labels

def pre_load_features_map(clip_model, images):
    features = []
        # target实际是类别的那个数字id.由DatasetWrapper(TorchDataset)的def __getitem__(self, idx):获得
    with torch.no_grad():
        image_features = clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        features.append(image_features)
    
    features = torch.cat(features)
    return features

def pre_load_features_test(cfg, split, clip_model, loader):
    # 当你设置 cfg['load_pre_feat'] = False 时，
    # 函数会遍历整个数据集，计算每张图像的特征，并将这些特征和对应的标签保存为 PyTorch 张量（.pt 文件）。
    if cfg['load_pre_feat'] == False:
        features = []

        with torch.no_grad():
            for i, (images) in enumerate(tqdm(loader)):
                images = images.cuda()
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
        features = torch.cat(features)
        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
    return features


def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None):

    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha
                _,acc = cls_acc(tip_logits, labels)
            
                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha


def search_hp2(cfg, features, labels, neg_logits,clip_weights, adapter=None):
# 最烦的两个矩阵的融合，并且不止一轮，FRN是最后所有轮的准确度取平均，所以不像CLIPlogist可以有一个完整的相似度矩阵，计算一个alpha，两者融合，
# 每一轮通过计算，得到一个我要的alpha,然后将两者矩阵权重融合TIP-LOGIST,然后所有轮的tiplogist取平均，，现在FRN的方法是每轮得到neglist,然后计算的准确度，取平均.
# 我就两个矩阵合起来,方法是在这个函数通过不断地找alpha,找到一个最好的alpha,使两个矩阵融合起来的logist矩阵,最终计算的准确度最高.
    if cfg['search_hp'] == True:
    
        
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_alpha = 0
        for alpha in alpha_list:
            clip_logits = 100. * features @ clip_weights
            tip_logits = clip_logits + neg_logits * alpha
            _,acc = cls_acc(tip_logits, labels)
            
            if acc > best_acc:
                # print("New best setting alpha: {:.2f}; accuracy: {:.2f}".format(alpha, acc))
                best_acc = acc
                best_alpha = alpha

        # print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_alpha

