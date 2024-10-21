import os
import random
import argparse
import yaml
from tqdm import tqdm
import json
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from trainers import trainer, frn_train
from functools import partial
from datasetsf import dataloaders
from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *
from Utils import util
from trainers.eval import meta_test
from clip import longclip
from clip.FRN import FRN

def save_hyperparameters(best_beta, best_alpha, filename='best_hyperparameters.txt'):
    with open(filename, 'w') as f:
        f.write(f'Best beta: {best_beta}\n')
        f.write(f'Best alpha: {best_alpha}\n')


def save_hyperparameters_F(best_beta, best_alpha, filename='best_hyperparameters_F.txt'):
    with open(filename, 'w') as f:
        f.write(f'Best beta: {best_beta}\n')
        f.write(f'Best alpha: {best_alpha}\n')


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    parser.add_argument("--opt",help="optimizer",choices=['adam','sgd'])
    parser.add_argument("--lr",help="initial learning rate",type=float)
    parser.add_argument("--gamma",help="learning rate cut scalar",type=float,default=0.1)
    parser.add_argument("--epoch",help="number of epochs before lr is cut by gamma",type=int)
    parser.add_argument("--stage",help="number lr stages",type=int)
    parser.add_argument("--weight_decay",help="weight decay for optimizer",type=float)
    parser.add_argument("--gpu",help="gpu device",type=int,default=0)
    parser.add_argument("--seed",help="random seed",type=int,default=42)
    parser.add_argument("--val_epoch",help="number of epochs before eval on val",type=int,default=20)
    parser.add_argument("--resnet", help="whether use resnet12 as backbone or not",action="store_true")
    parser.add_argument("--nesterov",help="nesterov for sgd",action="store_true")
    parser.add_argument("--batch_size",help="batch size used during pre-training",type=int)
    parser.add_argument('--decay_epoch',nargs='+',help='epochs that cut lr',type=int)
    parser.add_argument("--pre", help="whether use pre-resized 84x84 images for val and test",action="store_true")
    parser.add_argument("--no_val", help="don't use validation set, just save model at final timestep",action="store_true")
    parser.add_argument("--train_way",help="training way",type=int)
    # 训练类别数
    parser.add_argument("--test_way",help="test way",type=int,default=5)
    # train_shot：在元学习的验证过程中，对于元训练和元测试阶段，每个类别所使用的支持集图像的数量”
    parser.add_argument("--train_shot",help="number of support images per class for meta-training and meta-testing during validation",type=int)
    parser.add_argument("--test_shot",nargs='+',help="number of support images per class for meta-testing during final test",type=int)
    parser.add_argument("--train_query_shot",help="number of query images per class during meta-training",type=int,default=15)
    parser.add_argument("--test_query_shot",help="number of query images per class during meta-testing",type=int,default=16)
    parser.add_argument("--train_transform_type",help="size transformation type during training",type=int)
    parser.add_argument("--test_transform_type",help="size transformation type during inference",type=int)
    parser.add_argument("--val_trial",help="number of meta-testing episodes during validation",type=int,default=1000)
    parser.add_argument("--detailed_name", help="whether include training details in the name",action="store_true")

    args = parser.parse_args()
# 这些参数什么时候接受，就是从命令行运行python main.py时候输入这些参数的值时接收,输入test_query_shot不指定就使用默认值，而config文件，在文件中一定要写，不写就是没有引用参数，不会自动用默认值
    return args

def run_FRN_train(args):
    # 用frn，要么从头训练，训练出来模型提取特征图
    # 之后还有一个，用已经训练好的模型提取特征

    with open('config.yml', 'r') as f:
        temp = yaml.safe_load(f)
    # # 先加载trainer的那些参数
    data_path = os.path.abspath(temp['data_path'])
    fewshot_path = os.path.join(data_path, 'Aircraft_fewshot')
    print("使用数据集为{}".format(fewshot_path))

    pm = trainer.Path_Manager(fewshot_path=fewshot_path, args=args)
    # 构建数据集的验证集和测试集
    train_way = args['train_way']
    shots = [args['train_shot'], args['train_query_shot']]
    # 一个数据集加载器 样本数
    train_loader = dataloaders.meta_train_dataloader(data_path=pm.train,
                                                     way=train_way,
                                                     shots=shots,
                                                     transform_type=args['train_transform_type'])
    # print("\nConstructing cache model by few-shot visual features and labels.")
    # cache_keys, cache_values = build_cache_model_map(cfg, model, train_loader_cache)
    model = FRN(way=train_way,
                shots=[args['train_shot'], args['train_query_shot']],
                resnet=args['resnet'])

    train_func = partial(frn_train.default_train, train_loader=train_loader)
    # 创建一个 Train_Manager 类的实例 tm，就是一个训练管理器
    tm = trainer.Train_Manager(args, path_manager=pm, train_func=train_func)
    # 使用训练管理器中的训练功能，也就是训练的函数，下面就是评估的函数，
    tm.train(model)

    tm.evaluate(model)

def run_FRN_trained(args,clip_model):
    # 之后还有一个，用已经训练好的模型提取特征图，预先提取好，之后
    with open('config.yml', 'r') as f:
        temp = yaml.safe_load(f)
    data_path = os.path.abspath(temp['data_path'])
    alpha = args['init_alpha']
    test_path1 = os.path.join(data_path,'Aircraft_fewshot/test_pre')
    test_path2 = os.path.join(data_path,'Aircraft_fewshot/test')
#     测试集所用图片的路径
# 使用的模型
    # model_path = './model_ResNet-12.pth'
    model_path = 'trained_model_weights/CUB_fewshot_raw/FRN/ResNet-12/model.pth'

    gpu = 0
    torch.cuda.set_device(gpu)
# 实例化这个模型架构，然后下面把参数导入进去,模型加载好了，然后就可以提取特征图了，首先既然能提取了，那就构建cache model.
    model = FRN(resnet=True) 
    model.cuda()
    model.load_state_dict(torch.load(model_path,map_location=util.get_device_map(gpu)),strict=True)
    model.eval()   
    #  # Construct the cache model by few-shot training set，就算训练了，使用训练集的所有数据构成
    # print("\nConstructing cache model by few-shot visual features and labels.")
    # cache_keys, cache_values = build_cache_model_map(cfg,model, train_loader_cache)
    # # Pre-load val features maps
    # # 通过pre_load_features map函数，从验证集（val set）经过模型中提取视觉特征和标签。
    # # 这些特征通常是通过一个预训练的模型（如clip_model）对验证集中的图片进行前向传播得到的。
    # print("\nLoading visual features maps and labels from val set.")
    # val_features, val_labels = pre_load_features(cfg, "val", model, val_loader)
    # # Pre-load test features
    # # 与验证集类似，但这次是从测试集中加载。测试集通常用于评估模型的性能，但在某些场景下（如少样本学习），它也可能被用于模型训练或微调。
    # print("\nLoading visual features and labels from test set.")
    # test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)
#   只用了FRN
    with torch.no_grad():
        way = 5
#         计算测试集时中的支持集选择使用50个类别测试，每个类别都有16作为支持集，还有用16个样本作为查询集时的准确率
# 但这个太大，每次随机5个，10000次终归50个类别都会选到吧,,他这个测试集，完全就是测试效率的，不看预测结果。
# 这个代码好像只能识别84*84像素，所以pre是处理过的，84*84，没有pre是保留的原图
        for shot in [1,5,16]:
            mean,interval = meta_test(cfg=args,
                                      data_path1=test_path1,
                                      data_path2=test_path2,
                                      clip_model=clip_model,
                                      alpha=alpha,
                                      init_alpha=args['init_alpha'],
                                      model=model,
                                      way=way,
                                      shot=shot,
                                      pre=True,
                                      transform_type=None,
                                      trial=1000)
#             计算测试集时中的支持集可以选择使用随机5个类别测试，每个类别都有1，5，10个样本时的准确率
# 说白了就是测试时，做10000次相同的操作。先从所有类别中随机5个类别，其中每个类别使用1，5，10个样本作为支持集，还有默认16个作为预测的查询集，一次任务下来，
# 得到一个neg_l2_dist，inp是这次所有查询集的样本
            print('%d-way-%d-shot acc: %.3f\t%.3f'%(way,shot,mean,interval))
#     现在看能不能得到相似度矩阵然后和那个CLIP结果加权
            # print(matrix.size())
    
    
def run_test(args,clip_model):
    # 之后还有一个，用已经训练好的模型提取特征图，预先提取好，之后
    with open('config.yml', 'r') as f:
        temp = yaml.safe_load(f)
    data_path = os.path.abspath(temp['data_path'])
    alpha = args['init_alpha']
    test_path1 = os.path.join(data_path,'CUB_fewshot_raw/test_pre')
    test_path2 = os.path.join(data_path,'CUB_fewshot_raw/test')
#     测试集所用图片的路径
# 使用的模型
    # model_path = './model_ResNet-12.pth'
    model_path = 'trained_model_weights/CUB_fewshot_raw/FRN/ResNet-12/model.pth'

    gpu = 0
    torch.cuda.set_device(gpu)
# 实例化这个模型架构，然后下面把参数导入进去,模型加载好了，然后就可以提取特征图了，首先既然能提取了，那就构建cache model.
    model = FRN(resnet=True) 
    model.cuda()
    model.load_state_dict(torch.load(model_path,map_location=util.get_device_map(gpu)),strict=True)
    model.eval()   
#   只用了FRN
    with torch.no_grad():
        way = 20
#         这样测试输出图像的时候，有几个类别就写几个way，有几个shot就写几个，一般5个，然后跑一次
        for shot in [5]:
            mean,interval = meta_test(cfg=args,
                                      data_path1=test_path1,
                                      data_path2=test_path2,
                                      clip_model=clip_model,
                                      alpha=alpha,
                                      init_alpha=args['init_alpha'],
                                      model=model,
                                      way=way,
                                      shot=shot,
                                      pre=True,
                                      transform_type=None,
                                      trial=1)
#             计算测试集时中的支持集可以选择使用随机5个类别测试，每个类别都有1，5，10个样本时的准确率
# 说白了就是测试时，做10000次相同的操作。先从所有类别中随机5个类别，其中每个类别使用1，5，10个样本作为支持集，还有默认16个作为预测的查询集，一次任务下来，
# 得到一个neg_l2_dist，inp是这次所有查询集的样本
            print('%d-way-%d-shot acc: %.3f\t%.3f'%(way,shot,mean,interval))
#     现在看能不能得到相似度矩阵然后和那个CLIP结果加权
            # print(matrix.size())    

    

def run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights):
    print("\n-------- Searching hyperparameters on the val set. --------")

    # 是提取好的， val_features验证集所有图像的特征clip_weights所有文本的特征
    # Zero-shot CLIP
    clip_logits = 100. * val_features @ clip_weights
    # print(clip_logits)
    _,acc = cls_acc(clip_logits, val_labels)
    print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(acc))
    # acc = cls_acc(trainclip_vallogits, val_labels)
    # print("\n****Trained CLIP's val accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    # 初始权重或偏置
    beta, alpha = cfg['init_beta'], cfg['init_alpha']

    # 我们计算这个测试图像的特征与cache model中所有keys（即few-shot训练图像的特征）的相似度,使用余弦相似度点乘来计算衡量。
    # 1.先对验证集图像进行重建，通过cache_keys中的每个类别的16张图片去重新构建验证集的图像，如果cache keys构建所用的类别和查询集的图像类别是一样的，
    # 那相对应其他类别重构的图像，正确类别构建的图像应该和查询集的原图像的相似度是最高的，通过这样的方法，我一样能获得测试图像对应每一个类别的相似度
    affinity = val_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    # 指数函数被用来将相似度转换为非负值,β调节其锐度

    # 总的分数,有clip和adapter的结果残差连接（加权求和），进行融合
    tip_logits = clip_logits + cache_logits * alpha
    _,acc = cls_acc(tip_logits, val_labels)
    # cls_acc 很可能是一个函数，它接受预测 logits 和真实标签作为输入，并返回分类准确度。
    # 该函数通常会首先将 logits 转换为概率分布（例如，通过 softmax 函数），然后计算预测类别（概率最高的类别）与真实标签之间的匹配度，以得到准确度。
    print("**** Tip-Adapter's val accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights)
    # 将超参数保存到本地文件
    # 调用函数保存超参数

    save_hyperparameters(best_beta, best_alpha)

    print("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    _,acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))
    # acc = cls_acc(trainclip_testlogits, test_labels)
    # print("\n****Trained CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values

    tip_logits = clip_logits + cache_logits * best_alpha

    _,acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))


def run_tip_adapter_F(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights,
                      clip_model, train_loader_F):
    # Enable the cached keys to be learnable,让keys可以被训练，所以创建一个线性层（即“adapter”）
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = nn.Parameter(cache_keys.t())  # 权重被初始化为 cache_keys 的转置

    # 初始化一个 AdamW 优化器和一个 CosineAnnealingLR 学习率调度器。
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    # 初始化一些变量来跟踪训练过程中的最佳准确率和当前学习率。
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(cfg['train_epoch']):
        # Train
        adapter.train()  # 这行代码是将 adapter（即您之前定义的线性层）设置为训练模式，有两种模式：训练模式和评估模式（通过调用 .eval() 方法设置）
        correct_samples, all_samples = 0, 0  # correct_samples 用于累加每个批次中正确分类的样本数，而 all_samples 用于记录处理过的总样本数。
        # 这两个变量将在每个批次后更新，以便在 epoch 结束时计算准确率。
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))  # 这行代码打印当前正在训练的 epoch 的信息，当前轮/总轮
        # 那边是训练好的，这边就实时训练
        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()  # 图像和标签数据从 CPU 转移到 GPU 上
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            affinity = adapter(image_features)  # 提好的特征经过定义的线性层,之后再经过TIP adapter的操作
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha

            loss = F.cross_entropy(tip_logits, target)  # 计算交叉熵损失

            _,acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)  # 更新正确样本数和总样本数
            all_samples += len(tip_logits)
            loss_list.append(loss.item())  # 将当前批次的损失值添加到 loss_list 列表中，以便后续可以分析或记录这些损失值。

            optimizer.zero_grad()
            # 使用优化器的 zero_grad 方法清除之前累积的梯度。这是因为在反向传播之前，我们需要确保梯度是干净的，否则之前的梯度会累积到新的梯度上，导致训练出现问题。
            loss.backward()
            # 反向传播
            optimizer.step()
            # 使用优化器的 step 方法来更新模型的参数
            scheduler.step()
            # 在此处调用 step 方法来更新学习率。这有助于在训练过程中动态地调整学习率，以提高模型的性能
        # 上面那个for 循环到此为止，这边就是训练的过程
        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples,
                                                                       correct_samples, all_samples,
                                                                       sum(loss_list) / len(loss_list)))

        # Eval
        adapter.eval()  # adapter.eval()

        # 前向传播:提取的特征在经过一层可学习的线性层
        affinity = adapter(test_features)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        clip_logits = 100. * test_features @ clip_weights
        tip_logits = clip_logits + cache_logits * alpha
        _,acc = cls_acc(tip_logits, test_labels)

        print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:  # best_acc最好的准确率最开始设置为0，所以后面慢慢更新，准确率变高，当然更新最好准确率best_acc，同时最好的epoch也更新我当前的训练的这轮
            best_acc = acc
            best_epoch = train_idx  # 然后保存准确率最高的这轮模型，准确率最高的这轮模型，权重肯定是好的，保存的仅仅是adapter的权重，而不是整个模型权重
            torch.save(adapter.weight, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    # 最大的for结束，这时候 adapter.weight里面加载训练中保存的最好的模型权重
    adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    print(f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    print("\n-------- Searching hyperparameters on the val set. --------")

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights,
                                      adapter=adapter)
    # 找到好的超参数
    save_hyperparameters_F(best_beta, best_alpha)
    print("\n-------- Evaluating on the test set. --------")

    affinity = adapter(test_features)
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    # 这边使用最高的超参数
    tip_logits = clip_logits + cache_logits * best_alpha
    _,acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(max(best_acc, acc)))


def main():
    # Load config file
    args = get_arguments()
#    args里存放的Namespace(config='configs/bird.yaml')，所以调用这个函数，就是存放你给的参数，然后他解析这些参数。
    
    # 通过调用get_arguments()函数，从命令行参数中获取配置文件的路径（args.config）
    assert (os.path.exists(args.config))
    print('所有你命令行没给出时所用的默认参数')
    print(args)
    # 使用assert语句来确保配置文件确实存在。如果配置文件不存在，程序将抛出异常并停止运行
    # 解析YAML配置文件
    # 使用yaml.load()
    # 函数读取配置文件的内容，并将其解析为一个Python字典对象（cfg）
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    # 根据配置字典cfg中的dataset键和预定义的caches目录前缀，构建一个完整的缓存目录路径（cache_dir）。
    # 使用os.makedirs()函数创建该目录，如果目录已存在则不会报错（exist_ok = True）。

    cache_dir = os.path.join('caches', cfg['dataset'])
    # 输出的模型权重文件的位置
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir
    # 更新配置字典cfg，添加或覆盖cache_dir键的值为刚才创建的缓存目录路径。
    print("\nRunning configs（config文件中给出的参数值）.")
    print(cfg, "\n")
    # template = 'a photo of a {}, a type of bird.'
    # with open('bird/cat_to_name.json', 'r',
    #           encoding='utf-8') as f:
    #     categories_with_descriptions = json.load(f)
    # text_inputs = list(categories_with_descriptions.values())
    # text_inputs = []
    # # 遍历text_inputs列表，并使用模板字符串填充每个类别名
    # for category in text_outputs:
    # # 使用.format()方法来插入类别名
    #     text_inputs.append(template.format(category))
    # CLIP
    # 从CLIP库中加载预训练的CLIP模型及其预处理函数。这里cfg['backbone']应该是一个指定了CLIP模型类型的字符串，例如'ViT-B/32'。
    # cfg就是获取配置yaml的字典，里面不就写了backbone嘛

    # clip_model, preprocess = longclip.load_from_clip(cfg['backbone'])
    clip_model, preprocess = longclip.load(cfg['backbone'])
    # clip_model, preprocess = clip.load(cfg['backbone'])

    # 调用clip_model.eval()将模型设置为评估模式，确保在后续的推理过程中不会进行如梯度计算等训练相关的操作。
    clip_model.eval()
    # 使用random.seed(1)和torch.manual_seed(1)来设置随机种子，以确保实验的可重复性
    # 确保在多次运行相同的实验或代码时.能够产生相同的结果，它允许研究人员更精确地比较不同的模型、算法或超参数设置，而不受随机性的影响。
    # Prepare dataset
    random.seed(1)
    torch.manual_seed(1)

    print("Preparing dataset.")
    dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])
    # 您调用了一个名为build_dataset的函数，该函数根据配置文件cfg中的参数来构建数据集。
    # build_data_loader函数来构建验证集、测试集的数据加载器。这个数据加载器将用于在验证阶段、测试阶段从数据集中加载数据
    val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess,
                                   shuffle=False)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess,
                                    shuffle=False)
   
    # 该流程专门用于训练过程中的数据增强
    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    # 加载训练集的数据train_loader_F 是由某个数据集和批处理大小创建的 DataLoader
    train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform,
                                           is_train=True, shuffle=False)
    train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True,
                                       shuffle=True)

    # Textual features
    # 这部分代码使用clip_classifier函数来获取CLIP模型的文本特征，并将其作为分类器的权重。dataset.classnames可能是数据集中所有类别的名称，
    # dataset.template可能是用于将类别名称转换为CLIP模型可以理解的文本描述的模板，而clip_model是已经加载的CLIP模型。
#     print("\nGetting textual features as CLIP's classifier.")
#     clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)
    
#     # Construct the cache model by few-shot training set
#     print("\nConstructing cache model by few-shot visual features and labels.")
#     cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache)
    # 准备FRN
   
    # run_FRN_train(cfg)
    run_FRN_trained(cfg,clip_model)
    # run_test(cfg,clip_model)
    # # Pre-load val features
    # # 通过pre_load_features函数，从验证集（val set）经过模型中提取视觉特征和标签。
    # # 这些特征通常是通过一个预训练的模型（如clip_model）对验证集中的图片进行前向传播得到的。
    # print("\nLoading visual features and labels from val set.")
    # val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader)
    # print("\n计算验证集图像和文本相似度，使用的是已经训练后的模型，提取图像特征和文本.")
    # # trainclip_vallogits=train_clip_features(cfg, "val",val_loader,text_inputs)
    # # Pre-load test features
    # # 与验证集类似，但这次是从测试集中加载。测试集通常用于评估模型的性能，但在某些场景下（如少样本学习），它也可能被用于模型训练或微调。
    # print("\nLoading visual features and labels from test set.")
    # test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)
    # # trainclip_testlogits=train_clip_features(cfg, "test",test_loader,text_inputs)

    # ------------------------------------------ Tip-Adapter ------------------------------------------
    # run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights,trainclip_vallogits,trainclip_testlogits)
    # run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights)
    # ------------------------------------------ Tip-Adapter-F ------------------------------------------
    # run_tip_adapter_F(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, clip_model, train_loader_F,trainclip_vallogits,trainclip_testlogits)
    # run_tip_adapter_F(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights,clip_model, train_loader_F)


if __name__ == '__main__':
    main()