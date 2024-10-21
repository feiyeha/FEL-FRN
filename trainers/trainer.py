import os
import sys
import torch
import torch.optim as optim
import logging
import numpy as np
import argparse
from tqdm import tqdm
from tensorboardX import SummaryWriter
from .eval import meta_test2
# 用 eval.py 文件中的 meta_test 函数。需要正确导入。
sys.path.append('..')
from datasetsf import dataloaders



def get_logger(filename):

    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",datefmt='%m/%d %I:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Formatter是一个格式化器：输出日志记录的布局 / 格式。
    fh = logging.FileHandler(filename,"w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
# %(asctime)s创建日志记录时的时间,形式：2024-1-22 13:50。
# %(asctime)s已经记录的消息。datefmt=' ' 时间。
# logger = logging.getLogger()：实例化，创建一个logger日志对象。
# setLever()：设置日志记录级别。
# logging.INFO输出日志的信息，程序正常运行产生的信息。
# 原文链接：https://blog.csdn.net/qq_42022648/article/details/135837965
# FileHandler()：将日志信息输入到磁盘文件filename上。
# 比较：StreamHandler()能够将日志信息输出到sys.stdout, sys.stderr 或者类文件对象（更确切点，就是能够支持write()和flush()方法的对象）
# setFormatter()：自定义日志格式
# addHandler()：logger日志对象加载FileHandler对象

def train_parser():
    # 命令行参数，创建解析器，ArgumentParser包含将命令行解析成Python数据类型所需的全部信息
    parser = argparse.ArgumentParser()

    ## general hyper-parameters
    # 添加参数
    # parser.add_argument('--sum'......)
    # 给一个ArgumentParser添加程序参数信息是通过调用add_argument()方法完成的。
    # opt是参数名
    # help： 一个此选项作用的简单描述，type：命令行参数应当被转换成的类型defualt：当参数未在命令行中出现时使用的值
    # choices：可用的参数的容器
    # action：当参数在命令行中出现时使用的动作基本类型
    # name or flags：一个命名或者一个选项字符串的列表，例如 foo 或 -f, --foo
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
    # parser.parse_args()解析参数

    return args



def get_opt(model,args):
    # 获得opt这个参数
    if args['opt'] == 'adam':
        optimizer = optim.Adam(model.parameters(),lr=args['lr'],weight_decay=args['weight_decay'])
    #     model.parameters()：模型的参数。
    # lr：学习率，默认值是0.001，控制每次参数更新的步长。
    # weight_decay：权重衰减，也称L2正则化项，默认值是0， 控制参数的幅度，防止过拟合
    elif args['opt'] == 'sgd':
        optimizer = optim.SGD(model.parameters(),lr=args['lr'],momentum=0.9,weight_decay=args['weight_decay'],nesterov=args['nesterov'])
        
    # SGD随机梯度下降
    # momentum(动量)：用过去梯度的moving average来更新参数，加快梯度下降的速度。
    # nesterov：Momentum的变种。与Momentum唯一区别就是，计算梯度的不同。Nesterov动量中，先用当前的速度临时更新一遍参数，再用更新的临时参数计算梯度。

    if args['decay_epoch'] is not None:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=args['decay_epoch'],gamma=args['gamma'])

    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=args['epoch'],gamma=args['gamma'])

    return optimizer,scheduler
# MultiStepLR：按需调整学习率。它会在指定的 milestones（即epoch的列表）处按 gamma 指定的因子调整学习率。
# milestones：每个元素代表何时调整学习率。
# gamma：学习率调整倍数，默认为 0.1 倍，即下降 10 倍。
# StepLR：等间隔调整学习率。它会在每个 step_size 后按 gamma 指定的因子调整学习率。
# step_size：学习率下降间隔数，若为 30，则会在 30、 60、 90…个 step 时，将学习率调整为 lr*gamma。
# last_epoch：上一个 epoch 数，这个变量用来指示学习率是否需要调整。当last_epoch 符合设定的间隔时，就会对学习率进行调整。当为-1 时，学习率设置为初始值。



class Path_Manager:
    # 跟据给定的根目录（fewshot_path）和一组参数（args），它构建了训练集、测试集和验证集的路径
    # 就是设置训练集，测试机和验证集，通过输入的路径
    def __init__(self,fewshot_path,args):
        # 训练集的路径总是被设置为 fewshot_path 下的 'train' 子目录。
        self.train = os.path.join(fewshot_path,'train')
        # 假如args.pre存在，说明用的是预处理数据集，以及是否忽略验证集
        if args['pre']:
            self.test = os.path.join(fewshot_path,'test_pre')
            self.val = os.path.join(fewshot_path,'val_pre') if not args['no_val'] else self.test
        # 如果 args.no_val 也为真（即不需要验证集），则验证集的路径被设置为与测试集相同（即 'test_pre'）。
        # 否则，验证集的路径被设置为 fewshot_path 下的 'val_pre' 子目录。
        else:
            self.test = os.path.join(fewshot_path,'test')
            self.val = os.path.join(fewshot_path,'val') if not args['no_val'] else self.test



class Train_Manager:
    # 就是根据获取的参数，实装选择那些东西,构造函数
    def __init__(self,args,path_manager,train_func):

        seed = args['seed']#设置随机种子是为了确保每次生成固定的随机数，这就使得每次实验结果显示一致了，有利于实验的比较和改进。使得每次运行该 .py 文件时生成的随机数相同。
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # torch.manual_seed(seed)：为CPU设置种子，生成随机数。
        # torch.cuda.manual_seed(seed)：为特定GPU设置种子，生成随机数。
        np.random.seed(seed)
        # 生成指定随机数。
        torch.cuda.set_device(args['gpu'])
        # 把模型和数据加载到对应的GPU

        if args['resnet']:
            name = 'ResNet-12'
        else:
            name = 'Conv-4'

        if args['detailed_name']:
            if args['decay_epoch'] is not None:
                temp = ''
                for i in args['decay_epoch']:
                    temp += ('_'+str(i))

                suffix = '%s-lr_%.0e-gamma_%.0e-epoch_%d-drop%s-decay_%.0e-way_%d' % (args['opt'],
                    args['lr'],args['gamma'],args['epoch'],temps,args['weight_decay'],args['train_way'])
            else:
                suffix = '%s-lr_%.0e-gamma_%.0e-epoch_%d-stage_%d-decay_%.0e-way_%d' % (args['opt'],
                    args['lr'],args['gamma'],args['epoch'],args['stage'],args['weight_decay'],args['train_way'])

            name = "%s-%s"%(name,suffix)
        # 首先，检查args.detailed_name是否为真（即，是否需要构建详细的名称）。如果是，继续执行以下步骤。
        # 接下来，检查是否指定了学习率衰减的周期（args.decay_epoch）。如果指定了，就遍历这些周期，将它们转换为字符串并拼接起来，每个周期前添加一个下划线（_），作为temp变量的值。
        # 如果指定了decay_epoch，则构建的后缀（suffix）将包括优化器类型（args.opt）、初始学习率（args.lr）、学习率衰减率（args.gamma）、总训练周期（args.epoch）、拼接好的衰减周期（temp）、权重衰减（args.weight_decay）和训练时的类别数（args.train_way）。这里使用%.0e来格式化浮点数，使其以科学计数法的形式显示，但不包含小数部分。
        # 如果没有指定decay_epoch，则构建的后缀将不包括衰减周期（temp），而是使用args.stage来代替。这表示可能有一个不同的参数或阶段来控制学习率或其他训练过程。
        # 最后，将原始名称（name）和构建好的后缀（suffix）通过%s-%s格式化字符串拼接起来，形成最终的详细名称。
        self.logger = get_logger('%s.log' % (name))
        self.save_path = 'model_%s.pth' % (name)
        self.writer = SummaryWriter('log_%s' % (name))

        self.logger.info('display all the hyper-parameters in args:')
        for key, value in args.items():  
            # 检查值是否非空  
            if value is not None:  
                # 使用日志记录器记录信息  
                self.logger.info(f'{key}: {value}')  
        self.logger.info('------------------------')
        self.args = args
        self.train_func = train_func
        self.pm = path_manager

    def train(self,model):
        # 获取训练的参数
        args = self.args
        train_func = self.train_func
        writer = self.writer
        save_path = self.save_path
        logger = self.logger

        optimizer,scheduler = get_opt(model,args)

        val_shot = args['train_shot']
        test_way = args['test_way']
# train 10way5shot为支撑，10way15shot为查询
# val   5way5shot为支撑，5way16shot为查询
# test  5way1shot,5shot 16shot为支撑，5way16shot为查询
        best_val_acc = 0
        best_epoch = 0

        model.train()
        model.cuda()

        iter_counter = 0

        if args['decay_epoch'] is not None:
            total_epoch = args['epoch']
        else:
            total_epoch = args['epoch']*args['stage']

        logger.info("start training!")

        for e in tqdm(range(total_epoch)):

            iter_counter,train_acc = train_func(model=model,
                                                optimizer=optimizer,
                                                writer=writer,
                                                iter_counter=iter_counter)

            if (e+1)%args['val_epoch']==0:

                logger.info("")
                logger.info("epoch %d/%d, iter %d:" % (e+1,total_epoch,iter_counter))
                logger.info("train_acc: %.3f" % (train_acc))

                model.eval()
                with torch.no_grad():
                    val_acc,val_interval = meta_test2(data_path=self.pm.val,
                                                    model=model,
                                                    way=test_way,
                                                    shot=val_shot,
                                                    pre=args['pre'],
                                                    transform_type=args['test_transform_type'],
                                                    query_shot=args['test_query_shot'] ,
                                                    trial=args['val_trial'])
                    writer.add_scalar('val_%d-way-%d-shot_acc'%(test_way,val_shot),val_acc,iter_counter)

                logger.info('val_%d-way-%d-shot_acc: %.3f\t%.3f'%(test_way,val_shot,val_acc,val_interval))

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = e+1
                    if not args['no_val']:
                        torch.save(model.state_dict(),save_path)
                    logger.info('BEST!')

                model.train()

            scheduler.step()

        logger.info('training finished!')
        if args['no_val']:
            torch.save(model.state_dict(),save_path)

        logger.info('------------------------')
        logger.info(('the best epoch is %d/%d') % (best_epoch,total_epoch))
        logger.info(('the best %d-way %d-shot val acc is %.3f') % (test_way,val_shot,best_val_acc))


    def evaluate(self,model):

        logger = self.logger
        args = self.args

        logger.info('------------------------')
        logger.info('evaluating on test set:')

        with torch.no_grad():

            model.load_state_dict(torch.load(self.save_path))
            model.eval()

            for shot in args['test_shot']:

                mean,interval = meta_test2(data_path=self.pm.test,
                                        model=model,
                                        way=args['test_way'],
                                        shot=shot,
                                        pre=args['pre'],
                                        transform_type=args['test_transform_type'],
                                        query_shot=args['test_query_shot'],
                                        trial=1000)

                logger.info('%d-way-%d-shot acc: %.2f\t%.2f'%(args['test_way'],shot,mean,interval))
