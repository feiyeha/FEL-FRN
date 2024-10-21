import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.nn import NLLLoss,BCEWithLogitsLoss,BCELoss


def auxrank(support):
    way = support.size(0)
    shot = support.size(1)
    support = support/support.norm(2).unsqueeze(-1)
    L1 = torch.zeros((way**2-way)//2).long().cuda()
    L2 = torch.zeros((way**2-way)//2).long().cuda()
    counter = 0
    for i in range(way):
        for j in range(i):
            L1[counter] = i
            L2[counter] = j
            counter += 1
    s1 = support.index_select(0, L1) # (s^2-s)/2, s, d
    s2 = support.index_select(0, L2) # (s^2-s)/2, s, d
    dists = s1.matmul(s2.permute(0,2,1)) # (s^2-s)/2, s, s
    assert dists.size(-1)==shot
    frobs = dists.pow(2).sum(-1).sum(-1)
    return frobs.sum().mul(.03)
# 它接受一个名为 support 的张量作为输入，(way, shot, d) 的张量，其中 way 是支持集中类别的数量，shot 是每个类别的样本数量，d 是样本的特征维度。
# 并计算一个基于该支持集（support set）中样本之间距离的度量，最终返回一个标量值。这个函数主要用于评估或比较支持集中样本之间的相似性或差异性，
# 尽管这里的“距离”并不是传统意义上的欧氏距离或曼哈顿距离，而是基于样本特征向量的点积（在归一化后）的平方和
# 默认训练
def default_train(train_loader,model,optimizer,writer,iter_counter):
    
    way = model.way
    query_shot = model.shots[-1]
    target = torch.LongTensor([i//query_shot for i in range(query_shot*way)]).cuda()
    # target该张量用于计算分类损失，它包含了查询集（query set）中每个样本的真实类别标签
    criterion = nn.NLLLoss().cuda()
    # 设置损失函数为负对数似然损失（NLLLoss），并获取当前学习率 lr。
    lr = optimizer.param_groups[0]['lr']

    writer.add_scalar('lr',lr,iter_counter)
    writer.add_scalar('scale',model.scale.item(),iter_counter)
    writer.add_scalar('alpha',model.r[0].item(),iter_counter)
    writer.add_scalar('beta',model.r[1].item(),iter_counter)
    # 使用 writer 记录当前的学习率、模型中的某些参数（如 scale、alpha、beta），这些参数可能是模型特有的超参数或学习到的参数。
    avg_frn_loss = 0
    avg_aux_loss = 0
    avg_loss = 0
    avg_acc = 0

    for i, (inp,_) in enumerate(train_loader):

        iter_counter += 1
        inp = inp.cuda()
        log_prediction, s = model(inp)
        frn_loss = criterion(log_prediction,target)
        aux_loss = auxrank(s)
        #：使用criterion(log_prediction, target)计算，其中criterion是一个损失函数（如交叉熵损失），target是真实标签
        loss = frn_loss + aux_loss
        # 计算分类损失（frn_loss）和辅助损失（aux_loss，这里使用 auxrank 函数）。
        optimizer.zero_grad()
        # 两个损失相加得到总损失 loss，首先，清除之前的梯度：optimizer.zero_grad()。
        # 然后，对总损失执行反向传播：loss.backward()。
        # 最后，使用优化器更新模型参数：optimizer.step()
        loss.backward()
        optimizer.step()

        _,max_index = torch.max(log_prediction,1)
        acc = 100*torch.sum(torch.eq(max_index,target)).item()/query_shot/way

        avg_acc += acc
        avg_frn_loss += frn_loss.item()
        avg_aux_loss += aux_loss.item()
        avg_loss += loss.item()

    avg_acc = avg_acc/(i+1)
    avg_loss = avg_loss/(i+1)
    avg_aux_loss = avg_aux_loss/(i+1)
    avg_frn_loss = avg_frn_loss/(i+1)

    writer.add_scalar('total_loss',avg_loss,iter_counter)
    writer.add_scalar('frn_loss',avg_frn_loss,iter_counter)
    writer.add_scalar('aux_loss',avg_aux_loss,iter_counter)
    writer.add_scalar('train_acc',avg_acc,iter_counter)

    return iter_counter,avg_acc
# 一个迭代计数器 iter_counter

def pre_train(train_loader,model,optimizer,writer,iter_counter):
    # 您定义了一个预训练过程，其中涉及到模型的预训练阶段，
    # 使用指定的优化器和损失函数来更新模型参数，并记录一些关键指标如学习率、模型参数（假设model中有scale、r等属性）、损失和准确率
    lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('lr',lr,iter_counter)
    writer.add_scalar('scale',model.scale.item(),iter_counter)
    writer.add_scalar('alpha',model.r[0].item(),iter_counter)
    writer.add_scalar('beta',model.r[1].item(),iter_counter)
    criterion = NLLLoss().cuda()
    # 您使用了NLLLoss()（负对数似然损失），这是处理分类问题时的一个常见选择，特别是当输出是类别的对数概率时。
    # 请确保log_prediction的输出与NLLLoss的要求相匹配（即每个类别的对数概率，并且目标target是类别索引）
    avg_loss = 0
    avg_acc = 0

    for i, (inp,target) in enumerate(train_loader):

        iter_counter += 1
        batch_size = target.size(0)
        target = target.cuda()

        inp = inp.cuda()
        log_prediction = model.forward_pretrain(inp)
        
        loss = criterion(log_prediction,target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _,max_index = torch.max(log_prediction,1)
        acc = 100*(torch.sum(torch.eq(max_index,target)).float()/batch_size).item()

        avg_acc += acc
        avg_loss += loss.item()

    avg_loss = avg_loss/(i+1)
    avg_acc = avg_acc/(i+1)

    writer.add_scalar('pretrain_loss',avg_loss,iter_counter)
    writer.add_scalar('train_acc',avg_acc,iter_counter)

    return iter_counter,avg_acc
