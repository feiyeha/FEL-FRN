import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .backbones import Conv_4, ResNet


class FRN(nn.Module):

    def __init__(self, way=None, shots=None, resnet=False, is_pretraining=False, num_cat=None):

        super().__init__()

        if resnet:
            num_channel = 640
            self.feature_extractor = ResNet.resnet12()

        else:
            num_channel = 64
            self.feature_extractor = Conv_4.BackBone(num_channel)

        self.shots = shots
        self.way = way
        self.resnet = resnet

        # number of channels for the feature map, correspond to d in the paper
        self.d = num_channel

        # temperature scaling, correspond to gamma in the paper
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

        # H*W=5*5=25, resolution of feature map, correspond to r in the paper
        self.resolution = 25

        # correpond to [alpha, beta] in the paper
        # if is during pre-training, we fix them to 0
        self.r = nn.Parameter(torch.zeros(2), requires_grad=not is_pretraining)

        if is_pretraining:
            # number of categories during pre-training
            self.num_cat = num_cat
            # category matrix, correspond to matrix M of section 3.6 in the paper
            self.cat_mat = nn.Parameter(torch.randn(self.num_cat, self.resolution, self.d), requires_grad=True)

    def get_feature_map(self, inp):
        # 总的来说，这个get_feature_map函数通过自定义的特征提取器提取特征图，
        # 并根据self.resnet的值可能进行归一化处理，然后将特征图重新塑形并调整维度顺序以适应后续处理的需求。
        batch_size = inp.size(0)      #我设置way=10，shot=5,queryshot=15,那么每一次就要选出10*（5+15）=200 way=5，那就是5*（15+5）=100这就是验证集100由来
        # print(batch_size)  #200                                                       #5*（5+16）=105   
        feature_map = self.feature_extractor(inp)
        # print(feature_map.shape)  #200*640*5*5
        # 通过上面骨干网络获得特征，我得改成VIT
        if self.resnet:
            feature_map = feature_map / np.sqrt(640)

        return feature_map.view(batch_size, self.d, -1).permute(0, 2, 1).contiguous()  # N,HW,C

    def get_recon_dist(self, query, support, alpha, beta, Woodbury=True):
        #     核心代码
        # query: way*query_shot*resolution, d   查询集
        # support: way, shot*resolution , d  支持集
        # Woodbury: whether to use the Woodbury Identity as the implementation or not
        # 实现了一个基于支持集（support set）和查询集（query set）之间的重构距离计算的过程
        # 这个函数的核心在于通过支持集来重构查询集，并计算重构后的查询集与原始查询集之间的距离，以此作为分类或识别的依据。
        # alpha, beta: 可学习的参数，用于调整正则化和重构的权重。
        # correspond to kr/d in the paper
        reg = support.size(1) / support.size(2)
        # 计算正则化项中的比例因子，reg可能是指support中每个类别的样本数与特征维度的比例
        # 计算正则化项lam，通过alpha的指数变换并乘以reg，然后加上一个小常数防止除以零
        # correspond to lambda in the paper
        lam = reg * alpha.exp() + 1e-6
        #    # 计算重构权重rho，通过beta的指数变换得到
        # correspond to gamma in the paper
        rho = beta.exp()
        # 对support进行转置
        st = support.permute(0, 2, 1)  # way, d, shot*resolution

        if Woodbury:
            # correspond to Equation 10 in the paper
            # 使用Woodbury恒等式优化计算
            # 首先计算support的转置与support的矩阵乘法，得到[way, d, d]的协方差矩阵
            sts = st.matmul(support)  # way, d, d
            m_inv = (sts + torch.eye(sts.size(-1)).to(sts.device).unsqueeze(0).mul(lam)).inverse()  # way, d, d
            hat = m_inv.matmul(sts)  # way, d, d

        else:
            # correspond to Equation 8 in the paper

            sst = support.matmul(st)  # way, shot*resolution, shot*resolution
            m_inv = (sst + torch.eye(sst.size(-1)).to(sst.device).unsqueeze(0).mul(
                lam)).inverse()  # way, shot*resolution, shot*resolutionsf
            hat = st.matmul(m_inv).matmul(support)  # way, d, d

        Q_bar = query.matmul(hat).mul(rho)  # way, way*query_shot*resolution, d
       
        # 计算重构后的查询集特征Q_bar与原始查询集特征query之间的距离
        # 首先，将query增加一个维度以匹配Q_bar的维度，然后计算两者之间的差的平方和，最后对最后一个维度求和得到距离
        # 注意：这里的permute(1,0)是为了将距离矩阵的维度从[way, way*query_shot*resolution*height*width]调整为[way*query_shot*resolution*height*width, way]
        dist = (Q_bar - query.unsqueeze(0)).pow(2).sum(2).permute(1, 0)  # way*query_shot*resolution, way

        return dist

    # 主功能函数，get_recon_dist，get_feature_map定义后，都是在这里面使用
    def get_neg_l2_dist(self, inp, way, shot, query_shot, return_support=False):
        # 这个函数的功能：通过使用get_recon_dist（这个是计算距离的方法定义），获得查询集（query set）样本对于每个支持集（support set）类别的重构距离（reconstruction distance），并将这些距离取负后，取平均，最终得到每个查询样本对每个支持集类别的平均负重构距离。
        # 类的实例变量中获取了resolution,特征图的分辨率,d（特征维度），以及两个正则化参数alpha和beta
        resolution = self.resolution
        d = self.d
        alpha = self.r[0]
        beta = self.r[1]
        # 提取特征图输入数据inp中提取特征图（feature map）
        feature_map = self.get_feature_map(inp)
        # print(feature_map.shape)   #200*25*640=3200000   200  batch_size*resolution*d
        # 这是训练的时候200*25*640,到了val验证的时候就变成100*25*640=1600000
        # 将特征图分割成支持集和查询集两部分
        # print(way)
        # print(shot)
        # print(query_shot)
        # print(resolution)
        # print(d)
#         训练时是10 5 15 25 640，和我设定的一样    10*5*25*640=800000    10*15*25*640=2400000 加起来3200000
# val时5 5 15 25 640
# 5*5*25*640 =400000 5*15*25*640=1200000    
# 花的数据集在运行的时候是100*25*640=1680000
                # 所以400000+1280000=1680000就可以跑，问题是为甚么，我这个数据集val的batchsize100,另外一个batchsize105，我懂了，这个根据你的数据自动分配的，而真正出的问题在下面。
    
    # 我懂了我的数据集的验证集有问题，我的飞机数据集中每个类别的图像只有20张，验证的时候5张作为支撑集，16张作为查询集，但我每个只有20张，5+16=21图像不够。
    # 我本来以为是根据我的query_shot调整，结果无论输多少都是105，
        support = feature_map[:way * shot].view(way, shot * resolution, d)
        query = feature_map[way * shot:].view(way * query_shot * resolution, d)
        # 从特征图 分出支持集和查询集的特征，支持集特征由0-way*shot元素的组成并重新塑形为(way, shot*resolution, d)的形状，
        # 其中way是类别数，shot*resolution是每个类别的样本数乘以分辨率（假设每个样本都被展平为一个向量），d是特征维度
        # 通过self.get_recon_dist方法，利用支持集特征来重构查询集特征，并计算重构后的距离
        # 调用self.get_recon_dist方法（之前定义的方法）来计算查询集特征和经过支持集特征重构后的图的特征的距离
        recon_dist = self.get_recon_dist(query=query, support=support, alpha=alpha,
                                         beta=beta)  # way*query_shot*resolution, way
        neg_l2_dist = recon_dist.neg().view(way * query_shot, resolution, way).mean(1)  # way*query_shot, way
        # 首先，对recon_dist取负值（因为通常我们希望最小化距离，而recon_dist可能表示的是某种形式的“距离”，其值越小表示越相似），
        # 然后使用view方法将其重新塑形为(way*query_shot, resolution, way)的形状。
        # 这里可能有一个假设，即原始的resolution实际上是指每个查询样本在展平前的空间维度（如高度和宽度的乘积），但在这一步中它被用作了一个新的维度来组织数据。
        # 这个mean(1)就是有点像在聚合所有的同类别样本。
        # 接着，对第二个维度（即resolution维度）求均值，得到每个查询样本对每个类别的平均负L2距离，最终形状为(way*query_shot, way)。
        # return_support指示是否返回支持集特征
        if return_support:
            return neg_l2_dist, support
        else:
            return neg_l2_dist

    def meta_test(self, inp, way, shot, query_shot):
        # 导入get_neg_l2_dist方法，就能得到
        neg_l2_dist = self.get_neg_l2_dist(inp=inp,
                                           way=way,
                                           shot=shot,
                                           query_shot=query_shot)
        
        _, max_index = torch.max(neg_l2_dist, 1)
        # 获得最大概率的索引，
        return max_index,neg_l2_dist

    def forward_pretrain(self, inp):

        feature_map = self.get_feature_map(inp)
        batch_size = feature_map.size(0)

        feature_map = feature_map.view(batch_size * self.resolution, self.d)

        alpha = self.r[0]
        beta = self.r[1]

        recon_dist = self.get_recon_dist(query=feature_map, support=self.cat_mat, alpha=alpha,
                                         beta=beta)  # way*query_shot*resolution, way

        neg_l2_dist = recon_dist.neg().view(batch_size, self.resolution, self.num_cat).mean(1)  # batch_size,num_cat

        logits = neg_l2_dist * self.scale
        log_prediction = F.log_softmax(logits, dim=1)

        return log_prediction

    def forward(self, inp):

        neg_l2_dist, support = self.get_neg_l2_dist(inp=inp,
                                                    way=self.way,
                                                    shot=self.shots[0],
                                                    query_shot=self.shots[1],
                                                    return_support=True)

        logits = neg_l2_dist * self.scale
        log_prediction = F.log_softmax(logits, dim=1)

        return log_prediction, support
