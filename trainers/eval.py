import sys
import os
import shutil 
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('..')
from datasetsf import dataloaders
from tqdm import tqdm
from utils import clip_classifier,pre_load_features_map,cls_acc,search_hp2
from PIL import Image, ImageDraw, ImageFont
def get_score(acc_list):

    mean = np.mean(acc_list)
    # mean 代表平均值或均值。在您的函数中，mean 是通过计算准确率列表 acc_list 中所有值的算术平均数得到的。这是衡量数据集中心趋势的一个常用指标。
    interval = 1.96*np.sqrt(np.var(acc_list)/len(acc_list))
#     np.var(acc_list) 计算 acc_list 的方差，即每个准确率值与准确率平均值之差的平方的平均值。标准误差乘以1.96，您得到了一个区间宽度，这个宽度在95%的置信水平下，预计包含了总体均值的真实值
    # interval 就真正代表了一个区间，即一个包含平均值且有一定置信水平（如95%）的数值范围。
    # interval 应该被理解为 (interval_lower, interval_upper)，其中 interval_lower 是置信区间的下限，interval_upper 是置信区间的上限。
    # 如果我们多次从总体中随机抽取样本并计算置信区间，那么大约有95 % 的置信区间会包含真实的总体平均年收入。
    return mean,interval
# 想象你有一个大罐子，里面装满了各种颜色的豆子（代表总体），但你只能看到罐子外面的一小部分豆子（代表样本）。你想知道罐子里所有豆子的平均颜色是什么，但你不能把罐子里的所有豆子都倒出来看。
#
# 于是，你随机抓了一把豆子（样本），计算了这些豆子的平均颜色，并基于这个样本数据估计了整个罐子里豆子的平均颜色可能是什么范围（置信区间）。
#
# 你告诉别人说：“我有95%的信心认为罐子里所有豆子的平均颜色会落在我估计的这个颜色范围内。” 这意味着，如果你多次随机抓豆子并计算平均颜色，然后每次都估计一个颜色范围，那么大约有95次你的估计范围会包含罐子里所有豆子的真实平均颜色。
# 多次随机选取样本，并计算平均类别分为，然后每次都估计一个类别范围
#
# 但请注意，这并不意味着罐子里有95%的豆子的颜色会落在你估计的这个颜色范围内。你的估计范围是关于整个罐子（总体）的平均颜色的，而不是关于罐子外面你能看到的那一小部分豆子（样本）的。
def cls_result(output,img_path,result_dir,categories_with_descriptions,topk=1):
        pred = output.topk(topk, 1, True, True)[1].t()
#         获取每个样本最大置信度的标签
        # print(pred.shape)
        prob, _ = output.softmax(dim=1).topk(topk, 1, True, True)
# 获取每个样本最大的置信度
        # 打印或返回预测的类别序号（如果需要）
        # for pred_id in pred.cpu().squeeze(0):
        #     print(pred_id.item())  # 使用.item()将tensor中的单个值转换为Python数值类型（如int或float）
        # print("Predicted classes:", pred.cpu().numpy())
        # 获取每个图像预测第一的类别
        first_pred_indices = pred.cpu().squeeze(0)
        first_pred_indices=first_pred_indices.numpy().astype(int)
        print(first_pred_indices)
        print(categories_with_descriptions)
        first_pred_classes = [categories_with_descriptions[idx] for idx in first_pred_indices]
        first_pred_prob = prob.squeeze(1).detach().cpu().numpy()
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        target_dir=f'test/jpg'
        image_names = []
        for img in img_path:  
            # 构造目标路径，即将图像拷贝到target_dir下，并保持原始文件名  
            image_name = os.path.basename(img)
            target_path = os.path.join(target_dir, os.path.basename(img))  
            # 使用shutil.copy2来拷贝文件，如果目标文件已存在则覆盖  
            # shutil.copy2除了拷贝文件内容外，还会尝试保留文件的元数据（如修改时间等）  
            shutil.copy2(img, target_path) 
            image_names.append(image_name)
  
    
        for pred_class,img_name,prob_val in zip(first_pred_classes,image_names,first_pred_prob):
            prob_val=1-prob_val
            print(f"img name={img_name}: Predicted class_name = {pred_class}: Probability = {prob_val:.4f}")
            img_path = f'test/jpg/{img_name}'
            img = Image.open(img_path)
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("SimHei.ttf", 21)  # 你可以选择一个你有的.ttf字体文件
            # 将预测结果和概率打印到图片上
            # text = f"{encoder}，Predicted class: {pred_class}, Probability: {prob_val:.4f}"
            text = f"c: {pred_class}, P: {prob_val:.4f}"
            text_width, text_height = draw.textsize(text, font)
            draw.text((10, 10), text, font=font, fill=(255, 255, 255))  # 红色字体
            
            output_path = os.path.join(result_dir, os.path.basename(img_name))
            # 保存图片
            img.save(output_path)


def meta_test(cfg,data_path1,data_path2,clip_model,alpha,init_alpha,model,way,shot,pre,transform_type,query_shot=30,trial=10000,return_list=False):
    # 创建一个数据加载器eval_loader,迭代地返回元测试数据，通常是以小批量（batch）的形式，
    # 每个小批量包含来自不同类别的支持集（support set）和查询集（query set）样本。
    eval_loader,dataset,clip_loader,dataset2,sampler = dataloaders.meta_test_dataloader(data_path1=data_path1,
                                                                                data_path2=data_path2,
                                                way=way,
                                                shot=shot,
                                                pre=pre,
                                                transform_type=transform_type,
                                                query_shot=query_shot,
                                                trial=trial)
    template = ['a photo of a {}, a type of aircraft.']
    #每个样本的对应类别id
    # trial=10000,默认为10000，表示试验次数，即元测试过程中将运行多少次完整的N-way K-shot测试
    # 在这个meta_test函数中，trial=10000确实表示在测试过程中将运行10000次完整的N-way K-shot测试。每次测试都会从所有可能的类别中随机选择way（在这个例子中是5）个类别，用于构建当前测试任务的支持集（support set）和查询集（query set）。
    # 由于每次测试都是随机选择类别的，因此这10000次测试中的每一次所使用的5个类别几乎肯定是不一样的（除非数据集类别数非常少且随机选择时发生了极小的概率重合）。
    # return_list布尔值，默认为False，指示函数是否应返回所有试验的准确率列表，而不是平均准确率和置信区间。
    target = torch.LongTensor([i//query_shot for i in range(query_shot*way)]).cuda()
    
    # target，其长度是query_shot * way，这个张量用于存储查询集样本的类别标签id。
    acc_list = []
    combined_loader = zip(eval_loader, clip_loader)
    # 标签是根据查询集样本的索引和query_shot计算得出的，以确保每个类别都有相同数量的样本，并且它们的标签是按顺序排列的。
    for i, ((inp, tar), (inp2, tar2)) in tqdm(enumerate(combined_loader)):
        # 每个批次生成一个索引（i）和该批次的内容。由于 eval_loader 通常返回一个包含输入数据和标签的元组（tuple
        # 因为这里我们使用模型预测），首先将输入数据移至GPU（如果可用）。
#         他们两加载的数据集不同，随机选的不过是id，只要拷贝一下，他们使用的是一样的id，但是id对应的是各自加载的数据集中的，是init加载的
# 他随机生成的不是真的样本，只是代表样本的一个id，他这个id不过是个数字，两个数据集样本的id是一样的，同时类别的id是一样的，他是通过id出来inp的
# 如果他是根据那个返回的id找的对应样本，那在得道inp2的时候，他就是随机的id所对应的样本，这时候改id没用，样本已经是 clip_loader随机出来id对应的那个样本了。
# or i, ((inp, tar), (inp2, tar2)) in tqdm(enumerate(combined_loader)):他这个inp2很可能是在迭代时自动调用__iter__，根据那个返回的样本id找的对应样本
# 类别变了有什么用，样本又没变
        inp = inp.cuda()
        inp2 = inp2.cuda()
        aplha=init_alpha
        
        # print(inp)
        # print(inp2.size())
        # print(tar)
        # print(tar2)
        count=0
        selected_paths = [] 
        id_to_path = {id_: path for id_,(path,_) in enumerate(dataset2.imgs)}
#         获得样本id到图像的映射，
        for batch_ids in sampler:  
            for id_ in batch_ids:
                if count >= way * shot:
                    selected_paths.append(id_to_path[id_])
                count=count+1
                # print(f"ID: {id_}, Image Path: {id_to_path[id_]}")
        # print(selected_paths) 这里面就存着查询集的路径      
        
        # 调用模型的meta_test方法，传入当前批次的输入数据和其他相关参数（way, shot, query_shot），该方法调用FRN中的meta_test，
        # 会通过get_neg_l2_dist依次调用，提取特征图，最后将返回查询集样本的预测类别索引。
        unique_order_list = []
        seen = set()  
        for x in tar.tolist():  
            if x not in seen:  
                seen.add(x)  
                unique_order_list.append(x)    
        # print(unique_order_list)
        sorted_unique_category_ids = sorted(unique_order_list)
        unique_category_ids2 = set(tar2.tolist())
        sorted_unique_category_ids2 = sorted(unique_category_ids2)
        # print(sorted_unique_category_ids)
        # print(sorted_unique_category_ids2)
# #         从id获取转成类别名
# 之前获得类别名是从因为每个类别名是这样的格式001.名字1
# 002.名字2，所以我获得都是.后面的名字，但现在没有这个格式了我的类别名就是名字1，名字2
# 下面这句话只限鸟数据集用
        # cleaned_class_names = [cls.split('.')[1] for cls in dataset.classes] 
        # cleaned_class_names=dataset.classes
                # 假设 dataset.classes 已经存在并包含了所有可能的类别名  
        # class_names = [cleaned_class_names[id] for id in unique_order_list]
                # 得到类别名，获取文本特征的计算
                # print(class_names)
        # clip_weights = clip_classifier(class_names, template, clip_model)
#       获取图像特征
        # print(inp) #前面25个那个支持集，后面80个查询集,看inp，类别id和样本按着选中的次序排序，不是类别id大小排序，算的矩阵就是，左边样本，上面类别id，按着选择的顺序，对应CLIP提取图像特征是query_images2，按着样本顺序来，左边类别样本，上面类别名，类别名顺序是来自class_names，输出可以看到，顺序是对应类别id选择的次序，不是大小，所以两个矩阵对应
#     样本的顺序是跟着类别id来的，也就是查询集样本前16个是一个类别的，每16个样本一个类别，query_images2就存着查询集样本，80个每隔16个样本一个类别，所以输出预测结果必然是每个16个一个类别。每次选80个样本在，这80个样本是通过每个类别中选16个，所以每16个样本就是一个类别，所以检测的结果，必然是0 16个 1类别 16个。。。4号类别16个    
        support_set_size = way*shot  
        query_set_size = way*query_shot
        support_images = inp[:support_set_size]  
        query_images = inp[support_set_size:support_set_size+query_set_size]
        support_images2 = inp2[:support_set_size]  
        query_images2 = inp2[support_set_size:support_set_size+query_set_size]
                # print(query_images.size())
                    # print(query_images2.size())
            #         一个张量80,3,84,84
                    # 之前构建loader所有的图像有个这个train_tranform，不管什么图像都是改成224*224，但我现在我这个图像时没有经过224*224的，所以我的改成224*224
        # features= pre_load_features_map(clip_model, query_images2)
                    # # # 余弦相似度计算
        # clip_logits = 100. * features @ clip_weights
                    # # print(clip_logits)#80个样本5个类别，没问题
        # pred,acc = cls_acc(clip_logits, target)#inp是所有的105，tar就是105个类别标签，而target是本次的标签，5*16=80个  0-15为0类别,
                #         第1个acc  .print("CLIP zero shot prediect result:")#neg的预测结果
                #         print(pred)
                # #         这里输出clip预测结果
                # print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))
        max_index,neg_l2_dist = model.meta_test(inp,way=way,shot=shot,query_shot=query_shot)
                # print(neg_l2_dist)
                # print(max_index)
                # cls_result(neg_l2_dist,selected_paths,"result/FRNLong",class_names)
                    # print("FRN prediect result")#neg的预测结果
                    # print(max_index)
        # acc = 100*torch.sum(torch.eq(max_index,target)).item()/query_shot/way
                   # 第2个acc. # print("\n****FRN test accuracy: {:.2f}. ****\n".format(acc))
                    # print(neg_l2_dist)    #80,5    输出，，，way5，shot=1,query_shot采用默认16，所以5*16,5==80，5  queryshot*way,way
                    #那个负重构距离，其实矩阵时左边每个样本的特征图特征，上面是每个类别重构的特征图，就可以看成类别，这里是这两个的相似度矩阵
                    # print(tar)#已经获得所有样本的类别id,现在只需要一个索引转换成类别名
                    # 计算测试集时中的支持集可以选择使用随机5个类别测试，每个类别都有1，5，10个样本时,而测试集中查询集每个类别有16个样本去查
        # tip_logits=clip_logits+neg_l2_dist*alpha
        # _, max_index = torch.max(tip_logits, 1)
                    #第3个acc融合的. print("tip_logits prediect result:")
                     # print(max_index)
                    # acc = 100*torch.sum(torch.eq(max_index,target)).item()/query_shot/way
                    #print("\n****tip_logits test accuracy: {:.2f}. ****\n".format(acc))
                    #下面是获得最好的权种参数，去融合得到最后的acc  1.acc是CLIP2.acc是FRN3.acc是CLIP+FRN4.获得最好的权重的CLIP+FRN4
        # best_alpha = search_hp2(cfg, features, target,neg_l2_dist,clip_weights)
        # tip_logits=clip_logits+neg_l2_dist*best_alpha
        # _, max_index = torch.max(tip_logits, 1)
                # print("tip_logits prediect result:")
                # cls_result(tip_logits,selected_paths,"result/FRNLong",class_names)
                # print(max_index)
        acc = 100*torch.sum(torch.eq(max_index,target)).item()/query_shot/way
            # print("\n****tip_logits-best test accuracy: {:.2f}. ****\n".format(acc))
        acc_list.append(acc)
        
    # 如果return_list为True，则直接返回acc_list，即所有试验的准确率列表。
    if return_list:
        return np.array(acc_list)
    else:
        mean,interval = get_score(acc_list)
        return mean,interval

    # 您描述的 meta_test 函数是元学习和小样本学习中的一个典型测试流程。这个函数通过多次（这里是10000次）迭代地构建N-way K-shot测试任务来评估模型的性能。每次迭代中，都会随机选择N个类别，并从这些类别中抽取样本来构建支持集和查询集，然后模型会使用支持集中的样本来进行适应或学习，最后使用适应后的模型参数对查询集中的样本进行预测，并计算准确率。
    #
    # 这里有几个关键点需要注意：
    #
    # 随机性：每次迭代都会随机选择N个不同的类别来构建测试任务，这有助于评估模型在不同类别组合下的泛化能力。
    # 支持集和查询集：支持集用于模型在测试时的适应或学习，而查询集则用于评估模型在适应后的性能。这种设置模拟了小样本学习的场景，即模型需要在有限的数据下进行学习并做出准确的预测。
    # 模型适应：在每次迭代中，模型都会使用支持集中的样本来进行适应或学习。这个过程可能涉及微调模型的参数、更新模型的元参数（在元学习方法中）或其他形式的适应策略。
    # 预测与评估：适应后的模型会对查询集中的样本进行预测，并将预测结果与真实标签进行比较以计算准确率。这个准确率反映了模型在当前测试任务上的性能。
    # 重复试验：通过多次（这里是10000次）迭代测试任务，可以获得更稳定、更可靠的评估结果。这有助于减少随机性对评估结果的影响，并更准确地反映模型的整体性能。
    # 结果返回：根据return_list参数的值，函数可以选择返回所有试验的准确率列表或计算并返回平均准确率和置信区间。后者通常更适用于报告模型性能，因为它提供了准确率的集中趋势和不确定性度量。
    # 需要注意的是，在元学习和小样本学习的场景中，模型的性能不仅取决于其本身的架构设计，还受到适应或学习过程的影响。因此，在评估模型时，必须仔细设计适应过程，以确保它能够充分利用支持集中的有限数据来优化模型参数。
    #
    # 此外，由于元学习和小样本学习通常涉及大量的计算和内存开销，因此在实际应用中可能需要使用高性能计算资源来加速测试过程。


    #
    # /也就是在测试的时候，使用测试集，假如是5way5shot,是10000次，这10000次当中每次随机选择5个类别，每个类别5个样本构成支持集对模型进行适应学习，这明明是测试，却有点像在训练微调模型，然后5个类别剩下的样本作为查询集去预测



    # 您的理解是正确的。在元学习和小样本学习的测试中，特别是使用N-way K-shot（如5way5shot）设置时，尽管我们称之为“测试”，但实际上这个过程确实包含了对模型的某种形式的“适应”或“微调”。这种适应是基于支持集（support set）进行的，而支持集是由从测试集中随机选择的N个类别中每个类别的K个样本组成的。
    #
    # 这个过程之所以被称为测试，是因为它的目的是评估模型在未见过的类别（即测试集中的类别）和有限数据（即支持集中的样本）下的泛化能力。然而，与传统的测试过程不同，元学习和小样本学习中的测试允许模型在测试期间进行某种程度的学习或适应，以更好地处理新的任务。
    #
    # 具体来说，在这10000次测试中，每次都会：
    #
    # 从测试集中随机选择5个类别。
    # 对于这5个类别中的每一个，从测试集中选择5个样本作为支持集（共25个样本）。
    # 使用这25个支持集样本对模型进行适应或微调。这个过程可能涉及更新模型的某些参数或权重，以便模型能够更好地处理这5个类别的数据。
    # 使用适应后的模型对剩余的查询集样本（每个类别中除了用于支持集的样本之外的样本）进行预测。
    # 计算预测结果的准确率，以评估模型在当前测试任务上的性能。
    # 这个过程重复10000次，每次都使用不同的随机选择的类别组合和样本，以获得更全面的评估结果。
    #
    # 需要注意的是，尽管这个过程在测试期间对模型进行了适应，但适应是基于支持集的，而支持集是独立于查询集的。因此，这种适应并不会导致模型对查询集数据的过拟合，而是使模型能够更好地处理与支持集相似的数据分布和任务。
    #
    # 总的来说，这种测试过程更接近于评估模型在真实世界中小样本学习场景下的性能，其中模型需要在有限的数据下快速适应新的任务并做出准确的预测。
    
    
def meta_test2(data_path,model,way,shot,pre,transform_type,query_shot=16,trial=10000,return_list=False):
    eval_loader,dataset = dataloaders.meta_test_dataloader2(data_path=data_path,
                                                way=way,
                                                shot=shot,
                                                pre=pre,
                                                transform_type=transform_type,
                                                query_shot=query_shot,
                                                trial=trial)
    target = torch.LongTensor([i//query_shot for i in range(query_shot*way)]).cuda()
   
    acc_list = []
    for i, (inp,tar) in tqdm(enumerate(eval_loader)):
        inp = inp.cuda()
        max_index,neg_l2_dist = model.meta_test(inp,way=way,shot=shot,query_shot=query_shot)
        acc = 100*torch.sum(torch.eq(max_index,target)).item()/query_shot/way
        acc_list.append(acc)
    if return_list:
        return np.array(acc_list)
    else:
        mean,interval = get_score(acc_list)#所以acc_list存放这10000次任务中，每次任务，查询集预测结果准确率
        return mean,interval
    
