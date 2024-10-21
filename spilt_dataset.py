from collections import defaultdict
import random
import os
import shutil
#
# # 设定你的图片所在的根目录
# root_dir = 'E:\PycharmProjects\Tip-Adapter-main\Tip-Adapter-main\RSTPReid\jpg'  # 替换为你的图片文件夹路径
#
# # 遍历根目录下的所有文件和文件夹
# for filename in os.listdir(root_dir):
#     # 检查文件是否是文件（不是文件夹）且文件名以数字开头
#     if os.path.isfile(os.path.join(root_dir, filename)) and filename[:4].isdigit():
#         # 提取文件名前四位作为子文件夹名
#         subdir_name = filename[:4]
#         # 构建子文件夹的完整路径
#         subdir_path = os.path.join(root_dir, subdir_name)
#
#         # 如果子文件夹不存在，则创建它
#         if not os.path.exists(subdir_path):
#             os.makedirs(subdir_path)
#
#             # 构建文件的完整路径
#         file_path = os.path.join(root_dir, filename)
#
#         # 将文件移动到相应的子文件夹中
#         shutil.move(file_path, subdir_path)
#
# print("处理完成！")

#







#
#
# 设定数据集根目录
dataset_root = 'E:\PycharmProjects\Tip-Adapter-main\Tip-Adapter-main\RSTPReid\jpg'  # 替换为你的数据集根目录路径
train_dir = os.path.join(dataset_root, 'train')
val_dir = os.path.join(dataset_root, 'val')
test_dir = os.path.join(dataset_root, 'test')
# 分好可以用于之后计算准确率之类的

# 创建train和val目录
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 遍历所有子目录（类别）
categories = [cat for cat in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, cat))]

# 存储每个类别的图片数量
category_image_counts = defaultdict(int)

# 计算每个类别的图片总数
for category in categories:
    category_dir = os.path.join(dataset_root, category)
    for filename in os.listdir(category_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # 假设图片是jpg或png格式
            category_image_counts[category] += 1

        # 分层划分图片到train和val目录
for category in categories:
    total_images = category_image_counts[category]
    train_images = int(0.6 * total_images)
    val_images = int(0.2 * total_images)  # 从剩下的30%中取20%作为验证集
    test_images = total_images-train_images-val_images  # 剩下的10%作为测试集

    # 创建对应的train和val类别目录
    train_category_dir = os.path.join(train_dir, category)
    val_category_dir = os.path.join(val_dir, category)
    test_category_dir = os.path.join(test_dir, category)
    os.makedirs(train_category_dir, exist_ok=True)
    os.makedirs(val_category_dir, exist_ok=True)
    os.makedirs(test_category_dir, exist_ok=True)

    # 获取该类别的所有图片路径
    category_dir = os.path.join(dataset_root, category)
    image_paths = [os.path.join(category_dir, filename) for filename in os.listdir(category_dir)
                   if filename.endswith('.jpg') or filename.endswith('.png')]

    # 随机打乱图片顺序
    random.shuffle(image_paths)

    # 将图片复制到train、val和test目录
    for i, image_path in enumerate(image_paths):
        if i < train_images:
            shutil.move(image_path, train_category_dir)
        elif i < train_images + val_images:
            shutil.move(image_path, val_category_dir)
        else:
            shutil.move(image_path, test_category_dir)

print("Data split into train, val, and test successfully!")

