import os
import json
# # 先分好数据集在来，
# # 假设以下是数据集文件夹和类别映射JSON的路径
# dataset_root = 'E:\PycharmProjects\Tip-Adapter-main\Tip-Adapter-main\RSTPReid\jpg'  # 替换为您的数据集根目录
# class_mapping_path = 'E:\PycharmProjects\Tip-Adapter-main\Tip-Adapter-main\RSTPReid\cat_to_name.json'  # 替换为类别映射JSON的路径
#
# # 加载类别映射JSON
# with open(class_mapping_path, 'r', encoding='utf-8') as f:
#     class_mapping = json.load(f)
#
# # 创建一个空的字典来存储数据
# data = {
#     "train": [],
#     "val": [],
#     "test": []
# }
#
# # 遍历train、val和test文件夹
# for split in ['train', 'val', 'test']:
#     split_dir = os.path.join(dataset_root, split)
#     for class_id in os.listdir(split_dir):
#         if class_id.isdigit() and int(class_id) < 4101:  # 确保是0-39的数字
#             class_dir = os.path.join(split_dir, class_id)
#             for filename in os.listdir(class_dir):
#                 if filename.endswith('.jpg'):
#                     # 获取类别名
#                     class_name = class_mapping.get('Unknown')
#                     # 假设文件名已经是唯一的，不需要额外的ID（如果需要，可以添加）
#                     image_data = [filename, int(class_id), class_name]
#                     data[split].append(image_data)
#
#                 # 将数据写入新的JSON文件
# with open('split_garbage.json', 'w', encoding='utf-8') as f:
#     json.dump(data, f,ensure_ascii=False,indent=4)
#
# print("JSON file 'split_garbage' has been created.")

# import json
#
# # 假设第一个JSON文件包含图像和描述信息
# with open('E:\PycharmProjects\Tip-Adapter-main\Tip-Adapter-main\RSTPReid\split_zhou_RSTPReid.json', 'r', encoding='utf-8') as f:
#     descriptions_data = json.load(f)
#
# # 创建一个字典，将文件名映射到描述列表
# filename_to_descriptions = {}
# for key in ['train', 'val', 'test']:
#     for item in descriptions_data[key]:
#         filename = item[0]
#         descriptions = item[2]
#         filename_to_descriptions[filename] = descriptions
#
#
# # 假设第二个JSON文件包含需要替换的Unknown描述
# with open('E:\PycharmProjects\Tip-Adapter-main\Tip-Adapter-main\split_garbage.json', 'r', encoding='utf-8') as f:
#     unknown_descriptions_data = json.load(f)
#
# # 遍历需要替换的文件，并替换Unknown
# for key in ['train', 'val', 'test']:
#     for item in unknown_descriptions_data[key]:
#         filename = item[0]
#         if filename in filename_to_descriptions:
#             # 假设我们只想要第一个描述，但你可以根据需要选择,这边可以减少描述
#             item[2] = filename_to_descriptions[filename]
#
#     # 将更新后的数据写入新的JSON文件
# with open('updated_descriptions.json', 'w', encoding='utf-8') as f:
#     json.dump(unknown_descriptions_data, f, ensure_ascii=False, indent=4)
#
# print("JSON文件 'updated_descriptions.json' 已创建。")

# 增强的图像没有描述，用那些同一个人的描述
import json
from collections import defaultdict

# 假设第一个JSON文件包含图像和描述信息
with open('E:\PycharmProjects\Tip-Adapter-main\Tip-Adapter-main\RSTPReid\split_zhou_RSTPReid.json', 'r',
          encoding='utf-8') as f:
    descriptions_data = json.load(f)

# 创建一个字典，将文件名前四位映射到描述列表
filename_prefix_to_descriptions = defaultdict(list)
for key in ['train', 'val', 'test']:
    for item in descriptions_data[key]:
        filename = item[0]
        prefix = filename[:4]  # 取前四位作为前缀
        descriptions = item[2]
        filename_prefix_to_descriptions[prefix].append((filename, descriptions))

    # 假设第二个JSON文件包含需要替换的Unknown描述
with open('E:\PycharmProjects\Tip-Adapter-main\Tip-Adapter-main\split_garbage.json', 'r', encoding='utf-8') as f:
    unknown_descriptions_data = json.load(f)

# 遍历需要替换的文件，并尝试使用前四位数字相同的描述来替换Unknown
for key in ['train', 'val', 'test']:
    for item in unknown_descriptions_data[key]:
        filename = item[0]
        prefix = filename[:4]  # 获取当前文件名前四位

        # 尝试直接查找当前文件名的描述
        if filename in {fn for fn, _ in filename_prefix_to_descriptions[prefix]}:
            for fn, desc in filename_prefix_to_descriptions[prefix]:
                if fn == filename:
                    item[2] = desc
                    break
        else:
            # 如果没有直接找到，则使用前四位相同的任意文件的描述（如果有多个，这里取第一个）
            if filename_prefix_to_descriptions[prefix]:
                item[2] = filename_prefix_to_descriptions[prefix][0][1]

            # 将更新后的数据写入新的JSON文件
with open('updated_descriptions.json', 'w', encoding='utf-8') as f:
    json.dump(unknown_descriptions_data, f, ensure_ascii=False, indent=4)

print("JSON文件 'updated_descriptions.json' 已创建。")