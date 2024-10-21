# import json
#
# # 读取JSON文件
# with open('E:\PycharmProjects\Tip-Adapter-main\Tip-Adapter-main\oxford_flowers\cat_to_name.json', 'r') as f:
#     flowers_dict = json.load(f)
#
# # 提取键，转换为整数，并减去1
# sorted_keys = sorted(flowers_dict, key=int)
# new_dict = {str(int(k) - 1): flowers_dict[k] for k in sorted_keys}
#
# # 将新的字典写回JSON文件
# with open('E:\PycharmProjects\Tip-Adapter-main\Tip-Adapter-main\oxford_flowers\cat_to_name.json', 'w') as f:
#     json.dump(new_dict, f, indent=4)

# import json
#
# # 初始化一个空字典来存储键值对
# data = {}
#
# # 填充从"0"到"4100"的键，每个键的值与其键相同（作为字符串）
# for i in range(4101):  # 从0到4100，共4101个数字
#     data[str(i)] = str(i)
#
# # 将字典转换为JSON字符串，并格式化输出（可选的格式化）
# json_string = json.dumps(data, indent=4, sort_keys=True)
#
# # 将JSON字符串写入文件
# with open('output.json', 'w') as f:
#     f.write(json_string)

import json
#
# # 假设你的原始JSON文件名为 input.json
# with open('output.json', 'r') as f:
#     data = json.load(f)
#
# # 分离键和值，按键的整数值排序
# sorted_items = sorted(data.items(), key=lambda x: int(x[0]))
#
# # 创建一个新的有序字典，或者简单地使用字典推导式（因为Python 3.7+中的普通字典会保持插入顺序）
# sorted_data = {str(k): v for k, v in sorted_items}
#
# # 将排序后的字典写入新的JSON文件
# with open('output.json', 'w') as f:
#     json.dump(sorted_data, f, indent=4)
# 解析JSON字符串为Python字典
import json

# import json
#
# # 读取原始JSON文件
# with open('E:\PycharmProjects\Tip-Adapter-main\Tip-Adapter-main\RSTPReid\split_zhou_RSTPReid.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)
#
#
# # 转换函数，将字典转换为列表
# def convert_dict_to_list(item):
#     return [item['img_path'], item['id'], item['captions']]  # 假设每个字典都有'img_path'、'id'和'category'这三个键
#
#
# # 遍历train、val和test列表，并转换每个元素
# for key in ['train', 'val', 'test']:
#     data[key] = [convert_dict_to_list(item) for item in data[key]]
#
# # 将处理后的数据写入新的JSON文件（或覆盖原始文件）
# with open('new_structure_data.json', 'w', encoding='utf-8') as f:
#     json.dump(data, f, ensure_ascii=False, indent=4)
# with open('E:\PycharmProjects\Tip-Adapter-main\Tip-Adapter-main\RSTPReid\split_zhou_RSTPReid.json', 'r',
#           encoding='utf-8') as f:
#     data = json.load(f)
#     # 提取所有的文本描述并放入列表中
#     # 初始化一个空列表来存储所有描述
# all_descriptions = []
# # 遍历JSON数据中的每个条目
# for split in ['train', 'val', 'test']:
#     # 遍历每个分割部分的条目
#     for item in data[split]:
#         # 提取描述列表
#         descriptions = item[2]
#         # 将描述添加到all_descriptions列表中
#         all_descriptions.extend(descriptions)
#
#     # 打印结果以验证
# for desc in all_descriptions:
#     print(desc)
# from transformers import BertTokenizer
#
# # 假设我们有一个BERT tokenizer的实例（这里用BERT作为示例）
# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
#
# # 你的文本描述
# text = "This woman's hair is just at shoulder level, neither too long nor too short.She was wearing a black down jacket, blue jeans and red loafers.She also wore a small bag with colored stripes on her left hand.She looked down at the phone she was holding in her left hand and a blue doll pendant in her right hand."
#
# # 使用tokenizer对文本进行编码
# encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True,
#                           max_length=512)  # 512是BERT的最大长度，CLIP可能不同
#
# # 获取input_ids，这些就是tokens的索引
# input_ids = encoded_input['input_ids']
#
# # input_ids的长度就是tokens的数量（不包括特殊标记如[CLS]和[SEP]）
# num_tokens = len(input_ids[0])
#
# print(f"The number of tokens is: {num_tokens}")

# with open('E:\PycharmProjects\Tip-Adapter-main\Tip-Adapter-main\RSTPReid\split_zhou_RSTPReid.json', 'r',
#           encoding='utf-8') as f:
#     json_data = json.load(f)
# # 提取所有文本描述（每个ID只提取一条）
# descriptions = []
# for dataset in ['val']:  # 遍历所有数据集
#     if dataset in json_data:
#         for item in json_data[dataset]:  # 遍历每个数据项
#             if len(item) > 2 and isinstance(item[2], list) and item[2]:
#                 # 提取第一条文本描述
#                 description = item[2][0]
#                 descriptions.append(description)
#
#             # 输出提取的文本描述
# with open('extracted_descriptions.json', 'w') as f:
#     json.dump(descriptions, f, indent=4)


import json

# 读取第一个JSON文件（包含文本描述的列表）
with open('E:\PycharmProjects\Tip-Adapter-main\Tip-Adapter-main\extracted_descriptions.json', 'r', encoding='utf-8') as f:
    descriptions = json.load(f)

# 初始化第二个JSON文件的模板
template = {f"{i}": "" for i in range(len(descriptions))}

# 更新模板，将文本描述添加到对应的键后面
for i, description in enumerate(descriptions):
    template[f"{i}"] = description

# 将更新后的字典保存为新的JSON文件
with open('updated_descriptions.json', 'w', encoding='utf-8') as f:
    json.dump(template, f, ensure_ascii=False, indent=2)
