# import json
#
# # 假设你的JSON文件名为 'data.json'
# filename = 'E:\PycharmProjects\Tip-Adapter-main\Tip-Adapter-main\RSTPReid\data_captions.json'
#
# # 使用 'r' 模式打开文件以进行读取
# with open(filename, 'r', encoding='utf-8') as f:
#     # 读取文件内容并解析为Python对象
#     data = json.load(f)
#
# # 使用 'json.dumps()' 函数格式化输出，并设置缩进为 4
# formatted_json = json.dumps(data, indent=4, ensure_ascii=False)
#
# # 打印格式化后的JSON
# print(formatted_json)
#
# # 如果你希望将格式化后的JSON写入一个新文件
# with open('formatted_data.json', 'w', encoding='utf-8') as f:
#     f.write(formatted_json)
import json

# 假设你的JSON文件名为 'data.json'
filename = 'E:\PycharmProjects\Tip-Adapter-main\Tip-Adapter-main\RSTPReid\data_captions.json'

# 读取原始JSON文件
with open(filename, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 创建三个新的列表来分别存储训练集、验证集和测试集数据
train_data = []
val_data = []
test_data = []

# 遍历原始数据，根据split键的值将数据添加到相应的列表中，并删除split键
for item in data:
    if item.get('split') == 'train':
        item_without_split = {k: v for k, v in item.items() if k != 'split'}
        train_data.append(item_without_split)
    elif item.get('split') == 'val':
        item_without_split = {k: v for k, v in item.items() if k != 'split'}
        val_data.append(item_without_split)
    elif item.get('split') == 'test':
        item_without_split = {k: v for k, v in item.items() if k != 'split'}
        test_data.append(item_without_split)

    # 创建一个新的字典，其中包含'train'、'val'和'test'键，其值为相应的数据集列表
formatted_data = {
    'train': train_data,
    'val': val_data,
    'test': test_data
}

# 打印格式化后的数据
print(json.dumps(formatted_data, indent=4, ensure_ascii=False))

# 如果你希望将格式化后的JSON写入一个新文件
with open('formatted_data.json', 'w', encoding='utf-8') as f:
    json.dump(formatted_data, f, indent=4, ensure_ascii=False)