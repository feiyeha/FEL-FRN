import json
import numpy as np
from scipy.io import savemat

# 读取JSON文件
with open('E:\PycharmProjects\Tip-Adapter-main\Tip-Adapter-main\RSTPReid\split_zhou_RSTPReid.json', 'r',encoding='utf-8') as f:
    data = json.load(f)

# 初始化一个空列表来存储所有可转换的ID
convertible_category_ids = []

# 遍历test、train、val并提取ID
for key, dataset in data.items():
    # 尝试将类别ID转换为float64，并检查是否成功
    try:
        category_ids = [float(row[1]) for row in dataset if isinstance(row[1], (int, float, str)) and (
                (isinstance(row[1], str) and row[1].replace('.', '', 1).isdigit()) or
                isinstance(row[1], (int, float))
        )]
        # 将这些ID添加到总列表中
        convertible_category_ids.extend(category_ids)
    except ValueError:
        print(f"无法将{key}中的某些类别ID转换为float64")

    # 将列表转换为NumPy数组（默认dtype为float64）
all_category_ids_array = np.array(convertible_category_ids, dtype=np.float64)

# 保存到.mat文件
savemat('imagelabels.mat', {'CategoryID': all_category_ids_array})
print("Category IDs saved to imagelabels.mat")
