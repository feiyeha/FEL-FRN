import cv2
import numpy as np
import os
import random
from PIL import Image, ImageEnhance
from torchvision import transforms

# # 定义一个色彩抖动变换
# color_jitter = transforms.ColorJitter(brightness=0.4,
#                                       contrast=0.4,
#                                       saturation=0.4,
#                                       hue=0.1)
# def horizontal_flip(image):
#     return cv2.flip(image, 1)
#
#
# def random_brightness_adjust(image, brightness_delta=30):
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(hsv)
#     random_brightness = random.uniform(-brightness_delta, brightness_delta)
#     v = cv2.addWeighted(v, 1, np.zeros(v.shape, v.dtype), 0, random_brightness)
#     v[v > 255] = 255
#     v[v < 0] = 0
#     final_hsv = cv2.merge((h, s, v))
#     return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
#
#
# def apply_augmentation(base_dir, num_images_per_folder=10):
#     # 遍历每个文件夹
#     for folder_id in range(130):
#         folder_path = os.path.join(base_dir, f'{folder_id:04d}')
#         if not os.path.exists(folder_path):
#             continue
#
#             # 获取文件夹中所有图片的文件名
#         image_files = [f for f in os.listdir(folder_path) if
#                        f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]
#         current_count = len(image_files)
#
#         # 如果当前文件夹的图片数量已经达到或超过目标数量，则跳过
#         if current_count >= num_images_per_folder:
#             continue
#
#             # 计算需要增强的图片数量
#         num_to_augment = num_images_per_folder - current_count
#
#         # 遍历文件夹中的所有图片文件
#         for filename in image_files:
#             img_path = os.path.join(folder_path, filename)
#             image = cv2.imread(img_path)
#             if image is None:
#                 continue  # 跳过无法读取的图片
#
#             # 最多增强num_to_augment次
#             for _ in range(min(num_to_augment, 2)):  # 这里假设每张图片至少增强两次
#                 # 根据概率选择增强方法
#                 if random.random() < 0.33:  # 33% 的概率进行色彩抖动
#                     image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#                     augmented_image_pil = color_jitter(image_pil)
#                     augmented_image = cv2.cvtColor(np.asarray(augmented_image_pil), cv2.COLOR_RGB2BGR)
#                 elif random.random() < 0.66:  # 额外33% 的概率进行水平翻转
#                     augmented_image = horizontal_flip(image)
#                 else:  # 剩下的34% 进行随机亮度调整
#                     augmented_image = random_brightness_adjust(image)
#
#                     # 保存增强后的图片
#                 base, ext = os.path.splitext(filename)
#                 output_filename = f'{base}_aug_{current_count + 1}{ext}'
#                 output_path = os.path.join(folder_path, output_filename)
#                 cv2.imwrite(output_path, augmented_image)  # 写入增强后的图片
#                 current_count += 1  # 更新已处理的图片数量
#
#                 # 如果已经生成了足够的图片，跳出循环
#                 if current_count >= num_images_per_folder:
#                     break
#
#                     # 如果已经生成了足够的图片，跳出内部循环
#             if current_count >= num_images_per_folder:
#                 break
#
#             # 使用函数
#
#
# apply_augmentation(r'E:\PycharmProjects\Tip-Adapter-main\Tip-Adapter-main\RSTPReid\jpg\train')


# 将文件夹下图片移动到和文件夹外面
import os
import shutil

# 设定你的jpg文件夹的路径（确保这个路径是正确的）
base_dir = 'E:\PycharmProjects\Tip-Adapter-main\Tip-Adapter-main\RSTPReid\jpg'  # 替换为你的jpg文件夹路径

# 遍历train, test, val文件夹
for subfolder in ['train', 'test', 'val']:
    full_subfolder_path = os.path.join(base_dir, subfolder)

    # 遍历每个子文件夹中的图片
    for root, dirs, files in os.walk(full_subfolder_path):
        for file in files:
            # 检查文件是否是图片（这里简单检查后缀，你可以根据需要添加更多条件）
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')):
                # 构造源文件和目标文件的完整路径
                src_file = os.path.join(root, file)
                dst_file = os.path.join(base_dir, file)

                # 移动文件，如果目标文件已存在则会被覆盖
                shutil.move(src_file, dst_file)

                # 完成后，train, test, val文件夹下的所有图片应该都被移动到了jpg文件夹的根目录下