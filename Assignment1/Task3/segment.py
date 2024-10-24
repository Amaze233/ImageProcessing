import os
from PIL import Image

# 指定要检索的目录
directory = 'data/matting_human_sample/clip_img'

# 存储图片路径的列表
image_paths = []

# 遍历目录
for root, dirs, files in os.walk(directory):
    for file in files:
        # 构建完整的文件路径
        full_path = os.path.join(root, file)
        try:
            # 尝试打开图片，如果是图片则添加到列表中
            with Image.open(full_path) as img:
                path = os.path.relpath(full_path, directory)
                image_paths.append(path.replace("\\", "/"))
        except IOError:
            # 如果不是图片，则忽略
            continue

print("image_paths.shape:", len(image_paths))
split = int(len(image_paths) * 0.8)
# 将图片路径写入txt文件
with open('data/select_data/matting_human_train.txt', 'w') as f:
    for path in image_paths[0:split]:
        f.write(path + '\n')
with open('data/select_data/matting_human_test.txt', 'w') as f:
    for path in image_paths[split:]:
        f.write(path + '\n')
print(f'Found {len(image_paths)} images.')
