import os
import random
import shutil
from collections import defaultdict


def split_dataset(input_dir, output_dir, train_ratio=0.8):
    """
    分割数据集并生成所需结构

    参数:
        input_dir: 原始数据集目录
        output_dir: 输出目录
        train_ratio: 训练集比例
    """
    # 创建输出目录结构
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)

    # 获取类别列表（原始子文件夹名称）
    classes = sorted([d for d in os.listdir(input_dir)
                      if os.path.isdir(os.path.join(input_dir, d))])

    # 写入classes.txt
    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        f.write('\n'.join(classes))

    # 创建训练集和验证集的标注文件
    train_file = open(os.path.join(output_dir, 'train.txt'), 'w')
    val_file = open(os.path.join(output_dir, 'val.txt'), 'w')

    # 为每个类别处理文件
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # 获取该类别所有图像文件
        images = [f for f in os.listdir(class_dir)
                  if os.path.isfile(os.path.join(class_dir, f))]
        random.shuffle(images)

        # 计算分割点
        split_point = int(len(images) * train_ratio)

        # 创建训练和验证子目录
        train_class_dir = os.path.join(output_dir, 'train', class_name)
        val_class_dir = os.path.join(output_dir, 'val', class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # 处理训练集
        for img in images[:split_point]:
            src = os.path.join(class_dir, img)
            dst = os.path.join(train_class_dir, img)
            shutil.copy(src, dst)
            train_file.write(f"{os.path.join('train', class_name, img)} {label}\n")

        # 处理验证集
        for img in images[split_point:]:
            src = os.path.join(class_dir, img)
            dst = os.path.join(val_class_dir, img)
            shutil.copy(src, dst)
            val_file.write(f"{os.path.join('val', class_name, img)} {label}\n")

    train_file.close()
    val_file.close()
    print(f"数据集已成功分割并保存到 {output_dir}")


if __name__ == '__main__':
    # 设置输入输出目录
    input_directory = 'flower_dataset'  # 原始数据集目录
    output_directory = 'flower_data'  # 输出目录

    # 执行分割
    split_dataset(input_directory, output_directory)