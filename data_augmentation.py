#!/usr/bin/env python3
"""
数据增强脚本：从现有CSV文件中随机抽取数据，进行增强，然后添加到原数据集中
"""

import pandas as pd
import numpy as np
import json
import random

# 配置参数
INPUT_FILE = '/root/CRODI-MSA/CRODI-MSA-op6-test/dataset/preTrain-myData/converted_dataset.csv'
TARGET_COUNT = 2000

# 读取现有数据集
df = pd.read_csv(INPUT_FILE)
print(f"原始数据集大小: {len(df)}")

# 计算需要添加的数据量
to_add = TARGET_COUNT - len(df)
if to_add <= 0:
    print(f"数据集已经达到或超过目标大小 {TARGET_COUNT}")
    exit(0)

print(f"需要添加 {to_add} 条数据")

# 随机抽取数据进行增强
sampled_df = df.sample(n=to_add, replace=True)

# 数据增强函数
def augment_features(feature_str):
    """
    对特征向量进行增强
    """
    # 将字符串转换为列表
    features = json.loads(feature_str)
    
    # 对特征向量进行随机扰动
    augmented = []
    for i, val in enumerate(features):
        # 只对前3个有意义的特征进行增强
        if i < 3:
            # 添加随机扰动，范围为原数值的±10%
            if val != 0:
                perturbation = val * random.uniform(-0.1, 0.1)
                augmented_val = max(0, int(val + perturbation))  # 确保非负整数
            else:
                augmented_val = 0
        else:
            # 其他特征保持不变
            augmented_val = val
    
    return json.dumps(augmented)

# 对抽取的数据进行增强
augmented_df = sampled_df.copy()
augmented_df['feature'] = augmented_df['feature'].apply(augment_features)

# 为增强的数据生成新的commit哈希（简单处理）
def generate_new_commit(original_commit):
    """
    生成新的commit哈希
    """
    # 保留前8位，后面随机生成新的哈希值
    new_hash = original_commit[:8] + ''.join(random.choices('0123456789abcdef', k=32))
    return new_hash

augmented_df['commit'] = augmented_df['commit'].apply(generate_new_commit)

# 将增强的数据添加到原数据集中
combined_df = pd.concat([df, augmented_df], ignore_index=True)

# 保存到原文件
combined_df.to_csv(INPUT_FILE, index=False, encoding='utf-8')

print(f"数据增强完成！")
print(f"增强后数据集大小: {len(combined_df)}")
print(f"添加了 {len(augmented_df)} 条增强数据")
print(f"输出文件: {INPUT_FILE}")
