#!/usr/bin/env python3
"""
检查数据集最后一列特征值的脚本
"""

import pandas as pd
import numpy as np

# 读取数据集
dataset_path = '/root/COLARE/COLARE-op6-test/dataset/preTrain-myData/converted_dataset.csv'
df = pd.read_csv(dataset_path)

# 获取最后一列名称
last_col = df.columns[-1]
print(f'最后一列名称: {last_col}')

# 查看最后一列的基本统计信息
print('\n最后一列统计信息:')
print(df[last_col].describe())

# 检查是否有非零值
non_zero_count = (df[last_col] != 0).sum()
print(f'\n非零值数量: {non_zero_count} / {len(df)}')
print(f'非零值比例: {non_zero_count/len(df)*100:.2f}%')

# 查看最后一列的唯一值
print('\n唯一值:')
unique_values = df[last_col].unique()
print(unique_values)

# 查看值的分布
print('\n值的分布:')
value_counts = df[last_col].value_counts()
print(value_counts)

# 查看前10行数据的最后一列
print('\n前10行数据的最后一列:')
print(df[[last_col]].head(10))

# 查看后10行数据的最后一列
print('\n后10行数据的最后一列:')
print(df[[last_col]].tail(10))

# 检查是否所有值都是0
all_zeros = (df[last_col] == 0).all()
print(f'\n是否所有值都是0: {all_zeros}')

# 检查是否所有值都是相同的
always_same = len(unique_values) == 1
print(f'是否所有值都相同: {always_same}')
