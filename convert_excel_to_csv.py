#!/usr/bin/env python3
"""
将Excel数据集转换为项目所需的CSV格式（支持五类commit分类，优化版）
"""

import pandas as pd
import json
import os
import re
import requests
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# GitHub API配置
GITHUB_TOKEN = "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"  # 用户提供的GitHub token
HEADERS = {
    "Accept": "application/vnd.github.diff",
    "Authorization": f"token {GITHUB_TOKEN}"
}

# 配置参数
MAX_WORKERS = 10  # 并行工作线程数
API_TIMEOUT = 10  # API请求超时时间（秒）
MAX_RETRIES = 2  # API请求最大重试次数
INITIAL_DELAY = 2  # 初始重试延迟（秒）

# 缓存机制
diff_cache = {}

# 缓存文件路径
CACHE_FILE = "diff_cache.json"

# 加载缓存
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "r") as f:
            diff_cache = json.load(f)
        print(f"加载了 {len(diff_cache)} 个diff缓存")
    except Exception as e:
        print(f"加载缓存失败: {e}")
        diff_cache = {}

# 直接使用原始5类标签，不进行映射
# 模型将直接处理这5类标签：Adaptive、Perfective、Preventive、Corrective、Other
LABEL_MAP = {
    "Adaptive": "Adaptive",
    "Corrective": "Corrective",
    "Perfective": "Perfective",
    "Preventive": "Preventive",
    "Other": "Other"
}


def extract_github_info(url):
    """
    从GitHub提交URL中提取user、repo和commit哈希
    
    参数：
    url: GitHub提交URL
    
    返回：
    (user, repo, commit_hash) 三元组，如果解析失败则返回(None, None, None)
    """
    # 支持不同格式的GitHub提交URL
    patterns = [
        r"https://github.com/([^/]+)/([^/]+)/commit/([a-f0-9]{40})",
        r"https://github.com/([^/]+)/([^/]+)/commit/([a-f0-9]{7,40})"  # 支持短哈希
    ]
    
    for pattern in patterns:
        match = re.match(pattern, url)
        if match:
            return match.group(1), match.group(2), match.group(3)
    
    return None, None, None


def get_cache_key(user, repo, commit_hash):
    """
    生成缓存键
    
    参数：
    user: 用户名
    repo: 仓库名
    commit_hash: 提交哈希
    
    返回：
    缓存键字符串
    """
    key_str = f"{user}/{repo}/{commit_hash}"
    return hashlib.md5(key_str.encode()).hexdigest()


def fetch_diff(user, repo, commit_hash):
    """
    使用GitHub API获取提交的diff内容（带缓存机制）
    
    参数：
    user: 用户名
    repo: 仓库名
    commit_hash: 提交哈希
    
    返回：
    diff内容字符串，如果获取失败则返回空字符串
    """
    # 生成缓存键
    cache_key = get_cache_key(user, repo, commit_hash)
    
    # 检查缓存
    if cache_key in diff_cache:
        return diff_cache[cache_key]
    
    url = f"https://api.github.com/repos/{user}/{repo}/commits/{commit_hash}"
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=HEADERS, timeout=API_TIMEOUT)
            
            # 处理速率限制
            if response.status_code == 403 and "rate limit" in response.text.lower():
                print(f"GitHub API速率限制，等待 {INITIAL_DELAY * (2 ** attempt)} 秒后重试...")
                time.sleep(INITIAL_DELAY * (2 ** attempt))
                continue
            
            # 处理成功响应
            if response.status_code == 200:
                diff_content = response.text
                # 保存到缓存
                diff_cache[cache_key] = diff_content
                return diff_content
            
            # 处理其他错误
            print(f"获取diff失败，状态码：{response.status_code} {user}/{repo}/{commit_hash}")
            return ""
            
        except requests.exceptions.RequestException as e:
            print(f"获取diff失败 (尝试 {attempt+1}/{MAX_RETRIES}) {user}/{repo}/{commit_hash}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(INITIAL_DELAY * (2 ** attempt))
            else:
                print(f"多次尝试后仍失败，跳过此提交")
                return ""
    
    return ""


def save_cache():
    """
    保存缓存到文件
    """
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(diff_cache, f)
        print(f"已保存 {len(diff_cache)} 个diff到缓存文件")
    except Exception as e:
        print(f"保存缓存失败: {e}")


def generate_dummy_features(diff):
    """
    生成简单的特征向量（21维，与项目要求一致）
    
    参数：
    diff: diff内容字符串
    
    返回：
    21维特征向量
    """
    # 简单统计diff中的一些基本信息
    lines = diff.splitlines()
    added_lines = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++"))
    deleted_lines = sum(1 for line in lines if line.startswith("-") and not line.startswith("---"))
    
    # 生成21维特征向量（与项目要求的特征维度一致）
    # 这里使用简单的统计信息和默认值
    features = [
        added_lines, deleted_lines, added_lines - deleted_lines,  # 代码相关
        0, 0, 0,  # 注释相关
        0, 0, 0,  # 空格相关
        0, 0, 0,  # 测试文件相关
        0, 0, 0,  # 测试文件注释相关
        0, 0, 0,  # 测试文件空格相关
        0, 0, 0   # 文档相关
    ]
    
    return features


def process_single_row(row, max_per_category, category_count, row_index):
    """
    处理单行数据
    
    参数：
    row: 数据行
    max_per_category: 每个类别最大数量
    category_count: 类别计数字典
    row_index: 行索引
    
    返回：
    处理后的行数据，如果不需要保留则返回None
    """
    # 获取原始类别
    original_label = row['Parent directory']
    
    # 检查该类别是否已达到最大数量
    if category_count.get(original_label, 0) >= max_per_category:
        return None
    
    # 从Commit Url中提取信息
    user, repo, commit_hash = extract_github_info(row['Commit Url'])
    if not user or not repo or not commit_hash:
        print(f"第 {row_index+1} 行URL解析失败: {row['Commit Url']}")
        return None
    
    # 构建提交信息
    commit_message = f"{row['docker']} | {row['What']}"
    if pd.notna(row['Why']):
        commit_message += f" | {row['Why']}"
    
    # 获取diff内容
    diff = fetch_diff(user, repo, commit_hash)
    if not diff:
        return None
    
    # 生成特征
    features = generate_dummy_features(diff)
    
    # 获取映射后的标签
    mapped_label = LABEL_MAP.get(original_label, "Other")
    
    # 构建转换后的数据行
    converted_row = {
        'user': user,
        'repo': repo,
        'commit': commit_hash,
        'labels': mapped_label,
        'msgs': commit_message,
        'diffs': diff,
        'feature': json.dumps(features)
    }
    
    return converted_row


def convert_excel_to_csv(excel_path, output_path, max_per_category=400):
    """
    将Excel文件转换为项目所需的CSV格式（并行版本）
    
    参数：
    excel_path: Excel文件路径
    output_path: 输出CSV文件路径
    max_per_category: 每个类别最多保留的数据条数
    """
    # 读取Excel文件
    df = pd.read_excel(excel_path)
    
    # 准备转换后的数据
    converted_data = []
    
    # 统计每个类别已处理的数据条数
    category_count = {
        "Adaptive": 0,
        "Corrective": 0,
        "Perfective": 0,
        "Preventive": 0,
        "Other": 0
    }
    
    # 准备所有需要处理的行
    rows_to_process = []
    for index, row in df.iterrows():
        original_label = row['Parent directory']
        if category_count.get(original_label, 0) < max_per_category:
            rows_to_process.append((index, row))
    
    print(f"总共需要处理 {len(rows_to_process)} 行数据")
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_row = {
            executor.submit(process_single_row, row, max_per_category, category_count, index): (index, row)
            for index, row in rows_to_process
        }
        
        # 处理完成的任务
        for future in tqdm(as_completed(future_to_row), total=len(future_to_row), desc="转换数据"):
            index, row = future_to_row[future]
            try:
                result = future.result()
                if result:
                    # 检查该类别是否还有名额
                    original_label = row['Parent directory']
                    if category_count.get(original_label, 0) < max_per_category:
                        converted_data.append(result)
                        category_count[original_label] += 1
                        
                        # 检查是否所有类别都已达到最大数量
                        if all(count >= max_per_category for count in category_count.values()):
                            print("\n所有类别已达到最大数量，取消剩余任务")
                            # 取消所有未完成的任务
                            for f in future_to_row:
                                f.cancel()
                            break
            except Exception as e:
                print(f"处理第 {index+1} 行失败: {e}")
    
    # 将转换后的数据保存为CSV
    converted_df = pd.DataFrame(converted_data)
    converted_df.to_csv(output_path, index=False, encoding='utf-8')
    
    # 保存缓存
    save_cache()
    
    print(f"转换完成！")
    print(f"原始数据行数: {len(df)}")
    print(f"转换后数据行数: {len(converted_df)}")
    print(f"类别分布:")
    for category, count in category_count.items():
        print(f"  {category}: {count} 条")
    print(f"输出文件: {output_path}")


if __name__ == "__main__":
    # 输入输出文件路径
    excel_path = '/root/COLARE/COLARE-op6-test/dataset/preTrain-myData/Mydata_5.xlsx'
    output_path = '/root/COLARE/COLARE-op6-test/dataset/preTrain-myData/converted_dataset.csv'
    
    # 转换数据，每个类别保留400条数据
    convert_excel_to_csv(excel_path, output_path, max_per_category=400)
