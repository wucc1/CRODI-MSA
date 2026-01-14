# 使用项目核心文件转换自定义数据集指南

## 1. 核心文件功能说明

### 1.1 `__init__.py`
- 提供了GitHub仓库和提交的正则表达式
- 实现了从GitHub API获取原始diff的功能
- 提供了获取commit分页器和处理commits的功能

### 1.2 `common.py`
- 定义了数据类：
  - `PaginatorCache`：用于缓存分页器信息
  - `CommitInfo`：包含提交的完整信息（sha、作者、消息、diff、文件名、日期、标签等）

### 1.3 `extractor.py`
- 实现了代码和文档的提取器
- 用于从diff中提取各种特征：
  - 代码添加/删除行数
  - 注释添加/删除行数
  - 空格添加/删除行数
- 提供了`process_whole_diff`函数，将完整diff转换为数值特征

### 1.4 `git_client.py`
- 基于pygit2库实现了Git仓库操作
- 提供了提取提交、分类提交、获取文件和贡献者统计等功能
- 核心类`RepoClient`用于管理本地Git仓库

### 1.5 `classify_zip.py`
- 提供了处理ZIP压缩包形式的Git仓库的功能
- 自动提取ZIP文件并查找Git仓库
- 对仓库中的提交进行分类

## 2. 使用示例脚本转换数据集

### 2.1 脚本功能

示例脚本`convert_dataset.py`提供了以下功能：
- 从本地Git仓库提取所有提交信息
- 使用项目提供的`process_whole_diff`函数生成特征
- 将数据转换为模型所需的CSV格式

### 2.2 脚本使用方法

#### 处理单个Git仓库

```bash
python convert_dataset.py --repo /path/to/git/repo --output dataset.csv
```

#### 处理多个Git仓库

```bash
python convert_dataset.py --repos /path/to/repo1,/path/to/repo2 --output dataset.csv
```

### 2.3 脚本参数说明

| 参数 | 说明 | 必须 |
|------|------|------|
| `--repo` | 单个Git仓库路径 | 是（与`--repos`二选一） |
| `--repos` | 多个Git仓库路径，用逗号分隔 | 是（与`--repo`二选一） |
| `--output` | 输出CSV文件路径 | 是 |

## 3. 自定义数据集转换步骤

### 3.1 步骤1：准备Git仓库

确保你的自定义数据集是Git仓库格式，可以是：
- 本地Git仓库
- Git仓库的ZIP压缩包

### 3.2 步骤2：提取提交信息

使用`RepoClient`类从Git仓库中提取提交信息：

```python
from dashboard.classifier.core.git_client import RepoClient

# 初始化RepoClient
repo_client = RepoClient(repo_path)

# 获取分支的提交
commits = repo_client.list_branch_commits(branch_name)
```

### 3.3 步骤3：生成特征

使用`process_whole_diff`函数从diff中生成特征：

```python
from dashboard.classifier.core.extractor import process_whole_diff

# 生成特征
features = process_whole_diff(commit_diff)
```

### 3.4 步骤4：构建数据集

将提取的信息和生成的特征组合成模型所需的格式：

```python
commit_info = {
    'user': commit.author,
    'repo': repo_name,
    'commit': commit.sha,
    'labels': commit_label,  # 根据实际情况设置标签
    'msgs': commit.commit_message,
    'diffs': commit.commit_diff,
    'feature': json.dumps(features)
}
```

### 3.5 步骤5：保存为CSV

使用pandas将数据保存为CSV格式：

```python
import pandas as pd

# 创建DataFrame
df = pd.DataFrame(commits_info_list)

# 保存为CSV
df.to_csv(output_path, index=False, encoding='utf-8')
```

## 4. 高级用法

### 4.1 处理ZIP格式的Git仓库

使用`classify_zip_repo`函数处理ZIP压缩包：

```python
from dashboard.classifier.core.classify_zip import classify_zip_repo

# 处理ZIP文件
repo_client = classify_zip_repo(zip_file_path, cache={})

# 获取提交信息
commits = repo_client.list_branch_commits()
```

### 4.2 从GitHub API获取数据

使用`fetch_commit_paginator`和`fetch_raw_diff`函数从GitHub API获取数据：

```python
from dashboard.classifier.core import fetch_commit_paginator, fetch_raw_diff

# 获取commit分页器
paginator_cache = fetch_commit_paginator(github_repo_url)

# 获取原始diff
diff = fetch_raw_diff(repo_url, commit_sha)
```

## 5. 示例：完整的转换流程

以下是一个完整的示例，展示了如何使用核心文件将自定义Git仓库转换为模型所需的数据集：

```python
import os
import sys
import pandas as pd
import json

# 添加项目根目录到Python路径
sys.path.insert(0, '/root/COLARE/COLARE-op6-test')

from dashboard.classifier.core.git_client import RepoClient
from dashboard.classifier.core.extractor import process_whole_diff

def convert_repo_to_dataset(repo_path, output_csv):
    """
    将Git仓库转换为模型所需的数据集
    """
    # 1. 初始化RepoClient
    repo_client = RepoClient(repo_path)
    
    # 2. 获取所有分支
    branches = repo_client.list_all_branches()
    print(f"找到分支: {branches}")
    
    # 3. 提取所有提交信息
    all_commits = []
    
    for branch in branches:
        print(f"处理分支: {branch}")
        commits = repo_client.list_branch_commits(branch)
        
        for commit in commits:
            try:
                # 4. 生成特征
                features = process_whole_diff(commit.commit_diff)
                
                # 5. 构建提交信息
                commit_info = {
                    'user': commit.author,
                    'repo': os.path.basename(repo_path),
                    'commit': commit.sha,
                    'labels': 'Service Functionality Defects',  # 根据实际情况修改标签
                    'msgs': commit.commit_message,
                    'diffs': commit.commit_diff,
                    'feature': json.dumps(features)
                }
                
                all_commits.append(commit_info)
                print(f"  处理提交: {commit.short_sha} - {commit.commit_message_to_display}")
            except Exception as e:
                print(f"  处理提交 {commit.short_sha} 失败: {e}")
                continue
    
    # 6. 保存为CSV
    df = pd.DataFrame(all_commits)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    
    print(f"转换完成！生成的CSV文件包含 {len(df)} 行数据")
    print(f"输出文件: {output_csv}")
    
    return df

# 示例用法
if __name__ == "__main__":
    repo_path = "/path/to/your/git/repo"
    output_csv = "custom_dataset.csv"
    
    convert_repo_to_dataset(repo_path, output_csv)
```

## 6. 注意事项

1. **标签处理**：
   - 示例脚本中使用了默认标签"Service Functionality Defects"
   - 实际使用时，应根据你的数据集情况设置正确的标签
   - 标签必须是项目支持的10类服务缺陷标签之一

2. **错误处理**：
   - 处理提交时可能会遇到各种错误（如空diff、格式错误等）
   - 示例脚本包含了基本的错误处理，但你可能需要根据实际情况进行调整

3. **性能考虑**：
   - 处理大型仓库时，提取所有提交可能需要较长时间
   - 可以考虑只处理特定分支或时间段的提交

4. **特征生成**：
   - `process_whole_diff`函数生成的特征是固定的21维数值特征
   - 如果你需要添加自定义特征，可以修改`extractor.py`中的相关代码

5. **数据集格式验证**：
   - 转换完成后，建议验证生成的CSV文件格式是否符合要求
   - 可以使用`pandas`库查看数据的基本信息和前几行数据

## 7. 扩展功能

### 7.1 添加自定义特征

如果你需要添加自定义特征，可以修改`extractor.py`文件：

1. 在`CodeInfo`或`DocumentInfo`类中添加新的特征字段
2. 在`extract`方法中实现特征提取逻辑
3. 在`to_dict`方法中添加新特征的处理
4. 在`convert_to_numerical_features`函数中调整特征维度

### 7.2 支持更多标签类型

如果你需要支持更多标签类型，可以修改`extractor.py`或`git_client.py`中的标签映射逻辑。

### 7.3 并行处理

对于大型仓库，可以考虑使用并行处理来提高转换速度：

```python
from concurrent.futures import ProcessPoolExecutor

def process_commit(commit):
    """处理单个提交"""
    try:
        features = process_whole_diff(commit.commit_diff)
        commit_info = {
            'user': commit.author,
            'repo': os.path.basename(repo_path),
            'commit': commit.sha,
            'labels': 'Service Functionality Defects',
            'msgs': commit.commit_message,
            'diffs': commit.commit_diff,
            'feature': json.dumps(features)
        }
        return commit_info
    except Exception as e:
        print(f"处理提交 {commit.short_sha} 失败: {e}")
        return None

# 使用并行处理
with ProcessPoolExecutor() as executor:
    results = executor.map(process_commit, commits)
    all_commits = [r for r in results if r is not None]
```

## 8. 总结

通过使用项目提供的核心文件，你可以轻松地将自定义Git仓库转换为模型所需的数据集。主要步骤包括：

1. 初始化`RepoClient`，连接到本地Git仓库
2. 提取仓库中的所有提交信息
3. 使用`process_whole_diff`函数从diff中生成特征
4. 将提取的信息和生成的特征组合成模型所需的格式
5. 保存为CSV文件

示例脚本`convert_dataset.py`提供了一个完整的实现，你可以直接使用或根据自己的需求进行修改。

希望本指南对你有所帮助！