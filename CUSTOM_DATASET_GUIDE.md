# 将自定义Commit数据集转换为项目所需格式指南

## 1. 项目期望的数据集格式

项目使用CSV格式的数据集，包含以下列：

| 列名 | 描述 | 示例 |
|------|------|------|
| user | 提交者用户名 | ponsonio |
| repo | 仓库名 | RxJava |
| commit | Commit哈希值 | 0531b8bff5c14d9504beefb4ad47f473e3a22932 |
| labels/maintenance_type | 提交标签（分类） | Perfective 或 Service Functionality Defects |
| msgs | 提交信息 | Change hasException to hasThrowable-- |
| diffs | 完整的git diff内容 | diff --git a/file1 b/file1... |
| feature | 数值特征（JSON格式） | [1, 0, 0, ...] |

## 2. 标签映射说明

项目支持两种标签格式，会自动根据列名识别：

### 2.1 长标签名映射（labels列）

| 标签名称 | 映射ID |
|----------|--------|
| Service Configuration Defects | 0 |
| Service Build and Dependency Defects | 1 |
| Service Functionality Defects | 2 |
| Service Communication Defects | 3 |
| Service Deployment Defects | 4 |
| Service Structure and Code Specification Defects | 5 |
| Service Data Consistency Defects | 6 |
| Cross-service Logging Defects | 7 |
| Service Security Defects | 8 |
| Service Exception Handling Defects | 9 |

### 2.2 短标签名映射（maintenance_type列）

| 短标签 | 映射ID | 对应长标签 |
|--------|--------|------------|
| SCD | 0 | Service Configuration Defects |
| SBDD | 1 | Service Build and Dependency Defects |
| SFD | 2 | Service Functionality Defects |
| SCMD | 3 | Service Communication Defects |
| SDD | 4 | Service Deployment Defects |
| SSCSD | 5 | Service Structure and Code Specification Defects |
| SDCD | 6 | Service Data Consistency Defects |
| CLD | 7 | Cross-service Logging Defects |
| SSD | 8 | Service Security Defects |
| SEHD | 9 | Service Exception Handling Defects |

## 3. 自定义数据集转换步骤

### 3.1 步骤1：准备原始数据

确保你的自定义数据集包含以下信息：
- 提交者信息
- 仓库信息
- Commit哈希
- 提交信息
- 完整的git diff内容
- 提交标签（分类）
- 可选的数值特征

### 3.2 步骤2：转换为CSV格式

将你的数据转换为CSV文件，包含项目所需的所有列。你可以使用Python的pandas库来处理：

```python
import pandas as pd

# 假设你有以下数据结构
data = [
    {
        'user': 'user1',
        'repo': 'repo1',
        'commit': 'commit1',
        'labels': 'Service Functionality Defects',  # 或使用短标签 'maintenance_type': 'SFD'
        'msgs': 'Fix bug in login function',
        'diffs': 'diff --git a/file1.py b/file1.py...',
        'feature': '[1, 0, 0, 1, 0]'  # JSON格式的数值特征
    },
    # 更多数据...
]

# 创建DataFrame并保存为CSV
df = pd.DataFrame(data)
df.to_csv('custom_dataset.csv', index=False)
```

### 3.3 步骤3：确保diff格式正确

确保你的git diff内容包含完整的diff信息，包括：
- `diff --git` 行
- 文件路径
- 索引信息
- 变更的行

### 3.4 步骤4：处理数值特征

如果你的数据集没有数值特征，可以：
1. 设置feature列为空字符串或默认值
2. 或者根据项目需要生成相关特征

## 4. 使用转换后的数据集

将转换后的CSV文件放入`dataset/`目录下，然后在模型训练时指定该数据集路径。

## 5. 示例转换脚本

以下是一个完整的示例脚本，用于将自定义commit数据集转换为项目所需格式：

```python
import pandas as pd
import json

def convert_commit_dataset(raw_data, output_path):
    """
    将原始commit数据转换为项目所需的CSV格式
    
    参数：
    raw_data: 原始数据列表，每个元素包含commit信息
    output_path: 输出CSV文件路径
    """
    converted_data = []
    
    for commit in raw_data:
        # 转换标签（根据你的原始数据格式调整）
        if 'label' in commit:
            # 假设原始数据使用数字标签，需要映射到项目的标签名
            label_map = {
                0: 'Service Configuration Defects',
                1: 'Service Build and Dependency Defects',
                2: 'Service Functionality Defects',
                # 其他标签映射...
            }
            label = label_map.get(commit['label'], 'Service Functionality Defects')
        else:
            label = commit.get('label_name', 'Service Functionality Defects')
        
        # 构建转换后的数据条目
        converted_item = {
            'user': commit.get('user', 'unknown'),
            'repo': commit.get('repo', 'unknown'),
            'commit': commit.get('commit_hash', 'unknown'),
            'labels': label,  # 或使用 maintenance_type 列和短标签
            'msgs': commit.get('message', ''),
            'diffs': commit.get('diff', ''),
            'feature': json.dumps(commit.get('features', [0]*10))  # 假设10维特征
        }
        
        converted_data.append(converted_item)
    
    # 创建DataFrame并保存
    df = pd.DataFrame(converted_data)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"数据集转换完成，保存到：{output_path}")
    print(f"转换后的数据量：{len(df)}")
    
    # 显示数据集信息
    print("\n数据集信息：")
    print(df.info())
    
    print("\n标签分布：")
    print(df['labels'].value_counts())

# 示例用法
if __name__ == "__main__":
    # 假设这是你的原始数据
    raw_commit_data = [
        {
            'user': 'developer1',
            'repo': 'my_project',
            'commit_hash': 'abc123',
            'label': 2,  # 对应 Service Functionality Defects
            'message': 'Fix null pointer exception in user service',
            'diff': 'diff --git a/user_service.py b/user_service.py\nindex 1234567..89abcde 100644\n--- a/user_service.py\n+++ b/user_service.py\n@@ -10,5 +10,7 @@ def get_user(id):\n     user = db.query(User).get(id)\n-    return user.name\n+    if user is None:\n+        return None\n+    return user.name',
            'features': [1, 0, 1, 0, 0, 1, 0, 0, 0, 1]
        },
        # 更多原始数据...
    ]
    
    # 转换数据集
    convert_commit_dataset(raw_commit_data, 'custom_commit_dataset.csv')
```

## 6. 注意事项

1. **标签一致性**：确保所有标签都能正确映射到项目支持的10类标签之一
2. **diff完整性**：确保git diff包含完整的变更信息，否则解析会失败
3. **编码问题**：确保CSV文件使用UTF-8编码
4. **特征维度**：确保数值特征的维度与模型期望一致
5. **数据量**：建议数据集至少包含数百个样本，以获得良好的模型性能

## 7. 验证转换结果

转换完成后，可以使用以下代码验证数据集格式是否正确：

```python
import pandas as pd

# 读取转换后的数据集
df = pd.read_csv('custom_dataset.csv')

# 检查列名是否正确
required_columns = ['user', 'repo', 'commit', 'labels', 'msgs', 'diffs', 'feature']
if all(col in df.columns for col in required_columns):
    print("✓ 所有必需列都存在")
else:
    missing_cols = [col for col in required_columns if col not in df.columns]
    print(f"✗ 缺少列：{missing_cols}")

# 检查前几行数据
print("\n前3行数据：")
print(df.head(3))

# 检查标签分布
print("\n标签分布：")
print(df['labels'].value_counts())
```

## 8. 常见问题解决

### 8.1 diff解析失败
- 确保diff包含完整的`diff --git`行
- 检查diff格式是否标准
- 避免在diff中包含特殊字符

### 8.2 标签映射错误
- 确保所有标签都在项目支持的标签列表中
- 检查标签大小写和拼写

### 8.3 数值特征格式错误
- 确保feature列是有效的JSON格式字符串
- 确保特征维度一致

通过以上步骤，你可以将自己的commit数据集转换为项目所需的格式，用于模型训练和评估。