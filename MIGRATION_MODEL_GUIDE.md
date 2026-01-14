# 迁移模型使用说明

## 1. 什么是迁移模型

该项目中的迁移模型是基于预训练的CodeBERT模型，针对代码提交分类任务进行微调的模型。迁移学习允许我们利用在大规模数据集上预训练的模型知识，快速适应新的任务或数据集。

## 2. 迁移模型的组成

项目中主要的迁移模型包括：

- **CCModel**: 综合使用代码变更、提交消息和特征的完整模型
- **CodeFeatModel**: 仅使用代码变更和特征的模型
- **MessageCodeModel**: 使用提交消息和代码变更的模型
- **MessageFeatModel**: 使用提交消息和特征的模型
- **CodeBERTBaseline**: CodeBERT基线模型

## 3. 如何启动迁移模型

### 3.1 直接使用finetune.py脚本

最简单的启动方式是直接运行项目中的`finetune.py`脚本：

```bash
cd /root/COLARE/COLARE-op5-toF1/commit_classifier
python finetune.py
```

### 3.2 使用提供的启动脚本

项目根目录下提供了一个启动脚本`start_migration_model.sh`，可以直接运行：

```bash
cd /root/COLARE/COLARE-op5-toF1
chmod +x start_migration_model.sh
./start_migration_model.sh
```

## 4. 迁移模型的配置

迁移模型的主要配置在`finetune.py`文件中，主要配置项包括：

### 4.1 模型配置

```python
config.model = "CCModel"  # 选择模型类型
config.device = "gpu"      # 设备选择: "gpu", "gpu2" 或 "cpu"
```

### 4.2 数据集配置

```python
config.dataset = "CommitDataset"  # 数据集类型
config.data_dir = "dataset/1793"   # 数据集目录
config.file_num_limit = 10         # 每个提交的最大文件数
config.hunk_num_limit = 10         # 每个文件的最大代码块数
config.code_num_limit = 256        # 每个代码块的最大代码行数
```

### 4.3 训练参数配置

```python
config.max_epochs = 10  # 训练轮数
config.batch_size = 16  # 批大小
config.lr = 1e-4        # 学习率
config.patience = 5     # 早停耐心值
```

### 4.4 预训练模型配置

```python
# 预训练模型路径
pretrained_path = os.path.join(os.path.dirname(__file__), "reproduction-results/reproduce5fold_java/checkpoint.pt")
```

## 5. 迁移模型的工作流程

1. **加载预训练模型**: 从`reproduction-results/reproduce5fold_java/checkpoint.pt`加载预训练的CodeBERT模型权重

2. **冻结预训练层**: 冻结CodeBERT的参数，只训练自定义的分类层和组合层

3. **数据集处理**: 加载并预处理代码提交数据集

4. **模型微调**: 在新的数据集上微调模型

5. **保存结果**: 将微调后的模型和配置保存到指定目录

## 6. 常见问题及解决方案

### 6.1 预训练模型不存在

如果出现"Pre-trained model not found"错误，请检查预训练模型路径是否正确：

```bash
ls -la /root/COLARE/COLARE-op5-toF1/commit_classifier/reproduction-results/reproduce5fold_java/
```

确保目录中存在`checkpoint.pt`文件。

### 6.2 CUDA内存不足

如果出现CUDA内存不足错误，可以尝试减小批大小：

```python
config.batch_size = 8  # 减小批大小
```

### 6.3 导入错误

项目中的`finetune.py`已经包含了导入错误的修复代码，如果仍然出现导入错误，可以尝试重新安装依赖：

```bash
pip install -r requirements.txt
```

## 7. 输出结果

迁移模型的训练结果将保存在`output/finetune_1793/`目录下，包括：

- **config.json**: 训练配置文件
- **best_model.pt**: 最佳模型权重
- **checkpoint.pt**: 训练检查点
- **runlong.log**: 训练日志

## 8. 自定义迁移任务

如果需要在其他数据集上进行迁移学习，可以修改`finetune.py`中的数据集路径：

```python
config.dataset_path = "your/dataset/path/dataset.csv"
config.data_dir = "your/dataset/path/"
```

并确保数据集格式与项目要求一致。

## 9. 模型评估

迁移模型训练完成后，可以使用`test_model.py`脚本进行评估：

```bash
cd /root/COLARE/COLARE-op5-toF1/commit_classifier
python trainer/test_model.py --name finetune_1793 --model CCModel --device gpu --data_dir ../dataset/1793/ --save_dir ../output/
```

## 10. 联系信息

如果在使用迁移模型过程中遇到问题，请参考项目的README.md文件或联系项目维护者。