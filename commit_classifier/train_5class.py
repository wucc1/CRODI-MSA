#!/usr/bin/env python3
"""
训练5分类模型的脚本
"""

# 直接在文件顶部添加补丁代码，解决导入错误
# 修复1: split_torch_state_dict_into_shards函数缺失
import sys
import types

# 先导入huggingface_hub模块
import huggingface_hub

# 定义缺失的函数
def split_torch_state_dict_into_shards(state_dict, max_shard_size):
    """将PyTorch的state_dict分割成多个shard"""
    shards = []
    current_shard = {}
    current_size = 0
    
    for key, tensor in state_dict.items():
        tensor_size = tensor.nelement() * tensor.element_size()
        
        if current_size + tensor_size > max_shard_size and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        
        current_shard[key] = tensor
        current_size += tensor_size
    
    if current_shard:
        shards.append(current_shard)
    
    return shards

# 将函数添加到huggingface_hub模块
setattr(huggingface_hub, 'split_torch_state_dict_into_shards', split_torch_state_dict_into_shards)

# 修复2: GradScaler导入错误
import torch

# 如果torch.amp模块存在但没有GradScaler，将torch.cuda.amp.GradScaler添加到torch.amp
if hasattr(torch, 'amp') and not hasattr(torch.amp, 'GradScaler'):
    # 检查torch.cuda.amp是否存在GradScaler
    if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'GradScaler'):
        setattr(torch.amp, 'GradScaler', torch.cuda.amp.GradScaler)
        print("已将torch.cuda.amp.GradScaler添加到torch.amp模块")

# 现在继续正常导入
import os
import logging
import json
from config import Config
from trainer import Trainer
from model import build_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_5class_model():
    """使用5分类数据集训练模型"""
    logger.info("=== 开始使用5分类数据集训练模型 ===")
    
    # 配置训练参数
    config = Config()
    
    # 设置模型类型
    config.model = "CCModel"
    
    # 设置5分类数据集路径
    config.dataset_path = "dataset/preTrain-myData/converted_dataset.csv"
    
    # 设置模型保存目录
    config.save_dir = "output/transfer_learning"
    os.makedirs(config.save_dir, exist_ok=True)
    
    # 设置设备参数 - 使用GPU加速训练
    config.device = "gpu"
    
    # 设置数据集参数
    config.dataset = "CommitDataset"
    config.data_dir = "dataset/preTrain-myData"
    config.use_roberta = False
    
    # 确保config.namespace包含所有必要的参数
    if not hasattr(config, 'namespace'):
        config.namespace = {}
    config.namespace['use_roberta'] = config.use_roberta
    config.namespace['do_train'] = True
    config.namespace['do_test'] = True
    config.num_workers = 1  # 减少workers数量
    
    # 设置文件和代码块限制，进一步减少内存使用
    config.file_num_limit = 2  # 进一步减少文件数量限制
    config.hunk_num_limit = 2  # 进一步减少hunk数量限制
    config.code_num_limit = 32  # 进一步减少代码块大小
    
    # 设置训练参数，优化内存使用
    config.max_epochs = 1  # 只训练1个epoch
    config.batch_size = 1  # 进一步减小批量大小，减少内存使用
    config.grad_acc = 8  # 进一步增加梯度累积，补偿批量大小减小
    config.lr = 1e-4  # 适当的学习率
    config.patience = 1  # 早停耐心值
    config.enable_checkpoint = False  # 禁用检查点保存，加速训练
    config.save_freq = 1  # 每1个epoch保存一次
    
    # 添加环境变量配置以优化CUDA内存使用
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # 构建模型
    logger.info("构建模型...")
    model = build_model(config)
    
    # 加载预训练权重（如果有）
    pretrained_path = os.path.join(os.path.dirname(__file__), "reproduction-results/reproduce5fold_java/checkpoint.pt")
    if os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        logger.info(f"加载了预训练权重: {pretrained_path}")
    
    # 初始化训练器
    trainer = Trainer(config)
    
    # 手动构建数据加载器
    from dataset.ccdataset import CommitDataset
    from dataset.ccdataset import collate_fn as cc_collate_fn
    from torch.utils.data import DataLoader, random_split
    import functools
    from collections import Counter
    
    # 加载5分类数据集
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "dataset/preTrain-myData/converted_dataset.csv")
    full_dataset = CommitDataset(dataset_path, config, use_roberta=False)
    
    # 输出数据集信息
    logger.info(f"5分类训练数据集大小: {len(full_dataset)} 样本")
    all_labels = [full_dataset.commit_labels[i] for i in range(len(full_dataset))]
    label_counts = Counter(all_labels)
    logger.info(f"训练数据集类别分布: {dict(label_counts)}")
    
    # 划分训练集、验证集和测试集
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_size = int(0.8 * len(train_dataset))
    eval_size = len(train_dataset) - train_size
    train_dataset, eval_dataset = random_split(train_dataset, [train_size, eval_size])
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=functools.partial(
            cc_collate_fn,
            msg_collator=full_dataset.message_collator,
        ),
        drop_last=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=functools.partial(
            cc_collate_fn,
            msg_collator=full_dataset.message_collator,
        ),
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=functools.partial(
            cc_collate_fn,
            msg_collator=full_dataset.message_collator,
        ),
    )
    
    # 设置训练器设备
    if config.device == "gpu":
        device = "cuda:0"
    elif config.device == "gpu2":
        device = "cuda:1"
    else:
        device = "cpu"
    
    trainer.device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # 设置模型
    trainer.model = model.to(trainer.device)
    
    # 设置损失函数和优化器
    trainer.criterion = trainer._build_criterion(config).to(trainer.device)
    trainer.optimizer = trainer._build_optimizer(trainer.model, config)
    
    # 设置数据加载器
    trainer.train_loader = train_loader
    trainer.test_loader = test_loader
    trainer.eval_loader = eval_loader
    
    # 设置数据集长度信息
    config.train_data_len = len(train_dataset)
    config.train_loader_len = len(train_loader)
    config.eval_data_len = len(eval_dataset)
    config.eval_loader_len = len(eval_loader)
    config.test_data_len = len(test_dataset)
    config.test_loader_len = len(test_loader)
    
    # 开始训练
    logger.info("开始训练5分类模型...")
    trainer.train(skip_prepare=True)
    
    # 保存最终模型权重（只保存状态字典，避免config对象的pickle问题）
    final_model_path = os.path.join(config.save_dir, "final_model.pt")
    torch.save({
        'state_dict': trainer.model.state_dict(),
        'epoch': config.max_epochs,
    }, final_model_path)
    
    logger.info(f"5分类模型训练完成，最终模型保存到: {final_model_path}")
    
    return final_model_path


if __name__ == "__main__":
    trained_model_path = train_5class_model()
    print(f"\n训练完成！模型保存路径: {trained_model_path}")
