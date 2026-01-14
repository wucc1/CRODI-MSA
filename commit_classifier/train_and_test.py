#!/usr/bin/env python3
"""
先使用converted_dataset.csv训练模型，然后在1793数据集上测试
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
from config import Config
from trainer import Trainer
from model import build_model
from dataset import build_data_loaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_with_converted_dataset():
    """使用converted_dataset.csv训练模型"""
    logger.info("=== 开始使用converted_dataset.csv训练模型 ===")
    
    # 配置训练参数
    config = Config()
    
    # 设置模型类型
    config.model = "CCModel"
    
    # 设置数据集路径为converted_dataset.csv
    config.dataset_path = "dataset/preTrain-myData/converted_dataset.csv"
    config.train_dataset_path = "dataset/preTrain-myData/converted_dataset.csv"
    config.test_dataset_path = "dataset/preTrain-myData/converted_dataset.csv"
    config.eval_dataset_path = "dataset/preTrain-myData/converted_dataset.csv"
    
    # 设置模型保存目录
    config.save_dir = "output/finetune_converted"
    os.makedirs(config.save_dir, exist_ok=True)
    
    # 设置设备参数
    config.device = "gpu" if torch.cuda.is_available() else "cpu"
    
    # 设置交叉验证参数
    config.enable_cv = False
    
    # 设置数据集参数
    config.dataset = "CommitDataset"
    config.data_dir = "dataset/preTrain-myData"
    config.use_roberta = False
    
    # 确保config.namespace包含所有必要的参数
    if not hasattr(config, 'namespace'):
        config.namespace = {}
    config.namespace['use_roberta'] = config.use_roberta
    config.namespace['enable_checkpoint'] = getattr(config, 'enable_checkpoint', False)
    config.namespace['do_train'] = True
    config.namespace['do_test'] = False
    config.namespace['enable_cv'] = config.enable_cv
    config.num_workers = 4
    
    # 设置文件和代码块限制
    config.file_num_limit = 10
    config.hunk_num_limit = 10
    config.code_num_limit = 256
    
    # 设置训练参数
    config.max_epochs = 10
    config.batch_size = 8
    config.grad_acc = 1
    config.lr = 1e-4
    config.patience = 5
    
    # 构建模型
    model = build_model(config)
    
    # 加载预训练权重（如果有）
    pretrained_path = os.path.join(os.path.dirname(__file__), "reproduction-results/reproduce5fold_java/checkpoint.pt")
    if os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        logger.info(f"加载了预训练权重: {pretrained_path}")
    
    # 冻结CodeBERT参数
    if config.model in ["CCModel", "CodeFeatModel", "MessageCodeModel", "MessageModel"]:
        for param in model.code_change_encoder.hunk_encoder.parameters():
            param.requires_grad = False
    elif config.model == "CodeBERTBaseline":
        for param in model.codebert.parameters():
            param.requires_grad = False
    elif config.model == "MessageFeatModel":
        for param in model.message_encoder.parameters():
            param.requires_grad = False
    
    # 保存配置
    config.to_json(os.path.join(config.save_dir, "config.json"))
    
    # 初始化训练器
    trainer = Trainer(config)
    
    # 手动构建数据加载器
    from dataset.ccdataset import CommitDataset
    from dataset.ccdataset import collate_fn as cc_collate_fn
    from torch.utils.data import DataLoader, random_split
    import functools
    from collections import Counter
    
    # 加载converted_dataset.csv
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "dataset/preTrain-myData/converted_dataset.csv")
    full_dataset = CommitDataset(dataset_path, config, use_roberta=False)
    
    # 输出数据集信息
    logger.info(f"训练数据集大小: {len(full_dataset)} 样本")
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
    
    # 开始训练
    logger.info("开始训练...")
    trainer.train(skip_prepare=True)
    
    # 保存训练后的模型
    model_save_path = os.path.join(config.save_dir, "final_model.pt")
    torch.save({
        'state_dict': trainer.model.state_dict(),
        'config': config,
    }, model_save_path)
    
    logger.info(f"模型训练完成，保存路径: {model_save_path}")
    logger.info(f"训练输出目录: {config.save_dir}")
    
    return model_save_path


def test_on_1793_dataset(trained_model_path):
    """使用训练后的模型在1793数据集上进行测试"""
    logger.info("=== 开始在1793数据集上测试模型 ===")
    
    # 配置测试参数
    config = Config()
    
    # 设置模型类型
    config.model = "CCModel"
    
    # 设置1793数据集路径
    config.dataset_path = "dataset/1793/dataset.csv"
    config.test_dataset_path = "dataset/1793/dataset.csv"
    
    # 设置模型保存目录
    config.save_dir = "output/test_1793"
    os.makedirs(config.save_dir, exist_ok=True)
    
    # 设置设备参数
    config.device = "gpu" if torch.cuda.is_available() else "cpu"
    
    # 设置数据集参数
    config.dataset = "CommitDataset"
    config.data_dir = "dataset/1793"
    config.use_roberta = False
    
    # 确保config.namespace包含所有必要的参数
    if not hasattr(config, 'namespace'):
        config.namespace = {}
    config.namespace['use_roberta'] = config.use_roberta
    config.namespace['do_test'] = True
    config.num_workers = 4
    
    # 设置文件和代码块限制
    config.file_num_limit = 10
    config.hunk_num_limit = 10
    config.code_num_limit = 256
    
    # 设置测试参数
    config.batch_size = 1
    
    # 构建模型
    model = build_model(config)
    
    # 加载训练后的模型权重
    logger.info(f"加载训练后的模型: {trained_model_path}")
    checkpoint = torch.load(trained_model_path)
    model.load_state_dict(checkpoint["state_dict"])
    
    # 初始化训练器
    trainer = Trainer(config)
    
    # 手动构建数据加载器
    from dataset.ccdataset import CommitDataset
    from dataset.ccdataset import collate_fn as cc_collate_fn
    from torch.utils.data import DataLoader
    import functools
    from collections import Counter
    
    # 加载1793数据集
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "dataset/1793/dataset.csv")
    test_dataset = CommitDataset(dataset_path, config, use_roberta=False)
    
    # 输出数据集信息
    logger.info(f"测试数据集大小: {len(test_dataset)} 样本")
    all_labels = [test_dataset.commit_labels[i] for i in range(len(test_dataset))]
    label_counts = Counter(all_labels)
    logger.info(f"测试数据集类别分布: {dict(label_counts)}")
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=functools.partial(
            cc_collate_fn,
            msg_collator=test_dataset.message_collator,
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
    
    # 设置损失函数
    trainer.criterion = trainer._build_criterion(config).to(trainer.device)
    
    # 设置测试数据加载器
    trainer.test_loader = test_loader
    
    # 设置测试集长度信息
    config.test_data_len = len(test_dataset)
    config.test_loader_len = len(test_loader)
    
    # 开始测试
    logger.info("开始测试...")
    test_results = trainer.test(skip_prepare=True)
    
    # 输出测试结果
    logger.info("测试结果:")
    for key, value in test_results.items():
        logger.info(f"{key}: {value}")
    
    # 保存测试结果
    import json
    test_results_path = os.path.join(config.save_dir, "test_results.json")
    with open(test_results_path, "w") as f:
        json.dump(test_results, f, indent=2)
    
    logger.info(f"测试完成，结果保存路径: {test_results_path}")
    logger.info(f"测试输出目录: {config.save_dir}")


if __name__ == "__main__":
    # 第一步：使用converted_dataset.csv训练模型
    trained_model_path = train_with_converted_dataset()
    
    # 第二步：使用训练后的模型在1793数据集上测试
    test_on_1793_dataset(trained_model_path)
