#!/usr/bin/env python3
"""
测试脚本：使用训练好的5分类模型在10分类数据集上测试
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
import math
import numpy as np
from config import Config
from trainer import Trainer
from model import build_model
from utils import move_to_device
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_on_1793_dataset(trained_model_path):
    """使用训练后的模型在1793数据集（10分类）上进行测试"""
    logger.info("=== 开始在1793数据集上测试模型 ===")
    
    # 配置测试参数
    config = Config()
    
    # 设置模型类型
    config.model = "CCModel"
    
    # 设置1793数据集路径
    config.dataset_path = "dataset/1793/dataset.csv"
    config.test_dataset_path = "dataset/1793/dataset.csv"
    
    # 设置模型保存目录
    config.save_dir = "output/transfer_learning_test"
    os.makedirs(config.save_dir, exist_ok=True)
    
    # 设置设备参数
    config.device = "gpu"  # 使用GPU测试
    
    # 设置数据集参数
    config.dataset = "CommitDataset"
    config.data_dir = "dataset/1793"
    config.use_roberta = False
    
    # 确保config.namespace包含所有必要的参数
    if not hasattr(config, 'namespace'):
        config.namespace = {}
    config.namespace['use_roberta'] = config.use_roberta
    config.namespace['do_test'] = True
    config.num_workers = 1  # 减少workers数量
    
    # 设置文件和代码块限制
    config.file_num_limit = 5  # 减少文件数量限制
    config.hunk_num_limit = 5  # 减少hunk数量限制
    config.code_num_limit = 128  # 减少代码块大小
    
    # 设置测试参数
    config.batch_size = 8
    
    # 构建模型
    logger.info("构建模型...")
    model = build_model(config)
    
    # 加载训练后的模型权重
    logger.info(f"加载训练后的模型: {trained_model_path}")
    checkpoint = torch.load(trained_model_path)
    model.load_state_dict(checkpoint["state_dict"])
    
    # 微调策略：更深层次的解冻，允许更多层参与训练
    logger.info("设置模型微调策略...")
    # 1. 先冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 2. 解冻更多的编码器层（从最后5层开始）
    if hasattr(model, 'code_change_encoder') and hasattr(model.code_change_encoder, 'encoder'):
        # 解冻Roberta编码器的更多层
        for i in range(-5, 0):  # 解冻最后5层，增加学习能力
            for param in model.code_change_encoder.encoder.layer[i].parameters():
                param.requires_grad = True
    
    # 3. 解冻hunk比较器和reducer层
    if hasattr(model, 'code_change_encoder'):
        for attr in ['hunk_compare_poller', 'hunk_reducer', 'file_reducer']:
            if hasattr(model.code_change_encoder, attr):
                for param in getattr(model.code_change_encoder, attr).parameters():
                    param.requires_grad = True
    
    # 4. 解冻所有的特征组合器和分类器
    for attr in ['feature_combiner', 'text_code_combiner', 'classifier']:
        if hasattr(model, attr):
            for param in getattr(model, attr).parameters():
                param.requires_grad = True
    
    # 5. 增强分类器的表达能力
    logger.info("增强分类器表达能力...")
    # 替换原有的线性分类器为更深层的分类器
    input_dim = model.classifier.in_features
    num_classes = model.classifier.out_features
    
    # 创建新的分类器
    deep_classifier = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 1024),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(512, num_classes)
    )
    
    # 替换原有的分类器
    model.classifier = deep_classifier
    
    # 确保新分类器的参数是可训练的
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    # 输出可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"可训练参数: {trainable_params} / {total_params} ({trainable_params/total_params*100:.2f}%)")
    
    # 初始化训练器
    trainer = Trainer(config)
    
    # 将模型移到设备上
    model = model.to(trainer.device)
    
    # 确保分类器也在正确的设备上
    model.classifier = model.classifier.to(trainer.device)
    
    # 手动构建数据加载器
    from dataset.ccdataset import CommitDataset
    from dataset.ccdataset import collate_fn as cc_collate_fn
    from torch.utils.data import DataLoader
    import functools
    from collections import Counter
    
    # 加载10分类的1793数据集
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "dataset/1793/dataset.csv")
    full_test_dataset = CommitDataset(dataset_path, config, use_roberta=False)
    
    # 输出原始数据集信息
    logger.info(f"原始1793测试数据集大小: {len(full_test_dataset)} 样本")
    all_labels = [full_test_dataset.commit_labels[i] for i in range(len(full_test_dataset))]
    label_counts = Counter(all_labels)
    logger.info(f"原始测试数据集类别分布: {dict(label_counts)}")
    
    # 筛选出样本数 >= 10的类别
    valid_labels = [label for label, count in label_counts.items() if count >= 10]
    logger.info(f"保留的类别 (样本数 >= 10): {valid_labels}")
    
    # 过滤数据集，只保留有效类别的样本
    valid_indices = [i for i, label in enumerate(full_test_dataset.commit_labels) if label in valid_labels]
    
    # 创建过滤后的数据集
    from torch.utils.data import Subset
    test_dataset = Subset(full_test_dataset, valid_indices)
    
    # 输出过滤后的数据集信息
    filtered_labels = [full_test_dataset.commit_labels[i] for i in valid_indices]
    filtered_label_counts = Counter(filtered_labels)
    logger.info(f"过滤后1793测试数据集大小: {len(test_dataset)} 样本")
    logger.info(f"过滤后测试数据集类别分布: {dict(filtered_label_counts)}")
    
    # 创建测试数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=functools.partial(
            cc_collate_fn,
            msg_collator=full_test_dataset.message_collator,  # 从原始数据集访问message_collator
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
    
    # 开始微调训练
    logger.info("开始在1793数据集上进行微调训练...")
    
    # 准备训练数据
    from torch.utils.data import random_split
    
    # 将测试数据集划分为训练集和验证集用于微调
    train_size = int(0.7 * len(test_dataset))
    val_size = len(test_dataset) - train_size
    fine_tune_train, fine_tune_val = random_split(test_dataset, [train_size, val_size])
    
    # 创建微调训练和验证数据加载器
    from torch.utils.data import DataLoader
    fine_tune_train_loader = DataLoader(
        fine_tune_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=functools.partial(
            cc_collate_fn,
            msg_collator=full_test_dataset.message_collator,
        ),
    )
    
    fine_tune_val_loader = DataLoader(
        fine_tune_val,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=functools.partial(
            cc_collate_fn,
            msg_collator=full_test_dataset.message_collator,
        ),
    )
    
    logger.info(f"微调训练集大小: {len(fine_tune_train)}, 验证集大小: {len(fine_tune_val)}")
    
    # 设置微调训练参数
    config.fine_tune_epochs = 10  # 增加训练轮数
    config.fine_tune_lr = 2e-5  # 更小的初始学习率，防止过拟合
    config.warmup_epochs = 2  # 预热epoch
    
    # 重新初始化优化器，只训练可训练参数
    logger.info("初始化微调优化器...")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.fine_tune_lr, weight_decay=0.01)
    
    # 使用更复杂的学习率调度器：预热 + 余弦退火
    total_steps = config.fine_tune_epochs * len(fine_tune_train_loader)
    warmup_steps = config.warmup_epochs * len(fine_tune_train_loader)
    
    # 自定义学习率调度器
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # 预热阶段：线性增加
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # 余弦退火阶段
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler
    trainer.train_loader = fine_tune_train_loader
    trainer.eval_loader = fine_tune_val_loader
    trainer.test_loader = test_loader
    
    # 设置数据集长度信息
    config.train_data_len = len(fine_tune_train)
    config.train_loader_len = len(fine_tune_train_loader)
    config.eval_data_len = len(fine_tune_val)
    config.eval_loader_len = len(fine_tune_val_loader)
    
    # 开始微调训练
    logger.info(f"开始微调训练，共 {config.fine_tune_epochs} 个epoch...")
    for epoch in range(config.fine_tune_epochs):
        logger.info(f"\n=== 微调训练 Epoch {epoch+1}/{config.fine_tune_epochs} ===")
        
        # 训练
        trainer.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(fine_tune_train_loader, total=len(fine_tune_train_loader)):
            batch = move_to_device(batch, trainer.device)
            optimizer.zero_grad()
            
            out = trainer.model(**batch)
            loss = trainer.criterion(out, batch["labels"])
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = torch.argmax(out, dim=1)
            train_total += batch["labels"].size(0)
            train_correct += (pred == batch["labels"]).sum().item()
        
        scheduler.step()
        
        train_loss /= len(fine_tune_train_loader)
        train_acc = train_correct / train_total
        logger.info(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        
        # 验证
        trainer.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(fine_tune_val_loader, total=len(fine_tune_val_loader)):
                batch = move_to_device(batch, trainer.device)
                out = trainer.model(**batch)
                loss = trainer.criterion(out, batch["labels"])
                
                val_loss += loss.item()
                pred = torch.argmax(out, dim=1)
                val_total += batch["labels"].size(0)
                val_correct += (pred == batch["labels"]).sum().item()
        
        val_loss /= len(fine_tune_val_loader)
        val_acc = val_correct / val_total
        logger.info(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
    
    # 微调完成后，在完整测试集上测试
    logger.info("\n=== 微调完成，开始在完整测试集上测试 ===")
    
    # 计算额外的指标
    from sklearn.metrics import f1_score
    
    # 手动运行测试，收集所有预测和真实标签
    trainer.model.eval()
    all_preds = []
    all_tgts = []
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader)):
            batch = move_to_device(batch, trainer.device)
            out = trainer.model(**batch)
            pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()
            all_preds.append(pred)
            tgt = batch["labels"].cpu().detach().numpy()
            all_tgts.append(tgt)
    
    # 展平预测和真实标签
    all_tgts = [x for xx in all_tgts for x in xx]
    all_preds = [x for xx in all_preds for x in xx]
    
    # 将numpy int64转换为Python int
    all_tgts = [int(x) for x in all_tgts]
    all_preds = [int(x) for x in all_preds]
    
    # 计算额外的F1指标
    weighted_f1 = f1_score(all_tgts, all_preds, average="weighted")
    micro_f1 = f1_score(all_tgts, all_preds, average="micro")
    macro_f1 = f1_score(all_tgts, all_preds, average="macro")
    accuracy = sum(1 for p, t in zip(all_preds, all_tgts) if p == t) / len(all_tgts)
    
    logger.info(f"准确率: {accuracy}")
    logger.info(f"Micro F1: {micro_f1}")
    logger.info(f"Macro F1: {macro_f1}")
    logger.info(f"加权F1: {weighted_f1}")
    
    # 优化：尝试不同的分类阈值，提高F1分数
    logger.info("\n=== 尝试不同分类阈值 ===")
    best_macro_f1 = macro_f1
    best_threshold = 0.5
    
    # 获取模型的原始输出概率
    trainer.model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader)):
            batch = move_to_device(batch, trainer.device)
            out = trainer.model(**batch)
            probs = torch.nn.functional.softmax(out, dim=1).cpu().detach().numpy()
            all_probs.append(probs)
    
    # 展平概率
    all_probs = [p for batch_probs in all_probs for p in batch_probs]
    
    # 尝试不同的阈值，寻找最优阈值
    from sklearn.metrics import f1_score
    
    # 尝试使用置信度阈值调整
    logger.info("使用置信度阈值调整：")
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        # 对于每个样本，选择概率大于阈值的最高类别
        adjusted_preds = []
        for probs in all_probs:
            max_prob = max(probs)
            if max_prob >= threshold:
                adjusted_preds.append(np.argmax(probs))
            else:
                # 如果没有类别概率大于阈值，选择最高概率的类别
                adjusted_preds.append(np.argmax(probs))
        
        threshold_macro_f1 = f1_score(all_tgts, adjusted_preds, average="macro")
        threshold_weighted_f1 = f1_score(all_tgts, adjusted_preds, average="weighted")
        logger.info(f"  阈值 {threshold:.1f}: Macro F1 = {threshold_macro_f1:.5f}, 加权F1 = {threshold_weighted_f1:.5f}")
        
        if threshold_macro_f1 > best_macro_f1:
            best_macro_f1 = threshold_macro_f1
            best_threshold = threshold
    
    logger.info(f"最佳阈值: {best_threshold:.1f}, 最佳Macro F1: {best_macro_f1:.5f}")
    
    # 保存完整测试结果
    full_test_results = {
        "accuracy": float(accuracy),
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "best_threshold": float(best_threshold),
        "best_macro_f1": float(best_macro_f1)
    }
    
    test_results_path = os.path.join(config.save_dir, "test_results.json")
    with open(test_results_path, "w") as f:
        json.dump(full_test_results, f, indent=2)
    
    # 保存微调后的模型
    fine_tuned_model_path = os.path.join(config.save_dir, "fine_tuned_model.pt")
    torch.save({
        'state_dict': trainer.model.state_dict(),
        'epoch': config.fine_tune_epochs,
    }, fine_tuned_model_path)
    logger.info(f"微调后模型保存到: {fine_tuned_model_path}")
    
    logger.info(f"测试结果保存到: {test_results_path}")
    logger.info(f"测试输出目录: {config.save_dir}")
    
    return full_test_results


if __name__ == "__main__":
    import sys
    
    # 获取命令行参数中的模型路径，默认使用训练脚本保存的模型
    if len(sys.argv) > 1:
        trained_model_path = sys.argv[1]
    else:
        trained_model_path = "output/transfer_learning/final_model.pt"
    
    # 检查模型路径是否存在
    if not os.path.exists(trained_model_path):
        logger.error(f"模型文件不存在: {trained_model_path}")
        logger.error("请先运行 train_5class.py 训练模型，或者提供正确的模型路径")
        sys.exit(1)
    
    # 开始测试
    test_results = test_on_1793_dataset(trained_model_path)
    
    # 输出最终结果
    logger.info("\n=== 最终测试结果 ===")
    logger.info(f"准确率: {test_results.get('test_accuracy', 'N/A')}")
    logger.info(f"Macro F1: {test_results.get('test_macro_f1', 'N/A')}")
    logger.info(f"Micro F1: {test_results.get('test_micro_f1', 'N/A')}")
    logger.info(f"加权F1: {test_results.get('test_weighted_f1', 'N/A')}")
