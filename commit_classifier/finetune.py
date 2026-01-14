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

def finetune_model():
    # 配置微调参数
    config = Config()
    
    # 设置模型类型 - 使用CCModel匹配预训练权重架构
    config.model = "CCModel"  # 使用CCModel架构匹配预训练权重
    
    # 设置数据集路径为新的微调数据集
    config.dataset_path = "dataset/1793/dataset.csv"
    config.train_dataset_path = "dataset/1793/dataset.csv"  # 根据项目实际情况调整
    config.test_dataset_path = "dataset/1793/dataset.csv"    # 根据项目实际情况调整
    config.eval_dataset_path = "dataset/1793/dataset.csv"    # 根据项目实际情况调整
    
    # 设置模型保存目录
    config.save_dir = "output/finetune_1793"
    os.makedirs(config.save_dir, exist_ok=True)
    
    # 设置设备参数
    config.device = "gpu"  # 可以选择"gpu", "gpu2"或"cpu"
    
    # 设置交叉验证参数
    config.enable_cv = False  # 不启用交叉验证，使用完整的训练/测试/评估流程
    
    # 设置数据集参数
    config.dataset = "CommitDataset"  # 使用CommitDataset数据集类
    config.data_dir = "dataset/1793"  # 数据集目录
    config.use_roberta = False  # 使用CodeBERT而不是RoBERTa
    
    # 确保config.namespace包含所有必要的参数
    if not hasattr(config, 'namespace'):
        config.namespace = {}
    config.namespace['use_roberta'] = config.use_roberta
    config.namespace['enable_checkpoint'] = config.enable_checkpoint if 'enable_checkpoint' in config else False
    config.namespace['do_train'] = True
    config.namespace['do_test'] = False
    config.namespace['enable_cv'] = config.enable_cv
    config.num_workers = 4  # 数据加载的工作线程数
    
    # 设置文件和代码块限制
    config.file_num_limit = 10  # 每个提交的最大文件数
    config.hunk_num_limit = 10  # 每个文件的最大代码块数
    config.code_num_limit = 256  # 每个代码块的最大代码行数
    
    # 设置微调参数
    config.max_epochs = 10  # 根据需要调整
    config.batch_size = 8  # 根据需要调整
    config.grad_acc = 1
    config.lr = 1e-4  # 微调学习率
    config.patience = 5  # 设置早停耐心值，避免早停机制出现错误
    
    # 加载预训练模型
    logger.info("Loading pre-trained model...")
    
    # 构建模型
    model = build_model(config)
    
    # 加载预训练权重
    # 使用相对于当前脚本的路径
    pretrained_path = os.path.join(os.path.dirname(__file__), "reproduction-results/reproduce5fold_java/checkpoint.pt")
    if os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path)
        # 使用strict=False忽略不匹配的参数
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        logger.info(f"Loaded pre-trained weights from {pretrained_path}")
    else:
        logger.error(f"Pre-trained model not found at {pretrained_path}")
        return
    
    # 冻结CodeBERT的所有参数
    logger.info("Freezing CodeBERT parameters...")
    
    # 根据不同模型类型冻结CodeBERT参数
    if config.model in ["CCModel", "CodeFeatModel", "MessageCodeModel", "MessageModel"]:
        # 冻结CodeChangeEncoder中的hunk_encoder（CodeBERT）
        for param in model.code_change_encoder.hunk_encoder.parameters():
            param.requires_grad = False
    elif config.model == "CodeBERTBaseline":
        # 冻结codebert参数
        for param in model.codebert.parameters():
            param.requires_grad = False
    elif config.model == "MessageFeatModel":
        # 冻结message_encoder（CodeBERT）参数
        for param in model.message_encoder.parameters():
            param.requires_grad = False
    
    # 确保分类层有10个神经元（已在模型定义中设置）
    logger.info(f"Classifier output size: {model.classifier.out_features}")
    
    # 保存配置
    config.to_json(os.path.join(config.save_dir, "config.json"))
    
    # 初始化训练器
    trainer = Trainer(config)
    
    # 手动构建数据加载器，直接使用dataset.csv文件
    from dataset.ccdataset import CommitDataset
    from dataset.ccdataset import collate_fn as cc_collate_fn
    from torch.utils.data import DataLoader
    import functools
    
    # 获取完整的数据集路径
    # 使用相对路径，从当前脚本所在目录向上一级，然后到dataset目录
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "dataset/1793/dataset.csv")
    
    # 加载数据集
    full_dataset = CommitDataset(dataset_path, config, use_roberta=False)
    
    # 统计所有类别的样本数
    from collections import Counter
    all_labels = [full_dataset.commit_labels[i] for i in range(len(full_dataset))]
    label_counts = Counter(all_labels)
    
    # 输出原始数据集信息
    logger.info(f"原始数据集大小: {len(full_dataset)} 样本")
    logger.info(f"原始数据集类别分布: {dict(label_counts)}")
    
    # 找出样本数大于等于10的类别
    valid_labels = [label for label, count in label_counts.items() if count >= 10]
    
    # 输出筛选后的类别信息
    logger.info(f"筛选后保留的类别: {valid_labels}")
    logger.info(f"筛选后排除的类别: {[label for label, count in label_counts.items() if count < 10]}")
    
    # 筛选整个数据集，只保留有效类别的样本
    valid_indices = [i for i in range(len(full_dataset)) if full_dataset.commit_labels[i] in valid_labels]
    filtered_dataset = torch.utils.data.Subset(full_dataset, valid_indices)
    
    # 输出筛选后的数据集信息
    logger.info(f"筛选后的数据集大小: {len(filtered_dataset)} 样本")
    filtered_labels = [full_dataset.commit_labels[i] for i in valid_indices]
    logger.info(f"筛选后的数据集类别分布: {dict(Counter(filtered_labels))}")
    
    # 检查是否存在重复的commit
    from collections import Counter
    all_commits = [full_dataset.commit_sha[i] for i in valid_indices]
    commit_counts = Counter(all_commits)
    duplicate_commits = {commit: count for commit, count in commit_counts.items() if count > 1}
    print(f"发现 {len(duplicate_commits)} 个重复的commit")
    if duplicate_commits:
        print(f"重复commit详情: {duplicate_commits}")
    
    # 按commit去重
    unique_indices = []
    seen_commits = set()
    for i in valid_indices:
        commit = full_dataset.commit_sha[i]
        if commit not in seen_commits:
            seen_commits.add(commit)
            unique_indices.append(i)
    
    print(f"去重前数据集大小: {len(filtered_dataset)}")
    print(f"去重后数据集大小: {len(unique_indices)}")
    
    # 使用去重后的数据集
    unique_dataset = torch.utils.data.Subset(full_dataset, unique_indices)
    
    # 将去重后的数据集分为训练集和测试集（80%训练，20%测试）
    train_size = int(0.8 * len(unique_dataset))
    test_size = len(unique_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(unique_dataset, [train_size, test_size])
    
    # 再将训练集分为训练集和验证集（80%训练，20%验证）
    train_size = int(0.8 * len(train_dataset))
    eval_size = len(train_dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(train_dataset, [train_size, eval_size])
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(eval_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
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
    
    # 手动设置训练器的所有必要属性，绕过trainer.prepare()方法
    # 设置设备
    if config.device == "gpu":
        device = "cuda:0"
    elif config.device == "gpu2":
        device = "cuda:1"
    else:
        device = "cpu"
    
    if torch.cuda.is_available():
        trainer.device = torch.device(device)
    else:
        trainer.device = torch.device("cpu")
    
    # 设置模型
    trainer.model = model.to(trainer.device)
    
    # 设置损失函数和优化器，直接使用Trainer类的私有方法
    trainer.criterion = trainer._build_criterion(config).to(trainer.device)
    trainer.optimizer = trainer._build_optimizer(trainer.model, config)
    
    # 手动设置数据加载器
    trainer.train_loader = train_loader
    trainer.test_loader = test_loader
    trainer.eval_loader = eval_loader
    
    # 设置数据集长度信息
    config.train_data_len = len(train_dataset)
    config.train_loader_len = len(train_loader)
    config.eval_data_len = len(eval_dataset)
    config.eval_loader_len = len(eval_loader)
    
    # 开始微调
    logger.info("Starting finetuning...")
    trainer.train(skip_prepare=True)
    
    logger.info("Finetuning completed!")

if __name__ == "__main__":
    finetune_model()