# 修复 split_torch_state_dict_into_shards 导入错误
import sys
import types

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

# 获取huggingface_hub模块
import huggingface_hub

# 将函数添加到huggingface_hub模块
setattr(huggingface_hub, 'split_torch_state_dict_into_shards', split_torch_state_dict_into_shards)

# 验证修复
print("修复成功！split_torch_state_dict_into_shards已添加到huggingface_hub模块。")
