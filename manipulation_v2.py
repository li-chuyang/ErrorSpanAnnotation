import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from comet import download_model, load_from_checkpoint
from ESA.annotation_loader import AnnotationLoader
import warnings
import os

# ==============================================================================
# [CONFIG] 实验参数配置面板
# ==============================================================================
CONFIG = {
    # 【改动 1】：支持传入列表，比如 [309, 490, 502]
    "target_neurons": [309],  
    "multiplier": 0.0,                 # 0.0=Mask, 5.0+=放大
    
    # --- 核心修改：指定干预哪些层 ---
    # 选项1: "all" (干预所有层)
    # 选项2: "last" (只干预最后一层)
    # 选项3: [0, 18, 35] (干预指定的层索引列表)

    "layers_to_patch": "all", 
    "model_name": "Unbabel/XCOMET-XL",
    "dataset_name": "ESA-1",
    "test_sample_size": 50,
    "batch_size": 8
}
# ==============================================================================

# --- 1. 修改后的 Hook 定义 (支持多个神经元) ---
def get_manipulation_hook(neuron_indices, multiplier, layer_idx):
    """
    neuron_indices: 接收一个列表，例如 [309, 490, 502]
    """
    def hook_fn(module, inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        
        # 【核心改动 2】：利用花式索引一次性修改多个神经元
        # hidden_states 形状: [batch, seq, 2560]
        # 下面这行会把列表里所有索引对应的列都乘以 multiplier
        hidden_states[:, :, neuron_indices] = hidden_states[:, :, neuron_indices] * multiplier
        
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states
    return hook_fn

# --- 2. 加载数据 (保持不变) ---
loader = AnnotationLoader(refresh_cache=False)
df = loader.get_view([CONFIG['dataset_name']], only_overlap=False)
error_df = df[df[f"{CONFIG['dataset_name']}_error_spans"].notna()].copy()
test_df = error_df.head(CONFIG['test_sample_size'])
data_list = [{"src": row['source'], "mt": row['hypothesis'], "ref": ""} for _, row in test_df.iterrows()]

# --- 3. 加载模型 (保持不变) ---
print(f"Loading {CONFIG['model_name']}...")
model_path = download_model(CONFIG['model_name'])
model = load_from_checkpoint(model_path)
model.eval()
if torch.cuda.is_available(): model.to("cuda")

# --- 4. 运行 Baseline (Phase 1) ---
print("\n[Phase 1] Running BASELINE Model...")
with torch.no_grad():
    baseline_pred = model.predict(data_list, batch_size=CONFIG['batch_size'], gpus=1 if torch.cuda.is_available() else 0)
    baseline_scores = baseline_pred.scores

# --- 5. 挂载 Hook (Phase 2) ---
all_layers = model.encoder.model.encoder.layer
num_total_layers = len(all_layers)

if CONFIG["layers_to_patch"] == "all":
    target_indices = list(range(num_total_layers))
elif CONFIG["layers_to_patch"] == "last":
    target_indices = [num_total_layers - 1]
elif isinstance(CONFIG["layers_to_patch"], list):
    target_indices = CONFIG["layers_to_patch"]
else:
    raise ValueError("Invalid layers_to_patch setting!")

print(f"\n[Phase 2] Running MANIPULATION on Layers: {target_indices}")
# 【打印改动】：显示所有干预目标
print(f"Target Neurons: {CONFIG['target_neurons']}, Multiplier: {CONFIG['multiplier']}")

handles = []
for idx in target_indices:
    # 传入神经元列表
    hook_fn = get_manipulation_hook(CONFIG['target_neurons'], CONFIG['multiplier'], idx)
    handle = all_layers[idx].register_forward_hook(hook_fn)
    handles.append(handle)

# 执行干预推理
with torch.no_grad():
    manipulated_pred = model.predict(data_list, batch_size=CONFIG['batch_size'], gpus=1 if torch.cuda.is_available() else 0)
    manipulated_scores = manipulated_pred.scores

# 拆除钩子
for h in handles:
    h.remove()

# --- 6. 结果对比 ---
print("\n" + "="*50)
print(f" MULTI-NEURON INTERVENTION SUMMARY")
print("-" * 50)
print(f"Neurons Targeted: {CONFIG['target_neurons']}")
print(f"Multiplier:       {CONFIG['multiplier']}")
print(f"Layers Patched:   {len(target_indices)} layers")
print("-" * 50)
avg_base = np.mean(baseline_scores)
avg_mani = np.mean(manipulated_scores)
print(f"Avg Baseline:    {avg_base:.4f}")
print(f"Avg Manipulated: {avg_mani:.4f}")
print(f"Score Shift:     {avg_mani - avg_base:+.4f}")
print("="*50)