import torch
import numpy as np
import pandas as pd
from comet import download_model, load_from_checkpoint
from ESA.annotation_loader import AnnotationLoader
import argparse
import warnings
import os
import csv

# 屏蔽无关警告
warnings.filterwarnings("ignore")
os.environ["LIT_LOGGER_LOG_LEVEL"] = "ERROR"

def get_args():
    parser = argparse.ArgumentParser(description="Multi-Neuron Manipulation Script")
    parser.add_argument("--neurons", type=str, required=True, help="Comma-separated list of neurons, e.g., 309,490,819")
    parser.add_argument("--multiplier", type=float, default=0.0, help="Manipulation multiplier")
    parser.add_argument("--layers", type=str, default="all", help="all, last, or comma-separated layers")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to test")
    parser.add_argument("--dataset", type=str, default="ESA-1", help="Dataset name")
    parser.add_argument("--model", type=str, default="Unbabel/XCOMET-XL", help="Model name")
    # 结果保存路径
    parser.add_argument("--res_dir", type=str, default="manipulation_result")
    return parser.parse_args()

def get_manipulation_hook(neuron_indices, multiplier):
    def hook_fn(module, inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        hidden_states[:, :, neuron_indices] = hidden_states[:, :, neuron_indices] * multiplier
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states
    return hook_fn

def main():
    args = get_args()
    target_neurons = [int(n) for n in args.neurons.split(",")]
    
    # 1. 确保结果文件夹存在
    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)

    # 2. 加载数据
    loader = AnnotationLoader(refresh_cache=False)
    df = loader.get_view([args.dataset], only_overlap=False)
    error_df = df[df[f"{args.dataset}_error_spans"].notna()].copy()
    test_df = error_df.head(args.samples)
    data_list = [{"src": row['source'], "mt": row['hypothesis'], "ref": ""} for _, row in test_df.iterrows()]

    # 3. 加载模型
    model_path = download_model(args.model)
    model = load_from_checkpoint(model_path)
    model.eval()
    if torch.cuda.is_available(): model.to("cuda")

    # 4. Phase 1: Baseline
    with torch.no_grad():
        baseline_pred = model.predict(data_list, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)
        baseline_scores = baseline_pred.scores

    # 5. Phase 2: Manipulation
    all_layers = model.encoder.model.encoder.layer
    if args.layers == "all":
        target_indices = list(range(len(all_layers)))
    elif args.layers == "last":
        target_indices = [len(all_layers) - 1]
    else:
        target_indices = [int(l) for l in args.layers.split(",")]

    handles = []
    hook_fn = get_manipulation_hook(target_neurons, args.multiplier)
    for idx in target_indices:
        handle = all_layers[idx].register_forward_hook(hook_fn)
        handles.append(handle)

    with torch.no_grad():
        manipulated_pred = model.predict(data_list, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)
        manipulated_scores = manipulated_pred.scores

    for h in handles: h.remove()

    # 6. 计算结果
    avg_base = np.mean(baseline_scores)
    avg_mani = np.mean(manipulated_scores)
    shift = avg_mani - avg_base

    # 7. 保存到 CSV (采用追加模式，方便画图脚本读取)
    # 文件名示例: results_ESA-1_XCOMET-XL.csv
    csv_name = f"results_{args.dataset}_{args.model.split('/')[-1]}.csv"
    csv_path = os.path.join(args.res_dir, csv_name)
    
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            # 写入表头
            writer.writerow(['neuron_set', 'multiplier', 'layers', 'samples', 'baseline', 'manipulated', 'shift'])
        # neuron_set 保存为字符串，如 "309+490" 方便画图分组
        neuron_label = "+".join(map(str, target_neurons))
        writer.writerow([neuron_label, args.multiplier, args.layers, args.samples, 
                         round(avg_base, 4), round(avg_mani, 4), round(shift, 4)])

    print(f"Result Saved to {csv_path}: Neurons={target_neurons}, Multiplier={args.multiplier}, Shift={shift:+.4f}")

if __name__ == "__main__":
    main()