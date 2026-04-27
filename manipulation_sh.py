import torch
import numpy as np
import pandas as pd
from comet import download_model, load_from_checkpoint
from ESA.annotation_loader import AnnotationLoader
import argparse
import warnings
import os
import logging
import csv # 用于保存结果

# 放在最前面：彻底关闭干扰提示
warnings.filterwarnings("ignore")
os.environ["LIT_LOGGER_LOG_LEVEL"] = "ERROR"
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)



def get_args():
    parser = argparse.ArgumentParser(description="Neuron Manipulation Test")
    parser.add_argument("--neuron", type=int, default=309, help="Target neuron index")
    parser.add_argument("--multiplier", type=float, default=1.0, help="Multiplier for manipulation")
    parser.add_argument("--layers", type=str, default="all", help="all, last, or comma-separated list")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples to test")
    parser.add_argument("--output_csv", type=str, default="intervention_results.csv", help="CSV file to save results")
    return parser.parse_args()

def get_manipulation_hook(neuron_idx, multiplier):
    def hook_fn(module, inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        # 活体篡改
        hidden_states[:, :, neuron_idx] = hidden_states[:, :, neuron_idx] * multiplier
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states
    return hook_fn

# --- 1. 修改加载数据的逻辑 ---
def main():
    args = get_args()
    
    loader = AnnotationLoader(refresh_cache=False)
    df = loader.get_view(["ESA-1"], only_overlap=False)
    error_df = df[df["ESA-1_error_spans"].notna()].copy()

    # --- 这里是核心修改 ---
    if args.samples == -1:
        # 如果是 -1，取全部数据
        test_df = error_df
        print(f"模式：运行全量数据，总计 {len(test_df)} 条样本")
    else:
        # 否则取前 N 条
        test_df = error_df.head(args.samples)
        print(f"模式：运行部分数据，共 {len(test_df)} 条样本")
    
    # 剩下的代码保持不变...
    data_list = [{"src": row['source'], "mt": row['hypothesis'], "ref": ""} for _, row in test_df.iterrows()]

    # 2. 加载模型
    model_path = download_model("Unbabel/XCOMET-XL")
    model = load_from_checkpoint(model_path)
    model.eval()
    if torch.cuda.is_available(): model.to("cuda")

    # 3. 运行 Baseline
    with torch.no_grad():
        baseline_pred = model.predict(data_list, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)
        baseline_scores = baseline_pred.scores

    # 4. 解析层索引并挂载 Hook
    all_layers = model.encoder.model.encoder.layer
    if args.layers == "all":
        target_indices = list(range(len(all_layers)))
    elif args.layers == "last":
        target_indices = [len(all_layers) - 1]
    else:
        target_indices = [int(x) for x in args.layers.split(",")]

    handles = []
    hook_fn = get_manipulation_hook(args.neuron, args.multiplier)
    for idx in target_indices:
        handle = all_layers[idx].register_forward_hook(hook_fn)
        handles.append(handle)

    # 5. 运行 Intervention
    with torch.no_grad():
        manipulated_pred = model.predict(data_list, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)
        manipulated_scores = manipulated_pred.scores

    for h in handles: h.remove()

    # 6. 计算并保存结果
    avg_base = np.mean(baseline_scores)
    avg_mani = np.mean(manipulated_scores)
    shift = avg_mani - avg_base
    
    # 将结果追加到 CSV 文件
    file_exists = os.path.isfile(args.output_csv)
    with open(args.output_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['neuron', 'multiplier', 'layers', 'avg_baseline', 'avg_manipulated', 'shift'])
        writer.writerow([args.neuron, args.multiplier, args.layers, f"{avg_base:.4f}", f"{avg_mani:.4f}", f"{shift:.4f}"])

    print(f"DONE: Neuron #{args.neuron} | Multiplier {args.multiplier} | Shift {shift:+.4f}")

if __name__ == "__main__":
    main()