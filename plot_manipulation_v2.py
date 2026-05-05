import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_manipulation(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # 按照神经元组合分组画线
    plt.figure(figsize=(10, 6))
    
    # 获取唯一的神经元组合
    groups = df['neuron_set'].unique()
    
    for group in groups:
        subset = df[df['neuron_set'] == group].sort_values('multiplier')
        plt.plot(subset['multiplier'], subset['shift'], marker='o', label=f'Neurons: {group}')

    plt.axhline(0, color='black', linestyle='--', alpha=0.3) # 零基准线
    plt.title('Causal Intervention: Multiplier vs. Score Shift', fontsize=14)
    plt.xlabel('Multiplier (0.0=Mask, 1.0=Original)', fontsize=12)
    plt.ylabel('Score Shift (Manipulated - Baseline)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = csv_path.replace(".csv", ".png")
    plt.savefig(save_path, dpi=300)
    print(f"图表已保存至: {save_path}")

if __name__ == "__main__":
    # 默认读取 ESA-1 的结果
    plot_manipulation("manipulation_result/results_ESA-1_XCOMET-XL.csv")