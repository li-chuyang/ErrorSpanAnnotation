import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_manipulation(csv_path):
    if not os.path.exists(csv_path):
        print(f"错误: 找不到文件 {csv_path}")
        return

    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 设置画布：比例建议 10:6
    plt.figure(figsize=(10, 7))
    
    # 按照神经元组合分组
    groups = df['neuron_set'].unique()
    
    # 设置一些好看的颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, group in enumerate(groups):
        # 提取该组数据并按倍率排序
        subset = df[df['neuron_set'] == group].sort_values('multiplier')
        
        # 将 "309+490+502" 这种格式美化一下显示在图例里
        label_name = f"Group: {group}"
        
        plt.plot(subset['multiplier'], subset['shift'], 
                 marker='o', markersize=8, linestyle='-', 
                 linewidth=2.5, label=label_name, color=colors[i % len(colors)])

    # --- 核心视觉优化 ---
    
    # 1. 绘制 y=0 的基准线（代表正常模型）
    plt.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.6)
    
    # 2. 绘制 x=1.0 的垂直线（代表原始状态）
    plt.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
    
    # 3. 添加文字说明（告诉读者上面是分数上升，下面是下降）
    plt.text(df['multiplier'].min(), 0.005, "Score Improves (Model becomes 'Blinder')", 
             fontsize=10, color='green', fontweight='bold', alpha=0.7)
    plt.text(df['multiplier'].min(), -0.015, "Score Drops (Model becomes 'Harsher')", 
             fontsize=10, color='red', fontweight='bold', alpha=0.7)

    # 4. 细节微调
    plt.title('Causal Circuit Intervention: Multi-Neuron Synergy', fontsize=16, pad=20)
    plt.xlabel('Multiplier (x Baseline Activation)', fontsize=13)
    plt.ylabel('Average Quality Score Shift (Δ Score)', fontsize=13)
    plt.legend(loc='best', fontsize=11, frameon=True)
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    
    # 5. 保存
    save_path = csv_path.replace(".csv", "_v2.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"实验趋势图已生成: {save_path}")

if __name__ == "__main__":
    # 确保这里的路径和你 .sh 脚本里定义的一致
    plot_manipulation("manipulation_result/results_ESA-1_XCOMET-XL.csv")