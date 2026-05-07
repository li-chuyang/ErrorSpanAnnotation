import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

def plot_manipulation(csv_path):
    if not os.path.exists(csv_path):
        print(f"错误: 找不到文件 {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # 按 layers 分组分子图（all / last 分开画）
    layer_conditions = df['layers'].unique()
    n_plots = len(layer_conditions)

    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 6), squeeze=False)
    fig.patch.set_facecolor('#f8f9fa')

    colors = [
        '#2196F3', '#FF5722', '#4CAF50', '#9C27B0',
        '#FF9800', '#00BCD4', '#E91E63', '#795548'
    ]
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']

    groups = df['neuron_set'].unique()

    for col, layer_cond in enumerate(layer_conditions):
        ax = axes[0][col]
        ax.set_facecolor('#ffffff')

        subset_layer = df[df['layers'] == layer_cond]

        for i, group in enumerate(groups):
            subset = subset_layer[subset_layer['neuron_set'] == group].sort_values('multiplier')
            if subset.empty:
                continue

            # 标签格式：#309, #490+502 这种
            label = '#' + group.replace('+', ' + #')

            ax.plot(
                subset['multiplier'], subset['shift'],
                marker=markers[i % len(markers)],
                markersize=8,
                linestyle='-',
                linewidth=2,
                label=label,
                color=colors[i % len(colors)],
                zorder=3
            )

            # 在每个数据点旁标注 shift 数值
            for _, row in subset.iterrows():
                ax.annotate(
                    f"{row['shift']:+.3f}",
                    xy=(row['multiplier'], row['shift']),
                    xytext=(0, 9),
                    textcoords='offset points',
                    ha='center', fontsize=7.5, color=colors[i % len(colors)],
                    zorder=4
                )

        # 基准线
        ax.axhline(0, color='#333333', linestyle='-', linewidth=1.2, alpha=0.7, zorder=2)
        ax.axvline(1.0, color='#888888', linestyle='--', linewidth=1, alpha=0.6, zorder=2)

        # 着色区域：上方绿 / 下方红
        y_min, y_max = ax.get_ylim()
        ax.axhspan(0, max(y_max, 0.01), alpha=0.04, color='green', zorder=1)
        ax.axhspan(min(y_min, -0.01), 0, alpha=0.04, color='red', zorder=1)

        # 轴标签与标题
        ax.set_title(
            f'Intervention Results — Layers: {layer_cond}',
            fontsize=14, fontweight='bold', pad=14
        )
        ax.set_xlabel('Multiplier (× Baseline Activation)', fontsize=12)
        ax.set_ylabel('Average Score Shift (Δ Score)', fontsize=12)

        # 图例放在右侧外部，不遮挡曲线
        ax.legend(
            loc='upper left',
            bbox_to_anchor=(1.01, 1),
            borderaxespad=0,
            fontsize=10,
            frameon=True,
            framealpha=0.9,
            title='Neuron Set',
            title_fontsize=10
        )

        ax.grid(True, linestyle='--', alpha=0.4, zorder=0)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(axis='both', labelsize=10)

        # x=1.0 标注
        ax.text(1.0, ax.get_ylim()[0], 'baseline\n(×1.0)',
                ha='center', va='bottom', fontsize=8.5,
                color='#666666', style='italic')

    plt.suptitle(
        'Causal Neuron Intervention in xCOMET-XL',
        fontsize=16, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    save_path = csv_path.replace(".csv", "_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图已保存: {save_path}")
    plt.show()


if __name__ == "__main__":
    plot_manipulation("manipulation_result/results_ESA-1_XCOMET-XL.csv")