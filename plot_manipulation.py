import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="intervention_results.csv")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    plt.figure(figsize=(10, 6))
    
    # 按照神经元分组画线
    for neuron_id, group in df.groupby('neuron'):
        group = group.sort_values('multiplier')
        plt.plot(group['multiplier'], group['shift'], marker='o', linestyle='-', linewidth=2, label=f'Neuron #{neuron_id}')

    # 绘制参考线
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.axvline(1, color='red', linestyle=':', label='Baseline (1.0x)')

    plt.title('Causal Intervention: Multiplier vs. Quality Score Shift', fontsize=14)
    plt.xlabel('Multiplier (Activation * x)', fontsize=12)
    plt.ylabel('Average Score Shift (Δ Score)', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    output_png = "intervention_curve.png"
    plt.savefig(output_png, dpi=300)
    print(f"Plot saved to {output_png}")

if __name__ == "__main__":
    main()