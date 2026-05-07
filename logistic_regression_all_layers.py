import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.dummy import DummyClassifier
import os
import warnings

# 屏蔽无关警告
warnings.filterwarnings("ignore")

# ==============================================================================
# [CONFIG] 实验控制面板 (直接在这里修改参数即可)
# ==============================================================================
PENALTY = 'l1'         # 可选: 'l1', 'l2', 'none'
C_VALUE = 0.1            # 正则化强度 (C越小，惩罚越重。'none'模式下此参数无效)
TOP_K = 5                # 每层你想记录前多少名的神经元？(建议 5 或 10)
INPUT_FILE = "extract_all_layers_output/experiment_esa1_all_layers.pt"
OUTPUT_CSV = f"layer_wise_probing_{PENALTY}_C{C_VALUE}_top{TOP_K}.csv"
# ==============================================================================

def main():
    print(f"Loading data with mmap: {INPUT_FILE}...")
    data = torch.load(INPUT_FILE, mmap=True)
    X_all = data["X"]  # [Tokens, 37, 2560]
    y = data["y"].numpy()

    num_layers = X_all.shape[1]
    probing_records = []

    print(f"\nStarting {PENALTY.upper()} probing across {num_layers} layers...")
    print(f"Will extract Top {TOP_K} neurons for each layer.\n")

    # ==============================================================================
    # Chance Rate：全预测1（全部标为error）时的 baseline F1
    # 只依赖标签分布 y，与层无关，算一次即可
    # ==============================================================================
    chance_f1 = f1_score(y, np.ones_like(y), zero_division=0)
    print(f"Chance Rate (all_ones baseline): F1(class=1) = {chance_f1:.4f}")
    print(f"Label distribution: class=0: {(y==0).sum()} | class=1: {(y==1).sum()}")
    print("=" * 80 + "\n")
    # ==============================================================================

    # 配置 Logistic Regression 的参数
    if PENALTY == 'none':
        model_params = {
            'penalty': None,
            'solver': 'lbfgs',
            'max_iter': 2000,
            'class_weight': 'balanced',
            'n_jobs': -1,
            'random_state': 42
        }
    elif PENALTY == 'l2':
        model_params = {
            'penalty': 'l2',
            'C': C_VALUE,
            'solver': 'lbfgs',
            'max_iter': 2000,
            'class_weight': 'balanced',
            'n_jobs': -1,
            'random_state': 42
        }
    else:  # l1
        model_params = {
            'penalty': 'l1',
            'C': C_VALUE,
            'solver': 'liblinear',
            'max_iter': 2000,
            'class_weight': 'balanced',
            'random_state': 42
        }

    for i in range(num_layers):
        # 1. 提取当前层并转换为 float32 numpy
        X_layer = X_all[:, i, :].float().numpy()

        # 2. 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_layer)

        # 3. 训练模型
        clf = LogisticRegression(**model_params)
        clf.fit(X_scaled, y)

        # 4. 评估性能
        y_pred = clf.predict(X_scaled)

        # 正类（error，class=1）指标
        f1_1    = f1_score(y, y_pred, pos_label=1, zero_division=0)
        prec_1  = precision_score(y, y_pred, pos_label=1, zero_division=0)
        rec_1   = recall_score(y, y_pred, pos_label=1, zero_division=0)

        # 负类（correct，class=0）指标
        f1_0    = f1_score(y, y_pred, pos_label=0, zero_division=0)
        prec_0  = precision_score(y, y_pred, pos_label=0, zero_division=0)
        rec_0   = recall_score(y, y_pred, pos_label=0, zero_division=0)

        # Macro F1（两类平均）
        f1_macro = f1_score(y, y_pred, average='macro', zero_division=0)

        delta_f1 = f1_1 - chance_f1

        # 5. 权重深度分析
        weights = clf.coef_[0]
        active_neurons = np.sum(np.abs(weights) > 1e-5)
        top_k_indices = np.argsort(np.abs(weights))[-TOP_K:][::-1]

        # 6. 记录数据
        record = {
            "layer":          i,
            # 正类指标
            "f1_class1":      round(f1_1, 4),
            "prec_class1":    round(prec_1, 4),
            "rec_class1":     round(rec_1, 4),
            # 负类指标
            "f1_class0":      round(f1_0, 4),
            "prec_class0":    round(prec_0, 4),
            "rec_class0":     round(rec_0, 4),
            # 整体指标
            "f1_macro":       round(f1_macro, 4),
            # Chance rate
            "chance_f1":      round(chance_f1, 4),
            "delta_f1":       round(delta_f1, 4),
            # 稀疏度
            "active_neurons": active_neurons,
            "sparsity_pct":   round((1 - active_neurons / 2560) * 100, 2)
        }

        # 7. Top K 神经元
        top_str_list = []
        for rank, idx in enumerate(top_k_indices, 1):
            w_val = weights[idx]
            role  = "Detector(+)" if w_val > 0 else "Sentinel(-)"
            record[f"top{rank}_id"]     = idx
            record[f"top{rank}_weight"] = round(w_val, 4)
            record[f"top{rank}_role"]   = role
            if rank <= 3:
                top_str_list.append(f"#{idx}({w_val:+.2f})")

        probing_records.append(record)

        # 8. 终端打印（格式清晰分行）
        top_str = ", ".join(top_str_list)
        print(f"L{i:02d} | "
              f"F1_macro: {f1_macro:.4f} | "
              f"[class=1] F1: {f1_1:.4f}  Prec: {prec_1:.4f}  Rec: {rec_1:.4f} | "
              f"[class=0] F1: {f1_0:.4f}  Prec: {prec_0:.4f}  Rec: {rec_0:.4f} | "
              f"Delta: {delta_f1:+.4f} | "
              f"Active: {active_neurons:4d}/2560 | "
              f"Top3: {top_str}")

    # ==========================================================================
    # 保存结果到 CSV
    # ==========================================================================
    df = pd.DataFrame(probing_records)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 60)
    print("DETAILED PROBING COMPLETE!")
    print(f"详细结果已保存至: {OUTPUT_CSV}")
    print("=" * 60)

if __name__ == "__main__":
    main()