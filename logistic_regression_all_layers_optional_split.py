import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

# ==============================================================================
# [CONFIG] 实验控制面板
# 默认保持旧脚本逻辑：不划分训练/测试，直接全量拟合并在全量数据上评估
# 如果你想切换到 holdout 评估，只需要把 USE_TEST_SPLIT 改成 True
# ==============================================================================
PENALTY = "l1"              # 可选: 'l1', 'l2', 'none'
C_VALUE = 0.1               # 正则化强度
TOP_K = 5                   # 每层记录前多少名神经元
INPUT_FILE = "extract_all_layers_output/experiment_esa1_all_layers.pt"
OUTPUT_DIR = "logistic_regression_output"

USE_TEST_SPLIT = False      # False = 完全复现旧逻辑；True = 划分 train/test
TEST_SIZE = 0.2
RANDOM_STATE = 42

OUTPUT_CSV = (
    f"layer_wise_probing_{PENALTY}_C{C_VALUE}_top{TOP_K}"
    f"{'_split' if USE_TEST_SPLIT else '_nosplit'}.csv"
)
# ==============================================================================


def build_model_params():
    if PENALTY == "none":
        return {
            "penalty": None,
            "solver": "lbfgs",
            "max_iter": 2000,
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
        }
    if PENALTY == "l2":
        return {
            "penalty": "l2",
            "C": C_VALUE,
            "solver": "lbfgs",
            "max_iter": 2000,
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
        }
    return {
        "penalty": "l1",
        "C": C_VALUE,
        "solver": "liblinear",
        "max_iter": 2000,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    }


def compute_binary_metrics(y_true, y_pred):
    return {
        "f1_class1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "prec_class1": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "rec_class1": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_class0": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "prec_class0": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "rec_class0": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def main():
    print(f"Loading data with mmap: {INPUT_FILE}...")
    data = torch.load(INPUT_FILE, mmap=True)
    x_all = data["X"]  # [Tokens, 37, 2560]
    y = data["y"].numpy()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV)

    num_layers = x_all.shape[1]
    probing_records = []
    model_params = build_model_params()

    print(f"\nStarting {PENALTY.upper()} probing across {num_layers} layers...")
    print(f"Will extract Top {TOP_K} neurons for each layer.")
    print(f"Split mode: {'train/test holdout' if USE_TEST_SPLIT else 'no split (legacy behavior)'}\n")

    for layer_idx in range(num_layers):
        x_layer = x_all[:, layer_idx, :].float().numpy()

        if USE_TEST_SPLIT:
            x_train, x_eval, y_train, y_eval = train_test_split(
                x_layer,
                y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y,
            )

            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            x_eval_scaled = scaler.transform(x_eval)

            clf = LogisticRegression(**model_params)
            clf.fit(x_train_scaled, y_train)

            y_train_pred = clf.predict(x_train_scaled)
            y_eval_pred = clf.predict(x_eval_scaled)

            train_metrics = compute_binary_metrics(y_train, y_train_pred)
            eval_metrics = compute_binary_metrics(y_eval, y_eval_pred)
            chance_f1 = f1_score(y_eval, np.ones_like(y_eval), zero_division=0)
            delta_f1 = eval_metrics["f1_class1"] - chance_f1
            eval_name = "test"
        else:
            scaler = StandardScaler()
            x_eval_scaled = scaler.fit_transform(x_layer)

            clf = LogisticRegression(**model_params)
            clf.fit(x_eval_scaled, y)

            y_eval = y
            y_eval_pred = clf.predict(x_eval_scaled)
            eval_metrics = compute_binary_metrics(y_eval, y_eval_pred)
            train_metrics = eval_metrics
            chance_f1 = f1_score(y_eval, np.ones_like(y_eval), zero_division=0)
            delta_f1 = eval_metrics["f1_class1"] - chance_f1
            eval_name = "all"

        weights = clf.coef_[0]
        active_neurons = int(np.sum(np.abs(weights) > 1e-5))
        top_k_indices = np.argsort(np.abs(weights))[-TOP_K:][::-1]

        record = {
            "layer": layer_idx,
            "split_mode": "holdout" if USE_TEST_SPLIT else "none",
            "evaluation_split": eval_name,
            "num_train_tokens": len(y_train) if USE_TEST_SPLIT else len(y),
            "num_eval_tokens": len(y_eval),
            "baseline_eval_f1_class1": round(chance_f1, 4),
            "f1_class1_improvement_over_baseline": round(delta_f1, 4),
            "active_neurons": active_neurons,
            "sparsity_percent": round((1 - active_neurons / 2560) * 100, 2),
        }

        metric_name_map = {
            "f1_class1": "eval_f1_class1",
            "prec_class1": "eval_precision_class1",
            "rec_class1": "eval_recall_class1",
            "f1_class0": "eval_f1_class0",
            "prec_class0": "eval_precision_class0",
            "rec_class0": "eval_recall_class0",
            "f1_macro": "eval_macro_f1",
        }
        for key, value in eval_metrics.items():
            record[metric_name_map[key]] = round(value, 4)

        if USE_TEST_SPLIT:
            record["train_f1_class1"] = round(train_metrics["f1_class1"], 4)
            record["train_f1_class0"] = round(train_metrics["f1_class0"], 4)
            record["train_macro_f1"] = round(train_metrics["f1_macro"], 4)

        top_str_list = []
        for rank, neuron_idx in enumerate(top_k_indices, 1):
            weight = float(weights[neuron_idx])
            role = "Detector(+)" if weight > 0 else "Sentinel(-)"
            record[f"top{rank}_id"] = int(neuron_idx)
            record[f"top{rank}_weight"] = round(weight, 4)
            record[f"top{rank}_role"] = role
            if rank <= 3:
                top_str_list.append(f"#{neuron_idx}({weight:+.2f})")

        probing_records.append(record)

        top_str = ", ".join(top_str_list)
        if USE_TEST_SPLIT:
            print(
                f"L{layer_idx:02d} | "
                f"Train F1_macro: {train_metrics['f1_macro']:.4f} | "
                f"Test F1_macro: {eval_metrics['f1_macro']:.4f} | "
                f"Test F1(1): {eval_metrics['f1_class1']:.4f} | "
                f"Delta: {delta_f1:+.4f} | "
                f"Active: {active_neurons:4d}/2560 | "
                f"Top3: {top_str}"
            )
        else:
            print(
                f"L{layer_idx:02d} | "
                f"F1_macro: {eval_metrics['f1_macro']:.4f} | "
                f"[class=1] F1: {eval_metrics['f1_class1']:.4f}  "
                f"Prec: {eval_metrics['prec_class1']:.4f}  "
                f"Rec: {eval_metrics['rec_class1']:.4f} | "
                f"[class=0] F1: {eval_metrics['f1_class0']:.4f}  "
                f"Prec: {eval_metrics['prec_class0']:.4f}  "
                f"Rec: {eval_metrics['rec_class0']:.4f} | "
                f"Delta: {delta_f1:+.4f} | "
                f"Active: {active_neurons:4d}/2560 | "
                f"Top3: {top_str}"
            )

    df = pd.DataFrame(probing_records)
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("OPTIONAL-SPLIT PROBING COMPLETE!")
    print(f"Detailed results saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
