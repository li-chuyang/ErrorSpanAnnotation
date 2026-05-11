import os
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

# ==============================================================================
# [CONFIG] 多标签层级回归控制面板
# ==============================================================================
REGRESSOR = "linear"        # 可选: 'linear', 'ridge', 'lasso', 'elasticnet'
ALPHA = 10.0               # ridge/lasso/elasticnet 强度
L1_RATIO = 0.5             # 仅 elasticnet 使用
TOP_K = 5                  # 每层记录前多少个神经元
USE_TEST_SPLIT = False     # True = holdout 评估；False = 全量拟合并在全量数据上评估
TEST_SIZE = 0.2            # holdout 测试集比例，仅 USE_TEST_SPLIT=True 时生效
RANDOM_STATE = 42
INPUT_FILE = "extract_multilabel_output/experiment_esa1_all_layers_multilabel.pt"
OUTPUT_DIR = "multilabel_regression_output"
OUTPUT_CSV = (
    f"layer_wise_regression_{REGRESSOR}_alpha{ALPHA}_top{TOP_K}"
    f"{'_split' if USE_TEST_SPLIT else '_nosplit'}.csv"
)
# ==============================================================================


def build_regressor():
    if REGRESSOR == "linear":
        return LinearRegression()
    if REGRESSOR == "ridge":
        return Ridge(alpha=ALPHA, random_state=RANDOM_STATE)
    if REGRESSOR == "lasso":
        return Lasso(alpha=ALPHA, max_iter=5000, random_state=RANDOM_STATE)
    if REGRESSOR == "elasticnet":
        return ElasticNet(
            alpha=ALPHA,
            l1_ratio=L1_RATIO,
            max_iter=5000,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unsupported REGRESSOR: {REGRESSOR}")


def safe_corr(fn, y_true, y_pred):
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    value, _ = fn(y_true, y_pred)
    if np.isnan(value):
        return 0.0
    return float(value)


def build_constant_baselines(y_true, label_ids, mean_value):
    baselines = {}

    mean_pred = np.full_like(y_true, fill_value=mean_value, dtype=np.float32)
    baselines["mean_label"] = {
        "prediction_value": float(mean_value),
        "mae": mean_absolute_error(y_true, mean_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, mean_pred)),
    }

    y_true_int = y_true.astype(int)
    for label_id in label_ids:
        constant_pred = np.full_like(y_true, fill_value=float(label_id), dtype=np.float32)
        constant_pred_int = np.full_like(y_true_int, fill_value=int(label_id))
        baselines[f"always_{int(label_id)}"] = {
            "prediction_value": float(label_id),
            "mae": mean_absolute_error(y_true, constant_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, constant_pred)),
            "macro_f1": f1_score(y_true_int, constant_pred_int, average="macro", zero_division=0),
            "accuracy": float(np.mean(y_true_int == constant_pred_int)),
        }

    return baselines


def main():
    print(f"Loading data with mmap: {INPUT_FILE}...")
    data = torch.load(INPUT_FILE, mmap=True)
    x_all = data["X"]
    y = data["y"].numpy().astype(np.float32)
    label_map = data.get("label_map", {})

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV)

    num_layers = x_all.shape[1]
    probing_records = []

    unique_labels, counts = np.unique(y.astype(int), return_counts=True)
    label_summary = ", ".join(
        f"{label_map.get(int(label), int(label))}:{count}" for label, count in zip(unique_labels, counts)
    )

    print(f"\nStarting {REGRESSOR.upper()} multilabel regression across {num_layers} layers...")
    print(f"Input tensor shape: {tuple(x_all.shape)}")
    print(f"Observed labels: {label_summary}")
    print(f"Split mode: {'train/test holdout' if USE_TEST_SPLIT else 'no split (full-data probing)'}")
    print(f"Will extract Top {TOP_K} neurons for each layer.\n")

    print("Chance-rate baselines that will be reported:")
    print(f"- Mean label baseline: predict mean(y)")
    for label_id in unique_labels:
        print(f"- Constant baseline: always predict {int(label_id)}")
    print("=" * 90 + "\n")

    for layer_idx in range(num_layers):
        x_layer = x_all[:, layer_idx, :].float().numpy()

        if USE_TEST_SPLIT:
            x_train, x_eval, y_train, y_eval = train_test_split(
                x_layer,
                y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y.astype(int),
            )

            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            x_eval_scaled = scaler.transform(x_eval)

            reg = build_regressor()
            reg.fit(x_train_scaled, y_train)

            y_train_pred = reg.predict(x_train_scaled)
            y_eval_pred = reg.predict(x_eval_scaled)

            baseline_mean = float(np.mean(y_train))
            eval_name = "test"
        else:
            scaler = StandardScaler()
            x_eval_scaled = scaler.fit_transform(x_layer)

            reg = build_regressor()
            reg.fit(x_eval_scaled, y)

            x_train = x_layer
            y_train = y
            y_train_pred = reg.predict(x_eval_scaled)
            y_eval = y
            y_eval_pred = y_train_pred
            baseline_mean = float(np.mean(y))
            eval_name = "all"

        y_eval_pred_rounded = np.rint(y_eval_pred)
        y_eval_pred_rounded = np.clip(y_eval_pred_rounded, y.min(), y.max()).astype(int)
        y_eval_int = y_eval.astype(int)

        chance_baselines = build_constant_baselines(
            y_true=y_eval,
            label_ids=unique_labels,
            mean_value=baseline_mean,
        )
        baseline_mae = chance_baselines["mean_label"]["mae"]
        baseline_rmse = chance_baselines["mean_label"]["rmse"]

        mae = mean_absolute_error(y_eval, y_eval_pred)
        rmse = np.sqrt(mean_squared_error(y_eval, y_eval_pred))
        r2 = r2_score(y_eval, y_eval_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        pearson = safe_corr(pearsonr, y_eval, y_eval_pred)
        spearman = safe_corr(spearmanr, y_eval, y_eval_pred)
        rounded_acc = float(np.mean(y_eval_pred_rounded == y_eval_int))
        rounded_macro_f1 = f1_score(y_eval_int, y_eval_pred_rounded, average="macro", zero_division=0)
        mae_gain = baseline_mae - mae
        rmse_gain = baseline_rmse - rmse

        weights = np.ravel(reg.coef_)
        active_neurons = int(np.sum(np.abs(weights) > 1e-5))
        top_k_indices = np.argsort(np.abs(weights))[-TOP_K:][::-1]

        record = {
            "layer": layer_idx,
            "split_mode": "holdout" if USE_TEST_SPLIT else "none",
            "evaluation_split": eval_name,
            "num_train_tokens": len(y_train),
            "num_eval_tokens": len(y_eval),
            "train_r2": round(train_r2, 4),
            "eval_r2": round(r2, 4),
            "eval_mae": round(mae, 4),
            "eval_rmse": round(rmse, 4),
            "eval_pearson": round(pearson, 4),
            "eval_spearman": round(spearman, 4),
            "eval_rounded_accuracy": round(rounded_acc, 4),
            "eval_rounded_macro_f1": round(rounded_macro_f1, 4),
            "baseline_mean_label": round(baseline_mean, 4),
            "chance_mean_label_eval_mae": round(baseline_mae, 4),
            "chance_mean_label_eval_rmse": round(baseline_rmse, 4),
            "mae_improvement_over_mean_label_baseline": round(mae_gain, 4),
            "rmse_improvement_over_mean_label_baseline": round(rmse_gain, 4),
            "active_neurons": active_neurons,
            "sparsity_percent": round((1 - active_neurons / 2560) * 100, 2),
        }

        for label_id in unique_labels:
            baseline = chance_baselines[f"always_{int(label_id)}"]
            record[f"chance_always_{int(label_id)}_eval_mae"] = round(baseline["mae"], 4)
            record[f"chance_always_{int(label_id)}_eval_rmse"] = round(baseline["rmse"], 4)
            record[f"chance_always_{int(label_id)}_eval_accuracy"] = round(baseline["accuracy"], 4)
            record[f"chance_always_{int(label_id)}_eval_macro_f1"] = round(baseline["macro_f1"], 4)
            record[f"accuracy_improvement_over_always_{int(label_id)}"] = round(
                rounded_acc - baseline["accuracy"], 4
            )
            record[f"macro_f1_improvement_over_always_{int(label_id)}"] = round(
                rounded_macro_f1 - baseline["macro_f1"], 4
            )

        top_str_list = []
        for rank, neuron_idx in enumerate(top_k_indices, 1):
            weight = float(weights[neuron_idx])
            role = "Severity(+)" if weight > 0 else "SeverityBuffer(-)"
            record[f"top{rank}_id"] = int(neuron_idx)
            record[f"top{rank}_weight"] = round(weight, 4)
            record[f"top{rank}_role"] = role
            if rank <= 3:
                top_str_list.append(f"#{neuron_idx}({weight:+.3f})")

        probing_records.append(record)

        top_str = ", ".join(top_str_list)
        print(
            f"L{layer_idx:02d} | "
            f"R2({eval_name}): {r2:.4f} | "
            f"MAE: {mae:.4f} | "
            f"RMSE: {rmse:.4f} | "
            f"Pearson: {pearson:.4f} | "
            f"Spearman: {spearman:.4f} | "
            f"RoundAcc: {rounded_acc:.4f} | "
            f"RoundF1: {rounded_macro_f1:.4f} | "
            f"vs Mean MAE: {mae_gain:+.4f} | "
            f"vs All-0 F1: {rounded_macro_f1 - chance_baselines[f'always_{int(unique_labels[0])}']['macro_f1']:+.4f} | "
            f"Active: {active_neurons:4d}/2560 | "
            f"Top3: {top_str}"
        )

    df = pd.DataFrame(probing_records)
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("MULTILABEL REGRESSION COMPLETE!")
    print(f"Detailed results saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
