import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
import os
import warnings

# 屏蔽无关警告
warnings.filterwarnings("ignore")

# ==============================================================================
# [CONFIG] 实验控制面板 (直接在这里修改参数即可)
# ==============================================================================
PENALTY = 'l2'         # 可选: 'l1', 'l2', 'none'
C_VALUE = 0.1            # 正则化强度 (C越小，惩罚越重。'none'模式下此参数无效)
TOP_K = 5                # 每层你想记录前多少名的神经元？(建议 5 或 10)
INPUT_FILE = "experiment_data_output/experiment_esa1_all_layers.pt"
OUTPUT_CSV = f"layer_wise_probing_{PENALTY}_C{C_VALUE}_top{TOP_K}.csv"
# ==============================================================================

def main():
    print(f"Loading data with mmap: {INPUT_FILE}...")
    # 使用 mmap 保证大文件秒开且不爆内存
    data = torch.load(INPUT_FILE, mmap=True)
    X_all = data["X"]  #[Tokens, 37, 2560]
    y = data["y"].numpy()

    num_layers = X_all.shape[1]
    probing_records =[]

    print(f"\nStarting {PENALTY.upper()} probing across {num_layers} layers...")
    print(f"Will extract Top {TOP_K} neurons for each layer.\n")

    # 配置 Logistic Regression 的参数
    if PENALTY == 'none':
        model_params = {
            'penalty': None,
            'solver': 'lbfgs',  # 极快
            'max_iter': 2000,
            'class_weight': 'balanced',
            'n_jobs': -1,       # 【新增】使用所有 CPU 核心并行计算，起飞！
            'random_state': 42
        }
    elif PENALTY == 'l2':
        model_params = {
            'penalty': 'l2',
            'C': C_VALUE,
            'solver': 'lbfgs',  # 【关键】L2 也要用 lbfgs，拒绝 liblinear！
            'max_iter': 2000,
            'class_weight': 'balanced',
            'n_jobs': -1,       # 【新增】利用多核加速
            'random_state': 42
        }
    else: # 这是给 L1 留的
        model_params = {
            'penalty': 'l1',
            'C': C_VALUE,
            'solver': 'liblinear', # L1 只能用这个，确实会比 L2 慢一些
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
        f1 = f1_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        
        # 5. 权重深度分析
        weights = clf.coef_[0]
        # 计算当前层有多少神经元被“激活”（权重不为 0，用于 L1 分析）
        active_neurons = np.sum(np.abs(weights) > 1e-5)
        
        # 获取绝对值最大的前 TOP_K 个神经元的索引
        top_k_indices = np.argsort(np.abs(weights))[-TOP_K:][::-1]
        
        # 记录基础数据
        record = {
            "layer": i,
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "active_neurons": active_neurons,
            "sparsity_pct": round((1 - active_neurons/2560)*100, 2)
        }
        
        # 动态将 Top K 神经元的详细信息写入字典
        top_str_list =[] # 用于终端打印
        for rank, idx in enumerate(top_k_indices, 1):
            w_val = weights[idx]
            role = "Detector(+)" if w_val > 0 else "Sentinel(-)"
            
            # 存入字典
            record[f"top{rank}_id"] = idx
            record[f"top{rank}_weight"] = round(w_val, 4)
            record[f"top{rank}_role"] = role
            
            # 拼接用于打印的字符串 (前3名)
            if rank <= 3:
                top_str_list.append(f"#{idx}({w_val:+.2f})")
                
        probing_records.append(record)
        
        # 在终端打印当前层结果，展示前3名的简要信息
        top_str = ", ".join(top_str_list)
        print(f"L{i:02d} | F1: {f1:.4f} | Active: {active_neurons:4d}/2560 | Top3: {top_str}")
        

    # ==========================================================================
    # 保存结果到 CSV
    # ==========================================================================
    df = pd.DataFrame(probing_records)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print("\n" + "="*60)
    print("DETAILED PROBING COMPLETE!")
    print(f"详细结果已保存至: {OUTPUT_CSV}")
    print("="*60)

if __name__ == "__main__":
    main()