import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

# 使用 mmap 加速加载
data = torch.load("experiment_data_output/experiment_esa1_all_layers.pt", mmap=True)
tokens = data["tokens"]
y = data["y"].tolist()

print(f"总 Token 数: {len(tokens)}")
print(f"前 20 个 Token 详情:")
print(f"{'Index':<5} | {'Token':<15} | {'Label':<5}")
print("-" * 35)

for i in range(20):
    label_str = "ERROR" if y[i] == 1 else "OK"
    print(f"{i:<5} | {tokens[i]:<15} | {label_str}")


y = data["y"].numpy()

# 2. 【核心动作】：只切出最后一层 (Index 36)
# 形状从 [73681, 37, 2560] 变成 [73681, 2560]
X_last = data["X"][:, 36, :].float().numpy() 

# 3. 跑一遍你最熟悉的回归
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_last)
clf = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
clf.fit(X_scaled, y)

# 4. 检查结果
y_pred = clf.predict(X_scaled)
f1 = f1_score(y, y_pred)
weights = clf.coef_[0]
top_neuron = np.argsort(np.abs(weights))[-1]

print(f"验证结果：")
print(f"-> 第 36 层 F1 分数: {f1:.4f} (预期应在 0.58-0.59 左右)")
print(f"-> 权重第一的神经元: #{top_neuron} (预期应为 #309)")