import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.dummy import DummyClassifier

# 1. 加载数据
data = torch.load("experiment_data.pt")
X = data["X"].numpy()
y = data["y"].numpy()

# 2. 特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==================== 实验参数配置 ====================
# 你可以在这里手动指定参数
PENALTY = 'l2'    # 可选: 'l1', 'l2', 'none'
C_VALUE = 0.01     # 正则化强度 (C越小，惩罚越重)
# ======================================================

# ======================================================
# 实验 1: 探测模型 (带参数化正则化)
# ======================================================
if PENALTY == 'none':
    # 无正则化模式
    clf = LogisticRegression(
        penalty=None,
        max_iter=2000, 
        class_weight='balanced', 
        solver='lbfgs', # none 模式通常配合 lbfgs
        random_state=42
    )
else:
    # L1 或 L2 正则化模式
    clf = LogisticRegression(
        penalty=PENALTY,
        C=C_VALUE,
        max_iter=2000, 
        class_weight='balanced', 
        solver='liblinear', # liblinear 支持 l1 和 l2
        random_state=42
    )

clf.fit(X_scaled, y)
y_pred = clf.predict(X_scaled)

print("\n" + "="*25 + f" 1. Probing Model (Penalty={PENALTY}, C={C_VALUE}) " + "="*25)
print(classification_report(y, y_pred, digits=4))

# 新增：统计当前有多少神经元还在干活（权重不为0）
weights = clf.coef_[0]
active_count = np.sum(np.abs(weights) > 1e-5)
print(f"Active Neurons (Non-zero): {active_count} / {len(weights)} (精简了 {100 - active_count/len(weights)*100:.2f}%)")


# ======================================================
# 实验 2: Chance Rate - 全猜出现频率最高的类 (全猜 0)
# ======================================================
dummy_major = DummyClassifier(strategy='most_frequent')
dummy_major.fit(X_scaled, y)
y_major = dummy_major.predict(X_scaled)

print("\n" + "="*25 + " 2. Chance Rate (Most Frequent) " + "="*25)
print(classification_report(y, y_major, digits=4, zero_division=0))


# ======================================================
# 实验 3: Chance Rate - 按比例随机猜测 (Stratified)
# ======================================================
dummy_rand = DummyClassifier(strategy='stratified', random_state=42)
dummy_rand.fit(X_scaled, y)
y_rand = dummy_rand.predict(X_scaled)

print("\n" + "="*25 + " 3. Chance Rate (Random Stratified) " + "="*25)
print(classification_report(y, y_rand, digits=4))


# ======================================================
# 实验 4: 重要神经元 (Neuron Weights)
# ======================================================
# 即使加了正则化，我们也看绝对值最大的前10个
top_indices = np.argsort(np.abs(weights))[-10:][::-1]

print("\n" + "="*25 + " Top 10 Neurons " + "="*25)
for idx in top_indices:
    # 增加一个标记，如果该神经元被正则化削成了0，会显示 [ZEROED]
    status = "" if np.abs(weights[idx]) > 1e-5 else " [ZEROED]"
    print(f"Neuron #{idx:4} | Weight: {weights[idx]:8.4f}{status}")