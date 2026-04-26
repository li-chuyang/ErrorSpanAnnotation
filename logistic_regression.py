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

# ======================================================
# 实验 1: 正常的线性探测模型 (Logistic Regression)
# ======================================================
clf = LogisticRegression(
    max_iter=2000, 
    class_weight='balanced', 
    solver='liblinear',
    random_state=42
)
clf.fit(X_scaled, y)
y_pred = clf.predict(X_scaled)

print("\n" + "="*25 + " 1. Probing Model Result " + "="*25)
print(classification_report(y, y_pred, digits=4))


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
weights = clf.coef_[0]
top_indices = np.argsort(np.abs(weights))[-10:][::-1]

print("\n" + "="*25 + " Top 10 Neurons " + "="*25)
for idx in top_indices:
    print(f"Neuron #{idx:4} | Weight: {weights[idx]:.4f}")