import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# --- 1. Load the binary data ---
# We use the .pt file created by our extraction script
print("Loading binary data from experiment_data.pt...")
data = torch.load("experiment_data.pt")

# Convert torch tensors to numpy arrays
X = data["X"].numpy()
y = data["y"].numpy()

print(f"Loaded X with shape: {X.shape}") # Should show (Total_Tokens, 2560)
print(f"Total Labels: {len(y)} (Errors: {sum(y)}, Correct: {len(y)-sum(y)})")

# --- 2. Feature Scaling ---
# Normalize neuron activations so weights are comparable
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. Train Logistic Regression ---
# This is our 'Linear Probe'
# class_weight='balanced' helps with the small number of error tokens
print("Training Linear Probe (Logistic Regression)...")
clf = LogisticRegression(
    max_iter=2000, 
    class_weight='balanced', 
    solver='liblinear',
    random_state=42
)
clf.fit(X_scaled, y)

# --- 4. Evaluation on Global Data ---
# Checking how well the neurons represent the labels in our dataset
y_pred = clf.predict(X_scaled)
print("\n=== Global Performance Report ===")
print(classification_report(y, y_pred))

# --- 5. Extract Top Neurons (The 'Sentinels') ---
# Weights tell us which neurons are most sensitive to errors
weights = clf.coef_[0]

# Get the indices of top 10 neurons with largest absolute weights
top_indices = np.argsort(np.abs(weights))[-10:][::-1]

print("\n=== Top 10 Most Influential Neurons ===")
for i, idx in enumerate(top_indices):
    w_val = weights[idx]
    # Positive weight (+) = detects errors
    # Negative weight (-) = detects correctness
    role = "Error Detector (+)" if w_val > 0 else "Correctness Sentinel (-)"
    print(f"Rank {i+1}: Neuron #{idx:4} | Weight: {w_val:8.4f} | Role: {role}")