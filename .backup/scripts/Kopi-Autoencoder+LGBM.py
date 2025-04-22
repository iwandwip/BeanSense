# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 19:47:03 2025

@author: WINDOWS 11
"""

# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from lightgbm import LGBMClassifier
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sb
import time
import tracemalloc

# Start tracking
start_time = time.time()
tracemalloc.start()

# Load dataset
dataset = pd.read_csv("dDisertasiV1-8.csv")
X = dataset.iloc[:, 1:10].values
y = dataset.iloc[:, 0].values

# Label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Define Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoded_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, encoded_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoded_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Initialize model
autoencoder = Autoencoder(input_dim=X_train.shape[1], encoded_dim=16)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    autoencoder.train()
    encoded, decoded = autoencoder(X_train_tensor)
    loss = criterion(decoded, X_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Feature extraction
autoencoder.eval()
with torch.no_grad():
    X_train_encoded = autoencoder.encoder(X_train_tensor).numpy()
    X_test_encoded = autoencoder.encoder(X_test_tensor).numpy()

# Train LightGBM
lgbm = LGBMClassifier(
    objective='multiclass',
    num_class=len(np.unique(y)),
    n_estimators=150,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)
lgbm.fit(X_train_encoded, y_train)
y_pred = lgbm.predict(X_test_encoded)

# Evaluate
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
y_test_bin = label_binarize(y_test, classes=np.unique(y))
y_score = lgbm.predict_proba(X_test_encoded)
n_classes = y_test_bin.shape[1]
fpr, tpr, roc_auc = {}, {}, {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f"{label_encoder.classes_[i]} (AUC={roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# Resource usage
end_time = time.time()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"Total Time: {end_time - start_time:.2f}s")
print(f"Memory Used: {current / 1e6:.2f} MB, Peak: {peak / 1e6:.2f} MB")
