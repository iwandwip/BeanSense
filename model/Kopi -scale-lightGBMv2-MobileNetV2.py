# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 2025

Final Program: MobileNetV2 Feature Extraction + LightGBM + K-Fold Evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import time
import tracemalloc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, classification_report
from lightgbm import LGBMClassifier
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset

# Tracking
start_time = time.time()
tracemalloc.start()

# Load Dataset
dataset = pd.read_csv("dDisertasiV1-8.csv")
X = dataset.iloc[:, 1:10].values
y = dataset.iloc[:, 0].values

# Label encoding dan normalisasi
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
X = MinMaxScaler().fit_transform(X)

# Load MobileNetV2 dan modifikasi classifier
mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
mobilenet.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(mobilenet.last_channel, 128)
)
mobilenet.eval()

# Fungsi ekstraksi fitur
def extract_features_tensor(X_data, y_data, model, batch_size=64):
    X_tensor = torch.tensor(X_data, dtype=torch.float32).unsqueeze(1).unsqueeze(1)
    y_tensor = torch.tensor(y_data, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size)
    features, labels = [], []
    with torch.no_grad():
        for data, target in loader:
            data = data.repeat(1, 3, 1, 1)
            data = torch.nn.functional.interpolate(data, size=(224, 224))
            out = model(data)
            features.append(out.numpy())
            labels.append(target.numpy())
    return np.vstack(features), np.hstack(labels)

# K-Fold Cross Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    X_train_feat, y_train_feat = extract_features_tensor(X_train, y_train, mobilenet)
    X_test_feat, y_test_feat = extract_features_tensor(X_test, y_test, mobilenet)

    model = LGBMClassifier(
        objective='multiclass',
        num_class=len(np.unique(y)),
        n_estimators=150,
        max_depth=7,
        learning_rate=0.05,
        colsample_bytree=0.9,
        subsample=0.9,
        random_state=42
    )
    model.fit(X_train_feat, y_train_feat)
    y_pred = model.predict(X_test_feat)
    y_prob = model.predict_proba(X_test_feat)

    acc = accuracy_score(y_test_feat, y_pred)
    f1 = f1_score(y_test_feat, y_pred, average='macro')

    y_bin = label_binarize(y_test_feat, classes=np.unique(y))
    aucs = []
    for i in range(y_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        aucs.append(auc(fpr, tpr))
    avg_auc = np.mean(aucs)

    print(f"[FOLD {fold}] Accuracy: {acc:.4f}, F1-Score: {f1:.4f}, AUC: {avg_auc:.4f}")
    results.append((acc, f1, avg_auc))

# Rata-rata K-Fold
accs, f1s, aucs = zip(*results)
print("\n=== K-FOLD CV SUMMARY ===")
print(f"Average Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"Average F1-Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"Average AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

# Resource
end_time = time.time()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"\nTotal Time: {end_time - start_time:.2f} seconds")
print(f"Memory Used: {current / 1e6:.2f} MB, Peak: {peak / 1e6:.2f} MB")
