# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 08:56:20 2025

@author: WINDOWS 11
"""


# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 08:39:26 2025
Optimized Version: LightGBM + ICCS Hybrid + F1, AUC, K-Fold CV with Reduced Latency
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, roc_auc_score
)
from lightgbm import LGBMClassifier
import time, tracemalloc, random, os

# ========== Monitoring Waktu & Memori ==========
start_time = time.time()
tracemalloc.start()

# ========== Load Data ==========
dataset = pd.read_csv('dDisertasiV1-8.csv')
X = dataset.iloc[:, 1:10].values
y = dataset.iloc[:, 0].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ========== Ekstraksi Fitur dengan MobileNetV2 ==========
if os.path.exists("X_feat.npy") and os.path.exists("y_encoded.npy"):
    X_feat = np.load("X_feat.npy")
    y_encoded = np.load("y_encoded.npy")
else:
    print("Extracting features with MobileNetV2...")
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).unsqueeze(1)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)
    full_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=False)

    model_cnn = mobilenet_v2(pretrained=True).features
    model_cnn.eval()

    def extract_features(loader):
        features, labels = [], []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.repeat(1, 3, 1, 1)
                outputs = model_cnn(inputs)
                outputs = torch.nn.functional.adaptive_avg_pool2d(outputs, 1).view(outputs.size(0), -1)
                features.append(outputs.numpy())
                labels.append(targets.numpy())
        return np.vstack(features), np.hstack(labels)

    X_feat, y_encoded = extract_features(full_loader)
    np.save("X_feat.npy", X_feat)
    np.save("y_encoded.npy", y_encoded)

# ========== ICCS for Feature Selection ==========
def fitness(individual, X, y):
    selected = [i for i, val in enumerate(individual) if val == 1]
    if len(selected) == 0:
        return 0
    X_sub, y_sub = X[:300, :], y[:300]
    model = LGBMClassifier(objective='multiclass', num_class=len(np.unique(y)), n_estimators=30)
    model.fit(X_sub[:, selected], y_sub)
    y_pred = model.predict(X_sub[:, selected])
    return accuracy_score(y_sub, y_pred)

def generate_nests(n, dim):
    return [np.random.randint(0, 2, dim).tolist() for _ in range(n)]

def levy_flight(ind):
    return [i if random.random() > 0.3 else 1 - i for i in ind]

def iccs_optimize(X, y, max_iter=5, nests_num=5):
    dim = X.shape[1]
    nests = generate_nests(nests_num, dim)
    fitness_scores = [fitness(n, X, y) for n in nests]
    best_nest = nests[np.argmax(fitness_scores)]

    for t in range(max_iter):
        new_nests = [levy_flight(n) for n in nests]
        new_scores = [fitness(n, X, y) for n in new_nests]
        for i in range(nests_num):
            if new_scores[i] > fitness_scores[i]:
                nests[i] = new_nests[i]
                fitness_scores[i] = new_scores[i]
        best_nest = nests[np.argmax(fitness_scores)]
        print(f"Iteration {t+1}, Best Accuracy: {max(fitness_scores):.4f}")
    return best_nest

print("\n[ICCS] Selecting optimal feature subset...")
selected_mask = iccs_optimize(X_feat, y_encoded)
selected_indices = [i for i, bit in enumerate(selected_mask) if bit == 1]
print(f"Selected {len(selected_indices)} features out of {X_feat.shape[1]}")

# ========== K-Fold Cross Validation & Evaluation ==========
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

f1_scores = []
auc_scores = []
accuracies = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X_feat, y_encoded), 1):
    X_train_fold = X_feat[train_idx][:, selected_indices]
    y_train_fold = y_encoded[train_idx]
    X_test_fold = X_feat[test_idx][:, selected_indices]
    y_test_fold = y_encoded[test_idx]

    model = LGBMClassifier(
        boosting_type='gbdt',
        objective='multiclass',
        num_class=len(np.unique(y_encoded)),
        n_estimators=50,
        max_depth=5,
        num_leaves=31,
        learning_rate=0.05,
        colsample_bytree=0.7,
        subsample=0.7,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train_fold, y_train_fold)
    y_pred = model.predict(X_test_fold)
    y_proba = model.predict_proba(X_test_fold)

    f1 = f1_score(y_test_fold, y_pred, average='macro')
    auc_score = roc_auc_score(label_binarize(y_test_fold, classes=np.unique(y_encoded)), y_proba, average='macro', multi_class='ovr')
    acc = accuracy_score(y_test_fold, y_pred)

    f1_scores.append(f1)
    auc_scores.append(auc_score)
    accuracies.append(acc)

    print(f"\n[Fold {fold}] Accuracy: {acc:.4f}, F1-score: {f1:.4f}, AUC: {auc_score:.4f}")
    print(classification_report(y_test_fold, y_pred, target_names=label_encoder.classes_))

# Summary
print("\n=== K-Fold CV Results ===")
print(f"Avg Accuracy: {np.mean(accuracies):.4f}")
print(f"Avg F1-score: {np.mean(f1_scores):.4f}")
print(f"Avg AUC: {np.mean(auc_scores):.4f}")

# ========== Selesai ==========
end_time = time.time()
current, peak = tracemalloc.get_traced_memory()
print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")
print(f"Current memory usage: {current / 10**6:.2f} MB")
print(f"Peak memory usage: {peak / 10**6:.2f} MB")
tracemalloc.stop()
