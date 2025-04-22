import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, label_binarize
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from catboost import CatBoostClassifier
import time
import tracemalloc

def extract_features_tensor(X_data, y_data, model, batch_size=64):
    X_tensor = torch.tensor(X_data, dtype=torch.float32).unsqueeze(1).unsqueeze(1)
    y_tensor = torch.tensor(y_data, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size)
    features, labels = [], []
    with torch.no_grad():
        for data, target in loader:
            data = data.repeat(1, 3, 1, 1)
            out = model(data)
            features.append(out.numpy())
            labels.append(target.numpy())
    return np.vstack(features), np.hstack(labels)

def proses(data_string):
    start_time = time.time()
    tracemalloc.start()

    dataset = pd.read_csv("database6.csv")
    X = dataset.iloc[:, 1:10].values
    y = dataset.iloc[:, 0].values

    le = LabelEncoder()
    y = le.fit_transform(y)
    X = MinMaxScaler().fit_transform(X)

    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Identity()
    resnet.eval()

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_feat, y_train_feat = extract_features_tensor(X_train, y_train, resnet)
        X_test_feat, y_test_feat = extract_features_tensor(X_test, y_test, resnet)

        clf = CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            loss_function='MultiClass',
            random_state=42,
            verbose=0,
            task_type='GPU',  # Aktifkan GPU
            devices='0'       # Pilih device GPU
        )
        clf.fit(X_train_feat, y_train_feat)
        y_pred = clf.predict(X_test_feat).flatten()
        y_prob = clf.predict_proba(X_test_feat)

        acc = accuracy_score(y_test_feat, y_pred)
        f1 = f1_score(y_test_feat, y_pred, average='macro')

        y_test_bin = label_binarize(y_test_feat, classes=np.unique(y))
        aucs = []
        for i in range(y_test_bin.shape[1]):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            aucs.append(auc(fpr, tpr))
        avg_auc = np.mean(aucs)

        results.append((acc, f1, avg_auc))

    accs, f1s, aucs = zip(*results)
    end_time = time.time()
    current, peak = tracemalloc.get_tracked_memory()
    tracemalloc.stop()

    return {
        "predicted_label": "N/A (This model only evaluates performance)",
        "average_accuracy": round(np.mean(accs), 4),
        "average_f1_score": round(np.mean(f1s), 4),
        "average_auc": round(np.mean(aucs), 4),
        "time_used": round(end_time - start_time, 2),
        "memory_used_MB": round(current / 1e6, 2),
        "peak_memory_MB": round(peak / 1e6, 2)
    }
