#  metode1 = scale-adaboost-resnet

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, label_binarize
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
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

def proses(data_input=None):
    print("predict menggunakan metode scale-adaboost-resnet")
    # Tracking waktu & memori
    start_time = time.time()
    tracemalloc.start()

    # Load dataset
    dataset = pd.read_csv("database6.csv")
    X = dataset.iloc[:, 1:10].values
    y = dataset.iloc[:, 0].values

    le = LabelEncoder()
    y = le.fit_transform(y)
    X = MinMaxScaler().fit_transform(X)

    # Load ResNet18 pretrained
    resnet = models.resnet18(pretrained=True)
    resnet.fc = nn.Identity()
    resnet.eval()

    # K-Fold CV
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for _, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_feat, y_train_feat = extract_features_tensor(X_train, y_train, resnet)
        X_test_feat, y_test_feat = extract_features_tensor(X_test, y_test, resnet)

        clf = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
        clf.fit(X_train_feat, y_train_feat)
        y_pred = clf.predict(X_test_feat)
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
    avg_acc = round(np.mean(accs), 4)
    avg_f1 = round(np.mean(f1s), 4)
    avg_auc = round(np.mean(aucs), 4)

    # Tracking selesai
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Prediksi untuk input data jika disediakan
    predicted_label = "N/A"
    if data_input:
        try:
            data_list = list(map(float, data_input.strip().split(',')))
            input_array = np.array(data_list).reshape(1, -1)
            input_scaled = MinMaxScaler().fit_transform(np.vstack([X, input_array]))[-1].reshape(1, -1)
            input_feat, _ = extract_features_tensor(input_scaled, [0], resnet)
            predicted = clf.predict(input_feat)
            predicted_label = le.inverse_transform(predicted)[0]
        except Exception as e:
            predicted_label = f"Error: {e}"

    return {
        "predicted_label": predicted_label,
        "average_accuracy": avg_acc,
        "average_f1_score": avg_f1,
        "average_auc": avg_auc,
        "time_used": round(end_time - start_time, 2),
        "memory_used_MB": round(current / 1e6, 2),
        "peak_memory_MB": round(peak / 1e6, 2)
    }
