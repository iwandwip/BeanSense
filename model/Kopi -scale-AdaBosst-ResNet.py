
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, label_binarize
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_curve, auc
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sb
import time
import tracemalloc

# === Tracking ===
start_time = time.time()
tracemalloc.start()

# === Load Dataset ===
dataset = pd.read_csv("dDisertasiV1-8.csv")
X = dataset.iloc[:, 1:10].values
y = dataset.iloc[:, 0].values

# === Preprocessing ===
le = LabelEncoder()
y = le.fit_transform(y)
X = MinMaxScaler().fit_transform(X)

# === Load pretrained ResNet18 ===
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Identity()
resnet.eval()

# === Extract Features with ResNet ===
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

# === K-Fold CV Evaluation ===
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Ekstraksi fitur dari ResNet
    X_train_feat, y_train_feat = extract_features_tensor(X_train, y_train, resnet)
    X_test_feat, y_test_feat = extract_features_tensor(X_test, y_test, resnet)
    print(X_test_feat)
    # # AdaBoost Classifier
    # clf = AdaBoostClassifier(
    #     base_estimator=DecisionTreeClassifier(max_depth=3),
    #     n_estimators=100,
    #     learning_rate=0.1,
    #     random_state=42
    # )
    # clf.fit(X_train_feat, y_train_feat)
   
    # y_pred = clf.predict(X_test_feat)
    # y_prob = clf.predict_proba(X_test_feat)

    # acc = accuracy_score(y_test_feat, y_pred)
    # f1 = f1_score(y_test_feat, y_pred, average='macro')

    # y_test_bin = label_binarize(y_test_feat, classes=np.unique(y))
    # aucs = []
    # for i in range(y_test_bin.shape[1]):
    #     fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    #     aucs.append(auc(fpr, tpr))
    # avg_auc = np.mean(aucs)

    # print(f"\n[FOLD {fold}] Accuracy: {acc:.4f}, F1-Score: {f1:.4f}, AUC: {avg_auc:.4f}")
    # results.append((acc, f1, avg_auc))

# === Ringkasan Evaluasi ===
accs, f1s, aucs = zip(*results)
print("\n=== K-FOLD CV SUMMARY ===")
print(f"Avg Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"Avg F1-Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"Avg AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

# === Resource Info ===
end_time = time.time()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"\nTotal Time: {end_time - start_time:.2f}s")
print(f"Memory Used: {current / 1e6:.2f} MB, Peak: {peak / 1e6:.2f} MB")
