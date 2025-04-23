"""
Created on Mon Jan 27 15:23:03 2025

@author: WINDOWS 11
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from lightgbm import LGBMClassifier
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
import time
import tracemalloc  # Untuk memonitor penggunaan memori

# Record start time
start_time = time.time()

# Start memory monitoring
tracemalloc.start()

# Load the dataset
try:
    dataset = pd.read_csv('dDisertasiV1-8.csv')
except FileNotFoundError:
    print("Dataset not found. Please make sure 'alat3_1 (4 sensor).csv' is in the current directory.")
    exit()

X = dataset.iloc[:, 1:10].values
y = dataset.iloc[:, 0].values

# Encode labels to integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Preprocessing
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to tensors for ResNet
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Reshape for ResNet input (batch, channel, height, width)
X_train_tensor = X_train_tensor.unsqueeze(1).unsqueeze(1)
X_test_tensor = X_test_tensor.unsqueeze(1).unsqueeze(1)

# Create DataLoader
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load ResNet and modify for feature extraction
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Identity()  # Remove the final classification layer

# Extract features using ResNet
def extract_features(loader, model):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.repeat(1, 3, 1, 1)  # Repeat channels to match ResNet input
            outputs = model(inputs)
            features.append(outputs.numpy())
            labels.append(targets.numpy())
    return np.vstack(features), np.hstack(labels)

# Extract features for training and testing sets
X_train_features, y_train = extract_features(train_loader, resnet)
X_test_features, y_test = extract_features(test_loader, resnet)

# LightGBM model
lgbm = LGBMClassifier(
    boosting_type='gbdt',
    objective='multiclass',
    num_class=len(np.unique(y)),
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    colsample_bytree=0.8,
    subsample=0.8,
    random_state=42
)

# Train LightGBM
lgbm.fit(
    X_train_features, y_train,
    eval_set=[(X_test_features, y_test)],
    eval_metric='multi_logloss',
    early_stopping_rounds=10,
    verbose=10
)

# Evaluate the model
y_pred = lgbm.predict(X_test_features)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ROC Curve and AUC
y_test_bin = label_binarize(y_test, classes=np.unique(y))
y_score = lgbm.predict_proba(X_test_features)
n_classes = y_test_bin.shape[1]
fpr, tpr, roc_auc = {}, {}, {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {label_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Record end time and calculate total time taken
end_time = time.time()
total_time = end_time - start_time
print(f"\n Total processing time: {total_time:.2f} seconds")

# Display memory usage
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 10**6:.2f} MB")
print(f"Peak memory usage: {peak / 10**6:.2f} MB")

# Stop memory monitoring
tracemalloc.stop()
