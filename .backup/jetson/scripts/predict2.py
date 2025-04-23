#  metode2 = scale-catboost-resnet
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, label_binarize
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from catboost import CatBoostClassifier
import time
import tracemalloc
import os
import gc

# Set environment variables for performance
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

# Optimize CUDA settings
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

def extract_features_tensor(X_data, y_data, model, batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    X_tensor = torch.tensor(X_data, dtype=torch.float16 if device.type == 'cuda' else torch.float32).unsqueeze(1).unsqueeze(1).to(device)
    y_tensor = torch.tensor(y_data, dtype=torch.long).to(device)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=(device.type == 'cuda')
    )
    
    features, labels = [], []
    model.eval()
    
    with torch.no_grad(), amp.autocast(enabled=(device.type == 'cuda')):
        for data, target in loader:
            data = data.repeat(1, 3, 1, 1)
            out = model(data)
            features.append(out.cpu().numpy())
            labels.append(target.cpu().numpy())
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return np.vstack(features), np.hstack(labels)

def proses(data_string=None):
    start_time = time.time()
    tracemalloc.start()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = pd.read_csv("database6.csv")
    X = dataset.iloc[:, 1:10].values
    y = dataset.iloc[:, 0].values
    
    # Encode label and scale features
    le = LabelEncoder()
    y = le.fit_transform(y)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Load MobileNetV2
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier = nn.Identity()
    model.eval()
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    model_path = "catboost_model.cbm"
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_scaled, y)):
        print(f"\n=== Fold {fold_idx+1}/5 ===")
        
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        print("Extracting training features...")
        X_train_feat, y_train_feat = extract_features_tensor(X_train, y_train, model)
        print("Extracting testing features...")
        X_test_feat, y_test_feat = extract_features_tensor(X_test, y_test, model)
        
        gc.collect()
        
        clf = CatBoostClassifier(
            iterations=50,
            depth=4,
            learning_rate=0.1,
            loss_function='MultiClass',
            random_state=42,
            verbose=0,
            task_type='GPU' if device.type == 'cuda' else 'CPU',
            devices='0',
            train_dir='catboost_info'
        )
        clf.fit(X_train_feat, y_train_feat)
        
        if fold_idx == 0:
            clf.save_model(model_path)
            print(f"Model saved to {model_path}")
        
        print("Predicting...")
        y_pred = clf.predict(X_test_feat).flatten()
        y_prob = clf.predict_proba(X_test_feat)
        
        acc = accuracy_score(y_test_feat, y_pred)
        f1 = f1_score(y_test_feat, y_pred, average='macro')
        
        y_test_bin = label_binarize(y_test_feat, classes=np.unique(y))
        if y_test_bin.shape[1] == 1:  # binary class fix
            y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))
        
        aucs = []
        for i in range(y_test_bin.shape[1]):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            aucs.append(auc(fpr, tpr))
        avg_auc = np.mean(aucs)
        
        results.append((acc, f1, avg_auc))
        print(f"Fold {fold_idx+1} → Accuracy={acc:.4f}, F1={f1:.4f}, AUC={avg_auc:.4f}")
        
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    accs, f1s, aucs = zip(*results)
    
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    result_dict = {
        "predicted_label": "N/A",
        "average_accuracy": round(np.mean(accs), 4),
        "average_f1_score": round(np.mean(f1s), 4),
        "average_auc": round(np.mean(aucs), 4),
        "time_used": round(end_time - start_time, 2),
        "memory_used_MB": round(current / 1e6, 2),
        "peak_memory_MB": round(peak / 1e6, 2)
    }

    # === Predict new data if provided ===
    if data_string:
        print("\n--- Predicting New Data ---")
        input_array = np.array(data_string.strip().split(','), dtype=np.float32).reshape(1, -1)
        input_array = scaler.transform(input_array)
        input_feat, _ = extract_features_tensor(input_array, np.zeros(1), model)
        
        clf = CatBoostClassifier()
        clf.load_model(model_path)
        y_pred = clf.predict(input_feat).flatten()
        y_pred_label = le.inverse_transform(y_pred.astype(int))
        
        result_dict["predicted_label"] = y_pred_label[0]

    print("\n=== Final Results ===")
    for k, v in result_dict.items():
        print(f"{k}: {v}")
    
    return result_dict
