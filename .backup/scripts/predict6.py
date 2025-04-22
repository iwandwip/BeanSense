#  metode6 = scale-RBF-GS-SVM

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
import time
import tracemalloc

def proses(input_string):
    print("predict menggunakan metode scale-RBF-GS-SVM")   
    # Start time and memory tracking
    start_time = time.time()
    tracemalloc.start()

    # Load Dataset
    df = pd.read_csv('database6.csv')
    X = df.iloc[:, 1:10].values
    y = df.iloc[:, 0].values
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Scaling
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # K-Fold CV
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    best_model = None

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        param_grid = {
            'C': np.logspace(-2, 2, 5),
            'gamma': np.logspace(-4, 0, 5),
            'kernel': ['rbf']
        }

        svm = SVC(probability=True)
        grid = GridSearchCV(svm, param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        y_test_bin = label_binarize(y_test, classes=np.unique(y))
        aucs = []
        for i in range(y_test_bin.shape[1]):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            aucs.append(auc(fpr, tpr))
        avg_auc = np.mean(aucs)

        results.append((acc, f1, avg_auc))
        best_model = model  # Simpan model terakhir (opsional: bisa juga yang skor terbaik)

    accs, f1s, aucs = zip(*results)
    avg_acc = np.mean(accs)
    avg_f1 = np.mean(f1s)
    avg_auc = np.mean(aucs)

    # Resource usage
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_time = end_time - start_time
    mem_used = current / 1e6
    peak_mem = peak / 1e6

    # === Predict from input_string ===
    input_data = np.array([[float(x) for x in input_string.strip().split(',')]])
    input_data = scaler.transform(input_data)
    prediction = best_model.predict(input_data)
    predicted_label = le.inverse_transform(prediction)[0]

    return {
        "predicted_label": predicted_label,
        "average_accuracy": round(avg_acc, 4),
        "average_f1_score": round(avg_f1, 4),
        "average_auc": round(avg_auc, 4),
        "time_used": round(total_time, 2),
        "memory_used_MB": round(mem_used, 2),
        "peak_memory_MB": round(peak_mem, 2)
    }
