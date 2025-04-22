

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, classification_report
import time
import tracemalloc

# === Start Tracking ===
start_time = time.time()
tracemalloc.start()

# === Load Data ===
df = pd.read_csv('database8.csv')
X = df.iloc[:, 1:10].values
y = df.iloc[:, 0].values
le = LabelEncoder()
y = le.fit_transform(y)

# === Feature Scaling ===
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# === K-Fold Cross Validation ===
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

fold = 1
for train_index, test_index in kf.split(X, y):
    print(f"\n=== Fold {fold} ===")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Grid Search for optimal SVM
    param_grid = {
        'C': np.logspace(-2, 2, 5),
        'gamma': np.logspace(-4, 0, 5),
        'kernel': ['rbf']
    }
    svm = SVC(probability=True)
    grid = GridSearchCV(svm, param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print("="*20)
    # print(X_test)
    tes_input_str = "aSem-L,3141,1428,2586,170,3969,275,224,351"
    tes_parts = tes_input_str.strip().split(',')
    tes_label = tes_parts[0]
    tes_features = np.array([float(x) for x in tes_parts[1:]]).reshape(1, -1)

    tes_prediction = best_model.predict(tes_features)[0]
    tes_prediction_label = le.inverse_transform([tes_prediction])[0]

    print("HASIIL : {}".format(tes_prediction))
    print("HASIIL CLASS : {}".format(tes_prediction_label))
    # proba = best_model.predict_proba(features)
    print("="*20)
    print(" "*20)
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)


    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    aucs = []
    for i in range(y_test_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        aucs.append(auc(fpr, tpr))
    avg_auc = np.mean(aucs)

    print(f"Accuracy: {acc:.4f}, F1-score: {f1:.4f}, AUC: {avg_auc:.4f}")
    results.append((acc, f1, avg_auc))
    fold += 1

# === Summary ===
accs, f1s, aucs = zip(*results)
print("\n=== K-FOLD SUMMARY ===")
print(f"Average Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"Average F1-score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"Average AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

# === Resource Usage ===
end_time = time.time()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"\nTotal Time: {end_time - start_time:.2f} seconds")
print(f"Memory Used: {current / 1e6:.2f} MB, Peak: {peak / 1e6:.2f} MB")
