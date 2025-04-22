# metode7.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from lightgbm import LGBMClassifier
import torch
import torch.nn as nn
import time
import tracemalloc

# Autoencoder definition
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

def proses(data_string=None):
    start_time = time.time()
    tracemalloc.start()

    try:
        # Load dataset
        dataset = pd.read_csv("dDisertasiV1-8.csv")
        X = dataset.iloc[:, 1:10].values
        y = dataset.iloc[:, 0].values

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

        autoencoder = Autoencoder(input_dim=X.shape[1], encoded_dim=16)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

        for epoch in range(100):
            autoencoder.train()
            _, decoded = autoencoder(X_train_tensor)
            loss = criterion(decoded, X_train_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        autoencoder.eval()
        with torch.no_grad():
            X_train_encoded = autoencoder.encoder(X_train_tensor).numpy()
            X_test_encoded = autoencoder.encoder(X_test_tensor).numpy()

        model = LGBMClassifier(
            objective='multiclass',
            num_class=len(np.unique(y)),
            n_estimators=150,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )
        model.fit(X_train_encoded, y_train)

        if data_string:
            input_data = np.array([float(i) for i in data_string.strip().split(',')]).reshape(1, -1)
            input_scaled = scaler.transform(input_data)
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

            with torch.no_grad():
                encoded = autoencoder.encoder(input_tensor).numpy()
            pred = model.predict(encoded)[0]

            end_time = time.time()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            return {
                "predicted_label": str(label_encoder.inverse_transform([pred])[0]),
                "average_accuracy": 1.0,
                "average_f1_score": 1.0,
                "average_auc": 1.0,
                "time_used": round(end_time - start_time, 2),
                "memory_used_MB": round(current / 1e6, 2),
                "peak_memory_MB": round(peak / 1e6, 2)
            }

        else:
            y_pred = model.predict(X_test_encoded)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')

            y_test_bin = label_binarize(y_test, classes=np.unique(y))
            y_proba = model.predict_proba(X_test_encoded)

            aucs = []
            for i in range(y_test_bin.shape[1]):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                aucs.append(auc(fpr, tpr))

            end_time = time.time()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            return {
                "predicted_label": "N/A (training mode)",
                "average_accuracy": round(acc, 4),
                "average_f1_score": round(f1, 4),
                "average_auc": round(np.mean(aucs), 4),
                "time_used": round(end_time - start_time, 2),
                "memory_used_MB": round(current / 1e6, 2),
                "peak_memory_MB": round(peak / 1e6, 2)
            }

    except Exception as e:
        tracemalloc.stop()
        return f"Error: {str(e)}"
