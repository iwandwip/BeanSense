import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
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


class AdaBoostResNetModel:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, batch_size=64, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.batch_size = batch_size
        self.random_state = random_state

        self.label_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()
        self.resnet.eval()

        self.classifier = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=self.max_depth),
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state
        )

        self.is_trained = False

    def load_data(self, csv_path, features_cols=None, target_col=0):
        dataset = pd.read_csv(csv_path)
        print(f"Dataset shape: {dataset.shape}")
        print(f"Dataset columns: {dataset.columns.tolist()}")

        if features_cols is None:
            # Jika dataset memiliki 10 kolom atau lebih, gunakan 1-9 sebagai fitur
            if dataset.shape[1] >= 10:
                features_cols = list(range(1, 10))
            else:
                # Jika kurang, gunakan semua kolom kecuali target
                features_cols = [i for i in range(dataset.shape[1]) if i != target_col]

        print(f"Using features columns: {features_cols}")
        print(f"Using target column: {target_col}")

        # Menggunakan try-except untuk menangkap error indexing
        try:
            X = dataset.iloc[:, features_cols].values
            y = dataset.iloc[:, target_col].values
        except IndexError as e:
            print(f"Index error: {e}")
            print("Trying alternative approach...")

            if isinstance(features_cols, list):
                # Hanya gunakan kolom yang valid
                valid_cols = [col for col in features_cols if col < dataset.shape[1]]
                X = dataset.iloc[:, valid_cols].values
            else:
                # Jika bukan list, coba gunakan semua kolom kecuali kolom pertama
                X = dataset.iloc[:, 1:].values

            # Untuk target, pastikan indeks valid
            if target_col < dataset.shape[1]:
                y = dataset.iloc[:, target_col].values
            else:
                y = dataset.iloc[:, 0].values
                print("Using first column as target")

        y = self.label_encoder.fit_transform(y)
        X = self.scaler.fit_transform(X)

        print(f"X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    def extract_features(self, X_data, y_data=None):
        if y_data is None:
            y_data = np.zeros(X_data.shape[0])

        X_tensor = torch.tensor(X_data, dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        y_tensor = torch.tensor(y_data, dtype=torch.long)
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=self.batch_size, shuffle=False)

        features, labels = [], []
        with torch.no_grad():
            for data, target in loader:
                data = data.repeat(1, 3, 1, 1)
                out = self.resnet(data)
                features.append(out.numpy())
                labels.append(target.numpy())

        return np.vstack(features), np.hstack(labels)

    def train(self, X, y, use_kfold=False, n_splits=5, test_size=0.2):
        start_time = time.time()
        tracemalloc.start()

        results = []

        if use_kfold:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

            for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                X_train_feat, y_train_feat = self.extract_features(X_train, y_train)
                X_test_feat, y_test_feat = self.extract_features(X_test, y_test)

                self.classifier.fit(X_train_feat, y_train_feat)

                y_pred = self.classifier.predict(X_test_feat)
                y_prob = self.classifier.predict_proba(X_test_feat)

                acc = accuracy_score(y_test_feat, y_pred)
                f1 = f1_score(y_test_feat, y_pred, average='macro')

                y_test_bin = label_binarize(y_test_feat, classes=np.unique(y))
                aucs = []
                for i in range(y_test_bin.shape[1]):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                    aucs.append(auc(fpr, tpr))
                avg_auc = np.mean(aucs)

                print(f"[FOLD {fold}] Accuracy: {acc:.4f}, F1-Score: {f1:.4f}, AUC: {avg_auc:.4f}")
                results.append((acc, f1, avg_auc))

            accs, f1s, aucs = zip(*results)
            print(f"Avg Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
            print(f"Avg F1-Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
            print(f"Avg AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )

            X_train_feat, y_train_feat = self.extract_features(X_train, y_train)
            X_test_feat, y_test_feat = self.extract_features(X_test, y_test)

            self.classifier.fit(X_train_feat, y_train_feat)

            y_pred = self.classifier.predict(X_test_feat)
            y_prob = self.classifier.predict_proba(X_test_feat)

            acc = accuracy_score(y_test_feat, y_pred)
            f1 = f1_score(y_test_feat, y_pred, average='macro')

            y_test_bin = label_binarize(y_test_feat, classes=np.unique(y))
            aucs = []
            for i in range(y_test_bin.shape[1]):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                aucs.append(auc(fpr, tpr))
            avg_auc = np.mean(aucs)

            print(f"Accuracy: {acc:.4f}, F1-Score: {f1:.4f}, AUC: {avg_auc:.4f}")
            results = [(acc, f1, avg_auc)]

        self.is_trained = True

        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"Total Training Time: {end_time - start_time:.2f}s")
        print(f"Memory Used: {current / 1e6:.2f} MB, Peak: {peak / 1e6:.2f} MB")

        return results

    def predict(self, X, return_proba=False):
        if not self.is_trained:
            raise Exception("Model has not been trained yet. Call train() first.")

        X_scaled = self.scaler.transform(X)
        X_feat, _ = self.extract_features(X_scaled)

        if return_proba:
            return self.classifier.predict_proba(X_feat)
        else:
            predictions = self.classifier.predict(X_feat)
            return self.label_encoder.inverse_transform(predictions)

    def predict_single(self, features, return_proba=False):
        if not self.is_trained:
            raise Exception("Model has not been trained yet. Call train() first.")

        features = np.array(features).reshape(1, -1)
        return self.predict(features, return_proba)

    def save_model(self, filepath):
        model_data = {
            'classifier': self.classifier,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'random_state': self.random_state
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.classifier = model_data['classifier']
        self.label_encoder = model_data['label_encoder']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        self.n_estimators = model_data['n_estimators']
        self.learning_rate = model_data['learning_rate']
        self.max_depth = model_data['max_depth']
        self.random_state = model_data['random_state']

        print(f"Model loaded from {filepath}")


def main():
    try:
        model = AdaBoostResNetModel(n_estimators=100, learning_rate=0.1)
        print("Attempting to load CSV file...")
        X, y = model.load_data("dataset4.csv")

        if X is not None and y is not None:
            print("Data loaded successfully, starting training...")
            model.train(X, y, use_kfold=True)
            model.save_model("adaboost_resnet_model.pkl")

            print("Testing prediction:")
            test_data = X[:1]
            prediction = model.predict(test_data)
            print(f"Prediction: {prediction}")
        else:
            print("Failed to load data properly")
    except Exception as e:
        import traceback
        print(f"Error occurred: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
