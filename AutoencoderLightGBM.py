import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, label_binarize
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, classification_report
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sb
import time
import tracemalloc
import pickle
import os
import inquirer
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=RuntimeWarning)


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


class AutoencoderLightGBMModel:
    def __init__(self, encoded_dim=16, ae_epochs=100, ae_lr=0.001, batch_size=32,
                 n_estimators=150, learning_rate=0.05, max_depth=7,
                 subsample=0.9, colsample_bytree=0.9, random_state=42):
        self.encoded_dim = encoded_dim
        self.ae_epochs = ae_epochs
        self.ae_lr = ae_lr
        self.batch_size = batch_size

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

        self.label_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
        self.autoencoder = None
        self.classifier = None
        self.is_trained = False

    def load_data(self, csv_path, features_cols=None, target_col=None):
        dataset = pd.read_csv(csv_path)

        if target_col is None:
            target_col = 0

        if features_cols is None:
            features_cols = [i for i in range(dataset.shape[1]) if i != target_col]

        X = dataset.iloc[:, features_cols].values
        y = dataset.iloc[:, target_col].values

        y = self.label_encoder.fit_transform(y)
        X = self.scaler.fit_transform(X)

        return X, y

    def _create_autoencoder(self, input_dim):
        self.autoencoder = Autoencoder(input_dim=input_dim, encoded_dim=self.encoded_dim)
        return self.autoencoder

    def _train_autoencoder(self, X_train, X_val=None, verbose=True):
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

        if X_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.ae_lr)

        train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.ae_epochs):
            self.autoencoder.train()
            total_loss = 0

            for x_batch, _ in train_loader:
                encoded, decoded = self.autoencoder(x_batch)
                loss = criterion(decoded, x_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            if X_val is not None:
                self.autoencoder.eval()
                with torch.no_grad():
                    _, decoded_val = self.autoencoder(X_val_tensor)
                    val_loss = criterion(decoded_val, X_val_tensor).item()

                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            elif verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        return self.autoencoder

    def extract_features(self, X_data):
        X_tensor = torch.tensor(X_data, dtype=torch.float32)
        self.autoencoder.eval()
        with torch.no_grad():
            encoded_features = self.autoencoder.encoder(X_tensor).numpy()
        return encoded_features

    def train(self, X, y, use_kfold=False, n_splits=5, test_size=0.2, show_plots=False):
        start_time = time.time()
        tracemalloc.start()

        num_classes = len(np.unique(y))
        input_dim = X.shape[1]
        results = []

        if use_kfold:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

            for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
                print(f"\n=== Fold {fold} ===")
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                self._create_autoencoder(input_dim)
                self._train_autoencoder(X_train, X_test)

                X_train_encoded = self.extract_features(X_train)
                X_test_encoded = self.extract_features(X_test)

                self.classifier = LGBMClassifier(
                    objective='multiclass',
                    num_class=num_classes,
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,
                    random_state=self.random_state,
                    verbosity=-1
                )

                self.classifier.fit(X_train_encoded, y_train)

                y_pred = self.classifier.predict(X_test_encoded)
                y_prob = self.classifier.predict_proba(X_test_encoded)

                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='macro')

                y_test_bin = label_binarize(y_test, classes=np.unique(y))
                aucs = []
                for i in range(y_test_bin.shape[1]):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                    aucs.append(auc(fpr, tpr))
                avg_auc = np.mean(aucs)

                print(f"[FOLD {fold}] Accuracy: {acc:.4f}, F1-Score: {f1:.4f}, AUC: {avg_auc:.4f}")
                print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

                if show_plots:
                    self._plot_confusion_matrix(y_test, y_pred, fold)
                    self._plot_roc_curve(y_test, y_prob, fold)

                results.append((acc, f1, avg_auc))

            accs, f1s, aucs = zip(*results)
            avg_accuracy = np.mean(accs)
            avg_f1_score = np.mean(f1s)
            avg_auc_score = np.mean(aucs)

            print(f"Avg Accuracy: {avg_accuracy:.4f} ± {np.std(accs):.4f}")
            print(f"Avg F1-Score: {avg_f1_score:.4f} ± {np.std(f1s):.4f}")
            print(f"Avg AUC: {avg_auc_score:.4f} ± {np.std(aucs):.4f}")

            self._create_autoencoder(input_dim)
            self._train_autoencoder(X)

            X_encoded = self.extract_features(X)

            self.classifier = LGBMClassifier(
                objective='multiclass',
                num_class=num_classes,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                verbosity=-1
            )

            self.classifier.fit(X_encoded, y)

        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )

            self._create_autoencoder(input_dim)
            self._train_autoencoder(X_train, X_test)

            X_train_encoded = self.extract_features(X_train)
            X_test_encoded = self.extract_features(X_test)

            self.classifier = LGBMClassifier(
                objective='multiclass',
                num_class=num_classes,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                verbosity=-1
            )

            self.classifier.fit(X_train_encoded, y_train)

            y_pred = self.classifier.predict(X_test_encoded)
            y_prob = self.classifier.predict_proba(X_test_encoded)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')

            y_test_bin = label_binarize(y_test, classes=np.unique(y))
            aucs = []
            for i in range(y_test_bin.shape[1]):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                aucs.append(auc(fpr, tpr))
            avg_auc = np.mean(aucs)

            avg_accuracy = acc
            avg_f1_score = f1
            avg_auc_score = avg_auc

            print(f"Accuracy: {acc:.4f}, F1-Score: {f1:.4f}, AUC: {avg_auc:.4f}")
            print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

            if show_plots:
                self._plot_confusion_matrix(y_test, y_pred)
                self._plot_roc_curve(y_test, y_prob)

            results = [(acc, f1, avg_auc)]

        self.is_trained = True

        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        execution_time = end_time - start_time
        memory_used = current / 1e6
        peak_memory = peak / 1e6

        print(f"Total Training Time: {execution_time:.2f}s")
        print(f"Memory Used: {memory_used:.2f} MB, Peak: {peak_memory:.2f} MB")

        return {
            "avg_accuracy": avg_accuracy,
            "avg_f1_score": avg_f1_score,
            "avg_auc": avg_auc_score,
            "memory_used": memory_used,
            "peak_memory": peak_memory,
            "execution_time": execution_time
        }

    def _plot_confusion_matrix(self, y_true, y_pred, fold=None):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sb.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        title = "Confusion Matrix"
        if fold:
            title += f" - Fold {fold}"
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

    def _plot_roc_curve(self, y_true, y_prob, fold=None):
        y_test_bin = label_binarize(y_true, classes=np.unique(y_true))
        n_classes = y_test_bin.shape[1]

        plt.figure(figsize=(8, 6))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{self.label_encoder.classes_[i]} (AUC={roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], 'k--')
        title = "ROC Curve"
        if fold:
            title += f" - Fold {fold}"
        plt.title(title)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def predict(self, X, return_metrics=False, y_true=None):
        if not self.is_trained:
            raise Exception("Model has not been trained yet. Call train() first.")

        start_time = time.time()
        tracemalloc.start()

        X_scaled = self.scaler.transform(X)
        X_encoded = self.extract_features(X_scaled)

        predictions = self.classifier.predict(X_encoded)
        probabilities = self.classifier.predict_proba(X_encoded)

        result = {
            "predictions": self.label_encoder.inverse_transform(predictions),
            "probabilities": probabilities
        }

        if return_metrics and y_true is not None:
            y_true_encoded = self.label_encoder.transform(y_true)

            acc = accuracy_score(y_true_encoded, predictions)
            f1 = f1_score(y_true_encoded, predictions, average='macro')

            y_true_bin = label_binarize(y_true_encoded, classes=np.unique(y_true_encoded))
            aucs = []
            for i in range(y_true_bin.shape[1]):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], probabilities[:, i])
                aucs.append(auc(fpr, tpr))
            avg_auc = np.mean(aucs)

            end_time = time.time()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            result.update({
                "avg_accuracy": acc,
                "avg_f1_score": f1,
                "avg_auc": avg_auc,
                "memory_used": current / 1e6,
                "peak_memory": peak / 1e6,
                "execution_time": end_time - start_time
            })
        else:
            tracemalloc.stop()

        return result

    def predict_single(self, features, return_proba=False):
        if not self.is_trained:
            raise Exception("Model has not been trained yet. Call train() first.")

        features = np.array(features).reshape(1, -1)
        result = self.predict(features)

        if return_proba:
            return result["probabilities"][0]
        else:
            return result["predictions"][0]

    def predict_custom_input(self, input_string):
        if not self.is_trained:
            raise Exception("Model has not been trained yet. Call train() first.")

        try:
            values = [float(x) for x in input_string.strip().split(',')]
            features = np.array(values).reshape(1, -1)

            if features.shape[1] != self.scaler.n_features_in_:
                raise ValueError(f"Input features count ({features.shape[1]}) does not match model ({self.scaler.n_features_in_})")

            result = self.predict(features)

            prediction_label = result["predictions"][0]
            probabilities = result["probabilities"][0]

            class_probs = {self.label_encoder.inverse_transform([i])[0]: prob
                           for i, prob in enumerate(probabilities)}

            return {
                "predicted_class": prediction_label,
                "probabilities": class_probs
            }
        except Exception as e:
            return {"error": str(e)}

    def save_model(self, filepath):
        model_data = {
            'classifier': self.classifier,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'autoencoder_state': self.autoencoder.state_dict() if self.autoencoder else None,
            'input_dim': self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else None,
            'encoded_dim': self.encoded_dim,
            'is_trained': self.is_trained,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'random_state': self.random_state
        }

        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.classifier = model_data['classifier']
        self.label_encoder = model_data['label_encoder']
        self.scaler = model_data['scaler']
        self.encoded_dim = model_data['encoded_dim']
        self.is_trained = model_data['is_trained']
        self.n_estimators = model_data.get('n_estimators', self.n_estimators)
        self.learning_rate = model_data.get('learning_rate', self.learning_rate)
        self.max_depth = model_data.get('max_depth', self.max_depth)
        self.subsample = model_data.get('subsample', self.subsample)
        self.colsample_bytree = model_data.get('colsample_bytree', self.colsample_bytree)
        self.random_state = model_data.get('random_state', self.random_state)

        input_dim = model_data.get('input_dim')
        if input_dim and model_data['autoencoder_state']:
            self._create_autoencoder(input_dim)
            self.autoencoder.load_state_dict(model_data['autoencoder_state'])
            self.autoencoder.eval()


def main():
    try:
        model = AutoencoderLightGBMModel(
            encoded_dim=16,
            ae_epochs=100,
            n_estimators=150,
            learning_rate=0.05,
            max_depth=7
        )

        while True:
            operation_questions = [
                inquirer.List('mode',
                              message="Select operation mode:",
                              choices=[
                                  ('Train new model', '1'),
                                  ('Load existing model and predict', '2'),
                                  ('Exit', '0')
                              ],
                              )
            ]
            answers = inquirer.prompt(operation_questions)
            mode = answers['mode']

            if mode == "0":
                print("Exiting program...")
                break

            if mode == "1":
                dataset_questions = [
                    inquirer.List('dataset',
                                  message="Select dataset to use:",
                                  choices=[
                                      ('dataset4.csv (4 sensors)', '4'),
                                      ('dataset6.csv (6 sensors)', '6'),
                                      ('dataset8.csv (8 sensors)', '8')
                                  ],
                                  )
                ]
                dataset_answer = inquirer.prompt(dataset_questions)
                dataset_choice = dataset_answer['dataset']

                if dataset_choice == "4":
                    csv_file = "datasets/dataset4.csv"
                elif dataset_choice == "6":
                    csv_file = "datasets/dataset6.csv"
                elif dataset_choice == "8":
                    csv_file = "datasets/dataset8.csv"
                else:
                    print(f"Invalid choice: {dataset_choice}, using dataset8.csv")
                    csv_file = "datasets/dataset8.csv"
                    dataset_choice = "8"

                plot_questions = [
                    inquirer.Confirm('show_plots',
                                     message="Show evaluation plots?",
                                     default=False)
                ]
                plot_answer = inquirer.prompt(plot_questions)
                show_plots = plot_answer['show_plots']

                print(f"Using dataset: {csv_file}")
                print("Loading data...")
                X, y = model.load_data(csv_file)

                if X is not None and y is not None:
                    print("Data loaded successfully, starting training...")
                    metrics = model.train(X, y, use_kfold=True, show_plots=show_plots)

                    print("\nTraining Results:")
                    print(f"Average Accuracy: {metrics['avg_accuracy']:.4f}")
                    print(f"Average F1-Score: {metrics['avg_f1_score']:.4f}")
                    print(f"Average AUC: {metrics['avg_auc']:.4f}")
                    print(f"Memory Used: {metrics['memory_used']:.2f} MB")

                    model_file = f"model/autoencoder_lightgbm_model_{dataset_choice}.pkl"
                    model.save_model(model_file)

                    print("Testing prediction:")
                    test_data = X[:1]
                    prediction = model.predict_single(test_data)
                    print(f"Prediction: {prediction}")
                else:
                    print("Failed to load data properly")

            elif mode == "2":
                model_questions = [
                    inquirer.List('model',
                                  message="Select model to use:",
                                  choices=[
                                      ('4 sensor model (autoencoder_lightgbm_model_4.pkl)', '4'),
                                      ('6 sensor model (autoencoder_lightgbm_model_6.pkl)', '6'),
                                      ('8 sensor model (autoencoder_lightgbm_model_8.pkl)', '8')
                                  ],
                                  )
                ]
                model_answer = inquirer.prompt(model_questions)
                dataset_choice = model_answer['model']
                model_file = f"model/autoencoder_lightgbm_model_{dataset_choice}.pkl"

                try:
                    model.load_model(model_file)
                    print(f"Model successfully loaded from {model_file}")

                    prediction_loop = True
                    while prediction_loop:
                        input_questions = [
                            inquirer.Text('sensor_values',
                                          message="Enter sensor values (comma separated) or 'q' to quit:")
                        ]
                        input_answer = inquirer.prompt(input_questions)
                        input_string = input_answer['sensor_values']

                        if input_string.lower() == 'q':
                            break

                        result = model.predict_custom_input(input_string)

                        if "error" in result:
                            print(f"Error: {result['error']}")
                        else:
                            print(f"Prediction result: {result['predicted_class']}")
                            print("Class probabilities:")
                            for cls, prob in result['probabilities'].items():
                                print(f"  {cls}: {prob:.4f}")

                        continue_questions = [
                            inquirer.Confirm('continue',
                                             message="Would you like to make another prediction?",
                                             default=True)
                        ]
                        continue_answer = inquirer.prompt(continue_questions)
                        if not continue_answer['continue']:
                            prediction_loop = False
                except FileNotFoundError:
                    print(f"Model file {model_file} not found. Train a model first.")
                except Exception as e:
                    print(f"Error loading model: {e}")
            else:
                print("Invalid choice")

            print("\n" + "-"*50 + "\n")

    except Exception as e:
        import traceback
        print(f"Error occurred: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
