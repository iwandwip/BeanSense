from lightgbm import LGBMClassifier
import inquirer
import tracemalloc
import time
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, label_binarize
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=RuntimeWarning)


class MobileNetLightGBMModel:
    def __init__(self, n_estimators=120, learning_rate=0.01, max_depth=5, colsample_bytree=0.8,
                 subsample=0.8, reg_alpha=0.1, reg_lambda=0.1, min_child_samples=20, batch_size=64, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_samples = min_child_samples
        self.batch_size = batch_size
        self.random_state = random_state

        self.label_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
        self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.mobilenet.last_channel, 128)
        )
        self.mobilenet.eval()

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
                data = torch.nn.functional.interpolate(data, size=(224, 224))
                out = self.mobilenet(data)
                features.append(out.numpy())
                labels.append(target.numpy())

        return np.vstack(features), np.hstack(labels)

    def train(self, X, y, use_kfold=False, n_splits=5, test_size=0.2):
        start_time = time.time()
        tracemalloc.start()

        results = []

        num_classes = len(np.unique(y))

        if use_kfold:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

            for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                X_train_feat, y_train_feat = self.extract_features(X_train, y_train)
                X_test_feat, y_test_feat = self.extract_features(X_test, y_test)

                self.classifier = LGBMClassifier(
                    objective='multiclass',
                    num_class=num_classes,
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    colsample_bytree=self.colsample_bytree,
                    subsample=self.subsample,
                    reg_alpha=self.reg_alpha,
                    reg_lambda=self.reg_lambda,
                    min_child_samples=self.min_child_samples,
                    random_state=self.random_state,
                    verbosity=-1
                )

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
            avg_accuracy = np.mean(accs)
            avg_f1_score = np.mean(f1s)
            avg_auc_score = np.mean(aucs)

            print(f"Avg Accuracy: {avg_accuracy:.4f} ± {np.std(accs):.4f}")
            print(f"Avg F1-Score: {avg_f1_score:.4f} ± {np.std(f1s):.4f}")
            print(f"Avg AUC: {avg_auc_score:.4f} ± {np.std(aucs):.4f}")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )

            X_train_feat, y_train_feat = self.extract_features(X_train, y_train)
            X_test_feat, y_test_feat = self.extract_features(X_test, y_test)

            self.classifier = LGBMClassifier(
                objective='multiclass',
                num_class=num_classes,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                colsample_bytree=self.colsample_bytree,
                subsample=self.subsample,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                min_child_samples=self.min_child_samples,
                random_state=self.random_state,
                verbosity=-1
            )

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

            avg_accuracy = acc
            avg_f1_score = f1
            avg_auc_score = avg_auc

            print(f"Accuracy: {acc:.4f}, F1-Score: {f1:.4f}, AUC: {avg_auc:.4f}")
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

    def predict(self, X, return_metrics=False, y_true=None):
        if not self.is_trained:
            raise Exception("Model has not been trained yet. Call train() first.")

        start_time = time.time()
        tracemalloc.start()

        X_scaled = self.scaler.transform(X)
        X_feat, _ = self.extract_features(X_scaled)

        predictions = self.classifier.predict(X_feat)
        probabilities = self.classifier.predict_proba(X_feat)

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
            'is_trained': self.is_trained,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'colsample_bytree': self.colsample_bytree,
            'subsample': self.subsample,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'min_child_samples': self.min_child_samples,
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
        self.is_trained = model_data['is_trained']
        self.n_estimators = model_data.get('n_estimators', self.n_estimators)
        self.learning_rate = model_data.get('learning_rate', self.learning_rate)
        self.max_depth = model_data.get('max_depth', self.max_depth)
        self.colsample_bytree = model_data.get('colsample_bytree', self.colsample_bytree)
        self.subsample = model_data.get('subsample', self.subsample)
        self.reg_alpha = model_data.get('reg_alpha', self.reg_alpha)
        self.reg_lambda = model_data.get('reg_lambda', self.reg_lambda)
        self.min_child_samples = model_data.get('min_child_samples', self.min_child_samples)
        self.random_state = model_data.get('random_state', self.random_state)


def main():
    try:
        model = MobileNetLightGBMModel(
            n_estimators=120,
            learning_rate=0.01,
            max_depth=5,
            reg_alpha=0.1,
            reg_lambda=0.1,
            min_child_samples=20
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

                print(f"Using dataset: {csv_file}")

                print("Loading data...")
                X, y = model.load_data(csv_file)

                if X is not None and y is not None:
                    print("Data loaded successfully, starting training...")
                    metrics = model.train(X, y, use_kfold=True)

                    print("\nTraining Results:")
                    print(f"Average Accuracy: {metrics['avg_accuracy']:.4f}")
                    print(f"Average F1-Score: {metrics['avg_f1_score']:.4f}")
                    print(f"Average AUC: {metrics['avg_auc']:.4f}")
                    print(f"Memory Used: {metrics['memory_used']:.2f} MB")

                    model_file = f"model/mobilenet_lightgbm_model_{dataset_choice}.pkl"
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
                                      ('4 sensor model (mobilenet_lightgbm_model_4.pkl)', '4'),
                                      ('6 sensor model (mobilenet_lightgbm_model_6.pkl)', '6'),
                                      ('8 sensor model (mobilenet_lightgbm_model_8.pkl)', '8')
                                  ],
                                  )
                ]
                model_answer = inquirer.prompt(model_questions)
                dataset_choice = model_answer['model']
                model_file = f"model/mobilenet_lightgbm_model_{dataset_choice}.pkl"

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
