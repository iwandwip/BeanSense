import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, classification_report
from lightgbm import LGBMClassifier
import time
import tracemalloc
import random
import os
import pickle
import inquirer
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=RuntimeWarning)


class MobileNetICCSLightGBMModel:
    def __init__(self, n_estimators=50, learning_rate=0.05, max_depth=5, num_leaves=31,
                 colsample_bytree=0.7, subsample=0.7, batch_size=32, random_state=42,
                 use_iccs=True, iccs_iterations=5, iccs_nests=5):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.batch_size = batch_size
        self.random_state = random_state

        self.use_iccs = use_iccs
        self.iccs_iterations = iccs_iterations
        self.iccs_nests = iccs_nests

        self.label_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
        self.mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
        self.mobilenet.eval()

        self.classifier = None
        self.selected_indices = None
        self.is_trained = False

    def load_data(self, csv_path, features_cols=None, target_col=None):
        dataset = pd.read_csv(csv_path)
        print(f"Dataset shape: {dataset.shape}")
        print(f"Dataset columns: {dataset.columns.tolist()}")

        if target_col is None:
            target_col = 0

        if features_cols is None:
            features_cols = [i for i in range(dataset.shape[1]) if i != target_col]

        print(f"Using features columns: {features_cols}")
        print(f"Using target column: {target_col}")

        X = dataset.iloc[:, features_cols].values
        y = dataset.iloc[:, target_col].values

        y = self.label_encoder.fit_transform(y)
        X = self.scaler.fit_transform(X)

        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"Unique classes: {self.label_encoder.classes_}")
        return X, y

    def extract_features(self, X_data, y_data=None, cache_file=None, force_recompute=False):
        if y_data is None:
            y_data = np.zeros(X_data.shape[0])

        if cache_file is not None and os.path.exists(cache_file) and not force_recompute:
            print(f"Loading cached features from {cache_file}")
            feature_data = np.load(cache_file, allow_pickle=True)
            return feature_data['features'], feature_data['labels']

        X_tensor = torch.tensor(X_data, dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        y_tensor = torch.tensor(y_data, dtype=torch.long)
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=self.batch_size, shuffle=False)

        features, labels = [], []
        with torch.no_grad():
            for data, target in loader:
                data = data.repeat(1, 3, 1, 1)
                outputs = self.mobilenet(data)
                outputs = torch.nn.functional.adaptive_avg_pool2d(outputs, 1).view(outputs.size(0), -1)
                features.append(outputs.numpy())
                labels.append(target.numpy())

        features_array = np.vstack(features)
        labels_array = np.hstack(labels)

        if cache_file is not None:
            print(f"Saving features to {cache_file}")
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            np.savez(cache_file, features=features_array, labels=labels_array)

        return features_array, labels_array

    def _fitness(self, individual, X, y):
        selected = [i for i, val in enumerate(individual) if val == 1]
        if len(selected) == 0:
            return 0

        # Use a smaller subset for faster evaluation
        sample_size = min(300, X.shape[0])
        X_sub, y_sub = X[:sample_size, :], y[:sample_size]

        model = LGBMClassifier(
            objective='multiclass',
            num_class=len(np.unique(y)),
            n_estimators=30,
            random_state=self.random_state,
            verbosity=-1
        )

        model.fit(X_sub[:, selected], y_sub)
        y_pred = model.predict(X_sub[:, selected])
        return accuracy_score(y_sub, y_pred)

    def _generate_nests(self, n, dim):
        return [np.random.randint(0, 2, dim).tolist() for _ in range(n)]

    def _levy_flight(self, ind):
        return [i if random.random() > 0.3 else 1 - i for i in ind]

    def feature_selection(self, X, y):
        print("\n[ICCS] Selecting optimal feature subset...")
        dim = X.shape[1]
        nests = self._generate_nests(self.iccs_nests, dim)
        fitness_scores = [self._fitness(n, X, y) for n in nests]
        best_nest = nests[np.argmax(fitness_scores)]

        for t in range(self.iccs_iterations):
            new_nests = [self._levy_flight(n) for n in nests]
            new_scores = [self._fitness(n, X, y) for n in new_nests]

            for i in range(self.iccs_nests):
                if new_scores[i] > fitness_scores[i]:
                    nests[i] = new_nests[i]
                    fitness_scores[i] = new_scores[i]

            best_nest = nests[np.argmax(fitness_scores)]
            print(f"Iteration {t+1}, Best Accuracy: {max(fitness_scores):.4f}")

        selected_indices = [i for i, bit in enumerate(best_nest) if bit == 1]
        print(f"Selected {len(selected_indices)} features out of {X.shape[1]}")

        return selected_indices

    def train(self, X, y, use_kfold=False, n_splits=5, test_size=0.2, use_cached_features=False, feature_cache_file="cache/features.npz"):
        start_time = time.time()
        tracemalloc.start()

        print("Extracting features...")
        X_feat, y_feat = self.extract_features(X, y, cache_file=feature_cache_file if use_cached_features else None)

        # Feature selection using ICCS
        if self.use_iccs:
            self.selected_indices = self.feature_selection(X_feat, y_feat)
            X_feat_selected = X_feat[:, self.selected_indices]
        else:
            self.selected_indices = list(range(X_feat.shape[1]))
            X_feat_selected = X_feat

        num_classes = len(np.unique(y_feat))
        results = []

        if use_kfold:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

            for fold, (train_idx, test_idx) in enumerate(kf.split(X_feat_selected, y_feat), 1):
                X_train_fold = X_feat_selected[train_idx]
                y_train_fold = y_feat[train_idx]
                X_test_fold = X_feat_selected[test_idx]
                y_test_fold = y_feat[test_idx]

                self.classifier = LGBMClassifier(
                    boosting_type='gbdt',
                    objective='multiclass',
                    num_class=num_classes,
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    num_leaves=self.num_leaves,
                    learning_rate=self.learning_rate,
                    colsample_bytree=self.colsample_bytree,
                    subsample=self.subsample,
                    random_state=self.random_state,
                    verbosity=-1
                )

                self.classifier.fit(X_train_fold, y_train_fold)

                y_pred = self.classifier.predict(X_test_fold)
                y_prob = self.classifier.predict_proba(X_test_fold)

                acc = accuracy_score(y_test_fold, y_pred)
                f1 = f1_score(y_test_fold, y_pred, average='macro')

                y_test_bin = label_binarize(y_test_fold, classes=np.unique(y_feat))
                aucs = []
                for i in range(y_test_bin.shape[1]):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                    aucs.append(auc(fpr, tpr))
                avg_auc = np.mean(aucs)

                print(f"[FOLD {fold}] Accuracy: {acc:.4f}, F1-Score: {f1:.4f}, AUC: {avg_auc:.4f}")
                print(classification_report(y_test_fold, y_pred, target_names=self.label_encoder.classes_))

                results.append((acc, f1, avg_auc))

            accs, f1s, aucs = zip(*results)
            print(f"Avg Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
            print(f"Avg F1-Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
            print(f"Avg AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_feat_selected, y_feat, test_size=test_size, random_state=self.random_state, stratify=y_feat
            )

            self.classifier = LGBMClassifier(
                boosting_type='gbdt',
                objective='multiclass',
                num_class=num_classes,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                num_leaves=self.num_leaves,
                learning_rate=self.learning_rate,
                colsample_bytree=self.colsample_bytree,
                subsample=self.subsample,
                random_state=self.random_state,
                verbosity=-1
            )

            self.classifier.fit(X_train, y_train)

            y_pred = self.classifier.predict(X_test)
            y_prob = self.classifier.predict_proba(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')

            y_test_bin = label_binarize(y_test, classes=np.unique(y_feat))
            aucs = []
            for i in range(y_test_bin.shape[1]):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                aucs.append(auc(fpr, tpr))
            avg_auc = np.mean(aucs)

            print(f"Accuracy: {acc:.4f}, F1-Score: {f1:.4f}, AUC: {avg_auc:.4f}")
            print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

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

        if self.selected_indices is None:
            raise Exception("Feature selection indices are not available. Train the model first.")

        X_scaled = self.scaler.transform(X)
        X_feat, _ = self.extract_features(X_scaled)
        X_feat_selected = X_feat[:, self.selected_indices]

        if return_proba:
            return self.classifier.predict_proba(X_feat_selected)
        else:
            predictions = self.classifier.predict(X_feat_selected)
            return self.label_encoder.inverse_transform(predictions)

    def predict_single(self, features, return_proba=False):
        if not self.is_trained:
            raise Exception("Model has not been trained yet. Call train() first.")

        features = np.array(features).reshape(1, -1)
        return self.predict(features, return_proba)

    def predict_custom_input(self, input_string):
        if not self.is_trained:
            raise Exception("Model has not been trained yet. Call train() first.")

        try:
            values = [float(x) for x in input_string.strip().split(',')]
            features = np.array(values).reshape(1, -1)

            if features.shape[1] != self.scaler.n_features_in_:
                raise ValueError(f"Input features count ({features.shape[1]}) does not match model ({self.scaler.n_features_in_})")

            X_scaled = self.scaler.transform(features)
            X_feat, _ = self.extract_features(X_scaled)
            X_feat_selected = X_feat[:, self.selected_indices]

            prediction_idx = self.classifier.predict(X_feat_selected)
            prediction_label = self.label_encoder.inverse_transform(prediction_idx)

            probabilities = self.classifier.predict_proba(X_feat_selected)[0]
            class_probs = {self.label_encoder.inverse_transform([i])[0]: prob
                           for i, prob in enumerate(probabilities)}

            return {
                "predicted_class": prediction_label[0],
                "probabilities": class_probs
            }
        except Exception as e:
            return {"error": str(e)}

    def save_model(self, filepath):
        model_data = {
            'classifier': self.classifier,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'selected_indices': self.selected_indices,
            'is_trained': self.is_trained,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'num_leaves': self.num_leaves,
            'colsample_bytree': self.colsample_bytree,
            'subsample': self.subsample,
            'random_state': self.random_state,
            'use_iccs': self.use_iccs
        }

        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

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
        self.selected_indices = model_data['selected_indices']
        self.is_trained = model_data['is_trained']
        self.n_estimators = model_data.get('n_estimators', self.n_estimators)
        self.learning_rate = model_data.get('learning_rate', self.learning_rate)
        self.max_depth = model_data.get('max_depth', self.max_depth)
        self.num_leaves = model_data.get('num_leaves', self.num_leaves)
        self.colsample_bytree = model_data.get('colsample_bytree', self.colsample_bytree)
        self.subsample = model_data.get('subsample', self.subsample)
        self.random_state = model_data.get('random_state', self.random_state)
        self.use_iccs = model_data.get('use_iccs', self.use_iccs)

        print(f"Model loaded from {filepath}")


def main():
    try:
        model = MobileNetICCSLightGBMModel(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            colsample_bytree=0.7,
            subsample=0.7,
            use_iccs=True,
            iccs_iterations=5,
            iccs_nests=5
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

                cache_questions = [
                    inquirer.Confirm('use_cached',
                                     message="Use cached features if available?",
                                     default=True)
                ]
                cache_answer = inquirer.prompt(cache_questions)
                use_cached = cache_answer['use_cached']

                print(f"Using dataset: {csv_file}")
                print("Loading data...")
                X, y = model.load_data(csv_file)

                if X is not None and y is not None:
                    print("Data loaded successfully, starting training...")
                    feature_cache = f"cache/features_{dataset_choice}.npz"
                    model.train(X, y, use_kfold=True, use_cached_features=use_cached, feature_cache_file=feature_cache)

                    model_file = f"model/mobilenet_iccs_lightgbm_model_{dataset_choice}.pkl"
                    model.save_model(model_file)

                    print("Testing prediction:")
                    test_data = X[:1]
                    prediction = model.predict(test_data)
                    print(f"Prediction: {prediction}")
                else:
                    print("Failed to load data properly")

            elif mode == "2":
                model_questions = [
                    inquirer.List('model',
                                  message="Select model to use:",
                                  choices=[
                                      ('4 sensor model (mobilenet_iccs_lightgbm_model_4.pkl)', '4'),
                                      ('6 sensor model (mobilenet_iccs_lightgbm_model_6.pkl)', '6'),
                                      ('8 sensor model (mobilenet_iccs_lightgbm_model_8.pkl)', '8')
                                  ],
                                  )
                ]
                model_answer = inquirer.prompt(model_questions)
                dataset_choice = model_answer['model']
                model_file = f"model/mobilenet_iccs_lightgbm_model_{dataset_choice}.pkl"

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
