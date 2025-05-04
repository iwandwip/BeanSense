from AdaBoostClassifier import AdaBoostResNetModel
from CatBoostClassifier import CatBoostResNetModel
from LightGBMResNet import LightGBMResNetModel
from LightGBMMobileNet import MobileNetLightGBMModel
from MobileNetICCSLightGBM import MobileNetICCSLightGBMModel
from AutoencoderLightGBM import AutoencoderLightGBMModel
from RBFSVMGridSearch import RBFSVMGridSearchModel

import os
import traceback
import numpy as np
import pandas as pd
import time
import tracemalloc


class ClassifierWrapper:
    def __init__(self):
        self.dataset_choice = None
        self.model_type = None
        self.current_model = None
        self.csv_file = None
        self.models = {
            'adaboost_resnet': self._create_adaboost_resnet,
            'catboost_resnet': self._create_catboost_resnet,
            'lightgbm_resnet': self._create_lightgbm_resnet,
            'lightgbm_mobilenet': self._create_lightgbm_mobilenet,
            'mobilenet_iccs_lightgbm': self._create_mobilenet_iccs_lightgbm,
            'autoencoder_lightgbm': self._create_autoencoder_lightgbm,
            'rbf_svm_gs': self._create_rbf_svm_gs
        }
        self.labels = [
            'aKaw-D', 'aKaw-L', 'aKaw-M',
            'aSem-D', 'aSem-L', 'aSem-M',
            'rGed-D', 'rGed-L', 'rGed-M',
            'rTir-D', 'rTir-L', 'rTir-M'
        ]

        self.column_order = {
            '4': ['NAMA', 'MQ135', 'MQ2', 'MQ3', 'MQ6'],
            '6': ['NAMA', 'MQ135', 'MQ2', 'MQ3', 'MQ6', 'MQ138', 'MQ7'],
            '8': ['NAMA', 'MQ135', 'MQ2', 'MQ3', 'MQ6', 'MQ138', 'MQ7', 'MQ136', 'MQ5']
        }

    def set_dataset(self, dataset_choice):
        self.dataset_choice = dataset_choice

        if self.dataset_choice == "4":
            self.csv_file = "datasets/dataset4.csv"
        elif self.dataset_choice == "6":
            self.csv_file = "datasets/dataset6.csv"
        elif self.dataset_choice == "8":
            self.csv_file = "datasets/dataset8.csv"

        return self.csv_file

    def _create_adaboost_resnet(self):
        return AdaBoostResNetModel(n_estimators=100, learning_rate=0.1)

    def _create_catboost_resnet(self, use_gpu=False):
        return CatBoostResNetModel(iterations=100, learning_rate=0.1, depth=6, use_gpu=use_gpu)

    def _create_lightgbm_resnet(self):
        return LightGBMResNetModel(n_estimators=100, learning_rate=0.1, max_depth=6)

    def _create_lightgbm_mobilenet(self):
        return MobileNetLightGBMModel(
            n_estimators=120,
            learning_rate=0.01,
            max_depth=5,
            reg_alpha=0.1,
            reg_lambda=0.1,
            min_child_samples=20
        )

    def _create_mobilenet_iccs_lightgbm(self):
        return MobileNetICCSLightGBMModel(
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

    def _create_autoencoder_lightgbm(self):
        return AutoencoderLightGBMModel(
            encoded_dim=16,
            ae_epochs=100,
            n_estimators=150,
            learning_rate=0.05,
            max_depth=7
        )

    def _create_rbf_svm_gs(self):
        return RBFSVMGridSearchModel(
            C_range=np.logspace(-2, 2, 5),
            gamma_range=np.logspace(-4, 0, 5),
            random_state=42
        )

    def create_model(self, model_type, use_gpu=False):
        self.model_type = model_type

        if model_type == 'catboost_resnet':
            self.current_model = self._create_catboost_resnet(use_gpu)
        else:
            self.current_model = self.models[model_type]()

        return self.current_model

    def get_model_filename(self):
        if self.model_type == 'adaboost_resnet':
            return f"model/adaboost_resnet_model_{self.dataset_choice}.pkl"
        elif self.model_type == 'catboost_resnet':
            return f"model/catboost_resnet_model_{self.dataset_choice}.pkl"
        elif self.model_type == 'lightgbm_resnet':
            return f"model/lightgbm_resnet_model_{self.dataset_choice}.pkl"
        elif self.model_type == 'lightgbm_mobilenet':
            return f"model/mobilenet_lightgbm_model_{self.dataset_choice}.pkl"
        elif self.model_type == 'mobilenet_iccs_lightgbm':
            return f"model/mobilenet_iccs_lightgbm_model_{self.dataset_choice}.pkl"
        elif self.model_type == 'autoencoder_lightgbm':
            return f"model/autoencoder_lightgbm_model_{self.dataset_choice}.pkl"
        elif self.model_type == 'rbf_svm_gs':
            return f"model/rbf_svm_gs_model_{self.dataset_choice}.pkl"

    def train_model(self, use_cached_features=True, show_plots=False):
        print(f"Training {self.model_type} model on {self.csv_file}...")
        X, y = self.current_model.load_data(self.csv_file)

        if X is not None and y is not None:
            print("Data loaded successfully, starting training...")

            if self.model_type == 'mobilenet_iccs_lightgbm':
                feature_cache = f"cache/features_{self.dataset_choice}.npz"
                metrics = self.current_model.train(X, y, use_kfold=True, use_cached_features=use_cached_features, feature_cache_file=feature_cache)
            elif self.model_type == 'autoencoder_lightgbm':
                metrics = self.current_model.train(X, y, use_kfold=True, show_plots=show_plots)
            else:
                metrics = self.current_model.train(X, y, use_kfold=True)

            model_file = self.get_model_filename()
            self.current_model.save_model(model_file)

            print("Testing prediction:")
            test_data = X[:1]
            prediction = self.current_model.predict_single(test_data)
            print(f"Prediction: {prediction}")

            return metrics
        else:
            print("Failed to load data properly")
            return None

    def predict_with_model(self, input_string):
        model_file = self.get_model_filename()
        try:
            self.current_model.load_model(model_file)
            print(f"Model successfully loaded from {model_file}")
        except FileNotFoundError:
            print(f"Model file {model_file} not found. Train a model first.")
            return None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

        if input_string.lower() == 'q':
            return 'quit'

        start_time = time.time()
        tracemalloc.start()

        try:
            values = [float(x) for x in input_string.strip().split(',')]
            features = np.array(values).reshape(1, -1)

            X_scaled = self.current_model.scaler.transform(features)
            X_feat = self.current_model.extract_features(X_scaled)[0]

            if hasattr(self.current_model, 'selected_indices'):
                if self.current_model.selected_indices is not None:
                    X_feat = X_feat[:, self.current_model.selected_indices]

            prediction_idx = self.current_model.classifier.predict(X_feat)[0]
            prediction_label = self.current_model.label_encoder.inverse_transform([prediction_idx])[0]

            probabilities = self.current_model.classifier.predict_proba(X_feat)[0]
            class_probs = {self.current_model.label_encoder.inverse_transform([i])[0]: prob
                           for i, prob in enumerate(probabilities)}

            predicted_probability = class_probs[prediction_label]

            end_time = time.time()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            execution_time = end_time - start_time
            memory_used = current / 1e6
            peak_memory = peak / 1e6

            return {
                "predicted_class": prediction_label,
                "predicted_probability": predicted_probability,
                "probabilities": class_probs,
                "memory_used": memory_used,
                "peak_memory": peak_memory,
                "execution_time": execution_time
            }
        except Exception as e:
            tracemalloc.stop()
            return {"error": str(e)}

    def organize_dataset_by_label_order(self):
        if not os.path.exists(self.csv_file):
            return False

        try:
            df = pd.read_csv(self.csv_file)

            result_df = pd.DataFrame(columns=df.columns)

            for label in self.labels:
                label_data = df[df['NAMA'] == label]
                if not label_data.empty:
                    result_df = pd.concat([result_df, label_data], ignore_index=True)

            if len(result_df) == len(df):
                result_df.to_csv(self.csv_file, index=False)
                return True
            else:
                print("Warning: Some rows were not included in the reordering.")
                return False

        except Exception as e:
            print(f"Error reorganizing dataset: {e}")
            return False

    def reorder_columns(self, df):
        if self.dataset_choice in self.column_order:
            cols = self.column_order[self.dataset_choice]
            return df[cols]
        return df

    def add_dataset_entry(self, selected_label, sensor_values):
        if not os.path.exists(self.csv_file):
            print(f"Dataset file {self.csv_file} does not exist. Creating new file.")
            os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)

            if self.dataset_choice in self.column_order:
                columns = self.column_order[self.dataset_choice]
                df = pd.DataFrame(columns=columns)
            else:
                num_features = int(self.dataset_choice)
                columns = ['NAMA'] + [f'feature{i+1}' for i in range(num_features)]
                df = pd.DataFrame(columns=columns)

            df.to_csv(self.csv_file, index=False)

        try:
            values = [int(float(x)) for x in sensor_values.split(',')]
            num_features = int(self.dataset_choice)

            if len(values) != num_features:
                print(f"Error: Expected {num_features} values, but got {len(values)}.")
                return False

            df = pd.read_csv(self.csv_file)

            if df.empty:
                new_row_data = {'NAMA': selected_label}
                cols = df.columns.tolist()
                for i, value in enumerate(values):
                    if i + 1 < len(cols):
                        new_row_data[cols[i+1]] = value
            else:
                new_row_data = {df.columns[0]: selected_label}
                for i, value in enumerate(values):
                    if i + 1 < len(df.columns):
                        new_row_data[df.columns[i+1]] = value

            new_row = pd.DataFrame([new_row_data])

            result_df = pd.DataFrame(columns=df.columns)
            found_label = False

            for label in self.labels:
                label_data = df[df['NAMA'] == label]

                if label == selected_label:
                    found_label = True
                    if not label_data.empty:
                        result_df = pd.concat([result_df, label_data, new_row], ignore_index=True)
                    else:
                        result_df = pd.concat([result_df, new_row], ignore_index=True)
                else:
                    if not label_data.empty:
                        result_df = pd.concat([result_df, label_data], ignore_index=True)

            if not found_label:
                result_df = pd.concat([result_df, new_row], ignore_index=True)

            if len(result_df) != len(df) + 1:
                print("Warning: Row count mismatch. Using simple append and sort.")
                df = pd.concat([df, new_row], ignore_index=True)
                df = self.reorder_columns(df)
                self.organize_dataset_by_label_order()
            else:
                result_df = self.reorder_columns(result_df)
                result_df.to_csv(self.csv_file, index=False)

            print(f"Added new data point with label '{selected_label}' to {self.csv_file}")
            print(f"Total records now: {len(result_df)}")
            return True

        except ValueError:
            print("Error: Invalid sensor values. Please enter comma-separated numbers.")
            return False

    def pop_dataset_entry(self, selected_label):
        if not os.path.exists(self.csv_file):
            print(f"Dataset file {self.csv_file} does not exist.")
            return False

        try:
            df = pd.read_csv(self.csv_file)
            label_column = df.columns[0]

            label_mask = df[label_column] == selected_label

            if not any(label_mask):
                print(f"No data points with label '{selected_label}' found in {self.csv_file}")
                return False

            last_index = df[label_mask].index[-1]

            df = df.drop(last_index).reset_index(drop=True)

            df = self.reorder_columns(df)
            df.to_csv(self.csv_file, index=False)

            print(f"Removed last data point with label '{selected_label}' from {self.csv_file}")
            print(f"Total records now: {len(df)}")
            return True

        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())
            return False

    def delete_dataset_entries(self, selected_label, confirm=False):
        if not os.path.exists(self.csv_file):
            print(f"Dataset file {self.csv_file} does not exist.")
            return False

        try:
            df = pd.read_csv(self.csv_file)

            label_column = df.columns[0]

            count_label = len(df[df[label_column] == selected_label])

            if count_label == 0:
                print(f"No data points with label '{selected_label}' found in {self.csv_file}")
                return False

            if not confirm:
                return count_label

            df = df[df[label_column] != selected_label]

            df = self.reorder_columns(df)
            df.to_csv(self.csv_file, index=False)

            print(f"Deleted {count_label} data points with label '{selected_label}' from {self.csv_file}")
            print(f"Total records now: {len(df)}")
            return True

        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())
            return False

    def get_available_models(self):
        return list(self.models.keys())

    def get_labels(self):
        return self.labels
