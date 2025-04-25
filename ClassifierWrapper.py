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
                self.current_model.train(X, y, use_kfold=True, use_cached_features=use_cached_features, feature_cache_file=feature_cache)
            elif self.model_type == 'autoencoder_lightgbm':
                self.current_model.train(X, y, use_kfold=True, show_plots=show_plots)
            else:
                self.current_model.train(X, y, use_kfold=True)

            model_file = self.get_model_filename()
            self.current_model.save_model(model_file)

            print("Testing prediction:")
            test_data = X[:1]
            prediction = self.current_model.predict(test_data)
            print(f"Prediction: {prediction}")

            return True
        else:
            print("Failed to load data properly")
            return False

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

        result = self.current_model.predict_custom_input(input_string)
        return result

    def add_dataset_entry(self, selected_label, sensor_values):
        if not os.path.exists(self.csv_file):
            print(f"Dataset file {self.csv_file} does not exist. Creating new file.")
            os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)

            num_features = int(self.dataset_choice)
            columns = ['label'] + [f'feature{i+1}' for i in range(num_features)]
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.csv_file, index=False)

        try:
            values = [int(float(x)) for x in sensor_values.split(',')]
            num_features = int(self.dataset_choice)

            if len(values) != num_features:
                print(f"Error: Expected {num_features} values, but got {len(values)}.")
                return False

            df = pd.read_csv(self.csv_file)

            new_row_data = {df.columns[0]: selected_label}
            for i, value in enumerate(values):
                new_row_data[df.columns[i+1]] = value

            new_row = pd.DataFrame([new_row_data])
            df = pd.concat([df, new_row], ignore_index=True)

            df.to_csv(self.csv_file, index=False)

            print(f"Added new data point with label '{selected_label}' to {self.csv_file}")
            print(f"Total records now: {len(df)}")
            return True

        except ValueError:
            print("Error: Invalid sensor values. Please enter comma-separated numbers.")
            return False

    def pop_dataset_entry(self):
        if not os.path.exists(self.csv_file):
            print(f"Dataset file {self.csv_file} does not exist.")
            return False

        try:
            df = pd.read_csv(self.csv_file)

            if len(df) == 0:
                print(f"Dataset {self.csv_file} is already empty.")
                return False

            last_row = df.iloc[-1]
            last_label = last_row.iloc[0]

            df = df.iloc[:-1]

            df.to_csv(self.csv_file, index=False)

            print(f"Removed last data point with label '{last_label}' from {self.csv_file}")
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
