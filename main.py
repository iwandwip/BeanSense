from AdaBoostClassifier import AdaBoostResNetModel
from CatBoostClassifier import CatBoostResNetModel
from LightGBMResNet import LightGBMResNetModel
from LightGBMMobileNet import MobileNetLightGBMModel
from MobileNetICCSLightGBM import MobileNetICCSLightGBMModel
from AutoencoderLightGBM import AutoencoderLightGBMModel
from RBFSVMGridSearch import RBFSVMGridSearchModel

import inquirer
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
        self.operation = None
        self.labels = [
            'aKaw-D', 'aKaw-L', 'aKaw-M',
            'aSem-D', 'aSem-L', 'aSem-M',
            'rGed-D', 'rGed-L', 'rGed-M',
            'rTir-D', 'rTir-L', 'rTir-M'
        ]

    def select_dataset(self):
        dataset_questions = [
            inquirer.List('dataset',
                          message="Select dataset to use:",
                          choices=[
                              ('dataset4.csv (4 sensors)', '4'),
                              ('dataset6.csv (6 sensors)', '6'),
                              ('dataset8.csv (8 sensors)', '8'),
                              ('Exit', 'exit')
                          ],
                          )
        ]
        dataset_answer = inquirer.prompt(dataset_questions)
        self.dataset_choice = dataset_answer['dataset']

        if self.dataset_choice == "exit":
            return "exit"

        if self.dataset_choice == "4":
            self.csv_file = "datasets/dataset4.csv"
        elif self.dataset_choice == "6":
            self.csv_file = "datasets/dataset6.csv"
        elif self.dataset_choice == "8":
            self.csv_file = "datasets/dataset8.csv"

        print(f"Using dataset: {self.csv_file}")
        return self.dataset_choice

    def select_operation(self):
        operation_questions = [
            inquirer.List('operation',
                          message="Select operation:",
                          choices=[
                              ('Train model', 'train'),
                              ('Predict with model', 'predict'),
                              ('Add dataset', 'add_data'),
                              ('Pop dataset', 'pop_data'),
                              ('Delete dataset', 'delete_data'),
                              ('Back', 'back')
                          ],
                          )
        ]
        operation_answer = inquirer.prompt(operation_questions)
        self.operation = operation_answer['operation']

        return self.operation

    def select_model_type(self):
        model_questions = [
            inquirer.List('model_type',
                          message=f"Select model type for {self.operation}:",
                          choices=[
                              ('AdaBoost + ResNet', 'adaboost_resnet'),
                              ('CatBoost + ResNet', 'catboost_resnet'),
                              ('LightGBM + ResNet', 'lightgbm_resnet'),
                              ('LightGBM + MobileNet', 'lightgbm_mobilenet'),
                              ('MobileNet + ICCS + LightGBM', 'mobilenet_iccs_lightgbm'),
                              ('Autoencoder + LightGBM', 'autoencoder_lightgbm'),
                              ('RBF SVM + GridSearch', 'rbf_svm_gs'),
                              ('Back', 'back')
                          ],
                          )
        ]
        model_answer = inquirer.prompt(model_questions)
        self.model_type = model_answer['model_type']

        if self.model_type == 'back':
            return 'back'

        self.create_model_instance()

        if self.operation == 'train':
            self.train_model()
        elif self.operation == 'predict':
            self.predict_with_model()

        return self.model_type

    def create_model_instance(self):
        if self.model_type == 'adaboost_resnet':
            self.current_model = AdaBoostResNetModel(n_estimators=100, learning_rate=0.1)
        elif self.model_type == 'catboost_resnet':
            gpu_questions = [
                inquirer.Confirm('use_gpu',
                                 message="Use GPU for training (if available)?",
                                 default=False)
            ]
            gpu_answer = inquirer.prompt(gpu_questions)
            use_gpu = gpu_answer['use_gpu']
            self.current_model = CatBoostResNetModel(iterations=100, learning_rate=0.1, depth=6, use_gpu=use_gpu)
        elif self.model_type == 'lightgbm_resnet':
            self.current_model = LightGBMResNetModel(n_estimators=100, learning_rate=0.1, max_depth=6)
        elif self.model_type == 'lightgbm_mobilenet':
            self.current_model = MobileNetLightGBMModel(
                n_estimators=120,
                learning_rate=0.01,
                max_depth=5,
                reg_alpha=0.1,
                reg_lambda=0.1,
                min_child_samples=20
            )
        elif self.model_type == 'mobilenet_iccs_lightgbm':
            self.current_model = MobileNetICCSLightGBMModel(
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
        elif self.model_type == 'autoencoder_lightgbm':
            self.current_model = AutoencoderLightGBMModel(
                encoded_dim=16,
                ae_epochs=100,
                n_estimators=150,
                learning_rate=0.05,
                max_depth=7
            )
        elif self.model_type == 'rbf_svm_gs':
            self.current_model = RBFSVMGridSearchModel(
                C_range=np.logspace(-2, 2, 5),
                gamma_range=np.logspace(-4, 0, 5),
                random_state=42
            )

        return self.current_model

    def train_model(self):
        print(f"Training {self.model_type} model on {self.csv_file}...")
        X, y = self.current_model.load_data(self.csv_file)

        if X is not None and y is not None:
            print("Data loaded successfully, starting training...")

            if self.model_type == 'mobilenet_iccs_lightgbm':
                cache_questions = [
                    inquirer.Confirm('use_cached',
                                     message="Use cached features if available?",
                                     default=True)
                ]
                cache_answer = inquirer.prompt(cache_questions)
                use_cached = cache_answer['use_cached']
                feature_cache = f"cache/features_{self.dataset_choice}.npz"
                self.current_model.train(X, y, use_kfold=True, use_cached_features=use_cached, feature_cache_file=feature_cache)
            elif self.model_type == 'autoencoder_lightgbm':
                plot_questions = [
                    inquirer.Confirm('show_plots',
                                     message="Show evaluation plots?",
                                     default=False)
                ]
                plot_answer = inquirer.prompt(plot_questions)
                show_plots = plot_answer['show_plots']
                self.current_model.train(X, y, use_kfold=True, show_plots=show_plots)
            else:
                self.current_model.train(X, y, use_kfold=True)

            model_file = self.get_model_filename()
            self.current_model.save_model(model_file)

            print("Testing prediction:")
            test_data = X[:1]
            prediction = self.current_model.predict(test_data)
            print(f"Prediction: {prediction}")

            print("\n" + "-"*50 + "\n")
            return True
        else:
            print("Failed to load data properly")
            print("\n" + "-"*50 + "\n")
            return False

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

    def predict_with_model(self):
        model_file = self.get_model_filename()
        try:
            self.current_model.load_model(model_file)
            print(f"Model successfully loaded from {model_file}")
        except FileNotFoundError:
            print(f"Model file {model_file} not found. Train a model first.")
            print("\n" + "-"*50 + "\n")
            return
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\n" + "-"*50 + "\n")
            return

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

            result = self.current_model.predict_custom_input(input_string)

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

        print("\n" + "-"*50 + "\n")

    def select_label(self):
        label_choices = [(label, label) for label in self.labels]

        label_questions = [
            inquirer.List('label',
                          message="Select label:",
                          choices=label_choices,
                          )
        ]
        label_answer = inquirer.prompt(label_questions)
        selected_label = label_answer['label']

        return selected_label

    def add_dataset(self):
        if not os.path.exists(self.csv_file):
            print(f"Dataset file {self.csv_file} does not exist. Creating new file.")
            os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)

            num_features = int(self.dataset_choice)
            columns = ['label'] + [f'feature{i+1}' for i in range(num_features)]
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.csv_file, index=False)

        selected_label = self.select_label()

        num_features = int(self.dataset_choice)

        print(f"Enter {num_features} sensor values (comma separated):")
        sensor_values = input().strip()

        try:
            values = [int(float(x)) for x in sensor_values.split(',')]

            if len(values) != num_features:
                print(f"Error: Expected {num_features} values, but got {len(values)}.")
                print("\n" + "-"*50 + "\n")
                return

            df = pd.read_csv(self.csv_file)

            new_row_data = {df.columns[0]: selected_label}
            for i, value in enumerate(values):
                new_row_data[df.columns[i+1]] = value

            new_row = pd.DataFrame([new_row_data])
            df = pd.concat([df, new_row], ignore_index=True)

            df.to_csv(self.csv_file, index=False)

            print(f"Added new data point with label '{selected_label}' to {self.csv_file}")
            print(f"Total records now: {len(df)}")

        except ValueError:
            print("Error: Invalid sensor values. Please enter comma-separated numbers.")

        print("\n" + "-"*50 + "\n")

    def pop_dataset(self):
        if not os.path.exists(self.csv_file):
            print(f"Dataset file {self.csv_file} does not exist.")
            print("\n" + "-"*50 + "\n")
            return

        try:
            df = pd.read_csv(self.csv_file)

            if len(df) == 0:
                print(f"Dataset {self.csv_file} is already empty.")
                print("\n" + "-"*50 + "\n")
                return

            last_row = df.iloc[-1]
            last_label = last_row.iloc[0]

            df = df.iloc[:-1]

            df.to_csv(self.csv_file, index=False)

            print(f"Removed last data point with label '{last_label}' from {self.csv_file}")
            print(f"Total records now: {len(df)}")

        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())

        print("\n" + "-"*50 + "\n")

    def delete_dataset(self):
        if not os.path.exists(self.csv_file):
            print(f"Dataset file {self.csv_file} does not exist.")
            print("\n" + "-"*50 + "\n")
            return

        selected_label = self.select_label()

        try:
            df = pd.read_csv(self.csv_file)

            label_column = df.columns[0]

            count_label = len(df[df[label_column] == selected_label])

            if count_label == 0:
                print(f"No data points with label '{selected_label}' found in {self.csv_file}")
                print("\n" + "-"*50 + "\n")
                return

            confirm_questions = [
                inquirer.Confirm('confirm',
                                 message=f"Are you sure you want to delete all {count_label} data points with label '{selected_label}'?",
                                 default=False)
            ]
            confirm_answer = inquirer.prompt(confirm_questions)

            if not confirm_answer['confirm']:
                print("Operation cancelled.")
                print("\n" + "-"*50 + "\n")
                return

            df = df[df[label_column] != selected_label]

            df.to_csv(self.csv_file, index=False)

            print(f"Deleted {count_label} data points with label '{selected_label}' from {self.csv_file}")
            print(f"Total records now: {len(df)}")

        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())

        print("\n" + "-"*50 + "\n")

    def run(self):
        while True:
            dataset_result = self.select_dataset()

            if dataset_result == "exit":
                break

            while True:
                operation = self.select_operation()

                if operation == "back":
                    break

                if operation == "add_data":
                    self.add_dataset()
                elif operation == "pop_data":
                    self.pop_dataset()
                elif operation == "delete_data":
                    self.delete_dataset()
                else:
                    while True:
                        model_result = self.select_model_type()

                        if model_result == "back":
                            break


def main():
    try:
        wrapper = ClassifierWrapper()
        wrapper.run()
        print("Exiting program...")
    except Exception as e:
        print(f"Error occurred: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
