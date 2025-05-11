import os
import sys
import requests
import time
import json
import traceback
from ClassifierWrapper import ClassifierWrapper


class CoffeeClassifierClient:
    def __init__(self, server_url="http://localhost:5000"):
        self.server_url = server_url
        self.dataset_names = ["dataset4", "dataset6", "dataset8"]
        self.model_types = [
            "adaboost_resnet", "catboost_resnet", "lightgbm_resnet",
            "lightgbm_mobilenet", "mobilenet_iccs_lightgbm",
            "autoencoder_lightgbm", "rbf_svm_gs"
        ]
        self.wrapper = ClassifierWrapper()
        os.makedirs("datasets", exist_ok=True)

    def download_dataset(self, dataset_name):
        file_path = os.path.join("datasets", f"{dataset_name}.csv")
        url = f"{self.server_url}/read-{dataset_name}"

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(response.text)
                print(f"{file_path} saved successfully.")
                return True
            else:
                print(f"Failed to download {dataset_name}. Status: {response.status_code}")
                return False
        except Exception as e:
            print(f"Download error: {str(e)}")
            return False

    def send_result(self, result, is_final=True):
        try:
            if isinstance(result, dict):
                if "error" in result:
                    result_str = json.dumps(result)
                else:
                    formatted_result = ""
                    for key, value in result.items():
                        if key == "probabilities" and isinstance(value, dict):
                            continue
                        if isinstance(value, (int, float)) or (hasattr(value, "dtype") and "float" in str(value.dtype)):
                            formatted_value = f"{float(value):.2f}"
                        else:
                            formatted_value = str(value)
                            if "seconds" in formatted_value:
                                formatted_value = formatted_value.split()[0]
                        if formatted_result:
                            formatted_result += ","
                        formatted_result += f"{key}:{formatted_value}"

                    result_str = formatted_result
            else:
                result_str = str(result)

            print(f"Sending result: {result_str}")
            response = requests.post(f"{self.server_url}/post-result", data=result_str, timeout=10)

            if is_final:
                time.sleep(1)
                requests.post(f"{self.server_url}/post-result", data="idle", timeout=5)

            return response.status_code == 200
        except Exception as e:
            print(f"Error sending result: {str(e)}")
            return False

    def handle_train_command(self, dataset_index, method_index):
        try:
            if dataset_index < 0 or dataset_index >= len(self.dataset_names) or \
               method_index < 0 or method_index >= len(self.model_types):
                self.send_result({"error": "Invalid index values"})
                return False

            dataset_name = self.dataset_names[dataset_index]
            model_type = self.model_types[method_index]

            if not self.download_dataset(dataset_name):
                self.send_result({"error": f"Failed to download {dataset_name}.csv"})
                return False

            self.wrapper.set_dataset(dataset_name.replace("dataset", ""))
            self.wrapper.create_model(model_type)

            start_time = time.time()

            if model_type == "mobilenet_iccs_lightgbm":
                metrics = self.wrapper.train_model(use_cached_features=True)
            elif model_type == "autoencoder_lightgbm":
                metrics = self.wrapper.train_model(show_plots=False)
            else:
                metrics = self.wrapper.train_model()

            training_time = time.time() - start_time

            if metrics:
                metrics["total_training_time"] = f"{training_time:.2f} seconds"
                self.send_result(metrics)
                return True
            else:
                self.send_result({"error": "Training failed"})
                return False

        except Exception as e:
            self.send_result({"error": f"Training error: {str(e)}"})
            return False

    def handle_predict_command(self, dataset_index, method_index, input_data):
        try:
            if dataset_index < 0 or dataset_index >= len(self.dataset_names) or \
               method_index < 0 or method_index >= len(self.model_types):
                self.send_result({"error": "Invalid index values"})
                return False

            dataset_name = self.dataset_names[dataset_index]
            model_type = self.model_types[method_index]

            self.wrapper.set_dataset(dataset_name.replace("dataset", ""))
            self.wrapper.create_model(model_type)

            result = self.wrapper.predict_with_model(input_data)

            if result == 'quit' or result is None or "error" in result:
                self.send_result(result or {"error": "Prediction failed"})
                return False

            self.send_result(result)
            return True

        except Exception as e:
            self.send_result({"error": f"Prediction error: {str(e)}"})
            return False

    def run(self):
        print("Starting client. Connecting to", self.server_url)

        while True:
            try:
                response = requests.get(f"{self.server_url}/get-command", timeout=5)

                if response.status_code == 200:
                    cmd = response.text.strip()

                    if not cmd:
                        time.sleep(2)
                        continue

                    print(f"Command received: {cmd}")
                    parts = cmd.split(',')

                    if parts[0] == "train" and len(parts) == 3:
                        try:
                            dataset_index = int(parts[1])
                            method_index = int(parts[2])
                            self.handle_train_command(dataset_index, method_index)
                        except ValueError:
                            self.send_result({"error": "Invalid command format"})

                    elif parts[0] == "predict" and len(parts) >= 4:
                        try:
                            dataset_index = int(parts[1])
                            method_index = int(parts[2])
                            input_data = ",".join(parts[3:])
                            self.handle_predict_command(dataset_index, method_index, input_data)
                        except ValueError:
                            self.send_result({"error": "Invalid command format"})

                    else:
                        self.send_result({"error": "Unknown command format"})

            except requests.exceptions.ConnectionError:
                print("Connection to server failed. Retrying...")
            except requests.exceptions.Timeout:
                print("Request timed out. Retrying...")
            except Exception as e:
                print(f"Error: {str(e)}")

            time.sleep(2)


if __name__ == "__main__":
    try:
        client = CoffeeClassifierClient(server_url="http://192.168.4.1")
        client.run()
    except KeyboardInterrupt:
        print("Shutting down client...")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
