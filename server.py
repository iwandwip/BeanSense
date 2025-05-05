import inquirer
import threading
import queue
import time
import json
import socket
import http.server
import socketserver
import urllib.parse
from http import HTTPStatus


class SimpleHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, server_instance=None, **kwargs):
        self.server_instance = server_instance
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == '/get-command':
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()

            if not self.server_instance.command_queue.empty():
                command = self.server_instance.command_queue.get()
                print(f"Sending command: {command}")
                self.server_instance.current_status = "busy"
                self.wfile.write(command.encode('utf-8'))
            else:
                self.wfile.write("".encode('utf-8'))

        elif self.path.startswith('/read-dataset'):
            dataset = self.path.split('-')[1]
            if dataset in self.server_instance.sample_data:
                self.send_response(HTTPStatus.OK)
                self.send_header('Content-Type', 'text/csv')
                self.end_headers()
                self.wfile.write(self.server_instance.sample_data[dataset].encode('utf-8'))
            else:
                self.send_response(HTTPStatus.NOT_FOUND)
                self.end_headers()
                self.wfile.write(b"Dataset not found")
        else:
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()
            self.wfile.write(b"Not found")

    def do_POST(self):
        if self.path == '/post-result':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')

            if post_data == "idle":
                self.server_instance.current_status = "idle"
            else:
                self.server_instance.result_queue.put(post_data)
                print(f"Received result:")
                print(json.dumps(json.loads(post_data), indent=4, sort_keys=True))

            self.send_response(HTTPStatus.OK)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()
            self.wfile.write(b"Not found")

    def log_message(self, format, *args):
        # Suppressing log messages from the server
        pass


class CoffeeClassifierServer:
    def __init__(self, host="localhost", port=5000):
        self.host = host
        self.port = port
        self.command_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.current_status = "idle"
        self.setup_sample_data()

        # Server handler with access to this instance
        handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(
            *args, server_instance=self, **kwargs)

        self.httpd = socketserver.TCPServer((host, port), handler)

    def setup_sample_data(self):
        self.sample_data = {
            "dataset4": "NAMA,MQ135,MQ2,MQ3,MQ6\naKaw-D,123,456,789,101\naKaw-M,234,567,890,112\naKaw-L,345,678,901,123\n",
            "dataset6": "NAMA,MQ135,MQ2,MQ3,MQ6,MQ138,MQ7\naKaw-D,123,456,789,101,112,131\naKaw-M,234,567,890,112,223,242\naKaw-L,345,678,901,123,234,353\n",
            "dataset8": "NAMA,MQ135,MQ2,MQ3,MQ6,MQ138,MQ7,MQ136,MQ5\naKaw-D,123,456,789,101,112,131,142,153\naKaw-M,234,567,890,112,223,242,253,264\naKaw-L,345,678,901,123,234,353,364,375\n"
        }

    def start_server(self):
        server_thread = threading.Thread(target=self.httpd.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        print(f"Server started on {self.host}:{self.port}")

    def add_command(self, command):
        self.command_queue.put(command)
        print(f"Added command: {command}")

    def get_latest_result(self, wait_time=30):
        start_time = time.time()
        while time.time() - start_time < wait_time:
            if not self.result_queue.empty():
                result = self.result_queue.get()
                try:
                    return json.loads(result)
                except:
                    return result
            time.sleep(0.5)
        return None

    def run_interactive(self):
        self.start_server()

        datasets = ["dataset4", "dataset6", "dataset8"]
        models = [
            "adaboost_resnet", "catboost_resnet", "lightgbm_resnet",
            "lightgbm_mobilenet", "mobilenet_iccs_lightgbm",
            "autoencoder_lightgbm", "rbf_svm_gs"
        ]

        while True:
            questions = [
                inquirer.List('action',
                              message="Select an action:",
                              choices=[
                                  ('Train model', 'train'),
                                  ('Make prediction', 'predict'),
                                  ('Check status', 'status'),
                                  ('View latest result', 'result'),
                                  ('Exit', 'exit')
                              ],
                              )
            ]
            answers = inquirer.prompt(questions)
            action = answers['action']

            if action == "exit":
                print("Shutting down server...")
                self.httpd.shutdown()
                break

            elif action == "status":
                print(f"Current status: {self.current_status}")

            elif action == "result":
                result = self.get_latest_result(wait_time=0.1)
                if result:
                    if isinstance(result, dict):
                        for key, value in result.items():
                            print(f"{key}: {value}")
                    else:
                        print(f"Latest result: {result}")
                else:
                    print("No results available")

            elif action == "train":
                dataset_questions = [
                    inquirer.List('dataset',
                                  message="Select dataset to use:",
                                  choices=[
                                      (f'dataset4 (4 sensors)', 0),
                                      (f'dataset6 (6 sensors)', 1),
                                      (f'dataset8 (8 sensors)', 2)
                                  ],
                                  )
                ]
                dataset_answer = inquirer.prompt(dataset_questions)
                dataset_idx = dataset_answer['dataset']

                model_questions = [
                    inquirer.List('model',
                                  message="Select model type:",
                                  choices=[
                                      ('AdaBoost + ResNet', 0),
                                      ('CatBoost + ResNet', 1),
                                      ('LightGBM + ResNet', 2),
                                      ('LightGBM + MobileNet', 3),
                                      ('MobileNet + ICCS + LightGBM', 4),
                                      ('Autoencoder + LightGBM', 5),
                                      ('RBF SVM + GridSearch', 6)
                                  ],
                                  )
                ]
                model_answer = inquirer.prompt(model_questions)
                model_idx = model_answer['model']

                train_cmd = f"train,{dataset_idx},{model_idx}"
                self.add_command(train_cmd)
                print("Training command sent. Check status or results later.")

            elif action == "predict":
                dataset_questions = [
                    inquirer.List('dataset',
                                  message="Select dataset:",
                                  choices=[
                                      (f'dataset4 (4 sensors)', 0),
                                      (f'dataset6 (6 sensors)', 1),
                                      (f'dataset8 (8 sensors)', 2)
                                  ],
                                  )
                ]
                dataset_answer = inquirer.prompt(dataset_questions)
                dataset_idx = dataset_answer['dataset']

                model_questions = [
                    inquirer.List('model',
                                  message="Select model type:",
                                  choices=[
                                      ('AdaBoost + ResNet', 0),
                                      ('CatBoost + ResNet', 1),
                                      ('LightGBM + ResNet', 2),
                                      ('LightGBM + MobileNet', 3),
                                      ('MobileNet + ICCS + LightGBM', 4),
                                      ('Autoencoder + LightGBM', 5),
                                      ('RBF SVM + GridSearch', 6)
                                  ],
                                  )
                ]
                model_answer = inquirer.prompt(model_questions)
                model_idx = model_answer['model']

                data_questions = [
                    inquirer.Text('data',
                                  message="Enter sensor values (comma separated):")
                ]
                data_answer = inquirer.prompt(data_questions)
                sensor_data = data_answer['data']

                predict_cmd = f"predict,{dataset_idx},{model_idx},{sensor_data}"
                self.add_command(predict_cmd)
                print("Prediction command sent. Check results later.")


def main():
    server = CoffeeClassifierServer()
    try:
        server.run_interactive()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.httpd.shutdown()


if __name__ == "__main__":
    main()
