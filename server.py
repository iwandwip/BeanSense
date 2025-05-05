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
                print(f"Received result: {post_data}")

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
            "dataset4": """NAMA,MQ135,MQ2,MQ3,MQ6
aKaw-D,123,456,789,101
aKaw-D,131,446,798,111
aKaw-D,142,467,775,98
aKaw-D,118,448,801,106
aKaw-D,118,448,801,102
aKaw-M,234,567,890,112
aKaw-M,245,582,873,108
aKaw-M,228,559,886,118
aKaw-M,239,575,895,105
aKaw-M,239,575,895,104
aKaw-L,345,678,901,123
aKaw-L,352,685,913,119
aKaw-L,337,672,898,127
aKaw-L,358,689,908,121
aKaw-L,358,689,908,124
aSem-D,211,501,811,95
aSem-D,223,512,809,91
aSem-D,208,489,820,97
aSem-D,219,508,815,93
aSem-D,219,508,815,932
aSem-M,309,598,926,131
aSem-M,315,612,935,128
aSem-M,301,587,919,136
aSem-M,320,603,930,134
aSem-M,320,603,930,1344
aSem-L,401,702,1027,153
aSem-L,413,715,1034,148
aSem-L,396,691,1019,158
aSem-L,408,709,1031,151
aSem-L,396,691,1019,158""",

            "dataset6": """NAMA,MQ135,MQ2,MQ3,MQ6,MQ138,MQ7
aKaw-D,123,456,789,101,112,131
aKaw-D,131,446,798,111,118,128
aKaw-D,142,467,775,98,108,137
aKaw-D,118,448,801,106,115,134
aKaw-D,118,448,801,106,115,1344
aKaw-M,234,567,890,112,223,242
aKaw-M,245,582,873,108,231,235
aKaw-M,228,559,886,118,218,249
aKaw-M,239,575,895,105,227,238
aKaw-M,239,575,895,105,227,2388
aKaw-L,345,678,901,123,234,353
aKaw-L,352,685,913,119,241,347
aKaw-L,337,672,898,127,228,359
aKaw-L,358,689,908,121,237,350
aKaw-L,358,689,908,121,237,3503
aSem-D,211,501,811,95,132,143
aSem-D,223,512,809,91,128,139
aSem-D,208,489,820,97,135,147
aSem-D,219,508,815,93,130,141
aSem-D,219,508,815,93,130,1412
aSem-M,309,598,926,131,241,263
aSem-M,315,612,935,128,235,257
aSem-M,301,587,919,136,248,269
aSem-M,320,603,930,134,239,261
aSem-M,320,603,930,134,239,2613
aSem-L,401,702,1027,153,347,385
aSem-L,413,715,1034,148,339,378
aSem-L,396,691,1019,158,354,392
aSem-L,408,709,1031,151,343,381
aSem-L,396,691,1019,158,354,3924""",

            "dataset8": """NAMA,MQ135,MQ2,MQ3,MQ6,MQ138,MQ7,MQ136,MQ5
aKaw-D,123,456,789,101,112,131,142,153
aKaw-D,131,446,798,111,118,128,149,158
aKaw-D,142,467,775,98,108,137,136,148
aKaw-D,118,448,801,106,115,134,145,156
aKaw-D,118,448,801,106,115,134,145,1562
aKaw-M,234,567,890,112,223,242,253,264
aKaw-M,245,582,873,108,231,235,259,271
aKaw-M,228,559,886,118,218,249,248,257
aKaw-M,239,575,895,105,227,238,261,268
aKaw-M,239,575,895,105,227,238,261,2682
aKaw-L,345,678,901,123,234,353,364,375
aKaw-L,352,685,913,119,241,347,371,382
aKaw-L,337,672,898,127,228,359,358,368
aKaw-L,358,689,908,121,237,350,372,379
aKaw-L,358,689,908,121,237,350,372,3792
aSem-D,211,501,811,95,132,143,152,159
aSem-D,223,512,809,91,128,139,148,155
aSem-D,208,489,820,97,135,147,157,163
aSem-D,219,508,815,93,130,141,150,157
aSem-D,219,508,815,93,130,141,150,1572
aSem-M,309,598,926,131,241,263,271,285
aSem-M,315,612,935,128,235,257,265,278
aSem-M,301,587,919,136,248,269,278,292
aSem-M,320,603,930,134,239,261,268,281
aSem-M,320,603,930,134,239,261,268,2812
aSem-L,401,702,1027,153,347,385,391,412
aSem-L,413,715,1034,148,339,378,384,405
aSem-L,396,691,1019,158,354,392,399,419
aSem-L,408,709,1031,151,343,381,388,409
aSem-L,396,691,1019,158,354,392,399,4192"""
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
