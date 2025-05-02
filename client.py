# request.py
import requests
import time
from main_controller import MainController

esp32_ip = "http://192.168.4.1"
datasetName = ["dataset4", "dataset6", "dataset8"]
controller = MainController()

import os

def download_and_save(dataset_name):
    folder_path = "datasets"
    os.makedirs(folder_path, exist_ok=True)  # Buat folder jika belum ada

    url = f"{esp32_ip}/read-{dataset_name}"
    response = requests.get(url)

    if response.status_code == 200:
        file_path = os.path.join(folder_path, f"{dataset_name}.csv")
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(response.text)
        print(f"{file_path} saved successfully.")
    else:
        print(f"Failed to download {dataset_name}. Status code: {response.status_code}")

def main():
    print("Polling ESP32 for commands...\n")
    while True:
        try:
            response = requests.get(f"{esp32_ip}/get-command", timeout=5)
            if response.status_code == 200:
                cmd = response.text.strip()
                print(f"📥 Command received: {cmd}")
                
                parts = cmd.split(',')

                if parts[0] == "train" and len(parts) == 3:
                    dataset_index = int(parts[1])
                    metode_index = int(parts[2])

                    # 1. Download dataset
                    if 0 <= dataset_index < len(datasetName):
                        dataset_file = datasetName[dataset_index]
                        download_and_save(dataset_file)
                        controller.setup_dataset(dataset_index)
                        # 2. Train
                        result = controller.train(metode_index)
                        result_str = ",".join(f"{k}:{v}" for k, v in result.items())
                        # 3. Send result
                        r = requests.post(f"{esp32_ip}/post-result", data=result_str, timeout=5)
                        print(f"📤 Result sent: {result_str} | Response: {r.text}")
                        time.sleep(2)
                        requests.post(f"{esp32_ip}/post-result", data="idle", timeout=5)
                    else:
                        print("❌ Invalid dataset index.")

                elif parts[0] == "predict" and len(parts) >= 4:
                    dataset_index = int(parts[1])
                    metode_index = int(parts[2])
                    input_data = ",".join(parts[3:])

                    controller.setup_dataset(dataset_index)
                    result = controller.predict(metode_index, input_data)

                    if result is None or "error" in result:
                        result_str = f"error:{result.get('error', 'Unknown error')}"
                    else:
                        pred = result['predicted_class']
                        result_str = f"predicted_class:{pred}"
                        for cls, prob in result['probabilities'].items():
                            result_str += f",{cls}:{prob:.4f}"

                    r = requests.post(f"{esp32_ip}/post-result", data=result_str, timeout=5)
                    print(f"📤 Prediction result sent: {result_str} | Response: {r.text}")

                else:
                    print("⚠️ Unknown or malformed command format.")

        except Exception as e:
            print("wait")

        time.sleep(2)

if __name__ == "__main__":
    main()
