import requests
import time
import random

esp32_ip = "http://192.168.4.1"

datasetName = ["dataset4", "dataset6", "dataset8"]

def download_and_save(dataset_name):
    url = f"{esp32_ip}/read-{dataset_name}"
    response = requests.get(url)

    if response.status_code == 200:
        with open(f"{dataset_name}.csv", 'w', encoding='utf-8') as file:
            file.write(response.text)
        print(f"{dataset_name}.csv saved successfully.")
    else:
        print(f"Failed to download {dataset_name}. Status code: {response.status_code}")

def simulate_task(command):
    print(f"🚀 Processing: {command}")
    time.sleep(2)
    data = "accuracy_avg:0.91,accuracy_f1_score:0.87"
    headers = {"Content-Type": "text/plain"}
    r = requests.post(f"{esp32_ip}/post-result", data=data, headers=headers)
    print("Status:", r.status_code)
    print("Response:", r.text)    
    return data

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
                    if 0 <= dataset_index < len(datasetName):
                        dataset_file = datasetName[dataset_index]
                        download_and_save(dataset_file)
                    else:
                        print("❌ Invalid dataset index for training.")
                
                elif parts[0] == "predict" and len(parts) == 7:
                    # parts[1] = datasetIndex, parts[2] = metodeIndex, rest = ints
                    acc = simulate_task(cmd)
                    acc_payload = f"accuracy:{acc}"
                    r = requests.post(f"{esp32_ip}/result", data=acc_payload, timeout=5)
                    print(f"📤 Result sent to ESP32: {acc_payload} | Response: {r.text}")
                
                else:
                    print("⚠️ Unknown or malformed command format.")
            
            elif response.status_code == 204:
                print("⏳ No command. Waiting...")
        
        except Exception as e:
            print("❌ Error:", e)

        time.sleep(2)

if __name__ == "__main__":
    main()
