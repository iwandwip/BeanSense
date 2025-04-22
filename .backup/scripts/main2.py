#  metode1 = scale-adaboost-resnet
#  metode2 = scale-catboost-resnet
#  metode3 = scale-mobilenet-iccs-lightGBMv2
#  metode4 = scale-lightGBMv2-mobilenetv2
#  metode5 = scale-lightGBMv2-resnet
#  metode6 = scale-RBF-GS-SVM
#  metode7 = scale-AutoEncoder+LGBM

from metode1 import predict1, predict2, predict3, predict4, predict5, predict6, predict7

# Mapping predict dan nama-namanya
predict_map = {
    1: predict1.proses,
    2: predict2.proses,
    3: predict3.proses,
    4: predict4.proses,
    5: predict5.proses,
    6: predict6.proses,
    7: predict7.proses
}

method_names = [
    "scale-adaboost-resnet",
    "scale-catboost-resnet",
    "scale-mobilenet-iccs-lightGBMv2",
    "scale-lightGBMv2-mobilenetv2",
    "scale-lightGBMv2-resnet",
    "scale-RBF-GS-SVM",
    "scale-AutoEncoder+LGBM"
]

dataset_names = [
    "dataset4.csv",
    "dataset6.csv",
    "dataset8.csv",
]

def parse_input(input_string):
    print(input_string)
    input_string = input_string.strip()
    parts = input_string.split(',')

    command = parts[0].lower()

    if command == "predict":
        method = int(parts[1])
        data = ",".join(parts[2:])
        return {"command": "predict", "method": method, "data": data}

    elif command == "train":
        method = int(parts[1])
        index_dataset = int(part[2])
        return {"command": "train", "method": method, "index_dataset": index_dataset}

    elif command == "create":
        label = parts[1]
        data = ",".join(parts[2:])
        return {"command": "create", "label": label, "data": data}

    elif command == "delete":
        label = parts[1]
        return {"command": "delete", "label": label}

    else:
        raise ValueError("Perintah tidak dikenali")

def main():
    input_string = "predict\n"
    parsed = parse_input(input_string)

    if parsed["command"] == "predict":
        method = parsed["method"]
        data = parsed["data"]

        if 1 <= method <= 7:
            print(f">> command: predict")
            print(f">> Menggunakan Metode ke {method} : {method_names[method - 1]}")
            print(f">> Data: {data}")
        else:
            print("Metode tidak valid")
        
        match method:
            case 1:
                result = predict1.proses(data)
            case 2:
                result = predict2.proses(data)
            case 3:
                result = predict3.proses(data)
            case 4:
                result = predict4.proses(data)
            case 5:
                result = predict5.proses(data)
            case 6:
                result = predict6.proses(data)
            case 7:
                result = predict7.proses(data)
            case _:
                result = "Metode tidak valid"

        if isinstance(result, dict):
            print("\n=== Hasil Prediksi ===")
            print(f"Label Prediksi     : {result['predicted_label']}")
            print(f"Accuracy Rata-rata : {result['average_accuracy']}")
            print(f"F1-score Rata-rata : {result['average_f1_score']}")
            print(f"AUC Rata-rata      : {result['average_auc']}")
            print(f"Waktu Eksekusi     : {result['time_used']} detik")
            print(f"Memory Digunakan   : {result['memory_used_MB']} MB (peak: {result['peak_memory_MB']} MB)")
        else:
            print(result)

    elif parsed["train"] == "train":
        method = parsed["method"]
        index_dataset = parsed["index_dataset"]
        print(f"Data {data} berhasil dibuat untuk label {label}")

        if 1 <= method <= 7:
            print(f">> command: train")
            print(f">> Menggunakan Metode ke {method} : {method_names[method - 1]}")
            print(f">> Dataset: {dataset_names[index_dataset - 1]}")
        else:
            print("Metode tidak valid")


    elif parsed["command"] == "create":
        label = parsed["label"]
        data = parsed["data"]
        print(f"Data {data} berhasil dibuat untuk label {label}")

    elif parsed["command"] == "delete":
        label = parsed["label"]
        print(f"Label {label} berhasil dihapus")


if __name__ == "__main__":
    main()
