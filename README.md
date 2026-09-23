# BeanSense

BeanSense classifies coffee samples from readings produced by MQ gas sensors. The repository contains the Python training and prediction code, an HTTP command bridge, datasets, trained models, and firmware for an ESP32 device with a TFT display.

The project is still an experimental system. The Python code supports local operation and a distributed setup. The ESP32 firmware provides the device-side interface for collecting samples and sending training or prediction commands.

## What the project can do

- Train seven classification pipelines on datasets with 4, 6, or 8 sensors.
- Predict a coffee label from a new sensor reading.
- Manage dataset rows from the local CLI or the ESP32 interface.
- Extract features with ResNet18 or MobileNetV2 before classification.
- Use an autoencoder or ICCS feature selection as alternative preprocessing steps.
- Run training through an HTTP command queue between a server and a client.
- Save trained models as pickle files and store training metrics as CSV files.

## System overview

The repository contains three ways to use BeanSense:

### Local Python CLI

`CoffeeClassifierMain.py` runs training, prediction, and dataset management in one process. It reads datasets from `datasets/` and writes models to `model/`.

### Python server and client

`CoffeeClassifierServer.py` exposes a small HTTP command queue. `CoffeeClassifierClient.py` polls that queue, downloads a dataset, runs the selected model, and posts the result back.

### ESP32 device

The files in `utils/dendaFirmware/` implement the device interface. The firmware reads MQ sensors, stores CSV files in SPIFFS, serves the dataset and command endpoints, and displays results on the TFT screen.

The Python server and ESP32 firmware implement similar endpoints, but they are separate implementations. Choose one server for a deployment instead of running both for the same client.

## Requirements

- Python 3.10 or newer is recommended.
- Python dependencies listed in [`utils/requirements.txt`](utils/requirements.txt).
- A working PyTorch and torchvision installation.
- Internet access on first use if torchvision must download ResNet18 or MobileNetV2 weights.
- An ESP32 toolchain and the libraries listed in `utils/dendaFirmware/library.h` when building the firmware.

The root `requirements.txt` is a legacy dependency export. Use `utils/requirements.txt` for a normal Python setup.

## Installation

```bash
git clone <repository-url>
cd BeanSense

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r utils/requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Quick start: local CLI

Run:

```bash
python3 CoffeeClassifierMain.py
```

The menu provides four operations:

1. Train all models on all datasets.
2. Train one model on one dataset.
3. Predict from sensor values.
4. Add or remove dataset rows.

Training writes a model to `model/`. Training-all mode writes a timestamped report to `results/`.

## Python server and client

Start the Python server in one terminal:

```bash
python3 CoffeeClassifierServer.py
```

The default server address is `http://localhost:5000`. The server stores commands in memory and includes small sample datasets for development.

Run a client from another Python process with the same server URL:

```python
from CoffeeClassifierClient import CoffeeClassifierClient

client = CoffeeClassifierClient(server_url="http://localhost:5000")
client.run()
```

For the ESP32 setup, use the device address instead:

```python
from CoffeeClassifierClient import CoffeeClassifierClient

client = CoffeeClassifierClient(server_url="http://192.168.4.1")
client.run()
```

The client accepts two command types:

```text
train,<dataset-index>,<model-index>
predict,<dataset-index>,<model-index>,<sensor-value-1>,<sensor-value-2>,...
```

Dataset indexes are `0` for dataset4, `1` for dataset6, and `2` for dataset8. Model indexes follow the order in `CoffeeClassifierClient.model_types`.

## Direct model usage

This example trains the MobileNet plus LightGBM pipeline on the four-sensor dataset:

```python
from LightGBMMobileNet import MobileNetLightGBMModel

model = MobileNetLightGBMModel()
X, y = model.load_data("datasets/dataset4.csv")
metrics = model.train(X, y, use_kfold=True)

prediction = model.predict_single([123, 456, 789, 101])
print(prediction)

model.save_model("model/example_mobilenet_lightgbm.pkl")
```

For prediction from text input, use the model's `predict_custom_input` method after loading a trained model.

## Datasets

Each CSV uses `NAMA` as its label column. Sensor values follow this order:

| File | Sensors |
|---|---|
| `datasets/dataset4.csv` | `MQ135`, `MQ2`, `MQ3`, `MQ6` |
| `datasets/dataset6.csv` | `MQ135`, `MQ2`, `MQ3`, `MQ6`, `MQ138`, `MQ7` |
| `datasets/dataset8.csv` | `MQ135`, `MQ2`, `MQ3`, `MQ6`, `MQ138`, `MQ7`, `MQ136`, `MQ5` |

Labels use this format:

```text
<variety>-<roast>
```

Examples include:

```text
aKaw-D   # Arabica Kawisari, dark roast
aSem-M   # Arabica Semeru, medium roast
rGed-L   # Robusta Gedung, light roast
rTir-D   # Robusta Tirtoyudo, dark roast
```

`datasets/origin/` contains the original dataset copies. The files directly under `datasets/` are the files used by the Python code and may be reordered or edited by the dataset management functions.

When adding prediction data, provide exactly 4, 6, or 8 numeric values for the selected dataset. Keep the sensor order unchanged.

## Available models

| Key | Pipeline |
|---|---|
| `adaboost_resnet` | ResNet18 feature extraction + AdaBoost |
| `catboost_resnet` | ResNet18 feature extraction + CatBoost |
| `lightgbm_resnet` | ResNet18 feature extraction + LightGBM |
| `lightgbm_mobilenet` | MobileNetV2 feature extraction + LightGBM |
| `mobilenet_iccs_lightgbm` | MobileNetV2 + ICCS feature selection + LightGBM |
| `autoencoder_lightgbm` | Autoencoder compression + LightGBM |
| `rbf_svm_gs` | RBF SVM with grid search |

The default wrapper configuration uses five-fold stratified cross-validation. Hyperparameters are defined in `ClassifierWrapper.py`.

## Model and experiment files

| Path | Purpose |
|---|---|
| `model/` | Trained pickle models used by the wrapper |
| `model/default_model/` | Additional copies of trained models |
| `cache/` | Cached MobileNet features for ICCS training |
| `results/` | Timestamped training reports |
| `catboost_info/` | CatBoost training logs |

These files depend on the Python package versions used during training. A pickle model should only be loaded from a trusted source.

## Repository layout

```text
BeanSense/
├── AdaBoostClassifier.py
├── AutoencoderLightGBM.py
├── CatBoostClassifier.py
├── ClassifierWrapper.py
├── CoffeeClassifierClient.py
├── CoffeeClassifierMain.py
├── CoffeeClassifierServer.py
├── LightGBMMobileNet.py
├── LightGBMResNet.py
├── MobileNetICCSLightGBM.py
├── RBFSVMGridSearch.py
├── datasets/
│   ├── origin/
│   ├── dataset4.csv
│   ├── dataset6.csv
│   └── dataset8.csv
├── model/
├── cache/
├── results/
└── utils/
    ├── dendaFirmware/
    ├── main.py
    ├── main_controller.py
    └── requirements.txt
```

The repository also contains older backup files under `.backup/`. They are not part of the current Python entry points.

## Known limitations

- The project has no automated test suite yet.
- Training and prediction depend on relative paths from the repository root.
- The Python and ESP32 server implementations do not share a formal protocol schema.
- The Python client default and its executable entry point target different server addresses.
- Training metrics are experiment outputs, not a published benchmark.
- Cross-validation currently fits preprocessing before the folds, so reported metrics need careful interpretation.
- The model files do not include a complete environment lockfile or dataset version record.
- The project does not currently declare a software license.

## Authors

- Iwan Dwi — iwan.dwp@gmail.com
- Ahmad Zainul — ahmadzainularifin6@gmail.com
