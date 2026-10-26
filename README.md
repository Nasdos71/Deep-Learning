# 🧠 Deep Learning Studio

An extensible, plugin-based Deep Learning and Machine Learning workbench with a modern desktop GUI, real-time terminal streaming, and GPU acceleration.

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11%20CUDA%2012.8-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🌟 Features

* **Desktop GUI Application (`app.py`)**: Built with `customtkinter`, featuring a modern dark theme, real-time unbuffered training console, and an embedded confusion matrix visualizer.
* **Plug-and-Play Model Architecture (`Models/`)**: Each algorithm is a standalone plugin. Just drop a new `.py` file into `Models/` and the system auto-registers it.
* **Dynamic Dataset Ingestion & Validation (`Data sets/`)**: Drop any CSV or image folder into `Data sets/`. The system auto-detects targets, imputes missing data, scales features, and validates sample sizes.
* **Full Hyperparameter Flexibility**:
  * Configurable **Train/Test split percentage** (10% to 50% test size slider).
  * **Random seed** for reproducible experiments.
  * Granular controls for each algorithm (e.g. $k$-neighbors, distance metric, weighting for KNN; kernels and $C$ for SVM; epochs, learning rate, batch size, and optimizers for ANN).
* **NVIDIA GPU Acceleration**: Pre-configured with PyTorch CUDA 12.8 support for modern NVIDIA GPUs (including RTX 50-series Blackwell `sm_120`).
* **Classified Outputs (`output/`)**: Automatically routes model checkpoints, confusion matrix heatmaps, and classification metrics reports into:
  $$\text{output} \longrightarrow \langle\text{dataset}\rangle \longrightarrow \langle\text{model}\rangle$$

---

## 📂 Project Structure

```text
Deep Learning/
├── app.py                      # Desktop GUI (Launcher & Visualizer)
├── run_gui.bat                 # 1-click Windows desktop launcher
├── requirements.txt            # Project dependencies
├── .gitignore                  # Git ignore rules (excludes .venv & heavy binaries)
│
├── Models/                     # PURE MODEL PLUGINS
│   ├── base_model.py           # Shared base interface & evaluation helpers
│   ├── knn.py                  # Standalone KNN algorithm
│   ├── svm.py                  # Standalone SVM algorithm
│   └── ann.py                  # Standalone PyTorch GPU Neural Network
│
├── utils/                      # CORE PIPELINE & UTILITIES
│   ├── dataset_loader.py       # Auto-discovers, validates & splits datasets
│   ├── model_discovery.py      # Auto-registers algorithm plugins
│   └── training_engine.py      # Universal CLI runner connecting data & models
│
├── output/                     # CLASSIFIED RESULTS & ARTIFACTS
│   ├── <dataset_name>/
│   │   └── <model_name>/
│   │       ├── model.joblib / ann_model.pt
│   │       ├── confusion_matrix.png
│   │       └── classification_report.txt
│   └── last_run_cm.png
│
└── Data sets/                  # DATASETS DIRECTORY
    ├── Iris.csv                # Tabular classification dataset
    └── Mnist/                  # MNIST handwritten digits dataset
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/deep-learning-studio.git
cd deep-learning-studio
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🖥️ Running the Application

### Option A: Launch the GUI (Recommended)
Double-click `run_gui.bat` or run:
```bash
python app.py
```

### Option B: Run via CLI (Headless / Scripting)
You can directly run the training engine from the command line:
```bash
# Train KNN on Iris with 70/30 split:
python utils/training_engine.py --dataset iris --model knn --test_size 0.30 --k 7

# Train SVM on Iris:
python utils/training_engine.py --dataset iris --model svm --svm_kernel rbf --svm_c 1.0

# Train GPU ANN on MNIST for 10 epochs:
python utils/training_engine.py --dataset mnist --model ann --epochs 10 --lr 0.001
```

---

## 🧩 How to Add New Assets

### Adding a New Dataset
1. Drop any classification `.csv` file into `Data sets/` (e.g. `Data sets/Wine.csv`).
2. Open the GUI and click **🔄 Rescan**. The dataset will appear in the dropdown menu.

### Adding a New Algorithm / Model
1. Create a new file in `Models/` (e.g. `Models/random_forest.py`).
2. Implement the standard interface:
```python
from Models.base_model import BaseModel, save_evaluation_results
from sklearn.ensemble import RandomForestClassifier

class RandomForestModel(BaseModel):
    MODEL_NAME = "Random Forest"
    DESCRIPTION = "Ensemble decision tree classifier"

    def train(self, data: dict, output_dir: str, n_estimators: int = 100, **kwargs) -> float:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(data["X_train"], data["y_train"])
        y_pred = model.predict(data["X_test"])
        
        return save_evaluation_results(
            output_dir=output_dir,
            dataset_name=data["name"],
            model_name="random_forest",
            y_true=data["y_test"],
            y_pred=y_pred,
            class_names=data["class_names"],
            model_obj=model,
            save_filename="random_forest_model.joblib"
        )

model_instance = RandomForestModel()
```
3. Click **🔄 Rescan** in the GUI. Your new model will immediately be ready to train on any dataset!

---

## 📜 License
MIT License. Free for academic, personal, and commercial use.
