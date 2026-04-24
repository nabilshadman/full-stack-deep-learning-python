# Full-Stack Deep Learning with Python

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.x%20%7C%203.x-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-6B4FBB)](https://optuna.org/)
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?logo=google-colab&logoColor=white)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-Educational%20Use%20Only-lightgrey)](#license)

Exercise files for the LinkedIn Learning course **[Full-Stack Deep Learning with Python](https://www.linkedin.com/learning/full-stack-deep-learning-with-python-34351133/)**, instructed by Janani Ravi (Loonycorn). The course covers the end-to-end ML lifecycle — from data preparation and model training to experiment tracking, hyperparameter optimization, and local model serving.

---

## Contents

```
├── datasets/
│   └── (EMNIST letters train/test CSV files)
├── demo_01_EMNISTClassificationUsingDNN.ipynb
├── demo_02_EMNISTClassificationUsingCNN.ipynb
└── demo_03_ModelDeployment.ipynb
```

| Notebook | Description |
|---|---|
| `demo_01_EMNISTClassificationUsingDNN.ipynb` | Loads and explores the EMNIST dataset, builds a DNN image classifier, and introduces MLflow experiment tracking with metrics and artifact logging. |
| `demo_02_EMNISTClassificationUsingCNN.ipynb` | Implements a CNN-based classifier using PyTorch Lightning, logs runs via MLflow autologging, registers the trained model, and achieves ~91.7% test accuracy. |
| `demo_03_ModelDeployment.ipynb` | Downloads a registered MLflow model, sets up a local virtual environment, and serves the model as a REST endpoint using `mlflow models serve`. |

---

## Prerequisites

- Python 3.10 or later
- A Google account (notebooks are designed for Google Colab with Google Drive)
- An [ngrok](https://ngrok.com/) account and auth token (to tunnel the MLflow UI from Colab)
- The EMNIST letters dataset from [Kaggle](https://www.kaggle.com/datasets/crawford/emnist) — `emnist-letters-train.csv` and `emnist-letters-test.csv`

---

## Setup

### Google Colab (recommended)

The first two notebooks are designed to run on Google Colab with a T4 GPU runtime.

1. Upload the EMNIST CSV files to your Google Drive under `MyDrive/emnist_data/`.
2. Open the notebook in Colab and set the runtime to **T4 GPU** (`Runtime > Change runtime type`).
3. Run the installation cells at the top of each notebook. All required packages are installed inline.
4. Set your ngrok auth token in the cell that references `NGROK_AUTH_TOKEN` before starting the MLflow UI.

### Local Setup (demo_03 only)

Model deployment is intended to run locally. From your terminal:

```bash
mkdir full_stack_deep_learning && cd full_stack_deep_learning
python3 -m venv fsdl_venv
source fsdl_venv/bin/activate          # Windows: fsdl_venv\Scripts\activate
pip install --upgrade pip
pip install torch matplotlib numpy pandas mlflow pytorch_lightning
```

Install the Jupyter kernel and launch the notebook:

```bash
pip install ipykernel
python -m ipykernel install --user --name=fsdl_venv
jupyter notebook
```

Place the downloaded MLflow model artifacts under:

```
full_stack_deep_learning/mlruns/best_model/
```

Serve the model:

```bash
mlflow models serve -m mlruns/best_model --env-manager local --host 127.0.0.1 --port 1234
```

---

## Key Dependencies

| Package | Purpose |
|---|---|
| `torch` / `pytorch_lightning` | Model definition and training |
| `mlflow` | Experiment tracking, model registry, and serving |
| `optuna` | Hyperparameter optimization (covered in course Chapter 4) |
| `pyngrok` | Tunnel MLflow UI from Colab to a public URL |
| `torchmetrics` | Accuracy and other training metrics |
| `matplotlib` / `seaborn` | Visualization |

---

## Course Overview

This repository accompanies the following chapters:

1. **An Overview of Full-Stack Deep Learning** — Components, artifacts, and tooling across the ML lifecycle.
2. **MLOps with MLflow** — Setting up MLflow, running the tracking UI, and understanding the ML operations workflow.
3. **Model Training and Evaluation Using MLflow** — Training DNN and CNN models on EMNIST, logging parameters, metrics, and artifacts.
4. **Hyperparameter Tuning with Optuna** — Defining objective functions, running optimization trials, identifying the best model, and registering it.
5. **Model Deployment and Predictions** — Serving a registered MLflow model locally and querying it over HTTP.

---

## Notes

- The ngrok auth tokens visible in the notebooks are session-specific and have been rotated. Replace them with your own token from the [ngrok dashboard](https://dashboard.ngrok.com/auth).
- Pixel values in the EMNIST CSVs are in the range [0, 255] and require normalization. Labels start at 1 and are shifted to 0-indexed in the dataset class.
- The CNN in `demo_02` reaches approximately **91.7% test accuracy** after 10 epochs on the EMNIST letters split.

---

## License

These materials are provided for educational use only. All rights belong to LinkedIn Learning and the original instructor. Redistribution for commercial purposes is strictly prohibited.
