# mlops-pytorch-pipeline
MTech MLOps project for PyTorch image classification for IITM MLOps Assignment 3.

---

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0185CA.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Version%20Control-945DD6.svg)](https://dvc.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerization-2496ED.svg)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Orchestration-326CE5.svg)](https://kubernetes.io/)

An end-to-end production-grade MLOps pipeline built around **PyTorch**. This repository standardizes data ingestion, versioning, automated model training, experiment tracking, artifact evaluation, and API deployment.

---

## System Architecture

The pipeline follows a modular MLOps architecture separating data management, model training/tracking, and model serving.
```
                ┌──────────────────────────────────────────┐
                │               Data Source - CIFAR10      │
                └────────────────────┬─────────────────────┘
                                     │
                                     ▼
                ┌──────────────────────────────────────────┐
                │        Data Ingestion & Versioning       │
                │               (DVC / S3)                 │
                └────────────────────┬─────────────────────┘
                                     │
                                     ▼
                ┌──────────────────────────────────────────┐
                │       Data Preprocessing & Splits        │
                └────────────────────┬─────────────────────┘
                                     │
                                     ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │                     PyTorch Training Pipeline                    │
 │  ┌──────────────────┐    ┌─────────────────┐    ┌─────────────┐  │
 │  │ Custom Dataset / │───>│ DataLoader &    │───>│ Model Loop  │  │
 │  │ Transform        │    │ Augmentation    │    │ & Checkpoint│  │
 │  └──────────────────┘    └─────────────────┘    └──────┬──────┘  │
 └────────────────────────────────────────────────────────┼─────────┘
                                                          │
                                                          ▼
┌───────────────────────────┐                      ┌───────────────────┐
│    Experiment Tracking    │<─────────────────────│  Model Artefacts  │
│     & Metrics (MLflow)    │                      │ & Registry (DVC)  │
└───────────────────────────┘                      └─────────┬─────────┘
                                                             |
                                                             ▼
                                                    ┌───────────────────┐
                                                    │ REST API Serving  │
                                                    │ (FastAPI/Docker)  │
                                                    └───────────────────┘
```

---

## Features

- **DVC - Data Versioning:** Track CIFAR10 and PyTorch model files using **DVC**.
- **Experiment Tracking:** Log metrics, hyper parameters, and artefacts automatically with **MLflow**.
- **PyTorch Training Engine:** Configurable training loop with support for GPU acceleration, early stopping, and checkpoint saving.
- **Modular Configuration:** Dynamic pipeline management powered by `YAML` configuration files for `Docker` and `Kubernetes`.
- **Containerized Deployment:** Dockerized API setup for reproducible inference environments. Deployment and scaling with k8s pipeline.

---

## Repository Structure
```text

mlops-pytorch-pipeline/
├── config/
│   └── config.yaml          # Pipeline hyperparameters, paths, and configurations
├── data/                    # Local data storage (tracked by DVC)
│   ├── raw/
│   └── processed/
├── models/                  # Saved model checkpoints (.pt / .pth)
├── src/
│   ├── components/
│   │   ├── data_ingestion.py   # Data downloading and validation
│   │   ├── data_prep.py        # Preprocessing & PyTorch Datasets
│   │   ├── model_trainer.py    # Training & evaluation loop
│   │   └── model_eval.py       # Metrics computation
│   ├── pipeline/
│   │   ├── train_pipeline.py   # Complete training orchestration
│   │   └── predict_pipeline.py # Inference pipeline execution
│   └── utils/                  # Helper utilities and logging setups
├── app.py                      # FastAPI endpoint for model serving
├── params.yaml                 # ML hyperparameter tracking
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Containerization setup
└── README.md
```

---

## Quick Start & Setup

### Prerequisites

* **Python 3.9+**
* **Git**
* **Virtualenv** / **Conda**

---

### Step 1: Clone the Repository

```bash
git clone [https://github.com/aditi-govindu/mlops-pytorch-pipeline.git](https://github.com/aditi-govindu/mlops-pytorch-pipeline.git)
cd mlops-pytorch-pipeline
```

---

### Step 2: Set Up Virtual Environment

#### Using `venv`:

```bash
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

```

#### Using `conda`:

```bash
conda create -n mlops-env python=3.9 -y
conda activate mlops-env

```

---

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

```

---

### Step 4: Pull Versioned Data (DVC)

If using DVC with remote storage (S3/GCS/DAGsHub):

```bash
dvc pull

```

---

## Execution

### 1. Execute Training Pipeline

To run data ingestion, preprocessing, training, and evaluation in sequence:

```bash
python src/pipeline/train_pipeline.py

```

### 2. View Experiment Metrics with MLflow

Launch the MLflow UI to compare runs, parameters, and loss curves:

```bash
mlflow ui

```

Access the dashboard at `http://localhost:5000` in your Chrome browser.

---

## Deployment (Docker)

### Build Docker Image

```bash
docker build -t mlops-pytorch-app .

```

### Run Containerized Endpoint

```bash
docker run -p 8000:8000 mlops-pytorch-app

```

Test the inference endpoint via `http://localhost:8000/docs` with FastAPI.

---
