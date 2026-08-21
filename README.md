# 🛰️ Satellite Imagery Natural Disaster Classifier & MLOps Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ed.svg)](https://www.docker.com/)
[![CI/CD ML Pipeline](https://github.com/KishanDeka/disaster_alert_system/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/KishanDeka/disaster_alert_system/actions/workflows/ci-cd.yml)

An end-to-end **Deep Learning system and MLOps pipeline** designed to classify satellite imagery into natural disaster events in real time.

The project combines a custom **PyTorch CNN**, automated dataset handling, interactive **Streamlit** inference, **Docker** containerization, and **GitHub Actions CI/CD** into a complete machine-learning application.

---

## Project Overview

Rapid disaster assessment requires fast and accurate computer vision models. This project provides an end-to-end pipeline for training and deploying a satellite-image classification model.

### Key Features

- **Custom PyTorch Model (`ScratchCNN`)**
  - Modular 4-layer convolutional neural network.
  - Batch normalization.
  - Adaptive pooling.
  - Classification dropout.
  - Built-in training and prediction functionality.

- **Dynamic Data Pipeline**
  - Automatic dataset downloading through `kagglehub`.
  - Dataset indexing.
  - Image preprocessing.
  - Automatic channel-wise mean and standard deviation calculation.

- **Interactive Web Interface**
  - Streamlit-based dashboard.
  - Single-image upload and inference.
  - Real-time prediction.
  - Probability visualization for all target classes.

- **Production Packaging**
  - Dockerized application.
  - Ready for deployment to Hugging Face Spaces or other cloud platforms.

- **Automated CI/CD**
  - GitHub Actions workflow.
  - Automated unit tests using `pytest`.
  - Code quality checks using `flake8`.
  - Model integrity checks.

---

## Target Categories

The model classifies satellite image crops into four categories:

| Class | Description |
|---|---|
| **Earthquake** | Satellite imagery associated with earthquake-related damage or areas |
| **Fire** | Imagery showing wildfire or fire-affected regions |
| **Flood** | Imagery showing flooded or water-affected regions |
| **Normal** | Normal satellite imagery without the target disaster event |

---

## Project Architecture

```text
                    ┌──────────────────────┐
                    │   Kaggle Dataset     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     kagglehub        │
                    │ Dataset Downloader   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Data Pipeline     │
                    │ Indexing / Transform │
                    │ Mean / Std Metrics   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      ScratchCNN      │
                    │      PyTorch CNN     │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴───────────┐
                    ▼                      ▼
          ┌──────────────────┐    ┌──────────────────┐
          │ Model Training   │    │ Model Inference  │
          │     main.py      │    │     app.py       │
          └────────┬─────────┘    └────────┬─────────┘
                   │                       │
                   ▼                       ▼
          ┌──────────────────┐    ┌──────────────────┐
          │ best_model.pth   │    │    Streamlit     │
          └──────────────────┘    │    Dashboard     │
                                   └────────┬─────────┘
                                            │
                                            ▼
                                   ┌──────────────────┐
                                   │ Docker Container  │
                                   └──────────────────┘
```

---

## Repository Structure

```text
.
├── .github/
│   └── workflows/
│       └── pipeline.yml       # CI/CD pipeline definition
│
├── src/
│   ├── __init__.py            # Package exports
│   ├── dataset.py             # CSVDataset & kagglehub downloader
│   ├── model.py               # ScratchCNN architecture, fit & predict
│   └── utils.py               # Device helpers, seed setters, constants
│
├── app.py                     # Streamlit web application
├── main.py                    # Model training entry point
├── Dockerfile                 # Container definition
├── requirements.txt           # Python dependencies
├── .dockerignore              # Docker build exclusions
└── README.md                  # Project documentation
```

---

# Getting Started Locally

## 1. Prerequisites

Make sure the following are installed:

- >= Python 3.10
- pip
- Git
- Docker — optional, for containerized deployment

---

## 2. Clone the Repository

```bash
git clone https://github.com/KishanDeka/DisasterLens.git
cd disasterlens
```

---

## 3. Create a Virtual Environment

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Model Training

The project uses a custom PyTorch CNN called `ScratchCNN`.

The training entry point is:

```bash
python main.py
```

The training pipeline automatically:

1. Downloads the dataset using `kagglehub`.
2. Discovers and indexes the image data.
3. Calculates channel-wise image statistics.
4. Preprocesses the images.
5. Initializes the CNN.
6. Trains the model.
7. Evaluates model performance.
8. Saves the best model weights.

### Train with Default Parameters

```bash
python main.py
```

### Train with Custom Hyperparameters

```bash
python main.py \
    --epochs 15 \
    --batch_size 64 \
    --lr 0.0005 \
    --weights_path best_model.pth
```

### Available Training Parameters

| Parameter | Example | Description |
|---|---:|---|
| `--epochs` | `15` | Number of training epochs |
| `--batch_size` | `64` | Number of images per batch |
| `--lr` | `0.0005` | Learning rate |
| `--weights_path` | `best_model.pth` | Path for saving model weights |

---

# Testing

The project uses **pytest** for automated testing.

Run the test suite with:

```bash
pytest
```

For more detailed output:

```bash
pytest -v
```

---

# Code Quality

`flake8` is used to check Python code quality and style.

Run:

```bash
flake8 .
```

---

# Streamlit Dashboard

After training the model and generating the model weights, launch the interactive application:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```


# Docker Deployment

The application can be packaged into a Docker container for reproducible deployment.

## 1. Build the Docker Image

```bash
docker build -t satellite-disaster-app .
```

## 2. Run the Container

```bash
docker run -p 8501:8501 satellite-disaster-app
```

The Streamlit application will then be available at:

```text
http://localhost:8501
```

# CI/CD Pipeline

The project includes a GitHub Actions workflow located at:

```text
.github/workflows/pipeline.yml
```

The CI/CD pipeline automatically validates the project whenever changes are pushed to the repository.

### Pipeline Checks

```text
Git Push / Pull Request
          │
          ▼
   GitHub Actions
          │
     ┌────┴────┐
     ▼         ▼
  PyTest     Flake8
     │         │
     └────┬────┘
          ▼
 Model Integrity Check
          │
          ▼
       Pipeline
        Passed
```

The workflow performs:

- Python environment setup.
- Dependency installation.
- Unit tests with `pytest`.
- Static code analysis with `flake8`.
- Model integrity validation.

---

# CNN Model Architecture

`ScratchCNN` is a custom convolutional neural network designed specifically for the four-class satellite-image classification problem.

The architecture includes:

- Four convolutional layers.
- Batch normalization.
- Non-linear activation functions.
- Adaptive pooling.
- Classification dropout.
- Final classification layer for four target categories.

Conceptually:

```text
Input Image
     │
     ▼
Convolution Block 1
     │
     ▼
Convolution Block 2
     │
     ▼
Convolution Block 3
     │
     ▼
Convolution Block 4
     │
     ▼
Adaptive Pooling
     │
     ▼
Dropout
     │
     ▼
Fully Connected Layer
     │
     ▼
4-Class Prediction
```

---

# Deployment

The Dockerized application can be deployed to cloud platforms that support Docker containers.

Potential deployment targets include:

- Hugging Face Spaces.
- Cloud VM/container services.
- Self-hosted infrastructure.
- Other Docker-compatible platforms.

Before deployment, ensure that:

1. The model weights are available to the application.
2. Required environment variables and credentials are configured.
3. The container exposes Streamlit's port `8501`.
4. Dataset credentials are configured if runtime dataset downloading is required.

---

# Development Workflow

A typical development workflow is:

```bash
# Create environment
python -m venv venv

# Activate environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Run linting
flake8 .

# Train model
python main.py

# Start application
streamlit run app.py
```

---

# Future Improvements

Potential extensions include:

- GPU-optimized training.
- Data augmentation.
- Transfer learning with pretrained architectures.
- Model explainability using Grad-CAM.
- Experiment tracking with MLflow or Weights & Biases.
- Model versioning.
- Automated model deployment.
- REST API inference endpoint.
- Batch image inference.
- Performance monitoring.
- Data and model drift detection.

---

# 📄 License

Distributed under the **MIT License**.

See the `LICENSE` file for more information.

---

# 👤 Author

**Kishan Deka** and my buddy *Gemini*

GitHub: `https://github.com/KishanDeka`

---

## Project Summary

This project demonstrates a complete computer-vision and MLOps workflow:

**Data → Training → Evaluation → Inference → Web Application → Docker → CI/CD**

It combines deep learning with practical software-engineering and deployment practices to create a reproducible satellite-image natural-disaster classification system.
