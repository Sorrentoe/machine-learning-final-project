# 🫁 Medical Imaging: Pneumonia & Anomaly Detector

**Course:** SE 3231 - Deep Learning Application
**Project Category:** Medical Imaging (Transfer Learning)


## 1. Project Overview
This project is a Deep Learning application designed to classify pediatric chest X-rays to solve a real-world medical problem. It utilizes **Transfer Learning** on a standard ResNet backbone to predict the probability of Pneumonia versus a Normal scan.

To ensure system robustness and handle "Out-of-Distribution" (OOD) anomalies—such as non-medical images or selfies—the architecture was explicitly designed with a third class: `OTHER`. This allows the model to reject invalid data gracefully, directly addressing the "handles errors gracefully" criteria.

## How to run the app

The UI is a **Streamlit** app (`app.py`). Run it from the **project root** so `pneumonia_model.pth` and `src/` resolve correctly.

### Prerequisites

- **Python 3.10+** (3.11 recommended)
- **`pneumonia_model.pth`** in the project root (same folder as `app.py`)

### Steps (copy and paste)

**1. Go to the project folder**

```bash
cd /path/to/pneumonia-detector
```

Use your real path, for example:

```bash
cd ~/Desktop/pneumonia-detector
```

**2. Create a virtual environment (recommended)**

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (Command Prompt):

```bash
python -m venv .venv
.venv\Scripts\activate.bat
```

**3. Install Python packages**

```bash
pip install -r requirements.txt
```

**4. Start the app**

```bash
streamlit run app.py
```

**5. Open the app in your browser**

Streamlit prints a local URL (usually `http://localhost:8501`). Open that link, then upload a chest X-ray (`.jpg`, `.jpeg`, or `.png`).

### Sample images for quick testing

The **`testing-dataset/`** folder holds example images organized like the model’s classes: **`NORMAL/`**, **`PNEUMONIA/`**, and **`OTHER/`** (non–chest-X-ray or anomaly-style samples). After the app is running, pick any file from those subfolders and upload it in Streamlit to try the model without preparing your own images.

To stop the server, press **Ctrl+C** in the terminal.

---

## 2. Technical Architecture & Design
To satisfy the course constraints, this model was built without pre-built high-level AI services like OpenAI or LangChain. All layers, loss functions, and optimizers were manually defined.

* **Framework:** PyTorch (v2.x).
* **Backbone:** Pre-trained **ResNet18** used as a feature extractor.
* **Custom Classification Head:**
    * `Linear(in_features, 512)` -> `ReLU()` -> `Dropout(0.3)` -> `Linear(512, 3)`.
    * **Dropout** (30%) was implemented to minimize overfitting.
* **Loss Function:** `CrossEntropyLoss` with manual **Class Weights**. Because the dataset was highly imbalanced (1,341 Normal vs. 3,875 Pneumonia), weights were calculated to prevent the model from ignoring minority classes.
* **Optimizer:** **Stochastic Gradient Descent (SGD)** with a learning rate of `0.001` and **Momentum** of `0.9` to ensure stable convergence.

## 3. Model Performance & Evaluation
The model was evaluated against an unseen test dataset of 774 images, achieving an **Overall Accuracy of 89%**.

### Classification Report:
* **NORMAL:** Precision 0.94 | Recall 0.67
* **OTHER (Anomaly):** Precision 1.00 | Recall 1.00
* **PNEUMONIA:** Precision 0.83 | Recall 0.97

**Evaluation Analysis:**
The model prioritizes **Recall for Pneumonia (97%)**, which is critical in a medical context to ensure infections are not missed. The perfect performance on the `OTHER` class proves the efficacy of the data pipeline and normalization strategy.

## 4. Project Structure
```text
pneumonia-detector/
├── data/
│   ├── test/      # Unseen test set (NORMAL, OTHER, PNEUMONIA)
│   ├── train/     # Training set
│   └── val/       # Validation set
├── src/
│   ├── dataset.py # Data preprocessing and normalization
│   ├── evaluate.py# Confusion matrix and performance metrics
│   ├── model.py   # Custom architecture design
│   └── train.py   # Training logic, loss, and optimizer
├── testing-dataset/  # Sample NORMAL / PNEUMONIA / OTHER images for trying the app
├── app.py         # Functional Streamlit UI
├── pneumonia_model.pth # Saved model weights
├── README.md      # Project documentation
└── requirements.txt
```
