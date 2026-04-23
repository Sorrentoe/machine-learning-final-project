# 🫁 Medical Imaging: Pneumonia & Anomaly Detector

**Course:** SE 3231 - Deep Learning Application
**Project Category:** Medical Imaging (Transfer Learning)

## Project Overview
This project is a Deep Learning application designed to classify pediatric chest X-rays. It utilizes Transfer Learning on a ResNet18 backbone to predict the probability of Pneumonia versus a Normal scan. 

To ensure system robustness and handle "Out-of-Distribution" (OOD) anomalies, the architecture was explicitly designed with a third class (`OTHER`). This prevents the model from attempting to diagnose irrelevant uploads (e.g., selfies, random objects, landscapes) and forces it to reject them gracefully, vastly improving the safety of the application.

## 🧠 Architecture & Engineering Notes
* **Framework:** PyTorch (Built from scratch, no pre-built wrapper APIs used).
* **Backbone:** Pre-trained ResNet18 (Feature extraction layers frozen).
* **Custom Classification Head:** `nn.Linear(in_features, 512)` -> `nn.ReLU()` -> `nn.Dropout(0.3)` -> `nn.Linear(512, 3)`
  * Dropout (30%) was implemented to prevent overfitting during training.
* **Loss Function:** `CrossEntropyLoss`. Because the dataset was highly imbalanced, mathematical class weights were calculated and applied directly to the loss function to heavily penalize misclassifications on the minority classes.
* **Optimizer:** Stochastic Gradient Descent (`optim.SGD`) with a learning rate of `0.001` and momentum of `0.9` to navigate local minima effectively.

## 📊 Model Performance
The model was evaluated against an unseen test dataset consisting of 774 images, achieving an **Overall Accuracy of 89%**. 

**Classification Report:**
* **NORMAL:** Precision 0.94 | Recall 0.67 | F1-Score 0.78
* **OTHER (Anomaly Class):** Precision 1.00 | Recall 1.00 | F1-Score 1.00
* **PNEUMONIA:** Precision 0.83 | Recall 0.97 | F1-Score 0.90

**Key Engineering Outcomes:**
1. **Out-of-Distribution Robustness:** The `OTHER` class achieved perfect precision and recall (150/150). The mathematical loss penalties successfully taught the model to intercept and reject invalid data.
2. **Clinical Cautiousness:** The model was intentionally tuned to prioritize catching severe infections. It achieved a **97% recall rate for Pneumonia**, successfully identifying 380 out of 390 true cases. While this high sensitivity slightly lowers precision (false positives), it is the mathematically and clinically preferred behavior for a first-line medical screening tool.

## 📂 Project Structure
```text
pneumonia-detector/
├── data/
│   ├── test/      # Unseen data for evaluation
│   ├── train/     # Training data (NORMAL, OTHER, PNEUMONIA)
│   └── val/       # Validation data for epoch checking
├── src/
│   ├── dataset.py # Data loader and verification script
│   ├── evaluate.py# Confusion matrix and precision/recall metrics
│   ├── model.py   # Neural network architecture definition
│   └── train.py   # Training loop and loss calculation
├── app.py         # Streamlit User Interface
├── pneumonia_model.pth # The trained model weights
└── requirements.txt