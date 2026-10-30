# 🧠 NeuroScan — Brain Tumor Classification Using MRI Images

A deep learning project for brain tumor classification from MRI images using transfer learning with MobileNetV2 and PyTorch.

## 📌 Project Overview

NeuroScan explores how deep learning can be used to classify brain MRI images into different categories. The project uses MobileNetV2, a pretrained convolutional neural network architecture, to learn image features and perform classification.

The objective is to develop an image classification pipeline that combines data preprocessing, transfer learning, model training, and performance evaluation.

**Project Type:** Deep Learning | Computer Vision | Medical Image Classification

## 🎯 Project Objectives

- Develop a deep learning model for classifying brain MRI images.
- Apply transfer learning using MobileNetV2.
- Prepare and preprocess MRI images for model training.
- Evaluate model performance using appropriate classification metrics.
- Explore the potential and limitations of AI-assisted medical image analysis.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Model Architecture | MobileNetV2 |
| Development Environment | Jupyter Notebook |
| Application Domain | Medical Image Classification |

## ⚙️ Methodology

The project follows a deep learning workflow:

1. **Data Collection:** Obtain the brain MRI image dataset.
2. **Data Preprocessing:** Prepare images for model input.
3. **Transfer Learning:** Use MobileNetV2 as the foundation for image classification.
4. **Model Training:** Train the classification model using the prepared dataset.
5. **Model Evaluation:** Assess performance using appropriate evaluation metrics.
6. **Prediction:** Use the trained model to classify MRI images.

## 📊 Model Evaluation

The model's performance should be assessed using appropriate metrics, including:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Detailed evaluation results and visualizations will be documented here after verification against the model's actual outputs.

## 📁 Project Structure

```text
NeuroScan-Brain-Tumor-Classification/
│
├── README.md
├── notebook/
│   └── NeuroScan_Brain_Tumor_Classification.ipynb
│
├── requirements.txt
├── .gitignore
├── outputs/
│   └── Evaluation visualizations
│
└── models/
    └── Instructions for obtaining model weights
```

*The directory structure above is a proposed organization and will be updated to match the actual repository.*

## 🚀 Getting Started

### Prerequisites

- Python
- Jupyter Notebook
- PyTorch
- Required Python dependencies

### Installation

Clone the repository:

```bash
git clone https://github.com/manishkachitra/NeuroScan-Brain-Tumor-Classification.git
```

Navigate to the project directory:

```bash
cd NeuroScan-Brain-Tumor-Classification
```

Install the dependencies after the project's `requirements.txt` has been verified:

```bash
pip install -r requirements.txt
```

Open the notebook in Jupyter Notebook or Google Colab and follow the documented execution steps.

## 🔬 Future Improvements

- Improve model evaluation and validation.
- Add visualizations for training performance.
- Display sample predictions with confidence scores, if supported by the implementation.
- Explore model explainability techniques.
- Develop an interactive interface for image classification.
- Evaluate generalization on independent datasets.

## ⚠️ Disclaimer

NeuroScan is an educational and research project. Its predictions are not medical diagnoses and must not be used as a substitute for evaluation by a qualified medical professional.

## 👨‍💻 Project Information

**Project:** NeuroScan — Brain Tumor Classification  
**Primary Model:** MobileNetV2  
**Framework:** PyTorch
