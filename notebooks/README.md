# NeuroScan Notebook

This folder contains the Jupyter Notebook used for the NeuroScan brain tumor classification project.

## Notebook

**`NeuroScan_Brain_Tumor_Classification.ipynb`**

The notebook covers:

- MRI dataset loading and exploration
- Image preprocessing and data augmentation
- Training and validation dataset preparation
- MobileNetV2 transfer learning
- Model training and validation
- Test-set evaluation and classification metrics
- Confusion matrix and sample predictions

## Dataset

The model uses the Brain Tumor Classification (MRI) dataset by Sartaj Bhuvaji and contributors.

Dataset source: [Brain Tumor Classification (MRI) on Kaggle](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)

The notebook attempts to locate the dataset in a Kaggle environment and can use `kagglehub` to download it if necessary. Dataset access and paths may require configuration in other environments.

## Running the Notebook

1. Install the dependencies listed in the root `requirements.txt`.
2. Obtain access to the MRI dataset.
3. Open the notebook in Jupyter Notebook or Kaggle.
4. Run the cells in order, ensuring the dataset paths are configured correctly.

The notebook saves the best validation model checkpoint as `neuroscan_best_mobilenetv2.pth` during training. The checkpoint must be available when running the evaluation cell.

## Disclaimer

This notebook is intended for educational and research purposes only. It is not a medical diagnostic tool.
