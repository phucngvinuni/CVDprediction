

# AI Model for Cardiovascular Disease Prediction

This repository contains the AI model implementation for predicting cardiovascular disease (CVD) as presented in the paper:

**"A Digital Healthcare Platform for Cardiovascular Disease Management: Architecture, Technology Integration, and Deployment"**  

## Overview

The AI model combines advanced machine learning techniques to achieve high accuracy and reliability in CVD prediction. Key components include:  

- **Feature Extraction:** Random Forest for identifying important features.  
- **Outlier Detection:** DBSCAN clustering for cleaning the dataset.  
- **Data Balancing:** SMOTE for oversampling and CTGAN for synthetic data generation.  
- **Predictive Modeling:** LightGBM for classification with hyperparameter tuning via Optuna.

## Features

1. **Outlier Detection:**  
   - Removes noise using DBSCAN clustering.  
   - Refines data for improved model performance.

2. **Data Augmentation:**  
   - Balances class distribution using SMOTE.  
   - Enhances dataset diversity with CTGAN-generated samples.  

3. **Model Training:**  
   - Uses LightGBM with a focus on computational efficiency and accuracy.  
   - Hyperparameter tuning performed using Optuna for optimal performance.

4. **Performance Metrics:**  
   - Accuracy: **96.61%**  
   - AUC: **0.95**  
   - Sensitivity: **93.33%**  
   - MCC: **0.93**

## Repository Structure

```
.
├── data/                   # Preprocessed dataset and synthetic data
├── src/                    # AI model code
│   ├── preprocess_data.py  # Data preprocessing and augmentation scripts
│   ├── train_model.py      # Model training with LightGBM
│   ├── evaluate_model.py   # Model evaluation and metrics
├── notebooks/              # Jupyter notebooks for experimentation
└── README.md               # Project overview
```

## Getting Started

### Prerequisites
- Python 3.8+
- Required packages:
  ```
  pip install -r requirements.txt
  ```

### Steps to Run
1. Clone the repository:
   ```
   git clone https://github.com/<username>/<repo-name>.git
   ```
2. Navigate to the project directory and install dependencies:
   ```
   cd <repo-name>
   pip install -r requirements.txt
   ```
3. Preprocess the dataset:
   ```
   python src/preprocess_data.py
   ```
4. Train the model:
   ```
   python src/train_model.py
   ```
5. Evaluate the model:
   ```
   python src/evaluate_model.py
   ```

## Key Techniques

1. **DBSCAN Clustering:**  
   Used to detect and remove outliers, ensuring the dataset's quality.

2. **SMOTE and CTGAN:**  
   - SMOTE oversamples the minority class by interpolating between existing samples.  
   - CTGAN generates synthetic data to expand the dataset, improving model generalization.

3. **LightGBM:**  
   - Gradient-boosting framework optimized for speed and performance.  
   - Features such as leaf-wise tree growth and gradient-based sampling enhance efficiency.

4. **Optuna Hyperparameter Tuning:**  
   - Automatically searches for the best hyperparameters.  
   - Key parameters include learning rate, max depth, and number of leaves.

## Results

- The LightGBM model outperforms traditional classifiers with the following metrics:
  - **Accuracy:** 96.61%  
  - **AUC:** 0.95  
  - **Sensitivity:** 93.33%  
  - **Precision:** 93.94%  

## Future Work

- Adapt the AI model for other diseases like diabetes and chronic respiratory conditions.  
- Incorporate localized datasets to improve predictions for specific populations.  

