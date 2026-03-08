# Molecular Toxicity Prediction Pipeline

A **supervised machine learning pipeline** for predicting whether molecular compounds are **Toxic** or **NonToxic** based on high-dimensional molecular descriptors.

This project demonstrates a **leakage-safe machine learning workflow**, integrating preprocessing, dimensionality reduction, class imbalance handling, hyperparameter tuning, and evaluation.

---

# Project Overview

Drug discovery and chemical safety assessment often require determining whether a compound is **toxic**.

This project builds a **predictive classification model** that learns toxicity patterns from molecular descriptor data.

The pipeline focuses on:

- Preventing **data leakage**
- Handling **high-dimensional features**
- Managing **class imbalance**
- Evaluating models using **precision-recall metrics**

---

# Machine Learning Pipeline

The pipeline follows a strict structure to ensure reliable evaluation.
```bash
Raw Dataset
│
▼
Data Cleaning & Exploration
│
▼
Feature Filtering (VarianceThreshold)
│
▼
Feature Scaling (StandardScaler)
│
▼
Dimensionality Reduction (PCA)
│
▼
Class Imbalance Handling (SMOTE)
│
▼
Model Training
│
▼
Cross Validation
│
▼
Model Evaluation
```


All steps are implemented inside a **Scikit-Learn Pipeline** to prevent **data leakage**.

---

# Dataset

The dataset contains **molecular descriptors** representing chemical properties of compounds.

## Target Variable

| Class | Meaning |
|------|--------|
| 0 | NonToxic |
| 1 | Toxic |

## Dataset Characteristics

- High-dimensional feature space
- Binary classification problem
- Class imbalance present
- No missing values

---

# Data Preprocessing

Several preprocessing techniques were applied.

## 1️⃣ Feature Filtering

Low variance features were removed using:

```python
VarianceThreshold(threshold=0.01)
```

This eliminates descriptors with little predictive power.

## 2️⃣ Feature Scaling

Features were standardized using:

```python
StandardScaler
```
This is necessary for models such as SVM that are sensitive to feature magnitude.

## 3️⃣ Dimensionality Reduction

High dimensional data was reduced using **Principal Component Analysis (PCA)**.

```python
PCA(n_components=0.95)
```

This preserves 95% of the dataset variance while reducing dimensionality.

## 4️⃣ Handling Class Imbalance

The dataset had imbalanced classes, so **SMOTE** was applied.

**SMOTE (Synthetic Minority Oversampling Technique)** generates synthetic samples for the minority class to improve model learning.

---

## Train-Test Split

The dataset was split using **stratified sampling**:

- **Training set:** 80%
- **Testing set:** 20%

---

## Cross-Validation

**StratifiedKFold (5 folds)** was used.

This preserves the **class distribution across folds**.

---

# Models Implemented

Three machine learning models were trained and optimized.

---

## 1️⃣ Random Forest Classifier

### Pipeline
StandardScaler
→ PCA
→ SMOTE
→ RandomForestClassifier


### Hyperparameters Tuned

- `n_estimators`
- `max_depth`

---

## 2️⃣ Support Vector Machine (SVC)

### Pipeline
StandardScaler
→ PCA
→ SMOTE
→ SVC


### Hyperparameters Tuned

- `C`
- `kernel`

---

## 3️⃣ HistGradientBoosting Classifier

### Pipeline
StandardScaler
→ PCA
→ SMOTE
→ HistGradientBoostingClassifier


### Hyperparameters Tuned

- `learning_rate`
- `max_iter`

---

# Model Performance

| Model | Accuracy | PR-AUC | Average Precision |
|------|------|------|------|
| Random Forest | 0.5143 | 0.3108 | 0.3450 |
| SVC | 0.5714 | 0.3032 | 0.3401 |
| HistGradientBoosting | 0.5714 | 0.3196 | 0.3640 |

---

# Key Observations

- **SVC** and **HistGradientBoosting** achieved the **highest accuracy (57.14%)**
- **HistGradientBoosting** produced the **best PR-AUC and Average Precision**
- **Random Forest** performed slightly worse on the **minority class**
- Since the dataset is **imbalanced**, **PR-AUC** is more informative than **accuracy**

---

# Example Prediction

Example of generating predictions from the trained model:

```python
test_samples = X_test.head(10)
actual_labels = le.inverse_transform(y_test[:10])

print(f"{'Actual':<15} | {'RF Pred':<15} | {'SVC Pred':<15} | {'HGB Pred':<15}")
print("-" * 65)

rf_preds = le.inverse_transform(rf_gs.predict(test_samples))
svc_preds = le.inverse_transform(svc_gs.predict(test_samples))
hgb_preds = le.inverse_transform(hgb_gs.predict(test_samples))

for actual, rf, svc, hgb in zip(actual_labels, rf_preds, svc_preds, hgb_preds):
    print(f"{actual:<15} | {rf:<15} | {svc:<15} | {hgb:<15}")
```

##### Output
```bash
Actual          | RF Pred         | SVC Pred        | HGB Pred       
-----------------------------------------------------------------
NonToxic        | NonToxic        | Toxic           | NonToxic       
NonToxic        | NonToxic        | NonToxic        | NonToxic       
Toxic           | NonToxic        | NonToxic        | NonToxic       
NonToxic        | NonToxic        | NonToxic        | NonToxic       
Toxic           | NonToxic        | NonToxic        | NonToxic       
Toxic           | NonToxic        | NonToxic        | Toxic          
NonToxic        | NonToxic        | NonToxic        | NonToxic       
NonToxic        | NonToxic        | NonToxic        | NonToxic       
NonToxic        | Toxic           | Toxic           | Toxic          
NonToxic        | Toxic           | Toxic           | Toxic    

```

## Technologies Used

- **Python**
- **Jupyter Notebook**
- **pandas**
- **numpy**
- **matplotlib**
- **seaborn**
- **scikit-learn**
- **imbalanced-learn**

---

## Project Structure

```bash
molecular-toxicity-prediction
│
├── supervised.ipynb # Main analysis notebook
├── data.csv # Molecular descriptor dataset
└── README.md # Project documentation
```


---

## How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/lucaskojo02/molecular-toxicity-prediction
cd molecular-toxicity-prediction
```

## 2️⃣ Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

## 3️⃣ Launch the Notebook

```bash
jupyter notebook
```

Open the notebook file:

```bash
supervised.ipynb
```