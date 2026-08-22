# ❤️ Heart Disease Risk Prediction

A machine learning classification project that predicts whether a patient is likely to have **heart disease** using demographic, clinical, lifestyle, and health-related features.

The project uses **Logistic Regression** as the primary classification model and focuses not only on overall accuracy but also on **precision, recall, F1-score, ROC-AUC, cross-validation, and decision-threshold tuning**.

---

## 📌 Project Overview

Heart disease prediction is a binary classification problem where the model predicts whether a patient:

* `0` → Does not have heart disease
* `1` → Has heart disease

The main objective of this project is to build a Logistic Regression model capable of distinguishing between the two classes and then investigate whether changing the classification threshold can improve detection of positive cases.

The complete workflow includes:

1. Data loading and inspection
2. Exploratory dataset analysis
3. Feature identification
4. Train-test splitting
5. Categorical feature encoding
6. Numerical feature scaling
7. Logistic Regression training
8. Prediction and probability estimation
9. ROC curve analysis
10. Decision-threshold tuning
11. Cross-validation
12. Classification report
13. Confusion matrix analysis
14. ROC-AUC evaluation

---

## 📂 Project Structure

```text
03-Heart-Disease-Risk/
│
├── Heart_Disease_Risk.ipynb
└── heart_disease_risk_2026.csv
```

---

## 📊 Dataset

The dataset contains:

* **9,000 observations**
* **27 columns**
* **26 input/features columns**
* **1 target column**

### Target Variable

```text
has_heart_disease
```

| Value | Meaning          |
| ----- | ---------------- |
| `0`   | No heart disease |
| `1`   | Heart disease    |

### Features

The dataset contains demographic, clinical, lifestyle, and health-related attributes such as:

* Age
* Sex
* Resting systolic blood pressure
* Resting diastolic blood pressure
* Total cholesterol
* HDL
* LDL
* Triglycerides
* Fasting blood sugar
* HbA1c
* BMI
* Resting heart rate
* Maximum heart rate achieved
* Chest pain type
* Exercise-induced angina
* ST depression
* Family history
* Smoking status
* Alcohol consumption
* Exercise duration
* Sleep duration
* Stress score
* Wearable ownership
* Daily steps
* Diet quality score

`patient_id` is treated as an identifier rather than a predictive feature.

---

# 🔧 Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Google Colab / Jupyter Notebook

### Machine Learning Techniques

* Train-Test Split
* One-Hot Encoding
* Standardization
* Logistic Regression
* ROC Curve
* ROC-AUC
* Cross-Validation
* Decision Threshold Optimization
* Confusion Matrix
* Classification Report

---

# 🧹 Data Preprocessing

## 1. Separating Features

The target variable is separated from the input features:

```python
X = hr.drop(columns='has_heart_disease')
y = hr['has_heart_disease']
```

The `patient_id` column is excluded from the numerical predictive features because it is only an identifier.

---

## 2. Train-Test Split

The dataset is divided into:

* **80% training data**
* **20% testing data**

A stratified split is used to preserve the class distribution:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

This produces:

```text
Training samples → 7,200
Testing samples  → 1,800
```

---

## 3. Categorical Feature Encoding

Categorical variables are converted into numerical representations using:

```python
OneHotEncoder(
    handle_unknown='ignore',
    sparse_output=False
)
```

This allows Logistic Regression to work with categorical variables such as:

* Sex
* Chest pain type
* Smoking status

---

## 4. Numerical Feature Scaling

Numerical features are standardized using:

```python
StandardScaler()
```

Scaling is important for Logistic Regression because the numerical features have different ranges.

---

# 🤖 Model

## Logistic Regression

The primary model used in this project is:

```python
LogisticRegression(max_iter=10000)
```

Logistic Regression is suitable for this binary classification problem because it predicts the probability of belonging to either class.

The model produces a probability:

```text
P(heart disease = 1)
```

which can then be converted into a class prediction using a classification threshold.

---

# 📈 Model Evaluation

Several evaluation metrics are used instead of relying only on accuracy.

## Accuracy

Measures the proportion of total predictions that are correct.

```text
Accuracy = Correct Predictions / Total Predictions
```

---

## Precision

Measures how many predicted positive cases were actually positive.

```text
Precision = TP / (TP + FP)
```

---

## Recall

Measures how many actual positive cases were correctly detected.

```text
Recall = TP / (TP + FN)
```

For a heart disease prediction problem, recall is particularly important because a false negative means a patient with heart disease was predicted as not having heart disease.

---

## F1 Score

The F1-score balances precision and recall.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

---

## ROC-AUC

ROC-AUC evaluates how well the model separates positive and negative classes across different classification thresholds.

The model achieved:

```text
Test ROC-AUC ≈ 0.9498
```

A ROC-AUC close to `1.0` indicates strong class-separation ability.

---

# 🔄 Cross-Validation

5-fold cross-validation is used to obtain a more reliable estimate of model performance.

The following metrics were evaluated:

```python
['accuracy', 'f1', 'roc_auc']
```

### Cross-Validation Results

| Metric   | Mean CV Score |
| -------- | ------------: |
| Accuracy |    **0.9008** |
| F1 Score |    **0.8314** |
| ROC-AUC  |    **0.9547** |

These results indicate that the Logistic Regression model performs consistently across different validation folds.

---

# 🎯 Decision Threshold Tuning

By default, binary classification commonly uses:

```text
threshold = 0.5
```

However, the optimal threshold depends on the problem.

For this project, the ROC curve was used to calculate:

```python
J = TPR - FPR
```

The threshold that maximizes Youden's J statistic was selected.

### Selected Threshold

```text
Best threshold ≈ 0.3194
```

Instead of:

```python
Threshold ≈ 0.5
```

This makes the model more willing to classify a patient as positive, increasing the detection of positive cases.

---

# 📊 Results Before Threshold Tuning

Using the default classification threshold:

```text
Threshold = 0.5
```

### Classification Report

| Class                | Precision | Recall | F1-Score |
| -------------------- | --------: | -----: | -------: |
| 0                    |      0.91 |   0.95 |     0.93 |
| 1                    |      0.86 |   0.79 |     0.82 |
| **Overall Accuracy** |           |        | **0.90** |

### Confusion Matrix

```text
[[1186   69]
 [ 116  429]]
```

Therefore:

```text
True Negatives  = 1186
False Positives = 69
False Negatives = 116
True Positives  = 429
```

---

# 📊 Results After Threshold Tuning

Using:

```text
Threshold ≈ 0.3194
```

### Classification Report

| Class                | Precision | Recall | F1-Score |
| -------------------- | --------: | -----: | -------: |
| 0                    |      0.93 |   0.89 |     0.91 |
| 1                    |      0.78 |   0.85 |     0.81 |
| **Overall Accuracy** |           |        | **0.88** |

### Confusion Matrix

```text
[[1123  132]
 [  80  465]]
```

Therefore:

```text
True Negatives  = 1123
False Positives = 132
False Negatives = 80
True Positives  = 465
```

---

# 🔍 Threshold Tuning Analysis

Threshold tuning produced an important trade-off.

| Metric            | Before Tuning | After Tuning |
| ----------------- | ------------: | -----------: |
| Accuracy          |      **0.90** |     **0.88** |
| Class 1 Precision |      **0.86** |     **0.78** |
| Class 1 Recall    |      **0.79** |     **0.85** |
| Class 1 F1        |      **0.82** |     **0.81** |
| False Negatives   |       **116** |       **80** |
| True Positives    |       **429** |      **465** |

### What changed?

The tuned threshold:

```text
0.5 → 0.3194
```

makes the model more sensitive toward the positive class.

As a result:

* True positives increased from **429 → 465**
* False negatives decreased from **116 → 80**
* Recall increased from **0.79 → 0.85**
* Accuracy decreased from **0.90 → 0.88**
* Precision decreased from **0.86 → 0.78**

This demonstrates an important concept in classification:

> Improving recall can come at the cost of precision and accuracy.

For a healthcare-related classification problem, reducing false negatives can be more important than maximizing raw accuracy, depending on the intended application.

---

# 🧠 Key Machine Learning Concepts Demonstrated

This project was built to understand several important machine learning concepts:

### 1. Feature Preprocessing

Handling numerical and categorical variables differently using:

```text
OneHotEncoder
StandardScaler
ColumnTransformer
```

### 2. Logistic Regression

Understanding how Logistic Regression performs binary classification and generates class probabilities.

### 3. Probability vs Classification

The model produces probabilities:

```text
0.91
0.73
0.42
0.18
...
```

which are converted into classes using a threshold.

### 4. Classification Threshold

Understanding that `0.5` is not always the optimal threshold.

### 5. ROC Curve

Studying the relationship between:

```text
True Positive Rate
False Positive Rate
```

across different thresholds.

### 6. ROC-AUC

Evaluating the model's ability to distinguish between the two classes independent of a single threshold.

### 7. Cross-Validation

Using 5-fold cross-validation to obtain a more robust estimate of model performance.

### 8. Confusion Matrix

Understanding:

```text
TP
TN
FP
FN
```

and how they affect different evaluation metrics.

---

# ⚠️ Important Limitation

This project is a **machine learning learning project** and should not be interpreted as a medical diagnostic system.

A high ROC-AUC or accuracy does not mean the model is clinically reliable.

Real-world medical applications require:

* Clinically validated datasets
* External validation
* Careful handling of missing data
* Bias and fairness analysis
* Calibration analysis
* Clinical expert review
* Prospective validation
* Regulatory and safety considerations

The model should therefore be considered an educational machine learning experiment rather than a medical diagnosis tool.

---

# 🚀 Future Improvements

Possible improvements for this project include:

* [ ] Perform more extensive hyperparameter tuning
* [ ] Compare Logistic Regression with Random Forest, XGBoost and other classifiers
* [ ] Analyze feature importance / model coefficients
* [ ] Perform probability calibration
* [ ] Compare different threshold-selection strategies
* [ ] Analyze class imbalance more deeply
* [ ] Use nested cross-validation for more rigorous evaluation
* [ ] Evaluate the model on an independent external dataset
* [ ] Build a prediction interface using Streamlit
* [ ] Add model explainability using SHAP
* [ ] Package the preprocessing and model into a single Pipeline
* [ ] Add automated model evaluation

---
```

using Jupyter Notebook, JupyterLab, VS Code, or Google Colab.

---

# 📚 What I Learned

Through this project, I learned that evaluating a classification model is not simply about getting the highest accuracy.

A model can have good accuracy while still missing a significant number of positive cases.

This project helped me understand the relationship between:

```text
Probability
     ↓
Classification Threshold
     ↓
Confusion Matrix
     ↓
Precision / Recall / F1
     ↓
ROC Curve / ROC-AUC
```

The most important takeaway was understanding that **the appropriate decision threshold depends on the objective of the problem**.

---

# 👨‍💻 Author

**Pushkar Sinha**

NIT Jamshedpur — Electronics & Communication Engineering

GitHub: `SinhaPushkar047`

---

## 📄 Project Notebook

The complete implementation and experimentation are available in:

```text
Heart_Disease_Risk.ipynb
```

This project is part of the **ML Learning Projects** repository.

