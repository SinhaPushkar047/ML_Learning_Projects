# CIFAR-10 Image Classification using Classical Machine Learning

## 📌 Overview

This project explores **image classification on the CIFAR-10 dataset using classical Machine Learning techniques**, without using Deep Learning or Convolutional Neural Networks.

The main objective of this project is to understand how traditional Machine Learning algorithms perform on image data and to compare different combinations of:

- Feature Scaling
- Principal Component Analysis (PCA)
- Logistic Regression
- Support Vector Machine (SVM)

Four different Machine Learning pipelines were trained and evaluated on the CIFAR-10 dataset.

The final comparison shows that **SVM performs significantly better than Logistic Regression** for this image classification task.

---

## 🎯 Objective

The primary objectives of this project are:

1. Load and understand the CIFAR-10 dataset.
2. Convert image data into a format suitable for classical Machine Learning algorithms.
3. Apply feature scaling.
4. Use PCA for dimensionality reduction.
5. Train multiple classical ML models.
6. Compare their classification performance.
7. Identify the best-performing model.

---
## 📥 Dataset Download

The CIFAR-10 dataset can be downloaded directly using TensorFlow/Keras.

```python
from tensorflow.keras.datasets import cifar10

# Download CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
```
---

# 📊 Dataset

The project uses the **CIFAR-10 dataset**.

CIFAR-10 contains small RGB images belonging to 10 different classes.

Each image has dimensions:

```text
32 × 32 × 3
```

Therefore, each image contains:

```text
32 × 32 × 3 = 3072 features
```

### Dataset Split

| Dataset | Samples |
|---|---:|
| Training Set | 50,000 |
| Testing Set | 10,000 |
| Total | 60,000 |

The original image tensors have the shape:

```text
X_train → (50000, 32, 32, 3)
```

The images are flattened before being passed to the classical ML models:

```text
X_train → (50000, 3072)
X_test  → (10000, 3072)
```

The notebook performs this transformation using NumPy reshaping.

---

# 🏷️ CIFAR-10 Classes

The dataset contains 10 classes:

| Label | Class |
|---:|---|
| 0 | Airplane |
| 1 | Automobile |
| 2 | Bird |
| 3 | Cat |
| 4 | Deer |
| 5 | Dog |
| 6 | Frog |
| 7 | Horse |
| 8 | Ship |
| 9 | Truck |

Each class contains 1,000 images in the test set, giving a total test set of 10,000 images.

---

# 🧠 Project Approach

The overall workflow used in this project is:

```text
CIFAR-10 Dataset
       │
       ▼
Image Preprocessing
       │
       ▼
Flatten 32×32×3 Images
       │
       ▼
3072 Features
       │
       ▼
Feature Scaling
       │
       ▼
PCA Dimensionality Reduction
       │
       ▼
Machine Learning Model
       │
       ▼
Predictions
       │
       ▼
Classification Report
       │
       ▼
Model Comparison
```

---

# 🔄 Image Preprocessing

Machine Learning algorithms such as Logistic Regression and SVM expect tabular feature vectors rather than image tensors.

Therefore, each RGB image:

```text
32 × 32 × 3
```

is flattened into:

```text
3072-dimensional vector
```

For example:

```python
x_train = X_train.reshape(X_train.shape[0], -1)
x_test = X_test.reshape(X_test.shape[0], -1)
```

The labels are also converted from a two-dimensional array into a one-dimensional array:

```python
y_train = y_train.ravel()
y_test = y_test.ravel()
```

This results in:

```text
Training features : 50,000 × 3,072
Testing features  : 10,000 × 3,072
```

---

# 📉 Why PCA?

Each image initially contains **3,072 features**.

Working directly with all these features can make classical ML models computationally expensive and can introduce redundant information.

Therefore, **Principal Component Analysis (PCA)** is used.

PCA transforms the original feature space into a smaller number of principal components while attempting to preserve the most important variance in the data.

The project first examines cumulative explained variance to determine an appropriate number of components.

The final pipelines use approximately:

```text
662 components
```

for the StandardScaler pipelines and:

```text
658 components
```

for the MinMaxScaler pipelines.

For example, the StandardScaler + PCA + Logistic Regression pipeline uses:

```python
PCA(n_components=662)
```

while the MinMaxScaler + PCA + Logistic Regression pipeline uses:

```python
PCA(n_components=658)
```

The PCA and scaling steps are incorporated into Scikit-learn pipelines. 
---

# ⚖️ Feature Scaling

Two different scaling techniques are investigated.

## 1. StandardScaler

StandardScaler transforms features approximately according to:

```text
z = (x - μ) / σ
```

where:

- `μ` = mean
- `σ` = standard deviation

It is particularly useful for algorithms such as SVM and Logistic Regression because these algorithms can be sensitive to feature scales.

---

## 2. MinMaxScaler

MinMaxScaler transforms values into a specified range, typically:

```text
[0, 1]
```

The project compares both scaling approaches to determine their effect on classification performance.

---

# 🤖 Models Compared

Four complete ML pipelines were evaluated.

---

## Model 1 — StandardScaler + PCA + Logistic Regression

### Pipeline

```text
StandardScaler
      ↓
PCA (662 components)
      ↓
Logistic Regression
```

The implementation uses:

```python
Pipeline([
    ('scale', StandardScaler()),
    ('pca', PCA(n_components=662)),
    ('lr', LogisticRegression(max_iter=10000))
])
```

### Performance

```text
Accuracy: 40%
```
The classification report gives approximately:

```text
Accuracy       0.40
Macro Average  0.40
Weighted Avg   0.40
```


The classification report shows an overall accuracy of approximately **0.40**.

### Analysis

Logistic Regression performs relatively poorly on CIFAR-10.

Although PCA reduces the dimensionality and scaling makes optimization more appropriate, Logistic Regression is still fundamentally a relatively simple linear classifier.

CIFAR-10 contains complex visual patterns and relationships that are difficult to separate using a linear decision boundary.

---

# Model 2 — MinMaxScaler + PCA + Logistic Regression

### Pipeline

```text
MinMaxScaler
      ↓
PCA (658 components)
      ↓
Logistic Regression
```

Implementation:

```python
Pipeline([
    ('scale', MinMaxScaler()),
    ('pca', PCA(n_components=658)),
    ('lr', LogisticRegression(max_iter=10000))
])
```

The notebook uses `MinMaxScaler`, PCA with 658 components, and Logistic Regression with `max_iter=10000`.

### Performance

```text
Accuracy: 40%
```

The classification report gives approximately:

```text
Accuracy       0.40
Macro Average  0.40
Weighted Avg   0.40
```



### Analysis

Changing StandardScaler to MinMaxScaler does not provide a meaningful improvement.

Both Logistic Regression approaches remain around 40% accuracy.

This indicates that the primary limitation is not simply the scaling method. The linear nature of Logistic Regression appears to be the larger limitation for this image classification problem.

---

# Model 3 — MinMaxScaler + PCA + SVM

### Pipeline

```text
MinMaxScaler
      ↓
PCA (658 components)
      ↓
SVM
      ↓
RBF Kernel
```

Implementation:

```python
Pipeline([
    ('scale', MinMaxScaler()),
    ('pca', PCA(n_components=658)),
    ('svm', SVC(kernel='rbf', C=10))
])
```

The notebook uses an RBF-kernel SVM with `C=10`.

### Performance

```text
Accuracy: 57%
```

The classification report shows:

```text
Accuracy       0.57
Macro Average  0.57
Weighted Avg   0.57
```



### Analysis

This is a major improvement over both Logistic Regression models.

The accuracy increases from approximately:

```text
40% → 57%
```

This is a **17 percentage-point improvement**.

The major difference is the use of the **RBF kernel SVM**, which can model nonlinear decision boundaries.

This makes SVM much better suited than Logistic Regression for the transformed CIFAR-10 feature space.

---

# Model 4 — StandardScaler + PCA + SVM

### Pipeline

```text
StandardScaler
      ↓
PCA (662 components)
      ↓
SVM
      ↓
RBF Kernel
```

Implementation:

```python
Pipeline([
    ('scale', StandardScaler()),
    ('pca', PCA(n_components=662)),
    ('svm', SVC(kernel='rbf', C=10))
])
```

The notebook uses StandardScaler, PCA with 662 components and an SVM with `C=10`.

### Performance

```text
Accuracy: 57%
```

The classification report reports:

```text
Accuracy       0.57
Macro Average  0.57
Weighted Avg   0.57
```



### Class-wise performance

| Class | Precision | Recall | F1-score |
|---:|---:|---:|---:|
| 0 | 0.62 | 0.67 | 0.65 |
| 1 | 0.65 | 0.68 | 0.67 |
| 2 | 0.45 | 0.47 | 0.46 |
| 3 | 0.39 | 0.42 | 0.40 |
| 4 | 0.51 | 0.49 | 0.50 |
| 5 | 0.51 | 0.46 | 0.48 |
| 6 | 0.63 | 0.62 | 0.62 |
| 7 | 0.66 | 0.59 | 0.63 |
| 8 | 0.71 | 0.69 | 0.70 |
| 9 | 0.63 | 0.62 | 0.63 |

The strongest performance is obtained on **class 8 (ship)** with an F1-score of approximately **0.70**, while classes such as cat and bird are considerably more difficult.

---

# 📊 Model Comparison

| Rank | Model | Scaling | PCA Components | Classifier | Accuracy |
|---:|---|---|---:|---|---:|
| 🥇 1 | StandardScaler + PCA + SVM | StandardScaler | 662 | RBF SVM | **57%** |
| 🥈 2 | MinMaxScaler + PCA + SVM | MinMaxScaler | 658 | RBF SVM | **57%** |
| 🥉 3 | StandardScaler + PCA + Logistic Regression | StandardScaler | 662 | Logistic Regression | **40%** |
| 4 | MinMaxScaler + PCA + Logistic Regression | MinMaxScaler | 658 | Logistic Regression | **40%** |

The two SVM models significantly outperform the two Logistic Regression models. The notebook reports approximately **57% accuracy for both SVM pipelines**, compared with approximately **40% for both Logistic Regression pipelines**.

---

# 📈 Performance Improvement

Comparing the best Logistic Regression model with the best SVM model:

```text
Logistic Regression → 40%
SVM                 → 57%
```

Absolute improvement:

```text
57% - 40% = 17 percentage points
```

Relative improvement over the 40% baseline:

```text
(57 - 40) / 40 × 100 ≈ 42.5%
```

Therefore, the SVM approach provides approximately a **42.5% relative improvement in accuracy** over the Logistic Regression baseline.

---

# 🏆 Best Model

## StandardScaler + PCA + RBF SVM

The best model in this project is:

```text
StandardScaler
       ↓
PCA (662 components)
       ↓
RBF SVM (C=10)
```

### Final Accuracy

**57%**

## ⏱️ Training Time vs Accuracy

Although the SVM models significantly outperform Logistic Regression in terms of accuracy, they require substantially more computational time to train.

The SVM models took approximately **1 hour** to train on the CIFAR-10 training dataset, whereas Logistic Regression completed in 7 min.

This highlights an important trade-off:

| Model | Accuracy | Training Time | Trade-off |
|---|---:|---:|---|
| StandardScaler + PCA + Logistic Regression | 40% | ~7 min | Faster, but lower accuracy |
| MinMaxScaler + PCA + Logistic Regression | 40% | ~7 min | Faster, but lower accuracy |
| MinMaxScaler + PCA + SVM | 57% | ~1 hour | Higher accuracy, slower training |
| StandardScaler + PCA + SVM | 57% | ~1 hour | Higher accuracy, slower training |

### Key Observation

The SVM models provide a significant improvement in classification performance:

**40% → 57% accuracy**

However, this improvement comes at the cost of considerably longer training time.

Therefore, the choice of model depends on the application:

- **If accuracy is the priority:** SVM is the better choice.
- **If training speed and computational efficiency are the priority:** Logistic Regression may be preferable.
- **For this project:** SVM is selected as the **best-performing model based on accuracy**, despite its approximately 1-hour training time.

> **Note:** Training time can vary depending on hardware, CPU, RAM, Scikit-learn version, and system load. The ~1 hour figure represents the approximate training time observed during this experiment.


### Why SVM?

The biggest improvement comes from replacing Logistic Regression with a nonlinear SVM.

Logistic Regression produces approximately:

```text
40% accuracy
```

while the RBF SVM produces:

```text
57% accuracy
```

This demonstrates that the nonlinear decision boundary provided by the RBF kernel is substantially more effective for the transformed CIFAR-10 data.

---

# 🔍 Why Logistic Regression Performs Worse

Logistic Regression is essentially a linear classifier.

After PCA, the model still needs to distinguish between visually similar classes such as:

```text
cat ↔ dog
bird ↔ airplane
deer ↔ horse
```

These classes can have highly complex and overlapping feature distributions.

A linear decision boundary is therefore insufficient to capture all the relationships present in the data.

This explains why both Logistic Regression pipelines remain around 40% accuracy.

---

# 🔍 Why SVM Performs Better

The SVM uses an **RBF kernel**:

```python
SVC(kernel='rbf', C=10)
```

The RBF kernel allows the classifier to construct nonlinear decision boundaries.

Instead of attempting to separate all classes using simple linear boundaries, SVM can model more complex relationships in the PCA-transformed feature space.

This results in a significant performance improvement.

---

# 🧮 Effect of PCA

The original feature space contains:

```text
3072 features
```

PCA reduces this to approximately:

```text
658–662 components
```

This provides dimensionality reduction while retaining a large amount of the useful information captured in the original features.

It also makes the downstream classical ML algorithms more manageable than operating directly on all 3,072 pixel features.

The project determines PCA dimensionality using cumulative explained variance before constructing the final pipelines.

---

# 🔬 Important Observation

An important conclusion from this experiment is that **changing the scaler has much less impact than changing the classifier**.

### Logistic Regression

```text
StandardScaler → 40%
MinMaxScaler   → 40%
```

### SVM

```text
StandardScaler → 57%
MinMaxScaler   → 57%
```

The dominant improvement comes from:

```text
Logistic Regression → RBF SVM
```

rather than:

```text
StandardScaler → MinMaxScaler
```

This is an important Machine Learning insight from the experiment.

---

# 🧪 Evaluation Metric

The models are evaluated using a classification report containing:

- Precision
- Recall
- F1-score
- Accuracy
- Macro average
- Weighted average

Since CIFAR-10 has balanced classes, accuracy provides a useful overall comparison, while macro F1-score helps evaluate whether performance is consistent across all classes.

---

# 💻 Technologies Used

### Programming Language

- Python

### Libraries

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow/Keras dataset loader

### Machine Learning Techniques

- Feature Scaling
- Principal Component Analysis
- Logistic Regression
- Support Vector Machine
- RBF Kernel
- Classification Report

---

# 📁 Project Structure

```text
CIFAR-10-ML/
│
├── CIFAR_10.ipynb
└── README.md
```

---

# 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Navigate into the project

```bash
cd CIFAR-10-ML
```

### 3. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

### 4. Open the notebook

```bash
jupyter notebook CIFAR_10.ipynb
```

Alternatively, the notebook can be opened using Google Colab.

---

# 🔮 Future Improvements

Several improvements could be explored in future versions.

### 1. HOG + SVM

Instead of directly using pixel values, extract **Histogram of Oriented Gradients (HOG)** features and train an SVM.

This could provide more meaningful shape and edge information.

### 2. Hyperparameter Tuning

Use `GridSearchCV` to optimize parameters such as:

```text
SVM C
SVM gamma
PCA n_components
```

### 3. Compare Different PCA Values

Experiment with multiple PCA dimensions:

```text
100
200
300
400
500
600
700
...
```

and evaluate the resulting accuracy.

### 4. Feature Engineering

Explore alternative image representations such as:

- HOG
- grayscale features
- color histograms
- edge features
- combinations of handcrafted features

### 5. Additional Classical ML Models

Other algorithms could be evaluated:

- K-Nearest Neighbors
- Random Forest
- Linear SVM
- Naive Bayes
- Gradient Boosting

### 6. Ensemble Methods

Multiple classical classifiers could potentially be combined to improve robustness.

---

# 🏁 Conclusion

This project demonstrates that classical Machine Learning can be applied to CIFAR-10, but model selection has a major effect on performance.

The experiment compared four different combinations of scaling, PCA and classification algorithms.

The results were:

```text
StandardScaler + PCA + Logistic Regression → 40%
MinMaxScaler + PCA + Logistic Regression   → 40%
MinMaxScaler + PCA + SVM                  → 57%
StandardScaler + PCA + SVM                → 57%
```

The best-performing approach is:

## 🏆 StandardScaler + PCA + RBF SVM

with an accuracy of approximately:

# **57%**

The key takeaway is that **the nonlinear RBF SVM substantially outperforms Logistic Regression on this PCA-transformed CIFAR-10 feature space**.

This project therefore provides a useful demonstration of the strengths and limitations of classical Machine Learning for image classification.
