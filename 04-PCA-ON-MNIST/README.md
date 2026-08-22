# MNIST Classification using PCA

## 📌 Overview

This project explores **Principal Component Analysis (PCA)** for dimensionality reduction on the **MNIST handwritten digit dataset**, followed by classification using **Logistic Regression** and **K-Nearest Neighbors (KNN)**.

The original MNIST images contain **784 features (28 × 28 pixels)**. PCA is used to reduce the feature space to **100 principal components** while retaining approximately **70.59% of the variance**.

Four different experiments are performed to understand how the choice of scaler and classifier affects performance.

---

## 📂 Dataset

The project uses the MNIST dataset:

- **Training samples:** 60,000
- **Test samples:** 10,000
- **Original features:** 784 pixels
- **Classes:** 10 digits (`0`–`9`)
- **Reduced features after PCA:** 100

The notebook loads the training and test CSV files directly from the project's GitHub repository.

---

## 🔄 Workflow

The overall workflow is:

1. Load the MNIST training and test datasets.
2. Separate features and target labels.
3. Visualize an MNIST digit.
4. Scale the pixel values.
5. Apply PCA with `n_components=100`.
6. Train classification models on the PCA-transformed data.
7. Evaluate each model using a classification report.
8. Compare all four experiments and select the best-performing approach.

### PCA Configuration

```python
PCA(n_components=100)
```

The 784 original pixel features are reduced to 100 components.

The notebook reports an explained variance of approximately:

**70.59%**

---

## 🧪 Experiments

### Experiment 1 — StandardScaler + PCA + Logistic Regression

Pipeline:

```text
StandardScaler
      ↓
PCA (100 components)
      ↓
Logistic Regression
```

**Test Accuracy: 92%**

This provides a strong baseline, but Logistic Regression has difficulty capturing the nonlinear patterns present in handwritten digits.

---

### Experiment 2 — MinMaxScaler + PCA + Logistic Regression

Pipeline:

```text
MinMaxScaler
      ↓
PCA (100 components)
      ↓
Logistic Regression
```

**Test Accuracy: 92%**

Changing the scaler from StandardScaler to MinMaxScaler does not produce a meaningful improvement for Logistic Regression in this experiment.

---

### Experiment 3 — StandardScaler + PCA + KNN

Pipeline:

```text
StandardScaler
      ↓
PCA (100 components)
      ↓
KNN (n_neighbors=5)
```

**Test Accuracy: 96%**

KNN performs considerably better than Logistic Regression because it can use the similarity between digit representations in the PCA feature space.

---

### Experiment 4 — MinMaxScaler + PCA + KNN

Pipeline:

```text
MinMaxScaler
      ↓
PCA (100 components)
      ↓
KNN (n_neighbors=5)
```

**Test Accuracy: 97%**

This is the best-performing experiment in the notebook.

---

## 📊 Results Comparison

| Experiment | Scaling | PCA | Model | KNN Neighbors | Accuracy |
|---|---|---:|---|---:|---:|
| Experiment 1 | StandardScaler | 100 | Logistic Regression | — | **92%** |
| Experiment 2 | MinMaxScaler | 100 | Logistic Regression | — | **92%** |
| Experiment 3 | StandardScaler | 100 | KNN | 5 | **96%** |
| Experiment 4 | MinMaxScaler | 100 | KNN | 5 | **97%** |

### 🏆 Best Model

**Experiment 4 — MinMaxScaler + PCA + KNN**

It achieves the highest test accuracy of **97%**.

Compared with the other experiments:

- **+5 percentage points** over both Logistic Regression experiments.
- **+1 percentage point** over StandardScaler + PCA + KNN.
- It also achieves a **97% macro-average F1-score** and **97% weighted-average F1-score** in the notebook's classification report.

Therefore, among the four tested approaches, **MinMaxScaler + PCA + KNN is the best model for this experiment**.

---

## 📈 Classification Performance

The best model, **MinMaxScaler + PCA + KNN**, achieves the following overall metrics on the 10,000-image test set:

| Metric | Score |
|---|---:|
| Accuracy | **0.97** |
| Macro Precision | **0.97** |
| Macro Recall | **0.97** |
| Macro F1-score | **0.97** |
| Weighted Precision | **0.97** |
| Weighted Recall | **0.97** |
| Weighted F1-score | **0.97** |

The model performs consistently across all ten digit classes, with class-wise F1-scores ranging from approximately **0.96 to 0.98**.

---

## 💡 Key Observations

1. **PCA significantly reduces dimensionality** from 784 features to 100 features.
2. The selected 100 components retain approximately **70.59% of the variance**.
3. Logistic Regression achieves around **92% accuracy** with both scaling approaches.
4. KNN performs substantially better than Logistic Regression on the PCA-transformed MNIST data.
5. MinMaxScaler gives KNN a small improvement over StandardScaler, increasing accuracy from **96% to 97%**.
6. Based on the four experiments conducted, **MinMaxScaler + PCA + KNN** is the best-performing approach.

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
  - `StandardScaler`
  - `MinMaxScaler`
  - `PCA`
  - `LogisticRegression`
  - `KNeighborsClassifier`
  - `classification_report`

---

## 📁 Project Structure

```text
04-PCA-ON-MNIST/
│
├── Untitled4 (1)(1).ipynb
├── mnist_train.csv
├── mnist_test.csv
└── README.md
```

---

## 🚀 Conclusion

This project demonstrates that dimensionality reduction with **PCA** can make the MNIST feature space much smaller while still allowing strong classification performance.

Among the four approaches tested, **MinMaxScaler + PCA + KNN** achieved the highest accuracy of **97%**, making it the best model for this notebook.

The experiments also show that model choice can have a larger impact on performance than changing the scaling method: switching from Logistic Regression to KNN improved accuracy from **92% to 96–97%**.
