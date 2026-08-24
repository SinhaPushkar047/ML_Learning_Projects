# Breast Cancer Prediction using KNN, PCA & GridSearchCV

## Overview

This project builds a **breast cancer classification model using K-Nearest Neighbors (KNN)** on the Breast Cancer Wisconsin dataset.

The notebook focuses on a practical machine-learning workflow:

- Loading and cleaning the dataset
- Encoding the target variable
- Splitting the data into training and testing sets
- Scaling features with `MinMaxScaler`
- Applying **Principal Component Analysis (PCA)** for dimensionality reduction
- Selecting the number of PCA components using explained variance
- Building a KNN model inside a `Pipeline`
- Tuning the KNN `n_neighbors` hyperparameter using `GridSearchCV`
- Evaluating the final model using a classification report

The final model achieves **96% accuracy on the held-out test set**.

---

## Dataset

The notebook loads the data directly from the project's GitHub repository:

```text
https://raw.githubusercontent.com/SinhaPushkar047/ML_Learning_Projects/refs/heads/main/06-Breast-Cancer-Prediction/data.csv
```

The dataset contains measurements computed from breast-cell images and a binary `diagnosis` target.

The original dataset includes identifier/non-feature columns that are removed:

```python
.drop(columns=['id', 'Unnamed: 32'])
```

After removing those columns, the remaining feature columns are used for classification.

### Target Encoding

The `diagnosis` column is converted from categorical labels to numerical labels using `LabelEncoder`:

```python
lc = LabelEncoder()
brcn['diagnosis'] = lc.fit_transform(brcn['diagnosis'])
```

This makes the target suitable for the scikit-learn classification pipeline.

---

## Machine Learning Workflow

The notebook follows this sequence:

```text
Raw Dataset
     ↓
Remove ID / Unused Columns
     ↓
Label Encode Target
     ↓
Train-Test Split
     ↓
MinMax Scaling
     ↓
PCA
     ↓
KNN
     ↓
GridSearchCV
     ↓
Final Prediction
     ↓
Classification Report
```

---

## 1. Data Preprocessing

Two columns are removed before modeling:

- `id`
- `Unnamed: 32`

These columns are not used as predictive features.

The target column `diagnosis` is then label encoded.

---

## 2. Train-Test Split

The dataset is split using:

```python
train_test_split(
    ...,
    test_size=0.2,
    random_state=42
)
```

This creates:

- **80% training data**
- **20% testing data**

The `random_state=42` makes the split reproducible.

The notebook's test set contains **114 samples**.

---

## 3. Feature Scaling

KNN is a distance-based algorithm, so feature scaling is important.

The notebook uses:

```python
MinMaxScaler()
```

which scales each feature into a common range.

This prevents features with larger numerical magnitudes from dominating the distance calculation.

Scaling is also placed directly inside the final pipeline:

```python
Pipeline([
    ('scaling', MinMaxScaler()),
    ...
])
```

This is important because the transformations are learned only from the appropriate training data during cross-validation.

---

## 4. PCA for Dimensionality Reduction

The notebook first fits PCA on the scaled training features:

```python
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(x_train)

pca_full = PCA()
pca_full.fit(X_train_scaled)
```

It then calculates cumulative explained variance:

```python
cumulative_variance = np.cumsum(
    pca_full.explained_variance_ratio_
)
```

A graph is created to visualize how the retained variance changes with the number of principal components.

### Selected Number of Components

The notebook sets:

```python
threshold = 0.99
```

and finds the smallest number of components that retain at least 99% of the variance.

The result is:

```text
Components required for 99% variance: 16
```

Therefore, the final KNN pipeline uses:

```python
PCA(n_components=16)
```

This reduces the feature space while retaining approximately 99% of the variance present in the scaled training data.

---

## 5. KNN Pipeline

The final modeling pipeline is:

```python
pipe2 = Pipeline([
    ('scaling', MinMaxScaler()),
    ('pca', PCA(n_components=16)),
    ('knn', KNeighborsClassifier())
])
```

The pipeline combines:

1. **MinMaxScaler**
2. **PCA with 16 components**
3. **KNeighborsClassifier**

Keeping these steps inside one pipeline makes the preprocessing and model training operate together during cross-validation.

The notebook also creates another pipeline:

```python
pipe = Pipeline([
    ('scaling', MinMaxScaler()),
    ('knn', KNeighborsClassifier())
])
```

However, this pipeline is not used in the final `GridSearchCV` experiment. The actual tuned model is `pipe2`, which includes PCA.

---

## 6. Hyperparameter Tuning with GridSearchCV

The notebook tunes the KNN `n_neighbors` parameter.

The tested values are:

```python
n_neighbors = [5, 10, 15, 20, 25]
```

Because the model is inside a pipeline, the parameter is specified as:

```python
param_grid = {
    'knn__n_neighbors': n_neighbors
}
```

This tells GridSearchCV to tune the `n_neighbors` parameter inside the `knn` step of the pipeline.

### Cross-Validation Setup

The notebook uses:

```python
GridSearchCV(
    estimator=pipe2,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)
```

This means:

- **5-fold cross-validation**
- Evaluation metric: **F1-score**
- All available CPU cores are used with `n_jobs=-1`
- Five candidate neighbor values are evaluated

The notebook reports:

```text
Fitting 5 folds for each of 5 candidates, totalling 25 fits
```

So GridSearchCV performs **25 model fits** in total.

The notebook does not explicitly print the selected `best_params_`, so this README does not claim a specific optimal `n_neighbors` value.

---

## 7. Model Evaluation

After hyperparameter tuning, predictions are generated on the unseen test set:

```python
y_pred = grid.predict(x_test)
```

The notebook evaluates the result using:

```python
classification_report(y_test, y_pred)
```

### Test Set Results

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 | 0.96 | 0.99 | 0.97 | 71 |
| 1 | 0.98 | 0.93 | 0.95 | 43 |
| **Accuracy** | | | **0.96** | **114** |
| Macro Avg | 0.97 | 0.96 | 0.96 | 114 |
| Weighted Avg | 0.97 | 0.96 | 0.96 | 114 |

### Key Results

- **Test Accuracy:** 96%
- **Macro F1-score:** 0.96
- **Weighted F1-score:** 0.96

The model performs strongly on both classes. The reported recall is particularly high for class `0` (0.99), while class `1` has a recall of 0.93.

---

## Why Scaling and PCA Are Used

### Why MinMaxScaler?

KNN determines predictions using distances between samples.

If one feature has values in the hundreds or thousands while another feature has values close to zero, the larger-valued feature can disproportionately influence the distance calculation.

Min-max scaling puts the features onto a comparable scale before KNN calculates distances.

### Why PCA?

The dataset contains many correlated features.

PCA transforms the original features into a smaller set of principal components while preserving as much variance as possible.

In this notebook:

```text
Original feature space
        ↓
      PCA
        ↓
16 principal components
        ↓
      KNN
```

The 16 components were selected because they retain at least **99% cumulative explained variance** on the scaled training data.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

### Important Scikit-learn Components

- `LabelEncoder`
- `train_test_split`
- `MinMaxScaler`
- `PCA`
- `KNeighborsClassifier`
- `Pipeline`
- `GridSearchCV`
- `classification_report`

---

## Project Structure

A suitable project structure is:

```text
06-Breast-Cancer-Prediction/
│
├── data.csv
├── Breast_Cancer_Prediction.ipynb
└── README.md
```

---

## Key Takeaways

This project demonstrates several important classical machine-learning concepts together:

- **Label encoding** for categorical targets
- **Train-test splitting** for evaluating generalization
- **Feature scaling** for a distance-based algorithm
- **PCA** for dimensionality reduction
- **Pipeline** for combining preprocessing and modeling
- **GridSearchCV** for hyperparameter tuning
- **F1-score** as the optimization metric
- **Classification report** for detailed evaluation

The final workflow combines these concepts into a single reproducible KNN model and achieves **96% test accuracy**.

---

## Limitations / Possible Improvements

The notebook is focused on the core KNN + PCA workflow, so several useful extensions are not included yet:

- Print and report `grid.best_params_`
- Report `grid.best_score_`
- Compare KNN with and without PCA
- Compare different scaling methods
- Test a wider range of `n_neighbors`
- Evaluate additional KNN parameters such as distance metric and weighting
- Add a confusion matrix
- Add ROC-AUC and ROC curve
- Compare KNN with other classical ML models such as Logistic Regression, SVM, or Random Forest
- Add a separate validation/test analysis for more robust model comparison

These would make the experiment more comprehensive without changing the main approach.

---

## Conclusion

This project implements a complete classical machine-learning pipeline for breast cancer classification using **KNN + MinMaxScaler + PCA + GridSearchCV**.

PCA reduces the feature space to **16 principal components while retaining 99% variance**, and GridSearchCV evaluates multiple KNN neighborhood sizes using **5-fold cross-validation with F1-score as the optimization metric**.

On the held-out test set, the final workflow achieves **96% accuracy**, with balanced precision, recall, and F1 performance across the two encoded classes.
