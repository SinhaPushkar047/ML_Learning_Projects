# Bank Deposit Prediction

A machine learning classification project that predicts whether a bank customer will subscribe to a term deposit (`yes` / `no`) based on customer and campaign-related features.

## Project Overview

This project follows a typical supervised machine learning workflow:

1. Load the bank marketing dataset.
2. Perform basic exploratory data analysis (EDA).
3. Encode the target and categorical features.
4. Scale and transform numerical features.
5. Apply Principal Component Analysis (PCA) for dimensionality reduction.
6. Train and tune multiple classification models using cross-validated `GridSearchCV`.
7. Evaluate models using classification metrics and ROC-AUC.
8. Experiment with probability-threshold tuning using ROC curves.

## Dataset

The notebook loads the dataset from the project's GitHub repository:

`07-Bank-Deposit-Prediction/bank.csv`

The dataset contains **11,162 rows and 17 columns**.

### Features

| Feature | Type | Description |
|---|---|---|
| `age` | Numerical | Customer age |
| `job` | Categorical | Type of job |
| `marital` | Categorical | Marital status |
| `education` | Categorical | Education level |
| `default` | Categorical | Whether the customer has credit in default |
| `balance` | Numerical | Average yearly balance |
| `housing` | Categorical | Whether the customer has a housing loan |
| `loan` | Categorical | Whether the customer has a personal loan |
| `contact` | Categorical | Contact communication type |
| `day` | Numerical | Day of the month of the last contact |
| `month` | Categorical | Month of the last contact |
| `duration` | Numerical | Duration of the last contact |
| `campaign` | Numerical | Number of contacts performed during the campaign |
| `pdays` | Numerical | Days since the customer was last contacted in a previous campaign |
| `previous` | Numerical | Number of contacts performed before this campaign |
| `poutcome` | Categorical | Outcome of the previous marketing campaign |
| `deposit` | Target | Whether the customer subscribed to a term deposit |

The notebook shows no missing values in the dataset.

## Exploratory Data Analysis

The notebook performs basic numerical analysis by:

- Separating numerical and categorical columns.
- Checking numerical-feature skewness.
- Plotting histograms with KDE curves for numerical features.

## Preprocessing

### Target Encoding

The target variable, `deposit`, is converted from `yes` / `no` to numeric labels using `LabelEncoder`.

### Train-Test Split

The data is split into:

- **80% training data**
- **20% test data**
- `random_state=42`

This gives **2,233 observations** in the test set.

### Feature Transformation

The notebook uses a `ColumnTransformer` with:

- `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` for categorical features.
- `StandardScaler()` for numerical features.
- `PowerTransformer()` for numerical features.

### PCA

PCA is fitted on the transformed training data and the cumulative explained variance is examined.

A **99% explained-variance threshold** results in **38 principal components**, which are then used by the Gaussian Naive Bayes and KNN pipelines.

## Models

Three classification approaches are evaluated.

### 1. Gaussian Naive Bayes

The notebook tunes `var_smoothing` with 5-fold cross-validation using F1 score.

Best parameter:

```text
var_smoothing = 1e-05
```

Best cross-validation F1 score:

```text
67.80%
```

Test-set performance:

```text
Accuracy: 72%
F1 (class 1): 0.69
```

### 2. K-Nearest Neighbors (KNN)

The following hyperparameters are tuned:

- `n_neighbors`
- `weights`
- `metric`
- `p`

Best configuration:

```text
n_neighbors = 21
weights = distance
metric = euclidean
p = 1
```

Best cross-validation F1 score:

```text
81.24%
```

Test-set performance at the default classification threshold:

```text
Accuracy: 82%
F1 (class 1): 0.81
```

ROC-AUC:

```text
0.9011
```

#### KNN Threshold Tuning

The notebook also selects a probability threshold using the point on the ROC curve closest to `(FPR=0, TPR=1)`.

Selected threshold:

```text
0.4462
```

At this threshold:

```text
TPR: 0.8669
FPR: 0.2033
Accuracy: 83%
F1 (class 1): 0.83
```

This improves the balance between precision and recall compared with the default threshold.

### 3. Decision Tree

The notebook tunes:

- `criterion`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

Best configuration:

```text
criterion = gini
max_depth = 10
min_samples_split = 2
min_samples_leaf = 10
```

Best cross-validation F1 score:

```text
82.32%
```

Test-set performance at the default threshold:

```text
Accuracy: 82%
F1 (class 1): 0.82
```

ROC-AUC:

```text
0.8845
```

#### Decision Tree Threshold Tuning

Selected probability threshold:

```text
0.4167
```

At this threshold:

```text
TPR: 0.8707
FPR: 0.2178
Accuracy: 82%
F1 (class 1): 0.82
```

## Model Comparison

| Model | CV F1 | Test Accuracy | Class 1 F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Gaussian Naive Bayes | 67.80% | 72% | 0.69 | — |
| KNN | 81.24% | 82% | 0.81 | 0.9011 |
| KNN + tuned threshold | — | **83%** | **0.83** | 0.9011 |
| Decision Tree | **82.32%** | 82% | 0.82 | 0.8845 |
| Decision Tree + tuned threshold | — | 82% | 0.82 | 0.8845 |

Based on the notebook's reported test metrics, **KNN with the ROC-based probability threshold performs best on the test set**, reaching **83% accuracy and 0.83 F1 for the positive class**.

## Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook / Google Colab

## Key Scikit-learn Components

The notebook makes use of:

```python
LabelEncoder
OneHotEncoder
StandardScaler
PowerTransformer
ColumnTransformer
Pipeline
train_test_split
GridSearchCV
PCA
GaussianNB
KNeighborsClassifier
DecisionTreeClassifier
classification_report
roc_curve
roc_auc_score
```

## Project Structure

```text
Bank-Deposit-Prediction/
│
├── Bank_Deposit_Prediction.ipynb
├── bank.csv
└── README.md
```

## How to Run

1. Clone or download the project.
2. Open `Bank_Deposit_Prediction.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
3. Run the cells from top to bottom.

The notebook currently loads the dataset directly from GitHub, so a local `bank.csv` file is not required when the URL is accessible.

## Conclusion

The project compares three different classification strategies and demonstrates that model selection alone is not the only way to improve classification performance. For KNN, adjusting the probability threshold based on the ROC curve increases the positive-class F1 score from **0.81 to 0.83** and test accuracy from **82% to 83%**.

The notebook therefore provides practical experience with:

- EDA
- categorical encoding
- numerical transformation
- scaling
- PCA
- pipelines
- hyperparameter tuning
- cross-validation
- ROC curves
- ROC-AUC
- probability threshold optimization

## Notes

The README reflects the workflow and results present in the notebook as provided. It does not introduce additional modeling steps that are not implemented there.
