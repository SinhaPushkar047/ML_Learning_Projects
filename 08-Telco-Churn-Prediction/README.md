# Telco Customer Churn Prediction

A machine learning project for predicting whether a telecom customer is likely to churn. The notebook walks through exploratory data analysis (EDA), preprocessing, baseline classification models, hyperparameter tuning, threshold tuning, and ensemble learning.

## Project Overview

The dataset contains **7,043 customer records** and the target variable is `Churn`.

The target distribution in the notebook is:

- `No`: 5,174 customers
- `Yes`: 1,869 customers

This makes churn prediction an imbalanced binary classification problem, so **F1-score** is used as the main optimization metric for most models. The notebook also evaluates accuracy, precision, recall, confusion matrices, and ROC-AUC where applicable.

## Workflow

The notebook follows this workflow:

1. Load the Telco Customer Churn dataset.
2. Remove the `customerID` column.
3. Convert `TotalCharges` to numeric and treat blank values as missing.
4. Perform basic EDA and inspect feature distributions/skewness.
5. Analyze the target imbalance between churned and non-churned customers.
6. Split the data into training and test sets using an 80/20 split.
7. Encode the target using `LabelEncoder`.
8. Preprocess numerical and categorical features using a `ColumnTransformer`.
9. Train and tune multiple classification algorithms.
10. Apply cost-sensitive learning through class weighting where supported.
11. Perform probability/threshold tuning for selected models.
12. Compare model performance using classification reports and confusion matrices.
13. Explore ensemble methods including Random Forest, AdaBoost, Gradient Boosting, XGBoost, Bagging, and Voting.

## Handling Class Imbalance

The target variable `Churn` is **imbalanced**. The dataset contains:

- `No Churn`: 5,174 customers (~73.5%)
- `Churn`: 1,869 customers (~26.5%)

This imbalance makes the minority **class 1 (churn)** harder to predict accurately. A model can achieve reasonable overall accuracy while still missing a significant number of churners.

### Current Limitation

The current notebook **does not apply dedicated data-level imbalance techniques such as SMOTE, Random Over-Sampling, or Random Under-Sampling**.

It also does not build a separate **cost-sensitive learning strategy specifically designed around the business cost of false negatives**.

Although ensemble models such as Random Forest, AdaBoost, Gradient Boosting, XGBoost, Bagging, and Voting are evaluated, they are primarily used as model-comparison/ensemble-learning techniques rather than as a complete imbalance-handling pipeline.

Random Forest hyperparameter tuning does include `class_weight` options such as `balanced` and `balanced_subsample`, but the project does not systematically compare class-weighting, SMOTE, and other imbalance strategies across all models.

### Why Class 1 Prediction Is Not Very Strong

Because the churn class is the minority class and dedicated imbalance handling is limited, the models tend to favor the majority class (`0 = No Churn`).

As a result:

- **Class 1 recall is only moderate** for several models.
- Some churners are predicted as non-churners (**false negatives**).
- Accuracy can look acceptable even when churn detection is not strong.
- The model may therefore be less effective at identifying customers who are actually likely to churn.

For example, the reported test results show class 1 F1-scores mostly around the **0.60–0.67** range rather than a very strong minority-class performance.

### Possible Improvements

The project can be improved by explicitly testing imbalance-handling approaches such as:

1. **SMOTE**
   - Generate synthetic minority-class samples from the training data.
2. **Random Over-Sampling**
   - Increase the number of minority-class training examples.
3. **Random Under-Sampling**
   - Reduce the number of majority-class training examples.
4. **Cost-Sensitive Learning**
   - Assign a higher penalty to misclassifying churners.
   - Tune class weights or use a business-specific cost matrix.
5. **Imbalance-Aware Ensemble Methods**
   - Compare models trained with class weighting or resampling against the current baseline ensembles.
6. **Threshold Optimization**
   - Select a probability threshold based on the desired recall/precision trade-off instead of automatically using `0.50`.

SMOTE or other sampling methods should be applied **only to the training folds/data**, never to the test set, to avoid data leakage.

These experiments would provide a more complete comparison of how imbalance handling affects **class 1 precision, recall, F1-score, and the number of false negatives**.

## Data Preprocessing

### Numerical features

The notebook treats these as numerical variables:

- `SeniorCitizen`
- `tenure`
- `MonthlyCharges`
- `TotalCharges`

`TotalCharges` is handled separately because the original data contains blank values.

### Categorical features

The categorical variables are one-hot encoded with:

```python
OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)
```

### Numerical transformation

The notebook uses:

```text
SimpleImputer
    ↓
StandardScaler
    ↓
PowerTransformer
```

for `TotalCharges`, while the other numerical variables use:

```text
StandardScaler
    ↓
PowerTransformer
```

The preprocessing is implemented with `Pipeline` and `ColumnTransformer`.

## Models

The notebook experiments with the following classifiers:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost
- Bagging
- Soft Voting Classifier

## Hyperparameter Tuning

Two tuning approaches are used.

### GridSearchCV

Grid search with **5-fold cross-validation** is used for:

- Logistic Regression
- KNN
- SVM
- Decision Tree

The primary scoring metric is **F1-score**.

### Optuna

Optuna is used for:

- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost
- Bagging
- Voting Classifier

Most Optuna studies use **100 trials**.

The notebook also visualizes:

- Optimization history
- Parameter importance
- Parameter slices

## Threshold Tuning

For models that provide probabilities or suitable decision scores, the notebook experiments with changing the default classification threshold.

This is useful for churn prediction because the cost of missing a potential churner can be different from the cost of incorrectly flagging a non-churning customer.

The notebook specifically explores threshold tuning for:

- Logistic Regression
- KNN
- SVM
- Random Forest
- AdaBoost

## Test Set Results

The following results are taken from the notebook's reported test-set classification reports.

| Model | Accuracy | Class 1 Precision | Class 1 Recall | Class 1 F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.82 | 0.58 | 0.68 | 0.63 |
| KNN | 0.80 | 0.64 | 0.57 | 0.60 |
| SVM | 0.82 | 0.71 | 0.55 | 0.62 |
| Decision Tree | 0.78 | 0.57 | 0.66 | 0.61 |
| Random Forest | 0.77 | 0.55 | 0.81 | 0.65 |
| AdaBoost | 0.81 | 0.63 | 0.72 | 0.67 |
| Gradient Boosting | 0.81 | 0.68 | 0.56 | 0.61 |
| XGBoost | 0.82 | 0.68 | 0.57 | 0.62 |
| Bagging | 0.81 | 0.69 | 0.53 | 0.60 |
| Voting Classifier | 0.82 | 0.68 | 0.58 | 0.63 |

### Threshold-tuned results reported in the notebook

| Model | Accuracy | Class 1 Precision | Class 1 Recall | Class 1 F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.78 | 0.56 | 0.79 | 0.66 |
| SVM | 0.74 | 0.50 | 0.87 | 0.64 |
| Random Forest | 0.76 | 0.52 | 0.86 | 0.65 |
| AdaBoost | 0.81 | 0.63 | 0.72 | 0.67 |

The threshold-tuned models show the expected trade-off: increasing churn recall can reduce precision and overall accuracy.

## ROC-AUC

The notebook reports the following ROC-AUC values for selected models:

| Model | ROC-AUC |
|---|---:|
| Logistic Regression | 0.8618 |
| KNN | 0.8434 |
| Random Forest | 0.8641 |
| AdaBoost | 0.8590 |

Among these reported values, **Random Forest has the highest ROC-AUC (0.8641)**.

## Best Hyperparameters Recorded

### Random Forest

Optuna best CV F1:

**0.6350**

```text
n_estimators=156
criterion='gini'
max_depth=9
class_weight='balanced_subsample'
min_samples_split=13
min_samples_leaf=4
max_features='sqrt'
bootstrap=True
```

### AdaBoost

Optuna searches over `n_estimators` and `learning_rate`. The notebook then evaluates the tuned classifier on the test set and applies threshold tuning.

### Gradient Boosting

Optuna best CV F1:

**0.5876**

```text
loss='log_loss'
learning_rate≈0.1454
n_estimators=100
max_depth=2
min_samples_split=19
min_samples_leaf=1
subsample≈0.6205
max_features=None
```

### XGBoost

Optuna best CV F1:

**0.5982**

```text
n_estimators=615
learning_rate≈0.0364
max_depth=8
min_child_weight=9
subsample≈0.5067
colsample_bytree≈0.9827
gamma≈5.1082
reg_alpha≈6.62e-06
reg_lambda≈8.6609
```

### Bagging

The Bagging Optuna study optimizes **accuracy** rather than F1-score.

Best recorded CV accuracy:

**0.8049**

```text
max_depth=7
min_samples_split=7
min_samples_leaf=2
criterion='log_loss'
n_estimators=204
max_samples≈0.6588
max_features≈0.5359
bootstrap=False
```

### Voting Classifier

The soft Voting Classifier combines:

- Logistic Regression
- Decision Tree
- SVM

Best recorded CV F1:

**0.5915**

## Confusion Matrix

The notebook uses confusion matrices to examine the number of:

- True Negatives
- False Positives
- False Negatives
- True Positives

This is particularly important for churn prediction because **false negatives represent customers who churned but were not identified by the model**.

## Installation

Install the main dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn optuna xgboost
```

## How to Run

1. Download or clone this project.
2. Place the dataset file expected by the notebook in the appropriate location.
3. Open `Telco_Customer_Churn_Pred.ipynb` in Jupyter Notebook, JupyterLab, Google Colab, or VS Code.
4. Install the required packages.
5. Run the notebook cells from top to bottom.

The notebook currently loads the dataset using:

```python
pd.read_csv("/content/WA_Fn-UseC_-Telco-Customer-Churn.csv")
```

When running locally, update this path to the location of your CSV file.

## Project Structure

```text
.
├── Telco_Customer_Churn_Pred.ipynb
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── README.md
```

## Key Takeaways

- The project compares a wide range of traditional and ensemble classification algorithms.
- The target is imbalanced, with churners representing only ~26.5% of the dataset.
- The dataset is imbalanced, with only ~26.5% of customers belonging to the churn class.
- Dedicated imbalance techniques such as **SMOTE and explicit over/under-sampling are not implemented** in the current notebook.
- **Cost-sensitive learning is not systematically applied as an imbalance-handling strategy** across the models, although Random Forest tuning explores `class_weight`.
- Ensemble models are evaluated, but ensemble learning alone does not guarantee good minority-class prediction.
- **Class 1 prediction remains limited**, with reported class 1 F1-scores mostly around 0.60–0.67.
- **AdaBoost achieved the highest reported class-1 test F1-score (0.67)** among the models shown.
- **Random Forest achieved the highest reported ROC-AUC (0.8641)** among the models for which ROC-AUC was recorded.
- Threshold tuning can improve churn recall, but usually introduces a precision/accuracy trade-off.
- A stronger next step would be to compare the current models against SMOTE, sampling, and cost-sensitive approaches using class 1 recall/F1 and false-negative counts.
- For this problem, looking only at accuracy would hide important differences in the model's ability to identify churners.

## Future Improvements

Possible extensions include:

- Stratified cross-validation for more consistent class representation.
- Class-weight tuning across more classifiers.
- SMOTE or other resampling methods.
- Calibration of predicted probabilities.
- Precision-recall curves and PR-AUC comparison.
- More systematic threshold selection based on a business cost function.
- Saving the final preprocessing pipeline and model for deployment.
