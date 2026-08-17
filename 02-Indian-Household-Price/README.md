# 🏠 House Price Prediction using Linear Regression

## 📌 Project Overview

This project focuses on predicting house prices using **Linear Regression**.

The main purpose of this project is to practice the machine learning concepts I have learned so far, including:

- Exploratory Data Analysis (EDA)
- Feature selection
- Missing-value handling
- Categorical encoding
- Feature transformation
- Feature scaling
- Outlier analysis
- Pipelines and transformers
- Linear Regression
- Polynomial Regression
- SGD Regression
- Cross-validation
- Model evaluation

The project also explores whether different preprocessing techniques actually improve model performance instead of applying them blindly.

---

## 📊 Dataset

The dataset contains information about houses and their prices.

Some of the important features include:

- Living area
- Lot area
- Number of bedrooms
- Number of bathrooms
- Number of floors
- Grade of the house
- Condition of the house
- Area of the basement
- Waterfront presence
- Latitude and Longitude
- Built year
- Renovation year
- Number of views
- Price

The target variable is:

**`Price`**

---

## 🔎 Exploratory Data Analysis

During EDA, relationships between numerical features and `Price` were investigated using:

- Correlation analysis
- Scatter plots
- Distribution analysis
- Outlier detection using the IQR method

Correlation was used to identify features that had stronger relationships with the target.

For example, `living area` showed a strong positive relationship with `Price`.

---

## 🚨 Outlier Analysis

Potential outliers were initially identified using the **Interquartile Range (IQR)** method.

However, an important observation from this project was:

> An IQR outlier is not necessarily incorrect data.

Therefore, outliers were not automatically removed.

Potentially suspicious observations were investigated using scatter plots and their relationship with the target variable.

A particularly unusual observation in `living area` was investigated, but removing it did not produce a convincing overall improvement in model performance. Therefore, the original observation was retained for the final model.

---

## ⚙️ Feature Scaling

Different scaling techniques were experimented with, including:

- `StandardScaler`

Scaling did produce a meaningful improvement in the Linear Regression model.

This demonstrated that preprocessing techniques should be evaluated rather than applied automatically.

---

## 🔄 Feature Transformation

Power transformation and distribution/skewness were also investigated as possible approaches for handling highly skewed numerical features.

The main goal was to understand when transformations are useful rather than applying them to every numerical column.

---

## 🤖 Models

Two linear regression approaches were experimented with:

### 1. Linear Regression

`LinearRegression()` from Scikit-learn was used as the baseline model.

It uses a least-squares optimization approach.

### 2. Linear Regression using Stochastic Gradient Descent

`SGDRegressor` was used with:

```python```
loss='squared_error'

### 2. Polynomial Regression

`PolynomialRegression()` from Scikit-learn was used which improve the model accuracy.

## 📈 Model Evaluation

The model was evaluated using:

### R² Score

Measures how much of the variation in house prices is explained by the model.

### MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted prices.

### RMSE — Root Mean Squared Error

Penalizes larger prediction errors more strongly than MAE.

### 5-Fold Cross-Validation

Cross-validation was used to check whether the model's performance was consistent across different subsets of the training data.

---

## 🏆 Final Linear Regression Results

| Metric | Result |
|---|---:|
| Test R² | **83.87%** |
| Mean CV R² | **81.32%** |
| MAE | **95,934** |
| RMSE | **150,723** |

The final Polynomial Regression model achieved an R² score of approximately **0.838 on the test set**.

The mean 5-fold cross-validation R² was approximately **0.813**, indicating that the model explains around 81% of the variation in house prices on average across the validation folds.

---

## 💡 Key Learnings

Through this project, I learned that:

1. More preprocessing does not automatically mean better model performance.
2. Scaling may have little effect on ordinary Linear Regression.
3. `RobustScaler` does not remove outliers; it only scales using robust statistics.
4. An outlier detected using IQR is not automatically bad data.
5. Outlier removal should be based on data understanding and model evaluation.
6. R² should not be the only evaluation metric.
7. Cross-validation gives a better idea of model generalization than relying on a single train-test split.
8. Feature engineering and preprocessing should be driven by the data rather than applied as a checklist.
9. Different models and preprocessing techniques should be compared systematically.

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SGDRegressor
- Polynomial Regression
- Scikit-learn
- Google Colab / Jupyter Notebook

---

## 📁 Project Structure

```text
House-Price-Prediction/
│
├── house_price_prediction.ipynb
├── README.md
└── dataset.csv
