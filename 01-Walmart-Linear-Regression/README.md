# Walmart Weekly Sales Prediction

## 📌 Project Overview

This project uses the Walmart Sales dataset to practice the complete workflow of a **supervised machine learning regression problem**.

The goal is to predict **Weekly Sales** using Linear Regression after performing exploratory data analysis and preprocessing.

## 🎯 Objective

Predict Walmart's weekly sales based on features such as:

* Store
* Holiday Flag
* Temperature
* Fuel Price
* CPI
* Unemployment

## 🗂️ Dataset

The dataset contains weekly sales information for different Walmart stores along with economic and environmental factors.

### Features

| Feature        | Description                    |
| -------------- | ------------------------------ |
| `Store`        | Store identifier               |
| `Date`         | Date of the observation        |
| `Weekly_Sales` | Target variable                |
| `Holiday_Flag` | Whether the week was a holiday |
| `Temperature`  | Temperature                    |
| `Fuel_Price`   | Fuel price                     |
| `CPI`          | Consumer Price Index           |
| `Unemployment` | Unemployment rate              |

## 🔍 Exploratory Data Analysis

The following were investigated:

* Data types and structure
* Missing values
* Numerical feature distributions
* Relationships between features and Weekly Sales
* Potential outliers
* Feature ranges

No significant invalid outliers were identified that required removal.

## ⚙️ Preprocessing

The following preprocessing techniques were practiced:

* Train-test split
* Standardization of numerical features
* One-hot encoding of `Store`
* `ColumnTransformer`
* Pipeline

The raw `Date` column was removed for this first Linear Regression model rather than directly passing timestamps to the model.

## 🤖 Model

### Linear Regression

Linear Regression was trained after preprocessing the features.

## 📊 Evaluation

The model was evaluated using:

* R² Score
* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* 5-Fold Cross-Validation

### Results

| Metric     |      Result |
| ---------- | ----------: |
| Test R²    |   **0.921** |
| Mean CV R² |   **0.918** |
| MAE        |  **91,010** |
| RMSE       | **159,506** |

The similarity between the test R² and mean cross-validation R² indicates that the model performed consistently across different training/validation splits.

## 🧰 Tools & Libraries

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Google Colab

## 📁 Project Files

```text
01-Walmart-Linear-Regression/
│
├── README.md
├── Walmart_Linear_Regression.ipynb
└── Walmart_Sales.csv
```

## 📚 What I Practiced

This project helped me practice:

* Exploratory Data Analysis
* Outlier investigation
* Feature selection and handling
* Feature scaling
* One-hot encoding
* ColumnTransformer
* PowerTransformer
* Pipelines
* Linear Regression
* Cross-validation
* Regression evaluation metrics
