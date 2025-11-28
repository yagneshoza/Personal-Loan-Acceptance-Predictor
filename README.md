# Personal Loan Acceptance Predictor

A machine learning project to predict whether a customer will accept a
personal loan based on demographic and financial features.

## 📌 Overview

This project uses **Logistic Regression** and **Random Forest
Classifier** to predict personal-loan acceptance. It includes model
training, evaluation, comparison using ROC curves, and interpretation of
feature importance.

## 📊 Dataset

The dataset contains the following key features:

-   **Income** -- Customer's annual income\
-   **CCAvg** -- Average credit card spending\
-   **CD Account** -- Whether the customer has a certificate of deposit\
-   **Education** -- Education level\
-   **Other demographic + account-related features**

**Target Variable:** - `Personal Loan` -- 1 if customer accepted the
loan, 0 otherwise

## 🧠 Models Used

### 1. Logistic Regression

-   Interpretable\
-   Shows feature coefficients\
-   Performs well on linearly separable data

### 2. Random Forest Classifier

-   More powerful model\
-   Handles non-linear patterns\
-   Provides feature importance\
-   Works well for imbalanced classification with tuning

## 📈 Performance Metrics

Evaluated using: - Accuracy\
- Precision, Recall, F1-score\
- Confusion Matrix\
- ROC Curve & AUC Score

Example AUC: - Logistic Regression: \~0.95\
- Random Forest: varies based on hyperparameters

## 📁 Project Structure

    Personal-Loan-Acceptance-Predictor/
    │── data/
    │── notebooks/
    │   ├── logistic_regression.ipynb
    │   ├── random_forest.ipynb
    │   └── comparison.ipynb
    │── models/
    │── README.md
    │── requirements.txt
    │── plots/

## 📦 Model Saving

Models are stored using `joblib`:

``` python
joblib.dump(logreg, "logistic_regression_model.pkl")
joblib.dump(rf, "random_forest_model.pkl")
joblib.dump((X_test, y_test), "test_data.pkl")
```

## 📊 Visual Comparison

The comparison notebook includes: - ROC curves for both models - Feature
importance plots - Confusion matrices - Metric tables

## 🚀 How to Run

1.  Install dependencies:

        pip install -r requirements.txt

2.  Open the notebook:

        jupyter notebook

3.  Run the Logistic Regression & Random Forest files

4.  Open the `comparison.ipynb` to visualize performance

## 📝 Future Improvements

-   Hyperparameter tuning\
-   Add more models (XGBoost, SVM)\
-   Deploy with Streamlit\
-   Use SMOTE for imbalance handling

## 👤 Author

Yagnеsh Oza
