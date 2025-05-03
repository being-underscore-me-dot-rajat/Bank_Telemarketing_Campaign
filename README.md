# Customer Subscription Prediction with XGBoost

This project builds an end-to-end machine learning pipeline to predict whether a customer will subscribe to a term deposit based on their banking and personal data. The solution includes data cleaning, feature engineering, encoding, and hyperparameter-tuned model training using XGBoost.

## Features

- Loads training data from `train.csv`
- Handles missing values using mode imputation (`SimpleImputer`)
- Extensive feature engineering, including:
  - Binning for `age`, `balance`, `campaign`, `date`, and `month`
  - Log transformation and inverse mapping for skewed columns
  - Custom features like `ongoing_loan`, `risky_customers`, `possibly_interested`, etc.
- Encodes categorical features using:
  - One-hot encoding for high-cardinality features
  - Ordinal encoding for others
- Scales numerical features using `RobustScaler`
- Implements an `XGBClassifier` model inside a `Pipeline`
- Uses `RandomizedSearchCV` for hyperparameter optimization
- Evaluates performance using `f1_macro` scoring

## Requirements

Install dependencies using pip:

```bash
pip install pandas numpy scikit-learn xgboost
