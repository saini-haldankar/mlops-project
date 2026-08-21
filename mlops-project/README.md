# OTT Analytics and Churn Prediction System

## Overview
This project predicts which subscribers of an OTT (streaming) platform are
likely to cancel their subscription (churn), based on their viewing behavior
and account details. The goal is to help the platform identify at-risk users
early so retention actions (offers, reminders) can be targeted at the right
customers.

## Problem Statement
Given customer data such as watch hours, days since last login, subscription
type, and region, predict whether the customer will churn (1) or stay (0).

## Dataset
The dataset (`data/ott_churn_data.csv`) contains the following columns:

| Column             | Description                                   |
|---------------------|-----------------------------------------------|
| customer_id         | Unique customer identifier                    |
| age                 | Age of the customer                           |
| gender              | Gender of the customer                        |
| subscription_type   | Basic / Standard / Premium                    |
| watch_hours         | Average hours of content watched              |
| last_login_days     | Days since the customer last logged in        |
| region              | Region the customer belongs to                |
| device              | Primary device used to watch content          |
| monthly_fee         | Monthly subscription amount                   |
| churned             | Target column: 1 = churned, 0 = active         |

## Project Structure
```
mlops-project/
├── data/
│   └── ott_churn_data.csv
├── src/
│   ├── train.py
│   └── predict.py
├── models/
│   ├── churn_model.pkl
│   └── encoders.pkl
├── requirements.txt
└── README.md
```

## How It Works
1. **train.py**
   - Loads the dataset
   - Cleans missing values
   - Encodes categorical columns (gender, subscription_type, region, device)
   - Splits data into train/test sets
   - Trains a Random Forest classifier
   - Prints accuracy, precision, recall, and F1 score
   - Saves the trained model and encoders to the `models/` folder

2. **predict.py**
   - Loads the saved model and encoders
   - Takes a customer's details as input
   - Returns whether the customer is likely to churn, along with the churn
     probability

## How to Run

Install dependencies:
```
pip install -r requirements.txt
```

Train the model:
```
python src/train.py
```

Run a prediction:
```
python src/predict.py
```

To predict for a different customer, edit the `sample_customer` dictionary
inside `predict.py` with the new customer's details.

## Model Used
Random Forest Classifier (scikit-learn) — chosen because it handles both
numeric and categorical (encoded) features well and gives a reasonable
baseline without heavy tuning.

## Future Improvements
- Add more behavioral features (e.g., number of sessions per week, content
  genre preference)
- Try other models (XGBoost, Logistic Regression) and compare performance
- Add a simple API (Flask/FastAPI) to serve predictions
- Set up automated retraining when new data arrives
