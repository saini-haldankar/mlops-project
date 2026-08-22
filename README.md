# mlops-project

# OTT Analytics and Churn Prediction System

Predicts which subscribers of an OTT (streaming) platform are likely to
cancel their subscription (churn), based on viewing behavior and account
details — so the platform can target retention offers at the right
customers before they leave.

## Problem Statement

Given customer data such as watch hours, days since last login,
subscription type, and region, predict whether a customer will churn
(`1`) or stay (`0`).

## Dataset

`data/ott_churn_data.csv` — 500 rows, 10 columns:

| Column             | Description                             |
|---------------------|-----------------------------------------|
| customer_id         | Unique customer identifier              |
| age                 | Age of the customer                     |
| gender              | Gender of the customer                  |
| subscription_type   | Basic / Standard / Premium              |
| watch_hours         | Average hours of content watched        |
| last_login_days     | Days since the customer last logged in  |
| region              | Region the customer belongs to          |
| device              | Primary device used to watch content    |
| monthly_fee         | Monthly subscription amount             |
| churned             | Target column: 1 = churned, 0 = active  |

No missing values in any column.

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

**`src/train.py`**
- Loads the dataset
- Cleans missing values
- Encodes categorical columns (`gender`, `subscription_type`, `region`,
  `device`)
- Splits data into train/test sets (80/20)
- Trains a Random Forest classifier
- Prints accuracy, precision, recall, and F1 score
- Saves the trained model and encoders to `models/`

**`src/predict.py`**
- Loads the saved model and encoders
- Takes a customer's details as input
- Returns whether the customer is likely to churn, along with the churn
  probability

## How to Run

Install dependencies:
```bash
pip install -r requirements.txt
```

Train the model:
```bash
python src/train.py
```

Run a prediction:
```bash
python src/predict.py
```

To predict for a different customer, edit the `sample_customer` dictionary
inside `predict.py` with the new customer's details.

## Results

Trained on 400 samples, evaluated on 100 held-out samples:

| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.93  |
| Precision | 0.95  |
| Recall    | 0.88  |
| F1 Score  | 0.91  |

### Sample Prediction

Input:
```json
{
  "age": 28,
  "gender": "Male",
  "subscription_type": "Basic",
  "watch_hours": 3.5,
  "last_login_days": 40,
  "region": "South",
  "device": "Mobile",
  "monthly_fee": 199
}
```

Output:
```
Prediction: Likely to churn
Churn probability: 0.89
```

This matches intuition — low watch hours, a long gap since last login,
and a low-tier plan are classic early churn signals.

## Model Used

Random Forest Classifier (scikit-learn) — chosen because it handles both
numeric and encoded categorical features well and gives a strong baseline
without heavy tuning.

## Future Improvements

- Add more behavioral features (sessions per week, content genre
  preference)
- Try other models (XGBoost, Logistic Regression) and compare performance
