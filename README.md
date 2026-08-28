# OTT Analytics and Churn Prediction System

A machine learning system that predicts whether an OTT subscriber is likely to **churn (cancel their subscription)** based on customer details and viewing behavior.

## Problem Statement

Predict whether a customer will **churn (1)** or **stay active (0)** using features such as watch hours, last login days, subscription type, region, and monthly fee.

## Dataset

`data/ott_churn_data.csv`

* **500 rows**
* **10 columns**
* No missing values

| Column              | Description                |
| ------------------- | -------------------------- |
| `customer_id`       | Unique customer ID         |
| `age`               | Customer age               |
| `gender`            | Customer gender            |
| `subscription_type` | Basic / Standard / Premium |
| `watch_hours`       | Average watch hours        |
| `last_login_days`   | Days since last login      |
| `region`            | Customer region            |
| `device`            | Primary viewing device     |
| `monthly_fee`       | Monthly subscription fee   |
| `churned`           | 1 = Churned, 0 = Active    |

## Project Structure

```text
mlops-project/
├── data/
│   └── ott_churn_data.csv
├── src/
│   ├── train.py
│   ├── predict.py
│   └── compare_models.py
├── models/
│   ├── churn_model.pkl
│   ├── encoders.pkl
│   ├── model_comparison_results.csv
│   └── model_comparison_chart.png
├── requirements.txt
└── README.md
```

## How It Works

* **`train.py`** — Preprocesses the data, trains the Random Forest model, evaluates it, and saves the model.
* **`predict.py`** — Predicts customer churn, churn probability, and risk level.
* **`compare_models.py`** — Compares Logistic Regression, Random Forest, and Gradient Boosting.

## How to Run

```bash
pip install -r requirements.txt
python src/train.py
python src/predict.py
python src/compare_models.py
```

## Results

### Random Forest

| Metric    | Score |
| --------- | ----: |
| Accuracy  |  0.93 |
| Precision |  0.95 |
| Recall    |  0.88 |
| F1 Score  |  0.91 |

### Model Comparison

| Model               | Accuracy | Precision | Recall |    F1 |
| ------------------- | -------: | --------: | -----: | ----: |
| Logistic Regression |     0.97 |     0.951 |  0.975 | 0.963 |
| Random Forest       |     0.93 |     0.946 |  0.875 | 0.909 |
| Gradient Boosting   |     0.96 |     0.950 |  0.950 | 0.950 |

**Logistic Regression achieved the best overall performance** on this dataset.

## Sample Prediction

**Input:** Customer with 3.5 watch hours and 40 days since last login.

**Output:**

* Prediction: Likely to churn
* Churn Probability: 0.89
* Risk Level: High Risk
* Recommendation: Immediate retention offer

## Future Improvements

* Add more customer behavior features.
* Perform cross-validation and hyperparameter tuning.
* Test additional models such as XGBoost.
* Deploy the system as a web application/API.
