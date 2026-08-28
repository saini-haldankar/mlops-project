import pandas as pd
import joblib

model = joblib.load("models/churn_model.pkl")
encoders = joblib.load("models/encoders.pkl")

sample_customer = {
    "age": 28,
    "gender": "Male",
    "subscription_type": "Basic",
    "watch_hours": 3.5,
    "last_login_days": 40,
    "region": "South",
    "device": "Mobile",
    "monthly_fee": 199
}

df = pd.DataFrame([sample_customer])

categorical_cols = ["gender", "subscription_type", "region", "device"]
for col in categorical_cols:
    le = encoders[col]
    df[col] = le.transform(df[col])

prediction = model.predict(df)[0]
probability = model.predict_proba(df)[0][1]

result = "Likely to churn" if prediction == 1 else "Likely to stay"

print("Customer details:", sample_customer)
print("Prediction:", result)
print("Churn probability:", round(probability, 3))

if probability > 0.7:
    print("Recommend immediate retention offer")

if probability >= 0.8:
    risk_level = "High Risk"
elif probability >= 0.5:
    risk_level = "Medium Risk"
else:
    risk_level = "Low Risk"

print("Risk Level:", risk_level)