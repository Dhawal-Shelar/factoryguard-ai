import joblib
import pandas as pd
from features import create_features

#model initialize
model = joblib.load("model/factoryguard_model.pkl")

#load dataset
df = pd.read_csv("data/iot_sensor_data.csv", parse_dates=["timestamp"])

df = create_features(df)

df = df.bfill().ffill().dropna()


latest = df.sort_values("timestamp").groupby("machine_id").tail(1)

X = latest.drop(columns=["machine_id", "timestamp", "failure"])

preds = model.predict_proba(X)[:, 1]

latest["failure_risk_score"] = preds

print(latest[["machine_id", "failure_risk_score"]].head())