import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, average_precision_score
import joblib
from features import create_features , create_target_label
import shap

# Load dataset
df = pd.read_csv("data/iot_sensor_data.csv", parse_dates=["timestamp"])

print("Original rows:", df.shape)

# Feature engineering
df = create_features(df)
print("After features:", df.shape)

# Target creation
df = create_target_label(df)
print("Columns after target creation:")
print(df.columns)

print("After target:", df.shape)

# Remove leakage column
df = df.drop(columns=["failure"])

# Drop NaNs only once here
df = df.bfill()
df = df.ffill()
df = df.dropna()

print("After dropna:", df.shape)


# Prepare X and y
X = df.drop(columns=["machine_id", "timestamp", "failure_next_24h"])
y = df["failure_next_24h"]

print("Final dataset shape:", X.shape)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# Model
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)


# Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("PR-AUC Score:", average_precision_score(y_test, y_prob))


joblib.dump(model, "models/factoryguard_model.pkl")
print("Model saved successfully!")

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# If it's a binary classifier, select class 1
if len(shap_values.values.shape) == 3:
    shap.summary_plot(shap_values.values[:, :, 1], X_test)
else:
    shap.summary_plot(shap_values.values, X_test)
