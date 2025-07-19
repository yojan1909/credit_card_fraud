import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load Dataset
df = pd.read_csv("creditcard.csv")
print(f"Dataset shape: {df.shape}")

# Step 2: Preprocessing
df.drop("Time", axis=1, inplace=True)
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])

# Step 3: Train-Test Split
X = df.drop("Class", axis=1)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 4: Model 1 - Isolation Forest (Anomaly Detection)
iso_forest = IsolationForest(contamination=0.001, random_state=42)
iso_forest.fit(X_train)
y_pred_if = iso_forest.predict(X_test)
y_pred_if = np.where(y_pred_if == -1, 1, 0)  # Convert -1 to 1 (fraud), 1 to 0 (normal)

print("\n=== Isolation Forest Report ===")
print(classification_report(y_test, y_pred_if))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_if))

# Step 5: Model 2 - Logistic Regression (Supervised Learning)
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train, y_train)
y_pred_lr = log_model.predict(X_test)

print("\n=== Logistic Regression Report ===")
print(classification_report(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

# Step 6: Save Models Safely
try:
    model_path = os.path.join(os.getcwd(), "logistic_model.pkl")
    scaler_path = os.path.join(os.getcwd(), "scaler.pkl")
    joblib.dump(log_model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\n✅ Models saved successfully in: {os.getcwd()}")
    print(f"→ logistic_model.pkl\n→ scaler.pkl")
except Exception as e:
    print(f"\n❌ Error saving model: {e}")
