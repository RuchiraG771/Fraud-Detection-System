# ======================================
# Fraud Transaction Detection - train.py
# ======================================

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# --------------------------------------
# Load & Merge PKL Dataset
# --------------------------------------
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "data")

print("Dataset path:", DATASET_PATH)
print("Files found:", os.listdir(DATASET_PATH))


pkl_files = [
    f for f in os.listdir(DATASET_PATH)
    if f.lower().endswith(".pkl")
]

df_list = []
for file in pkl_files:
    file_path = os.path.join(DATASET_PATH, file)
    df = pd.read_pickle(file_path)
    df_list.append(df)

data = pd.concat(df_list, ignore_index=True)

print("✅ Dataset loaded and merged")
print("Shape:", data.shape)

# --------------------------------------
# Datetime Feature Engineering
# --------------------------------------
data["TX_DATETIME"] = pd.to_datetime(data["TX_DATETIME"])

data["hour"] = data["TX_DATETIME"].dt.hour
data["day"] = data["TX_DATETIME"].dt.day
data["weekday"] = data["TX_DATETIME"].dt.weekday

# Drop unused columns
data.drop(
    columns=["TX_DATETIME", "TRANSACTION_ID"],
    errors="ignore",
    inplace=True
)

# --------------------------------------
# Features & Target
# --------------------------------------
X = data.drop("TX_FRAUD", axis=1)
y = data["TX_FRAUD"]

print("Fraud distribution:")
print(y.value_counts())

# --------------------------------------
# Train-Test Split (STRATIFIED)
# --------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------
# Feature Scaling (IMPORTANT for SMOTE)
# --------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------
# Handle Imbalance using SMOTE
# --------------------------------------
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train_scaled, y_train
)

print("After SMOTE:")
print(pd.Series(y_train_resampled).value_counts())

# --------------------------------------
# Train Model
# --------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_resampled, y_train_resampled)

# --------------------------------------
# Evaluation
# --------------------------------------
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# --------------------------------------
# Save Model, Scaler & Features
# --------------------------------------
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")

print("✅ model.pkl, scaler.pkl, features.pkl saved successfully")
