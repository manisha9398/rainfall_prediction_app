import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

# ---------------- LOAD DATA ----------------
data = pd.read_csv("Rainfall.csv")
data.columns = data.columns.str.strip().str.lower()

data = data.dropna()
data.drop_duplicates(inplace=True)

# Convert target
data['rainfall'] = data['rainfall'].map({'yes':1,'no':0})

# Drop unused columns
data = data.drop(columns=['maxtemp','temparature','mintemp'], errors='ignore')

# ---------------- BALANCE DATA ----------------
majority = data[data.rainfall == 1]
minority = data[data.rainfall == 0]

majority_downsampled = resample(
    majority,
    replace=False,
    n_samples=len(minority),
    random_state=42
)

balanced = pd.concat([majority_downsampled, minority])

# ---------------- SPLIT ----------------
X = balanced.drop("rainfall", axis=1)
y = balanced["rainfall"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ---------------- RANDOM FOREST ----------------
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# ---------------- XGBOOST ----------------
xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)

print("Random Forest Accuracy:", rf_acc)
print("XGBoost Accuracy:", xgb_acc)

# ---------------- SAVE EVERYTHING ----------------
model_package = {
    "rf_model": rf_model,
    "xgb_model": xgb_model,
    "features": X.columns.tolist(),
    "rf_accuracy": rf_acc,
    "xgb_accuracy": xgb_acc
}

joblib.dump(model_package, "model.pkl")

print("✅ Both Models Saved Successfully")