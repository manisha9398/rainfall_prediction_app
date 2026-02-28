import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# ---------------- LOAD DATA ----------------
data = pd.read_csv("Rainfall.csv")

data.columns = data.columns.str.strip().str.lower()

data = data.dropna()
data.drop_duplicates(inplace=True)

# Convert target
data['rainfall'] = data['rainfall'].map({'yes':1,'no':0})

# Drop unused columns (same as app)
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

# ---------------- TRAIN ----------------
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# ---------------- SAVE MODEL + FEATURES ----------------
model_package = {
    "model": model,
    "features": X.columns.tolist()
}

joblib.dump(model_package, "model.pkl")

print("✅ Model and Features Saved Successfully")
