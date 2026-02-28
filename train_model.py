import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("Rainfall.csv")

data.columns = data.columns.str.strip().str.lower()

# Cleaning
data = data.dropna()
data.drop_duplicates(inplace=True)

# Convert target
data['rainfall'] = data['rainfall'].map({'yes':1,'no':0})

# Drop unwanted columns
data = data.drop(columns=['maxtemp','temparature','mintemp'], errors='ignore')

# Balance dataset
majority = data[data.rainfall == 1]
minority = data[data.rainfall == 0]

majority_downsampled = resample(majority,
                               replace=False,
                               n_samples=len(minority),
                               random_state=42)

balanced = pd.concat([majority_downsampled, minority])

# Split features
X = balanced.drop("rainfall", axis=1)
y = balanced["rainfall"]

# ⭐ SAVE FEATURE NAMES
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Save both model + feature names
joblib.dump((model, feature_names), "model.pkl")

print("✅ Model & Features Saved")
