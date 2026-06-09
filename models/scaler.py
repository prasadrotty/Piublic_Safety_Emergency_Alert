import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load dataset
df = pd.read_csv("data/Emergency_Alert_Dataset.csv")

# Remove target and non-feature columns
X = df.drop(columns=["target_label", "alert_id"], errors="ignore")

# Convert categorical columns to numeric
X = pd.get_dummies(X, drop_first=True)

# Create scaler
scaler = StandardScaler()

# Fit scaler on data
scaler.fit(X)

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

print("scaler.pkl created successfully")
