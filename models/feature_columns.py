import pandas as pd
import joblib
# Define your exact training columns in the correct order
feature_columns = [
    "age", "location", "emergency_type", "severity", "reporting_source"
] # 👈 Replace these with your actual model's feature column names!

# Now Python can safely use it to reorder/filter the input data
X_scaled = user_input_df[feature_columns]
prediction = model.predict(X_scaled)
# Load dataset
df = pd.read_csv("data/Emergency_Alert_Dataset.csv")

# Remove target and ID columns
X = df.drop(columns=["target_label", "alert_id"], errors="ignore")

# Convert categorical columns to dummy variables
X = pd.get_dummies(X, drop_first=True)

# Get feature column names
feature_columns = X.columns.tolist()

# Save feature columns
joblib.dump(feature_columns, "models/feature_columns.pkl")

print("feature_columns.pkl created successfully")
print(f"Total Features: {len(feature_columns)}")
