import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

model = RandomForestClassifier()

scaler = StandardScaler()

label_encoder = LabelEncoder()

feature_columns = []

joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")
joblib.dump(feature_columns, "models/feature_columns.pkl")

print("Dummy files created")
