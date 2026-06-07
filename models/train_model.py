import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# LOAD DATASET
# ==========================================

DATA_PATH = "data/Emergency_Alert_Dataset.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("Dataset Shape:", df.shape)
print("=" * 60)

# ==========================================
# REMOVE UNUSED COLUMNS
# ==========================================

drop_cols = [
    "alert_id"
]

for col in drop_cols:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# ==========================================
# TARGET COLUMN
# ==========================================

TARGET = "target_label"

if TARGET not in df.columns:
    raise Exception(
        f"{TARGET} not found in dataset"
    )

# ==========================================
# HANDLE MISSING VALUES
# ==========================================

for col in df.columns:

    if df[col].dtype == "object":

        df[col] = df[col].fillna(
            df[col].mode()[0]
        )

    else:

        df[col] = df[col].fillna(
            df[col].median()
        )

# ==========================================
# SEPARATE X AND y
# ==========================================

X = df.drop(TARGET, axis=1)

y = df[TARGET]

# ==========================================
# ENCODE CATEGORICAL FEATURES
# ==========================================

categorical_cols = X.select_dtypes(
    include=["object"]
).columns

X = pd.get_dummies(
    X,
    columns=categorical_cols,
    drop_first=True
)

# Save column names
feature_columns = X.columns.tolist()

# ==========================================
# ENCODE TARGET
# ==========================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nTarget Classes:")
print(label_encoder.classes_)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# ==========================================
# MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

# ==========================================
# TRAIN
# ==========================================

print("\nTraining Model...")

model.fit(
    X_train_scaled,
    y_train
)

print("Training Completed")

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(
    X_test_scaled
)

# ==========================================
# METRICS
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print(f"Accuracy : {accuracy:.4f}")
print("=" * 60)

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix\n")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

feature_importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 Important Features\n")

print(
    feature_importance.head(20)
)

# ==========================================
# SAVE ARTIFACTS
# ==========================================

joblib.dump(
    model,
    "models/model.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

joblib.dump(
    label_encoder,
    "models/label_encoder.pkl"
)

joblib.dump(
    feature_columns,
    "models/feature_columns.pkl"
)

print("\nModel Saved Successfully")

print("models/model.pkl")
print("models/scaler.pkl")
print("models/label_encoder.pkl")
print("models/feature_columns.pkl")

print("\nPipeline Ready For Deployment")
