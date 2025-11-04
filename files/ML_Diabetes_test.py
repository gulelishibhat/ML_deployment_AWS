# ==============================================
# diabetes_model_building.py
# ==============================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Load the data
df = pd.read_csv("FloridaBlue_interview_dataset.csv")

# 2. Quick data cleaning
# Drop name columns (non-predictive)
df = df.drop(columns=["first_name", "last_name"])

# Convert TRUE/FALSE to binary
df["diabetes"] = df["diabetes"].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0})

# Handle missing numeric values
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Handle missing categorical values
df["region_demographic"] = df["region_demographic"].fillna("Unknown")
df["gender"] = df["gender"].fillna("Unknown")

# 3. Define features and target
X = df.drop(columns=["diabetes"])
y = df["diabetes"]

# 4. Identify categorical and numeric columns
categorical_features = ["gender", "region_demographic"]
numeric_features = [col for col in X.columns if col not in categorical_features]

# 5. Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# 6. Combine preprocessing with model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 8. Train the model
model.fit(X_train, y_train)

# 9. Evaluate
y_pred = model.predict(X_test)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# 10. Save trained model
joblib.dump(model, "diabetes_prediction_model.pkl")
print("\n✅ Model saved as 'diabetes_prediction_model.pkl'")
