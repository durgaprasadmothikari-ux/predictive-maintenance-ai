import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score

import feature_engineering as fe

# ==================================================
# LOAD DATASET
# ==================================================
print("\n" + "=" * 60)
print("AI4I INDUSTRIAL MACHINE - HIGH PROFICIENCY PIPELINE")
print("=" * 60)

print("\nLoading dataset from data/ai4i2020.csv...")
df = pd.read_csv("data/ai4i2020.csv")

# Remove identification columns
df = df.drop(["UDI", "Product ID"], axis=1)

# One-hot encode categorical machine type
df = pd.get_dummies(df, columns=["Type"], drop_first=True)

base_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type_L",
    "Type_M"
]

X = df[base_features]
y = df["Machine failure"]

print(f"Total records: {X.shape[0]}")
print(f"Base Features ({len(base_features)}): {base_features}")
print(f"Failure distribution:\n{y.value_counts()}")

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining records: {X_train.shape[0]}")
print(f"Testing records:  {X_test.shape[0]}")

# ==================================================
# PROFICIENT GRADIENT BOOSTING + DOMAIN PHYSICS
# ==================================================
print("\nTraining high-proficiency Gradient Boosting model with physics feature engineering...")

base_clf = HistGradientBoostingClassifier(
    class_weight="balanced",
    max_iter=300,
    learning_rate=0.07,
    max_leaf_nodes=31,
    min_samples_leaf=15,
    l2_regularization=1.0,
    random_state=42
)

model = fe.ProficientClassifierPipeline(
    base_estimator=base_clf,
    fe_func=fe.engineer_ai4i_features,
    base_features=base_features
)

model.fit(X_train, y_train)

# Make predictions and calculate probabilities
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
f1 = f1_score(y_test, y_pred)

print("\n" + "=" * 60)
print("AI4I MODEL EVALUATION RESULTS")
print("=" * 60)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(f"ROC-AUC Score:  {roc_auc:.4f}")
print(f"F1-Score:       {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Failure"]))

# Save the trained model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/predictive_maintenance_model.pkl")

print("=" * 60)
print("Model saved successfully to models/predictive_maintenance_model.pkl")
print("=" * 60 + "\n")