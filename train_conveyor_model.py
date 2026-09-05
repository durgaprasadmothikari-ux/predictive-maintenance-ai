import numpy as np
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# ==================================================
# DATASET GENERATION / LOADING
# ==================================================

data_path = "data/conveyor_system.csv"

if not os.path.exists(data_path):
    print(f"Generating realistic Conveyor System dataset at {data_path}...")
    np.random.seed(42)
    n_samples = 10000

    belt_speed = np.random.normal(2.2, 0.4, n_samples).clip(0.3, 4.5)
    belt_tension = np.random.normal(22.0, 4.5, n_samples).clip(4.0, 45.0)
    motor_current = np.random.normal(45.0, 10.0, n_samples).clip(10.0, 110.0)
    roller_temp = np.random.normal(48.0, 10.0, n_samples).clip(20.0, 110.0)
    idler_vibration = np.random.gamma(2.0, 1.0, n_samples).clip(0.3, 14.0)
    belt_slip = np.random.exponential(1.2, n_samples).clip(0.0, 16.0)
    load_weight = np.random.normal(500.0, 150.0, n_samples).clip(40.0, 1400.0)
    ambient_temp = np.random.normal(28.0, 7.0, n_samples).clip(5.0, 52.0)

    # Physics-based failure criteria:
    # 1. Belt slip > 6.0% with high motor current -> drive pulley slipping / wear
    # 2. Roller bearing seizure: roller_temp > 80.0 C and idler_vibration > 6.5 mm/s
    # 3. Severe overtension / overload: belt_tension > 35.0 kN and load_weight > 950 t/h
    # 4. Under-tension: belt_tension < 10.0 kN
    failure_score = (
        (belt_slip > 6.0).astype(int) * 3
        + ((roller_temp > 80.0) & (idler_vibration > 6.0)).astype(int) * 3
        + (belt_tension > 36.0).astype(int) * 2
        + (belt_tension < 9.0).astype(int) * 2
        + ((motor_current > 80.0) & (load_weight > 900.0)).astype(int) * 2
        + np.random.normal(0, 0.4, n_samples)
    )

    failure = (failure_score >= 3.0).astype(int)

    df = pd.DataFrame({
        "Belt_Speed": np.round(belt_speed, 2),
        "Belt_Tension": np.round(belt_tension, 2),
        "Motor_Current": np.round(motor_current, 1),
        "Roller_Bearing_Temperature": np.round(roller_temp, 1),
        "Idler_Vibration": np.round(idler_vibration, 2),
        "Belt_Slip_Percentage": np.round(belt_slip, 2),
        "Load_Weight": np.round(load_weight, 1),
        "Ambient_Temperature": np.round(ambient_temp, 1),
        "Conveyor_Failure": failure
    })

    os.makedirs("data", exist_ok=True)
    df.to_csv(data_path, index=False)
    print("Conveyor System dataset generated successfully!")
else:
    print(f"Loading existing Conveyor System dataset from {data_path}...")
    df = pd.read_csv(data_path)


# ==================================================
# DISPLAY DATASET INFORMATION
# ==================================================

print("\n" + "=" * 60)
print("CONVEYOR SYSTEM DATASET INFORMATION")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFailure Distribution:")
print(df["Conveyor_Failure"].value_counts())


# ==================================================
# DEFINE FEATURES AND TARGET
# ==================================================

features = [
    "Belt_Speed",
    "Belt_Tension",
    "Motor_Current",
    "Roller_Bearing_Temperature",
    "Idler_Vibration",
    "Belt_Slip_Percentage",
    "Load_Weight",
    "Ambient_Temperature"
]

target = "Conveyor_Failure"

X = df[features]
y = df[target]


# ==================================================
# TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} rows")
print(f"Testing set: {X_test.shape[0]} rows")


import feature_engineering as fe

# ==================================================
# CREATE AND TRAIN MODEL
# ==================================================

print("\nTraining Conveyor System AI model with belt friction feature engineering...")

base_clf = RandomForestClassifier(
    n_estimators=180,
    max_depth=16,
    min_samples_split=4,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model = fe.ProficientClassifierPipeline(
    base_estimator=base_clf,
    fe_func=fe.engineer_conveyor_features,
    base_features=features
)

model.fit(X_train, y_train)


# ==================================================
# MODEL EVALUATION
# ==================================================

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n" + "=" * 60)
print("CONVEYOR SYSTEM MODEL RESULTS")
print("=" * 60)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, predictions, target_names=["Normal", "Failure Risk"]))


# ==================================================
# SAVE MODEL & ASSETS
# ==================================================

os.makedirs("models", exist_ok=True)

model_file = "models/conveyor_model.pkl"
features_file = "models/conveyor_features.pkl"

joblib.dump(model, model_file)
joblib.dump(features, features_file)

print("\n" + "=" * 60)
print("CONVEYOR MODEL ASSETS SAVED")
print("=" * 60)
print(f"Model saved: {model_file}")
print(f"Features saved: {features_file}")
