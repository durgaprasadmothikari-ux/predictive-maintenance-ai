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

data_path = "data/industrial_pump.csv"

if not os.path.exists(data_path):
    print(f"Generating realistic Industrial Pump dataset at {data_path}...")
    np.random.seed(42)
    n_samples = 10000

    flow_rate = np.random.normal(250.0, 45.0, n_samples).clip(40.0, 500.0)
    suction_pressure = np.random.normal(2.5, 0.6, n_samples).clip(0.5, 6.0)
    discharge_pressure = np.random.normal(12.0, 2.5, n_samples).clip(4.0, 25.0)
    vibration_rms = np.random.gamma(2.0, 1.2, n_samples).clip(0.4, 16.0)
    bearing_temp = np.random.normal(55.0, 12.0, n_samples).clip(25.0, 115.0)
    motor_power = np.random.normal(60.0, 15.0, n_samples).clip(10.0, 150.0)
    fluid_temp = np.random.normal(40.0, 10.0, n_samples).clip(15.0, 95.0)
    cavitation_index = (suction_pressure / (vibration_rms * 0.4 + 0.5)).clip(0.3, 4.0)

    # Physics-based failure criteria:
    # 1. Severe cavitation: cavitation_index < 1.0 and suction_pressure < 1.5
    # 2. Bearing overheating: bearing_temp > 85.0
    # 3. High vibration / misalignment: vibration_rms > 7.1 (ISO 10816 Class II/III warning)
    # 4. Discharge deadhead / overload: discharge_pressure > 18.0 and motor_power > 85.0
    failure_score = (
        (vibration_rms > 7.1).astype(int) * 3
        + (bearing_temp > 85.0).astype(int) * 3
        + (cavitation_index < 1.0).astype(int) * 2
        + (discharge_pressure > 18.0).astype(int) * 2
        + (motor_power > 90.0).astype(int) * 1
        + np.random.normal(0, 0.5, n_samples)
    )

    failure = (failure_score >= 3.0).astype(int)

    df = pd.DataFrame({
        "Flow_Rate": np.round(flow_rate, 2),
        "Suction_Pressure": np.round(suction_pressure, 2),
        "Discharge_Pressure": np.round(discharge_pressure, 2),
        "Vibration_RMS": np.round(vibration_rms, 2),
        "Bearing_Temperature": np.round(bearing_temp, 2),
        "Motor_Power": np.round(motor_power, 2),
        "Fluid_Temperature": np.round(fluid_temp, 2),
        "Cavitation_Index": np.round(cavitation_index, 2),
        "Pump_Failure": failure
    })

    os.makedirs("data", exist_ok=True)
    df.to_csv(data_path, index=False)
    print("Industrial Pump dataset generated successfully!")
else:
    print(f"Loading existing Industrial Pump dataset from {data_path}...")
    df = pd.read_csv(data_path)


# ==================================================
# DISPLAY DATASET INFORMATION
# ==================================================

print("\n" + "=" * 60)
print("INDUSTRIAL PUMP DATASET INFORMATION")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFailure Distribution:")
print(df["Pump_Failure"].value_counts())


# ==================================================
# DEFINE FEATURES AND TARGET
# ==================================================

features = [
    "Flow_Rate",
    "Suction_Pressure",
    "Discharge_Pressure",
    "Vibration_RMS",
    "Bearing_Temperature",
    "Motor_Power",
    "Fluid_Temperature",
    "Cavitation_Index"
]

target = "Pump_Failure"

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

print("\nTraining Industrial Pump AI model with hydraulic feature engineering...")

base_clf = RandomForestClassifier(
    n_estimators=160,
    max_depth=16,
    min_samples_split=4,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model = fe.ProficientClassifierPipeline(
    base_estimator=base_clf,
    fe_func=fe.engineer_pump_features,
    base_features=features
)

model.fit(X_train, y_train)


# ==================================================
# MODEL EVALUATION
# ==================================================

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n" + "=" * 60)
print("INDUSTRIAL PUMP MODEL RESULTS")
print("=" * 60)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, predictions, target_names=["Normal", "Failure Risk"]))


# ==================================================
# SAVE MODEL & ASSETS
# ==================================================

os.makedirs("models", exist_ok=True)

model_file = "models/pump_model.pkl"
features_file = "models/pump_features.pkl"

joblib.dump(model, model_file)
joblib.dump(features, features_file)

print("\n" + "=" * 60)
print("PUMP MODEL ASSETS SAVED")
print("=" * 60)
print(f"Model saved: {model_file}")
print(f"Features saved: {features_file}")
