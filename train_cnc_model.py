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

data_path = "data/cnc_machine.csv"

if not os.path.exists(data_path):
    print(f"Generating realistic CNC Machine dataset at {data_path}...")
    np.random.seed(42)
    n_samples = 10000

    spindle_speed = np.random.normal(6000.0, 1500.0, n_samples).clip(800.0, 15000.0)
    feed_rate = np.random.normal(800.0, 200.0, n_samples).clip(150.0, 2500.0)
    cutting_force = np.random.normal(850.0, 220.0, n_samples).clip(150.0, 3200.0)
    tool_wear = np.random.uniform(10.0, 240.0, n_samples)
    spindle_vibration = np.random.gamma(2.2, 0.9, n_samples).clip(0.3, 15.0)
    axis_feed_error = np.random.normal(4.0, 2.5, n_samples).clip(0.2, 35.0)
    coolant_pressure = np.random.normal(20.0, 4.0, n_samples).clip(3.0, 35.0)
    motor_temp = np.random.normal(45.0, 9.0, n_samples).clip(20.0, 95.0)

    # Physics-based failure criteria:
    # 1. Critical tool wear + excessive cutting force -> tool chipping/breakage
    # 2. Spindle chatter / excessive vibration (> 6.0 mm/s)
    # 3. Coolant starvation (< 10.0 bar) + high motor temperature (> 70 C)
    # 4. Axis positioning error > 18.0 um
    failure_score = (
        ((tool_wear > 180.0) & (cutting_force > 1200.0)).astype(int) * 3
        + (spindle_vibration > 6.0).astype(int) * 3
        + ((coolant_pressure < 10.0) & (motor_temp > 68.0)).astype(int) * 3
        + (axis_feed_error > 16.0).astype(int) * 2
        + (cutting_force > 1800.0).astype(int) * 2
        + np.random.normal(0, 0.4, n_samples)
    )

    failure = (failure_score >= 3.0).astype(int)

    df = pd.DataFrame({
        "Spindle_Speed": np.round(spindle_speed, 1),
        "Feed_Rate": np.round(feed_rate, 1),
        "Cutting_Force": np.round(cutting_force, 1),
        "Tool_Wear_Index": np.round(tool_wear, 1),
        "Spindle_Vibration": np.round(spindle_vibration, 2),
        "Axis_Feed_Error": np.round(axis_feed_error, 2),
        "Coolant_Pressure": np.round(coolant_pressure, 2),
        "Motor_Temperature": np.round(motor_temp, 1),
        "CNC_Failure": failure
    })

    os.makedirs("data", exist_ok=True)
    df.to_csv(data_path, index=False)
    print("CNC Machine dataset generated successfully!")
else:
    print(f"Loading existing CNC Machine dataset from {data_path}...")
    df = pd.read_csv(data_path)


# ==================================================
# DISPLAY DATASET INFORMATION
# ==================================================

print("\n" + "=" * 60)
print("CNC MACHINE DATASET INFORMATION")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFailure Distribution:")
print(df["CNC_Failure"].value_counts())


# ==================================================
# DEFINE FEATURES AND TARGET
# ==================================================

features = [
    "Spindle_Speed",
    "Feed_Rate",
    "Cutting_Force",
    "Tool_Wear_Index",
    "Spindle_Vibration",
    "Axis_Feed_Error",
    "Coolant_Pressure",
    "Motor_Temperature"
]

target = "CNC_Failure"

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

print("\nTraining CNC Machine AI model with machining stress feature engineering...")

base_clf = RandomForestClassifier(
    n_estimators=180,
    max_depth=18,
    min_samples_split=4,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model = fe.ProficientClassifierPipeline(
    base_estimator=base_clf,
    fe_func=fe.engineer_cnc_features,
    base_features=features
)

model.fit(X_train, y_train)


# ==================================================
# MODEL EVALUATION
# ==================================================

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n" + "=" * 60)
print("CNC MACHINE MODEL RESULTS")
print("=" * 60)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, predictions, target_names=["Normal", "Failure Risk"]))


# ==================================================
# SAVE MODEL & ASSETS
# ==================================================

os.makedirs("models", exist_ok=True)

model_file = "models/cnc_model.pkl"
features_file = "models/cnc_features.pkl"

joblib.dump(model, model_file)
joblib.dump(features, features_file)

print("\n" + "=" * 60)
print("CNC MODEL ASSETS SAVED")
print("=" * 60)
print(f"Model saved: {model_file}")
print(f"Features saved: {features_file}")
