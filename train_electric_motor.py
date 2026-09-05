import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import feature_engineering as fe

# ==========================================
# LOAD ELECTRIC MOTOR DATASET
# ==========================================

print("Loading Electric Motor dataset...")
data_path = "data/models/electric_motor_model.pkl.csv"
if not os.path.exists(data_path):
    data_path = "data/models/induction_OTG_motor.csv"

df = pd.read_csv(data_path)

# ==========================================
# DISPLAY DATASET INFORMATION
# ==========================================

print("\nDataset Shape:", df.shape)
print("Columns:", df.columns.tolist())

# ==========================================
# DEFINE FEATURES AND TARGET
# ==========================================

features = [
    "DC_Bus_Voltage",
    "Frequency",
    "High_Resolution_Output_Current",
    "Output_Current",
    "Output_Voltage",
    "Speed",
    "Temperature",
    "Load_Index",
    "Power",
    "Thermal_Load"
]

target = "RUL_hours"

X = df[features]
y = df[target]

# ==========================================
# SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# CREATE & TRAIN PROFICIENT MODEL
# ==========================================

print("\nTraining Electric Motor AI model with electromechanical feature engineering...")

base_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=16,
    min_samples_split=4,
    random_state=42,
    n_jobs=-1
)

model = fe.ProficientRegressorPipeline(
    base_estimator=base_reg,
    fe_func=fe.engineer_motor_features,
    base_features=features
)

model.fit(X_train, y_train)

# ==========================================
# MODEL EVALUATION
# ==========================================

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n" + "=" * 45)
print("ELECTRIC MOTOR MODEL RESULTS")
print("=" * 45)
print(f"Mean Absolute Error: {mae:.2f} hours")
print(f"R² Score:            {r2:.4f}")

# ==========================================
# SAVE MODEL & ASSETS
# ==========================================

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/electric_motor_rul_model.pkl")
joblib.dump(features, "models/electric_motor_features.pkl")

print("\nElectric Motor AI Model Saved successfully to models/electric_motor_rul_model.pkl")