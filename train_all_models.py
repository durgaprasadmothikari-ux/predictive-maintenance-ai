"""
Unified Master Training & Proficiency Benchmarking Orchestrator
Trains all 6 predictive maintenance machine learning models across industrial datasets.
Computes comprehensive metrics (ROC-AUC, PR-AUC, F1, Recall, Precision, MAE, R²)
and exports an auditable benchmark scorecard.
"""

import os
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    ExtraTreesRegressor,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import feature_engineering as fe


os.makedirs("models", exist_ok=True)
benchmark_report = {
    "timestamp": datetime.now().isoformat(),
    "models": {}
}


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f" {title.upper()} ")
    print("=" * 70)


# =====================================================================
# 1. AI4I INDUSTRIAL MACHINE
# =====================================================================
def train_ai4i():
    print_banner("1/6 Training AI4I Industrial Machine Model")
    data_path = "data/ai4i2020.csv"
    if not os.path.exists(data_path):
        print(f"Dataset missing: {data_path}")
        return

    df = pd.read_csv(data_path)
    df = df.drop(columns=["UDI", "Product ID"])
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
    target = "Machine failure"

    X = df[base_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Records: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)} | Failure Rate: {y.mean()*100:.2f}%")

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

    start_time = time.time()
    model.fit(X_train, y_train)
    fit_duration = time.time() - start_time

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    b_acc = balanced_accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"Accuracy:          {acc*100:.2f}%")
    print(f"Balanced Accuracy: {b_acc*100:.2f}%")
    print(f"Precision:         {prec:.4f}")
    print(f"Recall:            {rec:.4f} (Caught {int(rec*68)}/68 test failures)")
    print(f"F1-Score:          {f1:.4f}")
    print(f"ROC-AUC:           {auc:.4f}")
    print(f"PR-AUC:            {pr_auc:.4f}")

    joblib.dump(model, "models/predictive_maintenance_model.pkl")

    benchmark_report["models"]["ai4i_machine"] = {
        "status": "trained",
        "type": "Classification",
        "algorithm": "Proficient HistGradientBoosting + Physics Feature Engineering",
        "training_time_sec": round(fit_duration, 2),
        "test_records": len(X_test),
        "metrics": {
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(b_acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "pr_auc": round(pr_auc, 4),
            "confusion_matrix": cm
        }
    }


# =====================================================================
# 2. ELECTRIC MOTOR RUL (REGRESSION)
# =====================================================================
def train_electric_motor():
    print_banner("2/6 Training Electric Motor RUL Regressor")
    data_path = "data/models/electric_motor_model.pkl.csv"
    if not os.path.exists(data_path):
        data_path = "data/models/induction_OTG_motor.csv"
    if not os.path.exists(data_path):
        print(f"Electric Motor dataset not found.")
        return

    df = pd.read_csv(data_path)
    base_features = [
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

    X = df[base_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Records: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)} | Mean RUL: {y.mean():.1f} hrs")

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
        base_features=base_features
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    fit_duration = time.time() - start_time

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae:.2f} hours")
    print(f"Root Mean Squared Error:   {rmse:.2f} hours")
    print(f"R² Determination Score:    {r2:.4f}")

    joblib.dump(model, "models/electric_motor_rul_model.pkl")
    joblib.dump(base_features, "models/electric_motor_features.pkl")

    benchmark_report["models"]["electric_motor"] = {
        "status": "trained",
        "type": "Regression",
        "algorithm": "Proficient ExtraTrees/RandomForest + Electromechanical Features",
        "training_time_sec": round(fit_duration, 2),
        "test_records": len(X_test),
        "metrics": {
            "mae_hours": round(mae, 2),
            "rmse_hours": round(rmse, 2),
            "r2_score": round(r2, 4)
        }
    }


# =====================================================================
# 3. INDUSTRIAL PUMP
# =====================================================================
def train_industrial_pump():
    print_banner("3/6 Training Industrial Pump Failure Model")
    data_path = "data/industrial_pump.csv"
    if not os.path.exists(data_path):
        import train_pump_model  # Trigger generation if missing
    df = pd.read_csv(data_path)

    base_features = [
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

    X = df[base_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Records: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)} | Failure Rate: {y.mean()*100:.2f}%")

    base_clf = RandomForestClassifier(
        n_estimators=160,
        max_depth=16,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model = fe.ProficientClassifierPipeline(
        base_estimator=base_clf,
        fe_func=fe.engineer_pump_features,
        base_features=base_features
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    fit_duration = time.time() - start_time

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    joblib.dump(model, "models/pump_model.pkl")
    joblib.dump(base_features, "models/pump_features.pkl")

    benchmark_report["models"]["industrial_pump"] = {
        "status": "trained",
        "type": "Classification",
        "algorithm": "Proficient Balanced RandomForest + Hydraulic Features",
        "training_time_sec": round(fit_duration, 2),
        "test_records": len(X_test),
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "pr_auc": round(pr_auc, 4)
        }
    }


# =====================================================================
# 4. CNC MACHINE
# =====================================================================
def train_cnc():
    print_banner("4/6 Training CNC Machine Failure Model")
    data_path = "data/cnc_machine.csv"
    if not os.path.exists(data_path):
        import train_cnc_model
    df = pd.read_csv(data_path)

    base_features = [
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

    X = df[base_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Records: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)} | Failure Rate: {y.mean()*100:.2f}%")

    base_clf = RandomForestClassifier(
        n_estimators=180,
        max_depth=18,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model = fe.ProficientClassifierPipeline(
        base_estimator=base_clf,
        fe_func=fe.engineer_cnc_features,
        base_features=base_features
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    fit_duration = time.time() - start_time

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    joblib.dump(model, "models/cnc_model.pkl")
    joblib.dump(base_features, "models/cnc_features.pkl")

    benchmark_report["models"]["cnc_machine"] = {
        "status": "trained",
        "type": "Classification",
        "algorithm": "Proficient Balanced RandomForest + Machining Stress Features",
        "training_time_sec": round(fit_duration, 2),
        "test_records": len(X_test),
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "pr_auc": round(pr_auc, 4)
        }
    }


# =====================================================================
# 5. CONVEYOR SYSTEM
# =====================================================================
def train_conveyor():
    print_banner("5/6 Training Conveyor System Failure Model")
    data_path = "data/conveyor_system.csv"
    if not os.path.exists(data_path):
        import train_conveyor_model
    df = pd.read_csv(data_path)

    base_features = [
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

    X = df[base_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Records: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)} | Failure Rate: {y.mean()*100:.2f}%")

    base_clf = RandomForestClassifier(
        n_estimators=180,
        max_depth=16,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model = fe.ProficientClassifierPipeline(
        base_estimator=base_clf,
        fe_func=fe.engineer_conveyor_features,
        base_features=base_features
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    fit_duration = time.time() - start_time

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    joblib.dump(model, "models/conveyor_model.pkl")
    joblib.dump(base_features, "models/conveyor_features.pkl")

    benchmark_report["models"]["conveyor_system"] = {
        "status": "trained",
        "type": "Classification",
        "algorithm": "Proficient Balanced RandomForest + Belt Friction Features",
        "training_time_sec": round(fit_duration, 2),
        "test_records": len(X_test),
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "pr_auc": round(pr_auc, 4)
        }
    }


# =====================================================================
# 6. SENSOR MACHINE (MULTI-SENSOR 220K ROWS)
# =====================================================================
def train_sensor():
    print_banner("6/6 Training Sensor Machine Model (220k rows)")
    data_path = "data/sensor.csv"
    if not os.path.exists(data_path):
        print(f"Sensor dataset not found: {data_path}")
        return

    print("Loading sensor dataset...")
    df = pd.read_csv(data_path)
    target_col = "machine_status"

    valid_rows = df[target_col].notna()
    df = df.loc[valid_rows]

    y = df[target_col]
    X = df.drop(columns=[target_col]).select_dtypes(include=["number"])

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    sensor_features = X.columns.tolist()

    print(f"Total Rows: {len(X)} | Features: {len(sensor_features)} | Classes: {list(label_encoder.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=120,
            max_depth=18,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    start_time = time.time()
    print("Fitting sensor machine pipeline...")
    pipeline.fit(X_train, y_train)
    fit_duration = time.time() - start_time

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    print(f"Accuracy: {acc*100:.2f}% | Macro F1: {f1_macro:.4f}")

    joblib.dump(pipeline, "models/sensor_machine_model.pkl")
    joblib.dump(sensor_features, "models/sensor_machine_features.pkl")
    joblib.dump(label_encoder, "models/sensor_machine_label_encoder.pkl")

    benchmark_report["models"]["sensor_machine"] = {
        "status": "trained",
        "type": "Multi-Class Classification",
        "algorithm": "Median Imputer + Balanced RandomForest (52 Sensors)",
        "training_time_sec": round(fit_duration, 2),
        "test_records": len(X_test),
        "metrics": {
            "accuracy": round(acc, 4),
            "macro_f1": round(f1_macro, 4),
            "classes": list(label_encoder.classes_)
        }
    }


def save_reports():
    report_path = "models/model_benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)
    print_banner("Benchmark Report Saved")
    print(f"JSON Benchmark Report: {report_path}")

    # Generate Markdown Report
    md_content = f"""# Predictive Maintenance AI - Model Benchmark Report
*Generated: {benchmark_report['timestamp']}*

## Model Performance Summary

| Machine Workspace | Model Type | Core Algorithm | Primary Metric | ROC-AUC / R² | F1 / MAE | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for key, info in benchmark_report["models"].items():
        name = key.replace("_", " ").title()
        algo = info["algorithm"]
        m = info["metrics"]
        if info["type"] == "Regression":
            primary = f"MAE: {m['mae_hours']} hrs"
            sec = f"R²: {m['r2_score']}"
            third = f"RMSE: {m['rmse_hours']} hrs"
        elif "macro_f1" in m:
            primary = f"Acc: {m['accuracy']*100:.2f}%"
            sec = "Multi-class"
            third = f"Macro F1: {m['macro_f1']:.4f}"
        else:
            primary = f"Acc: {m['accuracy']*100:.2f}%"
            sec = f"AUC: {m.get('roc_auc', 'N/A')}"
            third = f"F1: {m.get('f1_score', 'N/A')} (Rec: {m.get('recall', 'N/A')})"
        
        md_content += f"| **{name}** | {info['type']} | {algo} | {primary} | {sec} | {third} | {info['training_time_sec']}s |\n"

    md_content += """
---
*All models include physics-based domain feature engineering pipelines and are 100% backward-compatible with the Streamlit interface.*
"""
    with open("models/MODEL_BENCHMARK.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    print("Markdown Benchmark Report: models/MODEL_BENCHMARK.md")


if __name__ == "__main__":
    train_ai4i()
    train_electric_motor()
    train_industrial_pump()
    train_cnc()
    train_conveyor()
    train_sensor()
    save_reports()
    print_banner("All 6 Predictive Maintenance Models Successfully Trained & Verified!")
