"""
Feature Engineering Pipelines and Proficient Model Wrappers for Predictive Maintenance.
Provides physics-based feature transformations and backward-compatible model wrappers.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


# =====================================================================
# FEATURE ENGINEERING FUNCTIONS
# =====================================================================

def engineer_ai4i_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates thermodynamic, mechanical power, and wear stress features for AI4I."""
    df = df.copy()
    temp_diff = df["Process temperature [K]"] - df["Air temperature [K]"]
    power_w = df["Rotational speed [rpm]"] * (2.0 * np.pi / 60.0) * df["Torque [Nm]"]
    overstrain = df["Tool wear [min]"] * df["Torque [Nm]"]
    wear_speed = df["Tool wear [min]"] / (df["Rotational speed [rpm]"] + 1e-5)
    heat_dissip = ((temp_diff < 8.6) & (df["Rotational speed [rpm]"] < 1380)).astype(float)

    df["Temp_Diff"] = temp_diff
    df["Power_W"] = power_w
    df["Overstrain_Index"] = overstrain
    df["Wear_Speed_Ratio"] = wear_speed
    df["Heat_Dissipation_Stress"] = heat_dissip
    return df


def engineer_motor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates apparent power, power factor, thermal severity, and slip for Electric Motor."""
    df = df.copy()
    apparent = df["Output_Voltage"] * df["Output_Current"]
    pf = (df["Power"] / (apparent + 1e-5)).clip(0.05, 1.0)
    thermal_sev = df["Temperature"] * df["Thermal_Load"] / 1000.0
    imbalance = df["High_Resolution_Output_Current"] / (df["Output_Current"] + 1e-5)
    sync_speed = 120.0 * df["Frequency"] / 4.0  # 4-pole motor synchronous speed assumption
    slip = (np.abs(df["Speed"] - sync_speed) / (sync_speed + 1e-5)).clip(0.0, 1.0)

    df["Apparent_Power"] = apparent
    df["Power_Factor"] = pf
    df["Thermal_Severity"] = thermal_sev
    df["Current_Imbalance_Ratio"] = imbalance
    df["Mechanical_Slip"] = slip
    return df


def engineer_pump_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates head differential, hydraulic power, ISO-10816 vibration, and cavitation risk."""
    df = df.copy()
    head_diff = (df["Discharge_Pressure"] - df["Suction_Pressure"]).clip(0.1)
    hydraulic_power = (df["Flow_Rate"] * head_diff) / 36.7
    vibe_zone = np.where(
        df["Vibration_RMS"] > 7.1, 3.0,
        np.where(df["Vibration_RMS"] > 4.5, 2.0,
        np.where(df["Vibration_RMS"] > 2.8, 1.0, 0.0))
    )
    cav_risk = ((df["Cavitation_Index"] < 1.0) & (df["Suction_Pressure"] < 1.5)).astype(float)
    thermal_stress = (df["Bearing_Temperature"] - df["Fluid_Temperature"]).clip(0.0)

    df["Head_Pressure_Diff"] = head_diff
    df["Hydraulic_Power_kW"] = hydraulic_power
    df["Vibration_Severity_ISO"] = vibe_zone
    df["Cavitation_Risk"] = cav_risk
    df["Thermal_Stress_Pump"] = thermal_stress
    return df


def engineer_cnc_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates specific cutting energy, chatter indicator, and tool stress product."""
    df = df.copy()
    sec = df["Cutting_Force"] / (df["Feed_Rate"] + 1e-5)
    chatter = df["Spindle_Vibration"] * (df["Spindle_Speed"] / 1000.0)
    tool_stress = df["Tool_Wear_Index"] * df["Cutting_Force"] / 1000.0
    coolant_risk = ((df["Coolant_Pressure"] < 10.0) & (df["Motor_Temperature"] > 68.0)).astype(float)
    feed_load = df["Axis_Feed_Error"] * df["Feed_Rate"] / 100.0

    df["Specific_Cutting_Energy"] = sec
    df["Chatter_Indicator"] = chatter
    df["Tool_Stress_Product"] = tool_stress
    df["Coolant_Starvation_Risk"] = coolant_risk
    df["Feed_Load_Index"] = feed_load
    return df


def engineer_conveyor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates slip-tension risk, roller friction index, and motor work efficiency."""
    df = df.copy()
    slip_sev = np.where(df["Belt_Slip_Percentage"] > 6.0, 2.0, df["Belt_Slip_Percentage"] / 3.0)
    roller_friction = (df["Roller_Bearing_Temperature"] - df["Ambient_Temperature"]).clip(0.0) * df["Idler_Vibration"]
    tension_load = df["Belt_Tension"] / (df["Load_Weight"] / 100.0 + 1e-5)
    specific_work = df["Motor_Current"] / (df["Belt_Speed"] + 1e-5)
    overload = ((df["Belt_Tension"] > 35.0) | (df["Belt_Tension"] < 10.0)).astype(float)

    df["Slip_Severity"] = slip_sev
    df["Roller_Friction_Index"] = roller_friction
    df["Tension_Load_Ratio"] = tension_load
    df["Motor_Specific_Work"] = specific_work
    df["Tension_Overload_Risk"] = overload
    return df


# =====================================================================
# SKLEARN-COMPATIBLE WRAPPERS WITH EMBEDDED FEATURE PIPELINES
# =====================================================================

class ProficientClassifierPipeline(BaseEstimator, ClassifierMixin):
    """
    Transparently applies domain feature engineering to raw input DataFrames
    and delegates to an underlying scikit-learn classifier.
    """
    def __init__(self, base_estimator, fe_func, base_features=None):
        self.base_estimator = base_estimator
        self.fe_func = fe_func
        self.base_features = base_features
        self.classes_ = getattr(base_estimator, "classes_", np.array([0, 1]))

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X_fe = self.fe_func(X)
        else:
            X_fe = X
        self.base_estimator.fit(X_fe, y)
        self.classes_ = getattr(self.base_estimator, "classes_", np.array([0, 1]))
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X_fe = self.fe_func(X)
        else:
            X_fe = X
        return self.base_estimator.predict(X_fe)

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X_fe = self.fe_func(X)
        else:
            X_fe = X
        if hasattr(self.base_estimator, "predict_proba"):
            return self.base_estimator.predict_proba(X_fe)
        # Fallback if decision function exists
        if hasattr(self.base_estimator, "decision_function"):
            df = self.base_estimator.decision_function(X_fe)
            prob = 1.0 / (1.0 + np.exp(-df))
            return np.column_stack([1.0 - prob, prob])
        preds = self.predict(X)
        return np.column_stack([1.0 - preds, preds])

    @property
    def feature_importances_(self):
        if hasattr(self.base_estimator, "feature_importances_"):
            return self.base_estimator.feature_importances_
        return None


class ProficientRegressorPipeline(BaseEstimator, RegressorMixin):
    """
    Transparently applies domain feature engineering to raw input DataFrames
    and delegates to an underlying scikit-learn regressor.
    """
    def __init__(self, base_estimator, fe_func, base_features=None):
        self.base_estimator = base_estimator
        self.fe_func = fe_func
        self.base_features = base_features

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X_fe = self.fe_func(X)
        else:
            X_fe = X
        self.base_estimator.fit(X_fe, y)
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X_fe = self.fe_func(X)
        else:
            X_fe = X
        return self.base_estimator.predict(X_fe)

    @property
    def feature_importances_(self):
        if hasattr(self.base_estimator, "feature_importances_"):
            return self.base_estimator.feature_importances_
        return None
