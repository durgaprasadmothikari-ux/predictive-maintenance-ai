# Predictive Maintenance AI - Model Benchmark Report
*Generated: 2026-09-05T08:50:14.081895*

## Model Performance Summary

| Machine Workspace | Model Type | Core Algorithm | Primary Metric | ROC-AUC / R² | F1 / MAE | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ai4I Machine** | Classification | Proficient HistGradientBoosting + Physics Feature Engineering | Acc: 99.15% | AUC: 0.9849 | F1: 0.8702 (Rec: 0.8382) | 1.2s |
| **Electric Motor** | Regression | Proficient ExtraTrees/RandomForest + Electromechanical Features | MAE: 1085.25 hrs | R²: 0.9452 | RMSE: 1558.57 hrs | 0.75s |
| **Industrial Pump** | Classification | Proficient Balanced RandomForest + Hydraulic Features | Acc: 99.30% | AUC: 0.9761 | F1: 0.8627 (Rec: 0.8302) | 0.35s |
| **Cnc Machine** | Classification | Proficient Balanced RandomForest + Machining Stress Features | Acc: 98.45% | AUC: 0.9924 | F1: 0.6265 (Rec: 0.8125) | 0.45s |
| **Conveyor System** | Classification | Proficient Balanced RandomForest + Belt Friction Features | Acc: 99.65% | AUC: 0.998 | F1: 0.4615 (Rec: 0.6) | 0.4s |
| **Sensor Machine** | Multi-Class Classification | Median Imputer + Balanced RandomForest (52 Sensors) | Acc: 99.98% | Multi-class | Macro F1: 0.6662 | 7.18s |

---
*All models include physics-based domain feature engineering pipelines and are 100% backward-compatible with the Streamlit interface.*
