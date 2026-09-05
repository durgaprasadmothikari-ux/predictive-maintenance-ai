import os
import json
from datetime import datetime
import streamlit as st
import pandas as pd
import joblib

import feature_engineering as fe
from humanoid_assistant import HumanoidAssistant


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Predictive Maintenance AI · Dr. Nova",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>
    :root {
        --panel: rgba(255,255,255,0.035);
        --panel-soft: rgba(255,255,255,0.022);
        --border: rgba(148,163,184,0.18);
        --muted: #9CA3AF;
        --text: #E5E7EB;
        --green: #22C55E;
        --amber: #F59E0B;
        --red: #EF4444;
        --blue: #60A5FA;
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1480px;
    }

    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: 760;
        line-height: 1.1;
        margin-bottom: 6px;
        letter-spacing: -1px;
    }

    .subtitle {
        text-align: center;
        font-size: 15px;
        color: var(--muted);
        margin-top: 0;
        margin-bottom: 24px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 720;
        margin-top: 4px;
        margin-bottom: 8px;
        letter-spacing: -0.3px;
    }

    .status-good, .status-warning, .status-critical {
        padding: 16px 20px;
        border-radius: 12px;
        margin-top: 12px;
        line-height: 1.55;
    }

    .status-good {
        background: rgba(34,197,94,0.10);
        border: 1px solid rgba(34,197,94,0.25);
        border-left: 4px solid var(--green);
    }

    .status-warning {
        background: rgba(245,158,11,0.10);
        border: 1px solid rgba(245,158,11,0.25);
        border-left: 4px solid var(--amber);
    }

    .status-critical {
        background: rgba(239,68,68,0.10);
        border: 1px solid rgba(239,68,68,0.25);
        border-left: 4px solid var(--red);
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid var(--border);
    }

    div[data-testid="stMetric"] {
        padding: 4px 0;
    }

    .stButton > button {
        border-radius: 10px;
        min-height: 42px;
        font-weight: 650;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# SESSION STATE INITIALIZATION
# ==================================================

for history_key in [
    "prediction_history",
    "ai4i_prediction_history",
    "sensor_machine_prediction_history",
    "pump_prediction_history",
    "cnc_prediction_history",
    "conveyor_prediction_history",
    "nova_global_chat"
]:
    if history_key not in st.session_state:
        st.session_state[history_key] = []

for result_key in [
    "latest_motor_result",
    "latest_ai4i_result",
    "latest_sensor_machine_result",
    "latest_pump_result",
    "latest_cnc_result",
    "latest_conveyor_result"
]:
    if result_key not in st.session_state:
        st.session_state[result_key] = None


# ==================================================
# MODEL LOADERS (CACHED)
# ==================================================

@st.cache_resource
def load_ai4i_model():
    try:
        return joblib.load("models/predictive_maintenance_model.pkl")
    except Exception:
        return None

@st.cache_resource
def load_electric_motor_assets():
    try:
        model = joblib.load("models/electric_motor_rul_model.pkl")
        features = joblib.load("models/electric_motor_features.pkl")
        return model, list(features)
    except Exception:
        return None, []

@st.cache_resource
def load_sensor_machine_assets():
    try:
        model = joblib.load("models/sensor_machine_model.pkl")
        features = joblib.load("models/sensor_machine_features.pkl")
        label_enc = joblib.load("models/sensor_machine_label_encoder.pkl")
        return model, list(features), label_enc
    except Exception:
        return None, [], None

@st.cache_resource
def load_pump_assets():
    try:
        model = joblib.load("models/pump_model.pkl")
        features = joblib.load("models/pump_features.pkl")
        return model, list(features)
    except Exception:
        return None, []

@st.cache_resource
def load_cnc_assets():
    try:
        model = joblib.load("models/cnc_model.pkl")
        features = joblib.load("models/cnc_features.pkl")
        return model, list(features)
    except Exception:
        return None, []

@st.cache_resource
def load_conveyor_assets():
    try:
        model = joblib.load("models/conveyor_model.pkl")
        features = joblib.load("models/conveyor_features.pkl")
        return model, list(features)
    except Exception:
        return None, []

@st.cache_data
def load_sensor_baseline_values(features):
    defaults = {f: 0.0 for f in features}
    try:
        sample = pd.read_csv("data/sensor.csv", nrows=1)
        for f in features:
            if f in sample.columns:
                try:
                    defaults[f] = float(sample.iloc[0][f])
                except Exception:
                    defaults[f] = 0.0
    except Exception:
        pass
    return defaults


# Load all models
ai4i_model = load_ai4i_model()
electric_motor_model, electric_motor_features = load_electric_motor_assets()
sensor_machine_model, sensor_machine_features, sensor_machine_label_encoder = load_sensor_machine_assets()
pump_model, pump_features = load_pump_assets()
cnc_model, cnc_features = load_cnc_assets()
conveyor_model, conveyor_features = load_conveyor_assets()


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, rgba(30,58,95,0.7), rgba(15,23,42,0.85));
            border: 1px solid rgba(96,165,250,0.35);
            border-radius: 14px;
            padding: 14px 16px;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 12px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.25);
        ">
            <div style="font-size: 32px;">🤖</div>
            <div>
                <div style="font-weight: 750; font-size: 15px; color: #F8FAFC;">Dr. Nova</div>
                <div style="font-size: 11px; color: #4ADE80; font-weight: 700;">● Chief Reliability Engineer</div>
                <div style="font-size: 10px; color: #94A3B8;">Autonomous Triage Active</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.title("Predictive Maintenance")
    st.caption("AI-Powered Machinery Diagnostics & Triage")
    st.divider()

    st.subheader("Model Availability")
    models_status = [
        ("AI4I Industrial Machine", ai4i_model is not None),
        ("Electric Motor", electric_motor_model is not None),
        ("Sensor Machine (52 Sensors)", sensor_machine_model is not None),
        ("Industrial Pump", pump_model is not None),
        ("CNC Machine", cnc_model is not None),
        ("Conveyor System", conveyor_model is not None)
    ]

    online_count = sum(1 for _, ok in models_status if ok)
    for m_name, is_ok in models_status:
        if is_ok:
            st.success(f"{m_name} — ONLINE", icon="✅")
        else:
            st.error(f"{m_name} — OFFLINE", icon="⚠️")

    st.divider()
    st.subheader("Session Statistics")
    total_analyses = (
        len(st.session_state.ai4i_prediction_history)
        + len(st.session_state.prediction_history)
        + len(st.session_state.sensor_machine_prediction_history)
        + len(st.session_state.pump_prediction_history)
        + len(st.session_state.cnc_prediction_history)
        + len(st.session_state.conveyor_prediction_history)
    )
    st.metric("Total Telemetry Runs", total_analyses)
    st.metric("AI Models Online", f"{online_count}/6")

    st.divider()
    with st.expander("💬 Consult Dr. Nova (Sidebar)", expanded=False):
        sidebar_q = st.text_input("Ask Dr. Nova:", placeholder="e.g. What are the common failure modes?", key="sidebar_nova_query")
        if sidebar_q:
            resp = HumanoidAssistant.converse(
                user_message=sidebar_q,
                machine_name="General Plant Machinery",
                telemetry={},
                assessment={}
            )
            st.markdown(f"**Dr. Nova:**\n\n{resp}")


# ==================================================
# INPUT VALIDATION HELPER
# ==================================================

def validate_range(label, value, minimum, maximum):
    if value < minimum or value > maximum:
        st.error(f"⚠️ {label} value ({value}) is outside the valid operating range [{minimum}, {maximum}].")
        return False
    return True


# ==================================================
# APPLICATION HEADER
# ==================================================

st.markdown('<div class="main-title">🏭 Industrial Predictive Maintenance AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">High-Proficiency Machine Learning Diagnostics & Humanoid Reliability Engineering</div>', unsafe_allow_html=True)

st.subheader("Select a Workspace")
selected_machine = st.selectbox(
    "Choose a machine workspace or Command Center",
    [
        "Command Center",
        "AI4I Industrial Machine",
        "Electric Motor",
        "Sensor Machine",
        "Industrial Pump",
        "CNC Machine",
        "Conveyor System"
    ]
)
st.divider()


# ==================================================
# 1. COMMAND CENTER
# ==================================================

if selected_machine == "Command Center":
    HumanoidAssistant.render_command_center_briefing(total_analyses, online_count)

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Assessments Run", total_analyses)
    m2.metric("Models Online & Trained", f"{online_count}/6")
    m3.metric("Factory Plant Readiness", "100% OPERATIONAL")
    m4.metric("Active Reliability Engineer", "Dr. Nova (AI)")

    # Model Proficiency & Benchmark Report Expander
    if os.path.exists("models/model_benchmark_report.json"):
        with st.expander("📊 Model Proficiency Scorecard (Trained on Real Datasets)", expanded=False):
            try:
                with open("models/model_benchmark_report.json", "r", encoding="utf-8") as f:
                    bench_data = json.load(f)
                b_cols = st.columns(3)
                idx = 0
                for m_key, m_val in bench_data.get("models", {}).items():
                    with b_cols[idx % 3]:
                        with st.container(border=True):
                            st.markdown(f"**{m_key.replace('_', ' ').title()}**")
                            st.caption(f"{m_val.get('algorithm')}")
                            metrics = m_val.get("metrics", {})
                            if "accuracy" in metrics:
                                st.write(f"Accuracy: `{metrics['accuracy']*100:.2f}%`")
                            if "roc_auc" in metrics:
                                st.write(f"ROC-AUC: `{metrics['roc_auc']}`")
                            if "f1_score" in metrics:
                                st.write(f"F1-Score: `{metrics['f1_score']}`")
                            if "mae_hours" in metrics:
                                st.write(f"MAE: `{metrics['mae_hours']} hrs`")
                            if "r2_score" in metrics:
                                st.write(f"R² Score: `{metrics['r2_score']}`")
                    idx += 1
            except Exception as e:
                st.caption(f"Scorecard loaded: {e}")

    st.divider()
    st.subheader("Fleet Telemetry Intelligence")

    # Row 1: AI4I, Electric Motor, Sensor Machine
    r1_c1, r1_c2, r1_c3 = st.columns(3)
    with r1_c1:
        with st.container(border=True):
            st.markdown("### AI4I Industrial Machine")
            if st.session_state.latest_ai4i_result is not None:
                r = st.session_state.latest_ai4i_result
                st.metric("Machine Health", r.get("health", "Not available"))
                st.metric("Failure Probability", f'{r.get("failure_probability", 0):.2f}%')
            else:
                st.info("No analysis run in current session.")

    with r1_c2:
        with st.container(border=True):
            st.markdown("### Electric Motor")
            if st.session_state.latest_motor_result is not None:
                r = st.session_state.latest_motor_result
                st.metric("Machine Health", r.get("health", "Not available"))
                st.metric("Predicted RUL", f'{r.get("prediction", 0):.1f} Hours')
            else:
                st.info("No analysis run in current session.")

    with r1_c3:
        with st.container(border=True):
            st.markdown("### Sensor Machine")
            if st.session_state.latest_sensor_machine_result is not None:
                r = st.session_state.latest_sensor_machine_result
                st.metric("Detected Status", r.get("status", "Not available"))
                st.metric("Confidence", f'{r.get("confidence", 0):.1f}%')
            else:
                st.info("No analysis run in current session.")

    # Row 2: Industrial Pump, CNC Machine, Conveyor System
    r2_c1, r2_c2, r2_c3 = st.columns(3)
    with r2_c1:
        with st.container(border=True):
            st.markdown("### Industrial Pump")
            if st.session_state.latest_pump_result is not None:
                r = st.session_state.latest_pump_result
                st.metric("Machine Health", r.get("health", "Not available"))
                st.metric("Failure Probability", f'{r.get("failure_probability", 0):.2f}%')
            else:
                st.info("No analysis run in current session.")

    with r2_c2:
        with st.container(border=True):
            st.markdown("### CNC Machine")
            if st.session_state.latest_cnc_result is not None:
                r = st.session_state.latest_cnc_result
                st.metric("Machine Health", r.get("health", "Not available"))
                st.metric("Failure Probability", f'{r.get("failure_probability", 0):.2f}%')
            else:
                st.info("No analysis run in current session.")

    with r2_c3:
        with st.container(border=True):
            st.markdown("### Conveyor System")
            if st.session_state.latest_conveyor_result is not None:
                r = st.session_state.latest_conveyor_result
                st.metric("Machine Health", r.get("health", "Not available"))
                st.metric("Failure Probability", f'{r.get("failure_probability", 0):.2f}%')
            else:
                st.info("No analysis run in current session.")


# ==================================================
# 2. AI4I INDUSTRIAL MACHINE WORKSPACE
# ==================================================

elif selected_machine == "AI4I Industrial Machine":
    st.markdown('<div class="section-title">⚙️ AI4I Industrial Machine Diagnostics</div>', unsafe_allow_html=True)
    st.write("Physics-grounded failure prediction analyzing thermodynamic heat dissipation, power deviations, and overstrain.")

    if ai4i_model is None:
        st.error("AI4I Machine Model is offline. Train models using `python train_all_models.py`.")
        st.stop()

    # Presets
    preset_cols = st.columns(4)
    p_norm = preset_cols[0].button("🟢 Normal Operation Preset", key="ai4i_norm", use_container_width=True)
    p_hdf = preset_cols[1].button("🔥 Heat Dissipation Anomaly", key="ai4i_hdf", use_container_width=True)
    p_osf = preset_cols[2].button("⚡ Overstrain Failure Anomaly", key="ai4i_osf", use_container_width=True)
    p_twf = preset_cols[3].button("🔪 Tool Wear Critical", key="ai4i_twf", use_container_width=True)

    def_air_t = 300.0 if not (p_hdf or p_osf or p_twf) else (303.5 if p_hdf else 301.0)
    def_proc_t = 310.0 if not (p_hdf or p_osf or p_twf) else (308.2 if p_hdf else 311.5)
    def_rpm = 1550 if not (p_hdf or p_osf or p_twf) else (1250 if p_hdf else 1350)
    def_torque = 42.0 if not (p_hdf or p_osf or p_twf) else (58.0 if p_osf else 48.0)
    def_wear = 50.0 if not (p_hdf or p_osf or p_twf) else (215.0 if (p_twf or p_osf) else 45.0)

    inp_c1, inp_c2 = st.columns(2)
    with inp_c1:
        air_temperature = st.number_input("Air Temperature [K]", min_value=250.0, max_value=400.0, value=float(def_air_t), step=0.5)
        process_temperature = st.number_input("Process Temperature [K]", min_value=250.0, max_value=450.0, value=float(def_proc_t), step=0.5)
        rotational_speed = st.number_input("Rotational Speed [RPM]", min_value=100.0, max_value=20000.0, value=float(def_rpm), step=25.0)
    with inp_c2:
        torque = st.number_input("Torque [Nm]", min_value=0.1, max_value=500.0, value=float(def_torque), step=0.5)
        tool_wear = st.number_input("Tool Wear [Minutes]", min_value=0.0, max_value=10000.0, value=float(def_wear), step=5.0)
        machine_type = st.selectbox("Machine Type (Quality Grade)", ["L", "M", "H"], index=0)

    st.divider()
    if st.button("Run AI4I Machine Assessment", type="primary", use_container_width=True, key="btn_run_ai4i"):
        input_data = pd.DataFrame([{
            "Air temperature [K]": air_temperature,
            "Process temperature [K]": process_temperature,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
            "Type_L": 1 if machine_type == "L" else 0,
            "Type_M": 1 if machine_type == "M" else 0
        }])

        prediction = ai4i_model.predict(input_data)[0]
        if hasattr(ai4i_model, "predict_proba"):
            failure_prob = float(ai4i_model.predict_proba(input_data)[0][1])
        else:
            failure_prob = 1.0 if prediction == 1 else 0.0

        fail_pct = failure_prob * 100.0
        health = "GOOD 🟢" if fail_pct < 35.0 else ("ATTENTION 🟡" if fail_pct < 70.0 else "CRITICAL 🔴")

        st.session_state.latest_ai4i_result = {
            "prediction": int(prediction),
            "failure_probability": fail_pct,
            "health": health
        }

        # Save History
        st.session_state.ai4i_prediction_history.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Air Temp": air_temperature,
            "Process Temp": process_temperature,
            "Speed RPM": rotational_speed,
            "Torque Nm": torque,
            "Tool Wear": tool_wear,
            "Failure Probability %": round(fail_pct, 2),
            "Health": health
        })

    if st.session_state.latest_ai4i_result is not None:
        res = st.session_state.latest_ai4i_result
        col1, col2, col3 = st.columns(3)
        col1.metric("Failure Prediction", "FAILURE DETECTED ⚠️" if res["prediction"] == 1 else "NO FAILURE DETECTED ✅")
        col2.metric("Machine Health", res["health"])
        col3.metric("Failure Probability", f"{res['failure_probability']:.2f}%")
        st.progress(int(min(max(res["failure_probability"], 0.0), 100.0)))

        # Dr. Nova Humanoid Section
        ai4i_telemetry = {
            "Air temperature [K]": air_temperature,
            "Process temperature [K]": process_temperature,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear
        }
        HumanoidAssistant.render_humanoid_section(
            machine_name="AI4I Industrial Machine",
            telemetry=ai4i_telemetry,
            is_failure=(res["prediction"] == 1),
            failure_prob=res["failure_probability"] / 100.0,
            key_suffix="ai4i"
        )

    if st.session_state.ai4i_prediction_history:
        st.divider()
        st.subheader("AI4I Analysis History")
        h_df = pd.DataFrame(st.session_state.ai4i_prediction_history)
        st.dataframe(h_df, use_container_width=True, hide_index=True)


# ==================================================
# 3. ELECTRIC MOTOR WORKSPACE
# ==================================================

elif selected_machine == "Electric Motor":
    st.markdown('<div class="section-title">⚡ Electric Motor Remaining Useful Life (RUL)</div>', unsafe_allow_html=True)
    st.write("Electromechanical degradation estimation predicting Remaining Useful Life (RUL) hours and insulation stress.")

    if electric_motor_model is None:
        st.error("Electric Motor model is offline. Train models using `python train_all_models.py`.")
        st.stop()

    preset_cols = st.columns(3)
    p_norm = preset_cols[0].button("🟢 Healthy Motor Preset", key="mtr_norm", use_container_width=True)
    p_therm = preset_cols[1].button("🔥 Stator Overheating Preset", key="mtr_therm", use_container_width=True)
    p_load = preset_cols[2].button("⚠️ Mechanical Overload Preset", key="mtr_load", use_container_width=True)

    def_temp = 62.0 if not (p_therm or p_load) else (88.0 if p_therm else 75.0)
    def_curr = 24.0 if not (p_therm or p_load) else (48.0 if p_load else 32.0)
    def_pwr = 14.0 if not (p_therm or p_load) else (28.0 if p_load else 18.0)
    def_therm_load = 400.0 if not (p_therm or p_load) else (1450.0 if p_therm else 800.0)

    c1, c2 = st.columns(2)
    with c1:
        dc_bus_voltage = st.number_input("DC Bus Voltage [V]", min_value=0.0, max_value=1000.0, value=400.0)
        frequency = st.number_input("Frequency [Hz]", min_value=0.0, max_value=500.0, value=50.0)
        high_res_current = st.number_input("High Resolution Current [A]", min_value=0.0, max_value=10000.0, value=float(def_curr))
        output_current = st.number_input("Output Current [A]", min_value=0.0, max_value=1000.0, value=float(def_curr))
        output_voltage = st.number_input("Output Voltage [V]", min_value=0.0, max_value=2000.0, value=400.0)
    with c2:
        speed = st.number_input("Speed [RPM]", min_value=0.0, max_value=20000.0, value=1450.0)
        temperature = st.number_input("Temperature [°C]", min_value=0.0, max_value=250.0, value=float(def_temp))
        load_index = st.number_input("Load Index", min_value=0.0, max_value=1000.0, value=120.0)
        power = st.number_input("Power [kW]", min_value=0.0, max_value=1000000.0, value=float(def_pwr))
        thermal_load = st.number_input("Thermal Load [J]", min_value=0.0, max_value=1000000.0, value=float(def_therm_load))

    st.divider()
    if st.button("Run Motor RUL Assessment", type="primary", use_container_width=True, key="btn_run_motor"):
        input_data = pd.DataFrame([[
            dc_bus_voltage, frequency, high_res_current, output_current,
            output_voltage, speed, temperature, load_index, power, thermal_load
        ]], columns=electric_motor_features)

        predicted_rul = float(electric_motor_model.predict(input_data)[0])
        risk_score = round(max(0.0, min(100.0, (1.0 - (predicted_rul / 5000.0)) * 100.0)), 1)
        health = "GOOD 🟢" if risk_score < 35.0 else ("ATTENTION 🟡" if risk_score < 70.0 else "CRITICAL 🔴")

        st.session_state.latest_motor_result = {
            "prediction": predicted_rul,
            "risk_score": risk_score,
            "health": health
        }

        st.session_state.prediction_history.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Temperature": temperature,
            "Speed": speed,
            "Current": output_current,
            "Predicted RUL (Hours)": round(predicted_rul, 1),
            "Risk Score %": risk_score,
            "Health": health
        })

    if st.session_state.latest_motor_result is not None:
        res = st.session_state.latest_motor_result
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Remaining Useful Life", f"{res['prediction']:.1f} Hours")
        col2.metric("Machine Health", res["health"])
        col3.metric("Operating Risk Score", f"{res['risk_score']:.1f}%")
        st.progress(int(res["risk_score"]))

        motor_telemetry = {
            "Temperature": temperature,
            "Speed": speed,
            "Output_Current": output_current,
            "Output_Voltage": output_voltage,
            "Power": power,
            "Load_Index": load_index
        }
        HumanoidAssistant.render_humanoid_section(
            machine_name="Electric Motor",
            telemetry=motor_telemetry,
            is_failure=(res["risk_score"] >= 50.0 or res["prediction"] < 500.0),
            failure_prob=res["risk_score"] / 100.0,
            extra_metrics={"rul_hours": res["prediction"]},
            key_suffix="motor"
        )

    if st.session_state.prediction_history:
        st.divider()
        st.subheader("Motor Assessment History")
        st.dataframe(pd.DataFrame(st.session_state.prediction_history), use_container_width=True, hide_index=True)


# ==================================================
# 4. SENSOR MACHINE WORKSPACE
# ==================================================

elif selected_machine == "Sensor Machine":
    st.markdown('<div class="section-title">📊 Multi-Sensor Machine Health Classification</div>', unsafe_allow_html=True)
    st.write("Pattern recognition across 52 industrial sensor feeds to classify machine status (NORMAL, RECOVERING, BROKEN).")

    if sensor_machine_model is None or not sensor_machine_features:
        st.error("Sensor Machine model is offline. Train models using `python train_all_models.py`.")
        st.stop()

    baseline = load_sensor_baseline_values(sensor_machine_features)
    sensor_df = pd.DataFrame({
        "Sensor Feature": sensor_machine_features,
        "Current Reading": [baseline.get(f, 0.0) for f in sensor_machine_features]
    })

    st.caption("Inspect or edit the multi-sensor readings below:")
    edited_sensor_table = st.data_editor(sensor_df, use_container_width=True, height=260, key="sensor_table")

    st.divider()
    if st.button("Run Multi-Sensor Assessment", type="primary", use_container_width=True, key="btn_run_sensor"):
        sensor_values = pd.to_numeric(edited_sensor_table["Current Reading"], errors="coerce")
        input_data = pd.DataFrame([sensor_values.tolist()], columns=sensor_machine_features)

        pred_enc = sensor_machine_model.predict(input_data)[0]
        detected_status = str(sensor_machine_label_encoder.inverse_transform([int(pred_enc)])[0]) if sensor_machine_label_encoder else str(pred_enc)

        confidence = 95.0
        if hasattr(sensor_machine_model, "predict_proba"):
            probs = sensor_machine_model.predict_proba(input_data)[0]
            confidence = float(max(probs) * 100.0)

        health = "GOOD 🟢" if "normal" in detected_status.lower() else ("ATTENTION 🟡" if "recovering" in detected_status.lower() else "CRITICAL 🔴")

        st.session_state.latest_sensor_machine_result = {
            "status": detected_status,
            "confidence": confidence,
            "health": health
        }

        st.session_state.sensor_machine_prediction_history.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Detected Status": detected_status,
            "Confidence %": round(confidence, 1),
            "Machine Health": health
        })

    if st.session_state.latest_sensor_machine_result is not None:
        res = st.session_state.latest_sensor_machine_result
        col1, col2, col3 = st.columns(3)
        col1.metric("Detected Machine Status", res["status"])
        col2.metric("Machine Health", res["health"])
        col3.metric("Prediction Confidence", f"{res['confidence']:.1f}%")

        sensor_prob = 0.90 if "broken" in res["status"].lower() else (0.45 if "recovering" in res["status"].lower() else 0.05)
        sensor_sample_telemetry = {col: round(float(val), 2) for col, val in zip(sensor_machine_features[:8], edited_sensor_table["Current Reading"].tolist()[:8])}

        HumanoidAssistant.render_humanoid_section(
            machine_name="Sensor Machine",
            telemetry=sensor_sample_telemetry,
            is_failure=("broken" in res["status"].lower()),
            failure_prob=sensor_prob,
            extra_metrics={"status": res["status"]},
            key_suffix="sensor"
        )

    if st.session_state.sensor_machine_prediction_history:
        st.divider()
        st.subheader("Sensor Analysis History")
        st.dataframe(pd.DataFrame(st.session_state.sensor_machine_prediction_history), use_container_width=True, hide_index=True)


# ==================================================
# 5. INDUSTRIAL PUMP WORKSPACE
# ==================================================

elif selected_machine == "Industrial Pump":
    st.markdown('<div class="section-title">🌊 Industrial Pump Cavitation & Vibration Analysis</div>', unsafe_allow_html=True)
    st.write("Hydraulic performance monitoring predicting cavitation risk, seal failure, and ISO 10816 bearing vibration.")

    if pump_model is None or not pump_features:
        st.error("Industrial Pump model is offline. Train models using `python train_all_models.py`.")
        st.stop()

    preset_cols = st.columns(3)
    p_norm = preset_cols[0].button("🟢 Normal Pumping Preset", key="pmp_norm", use_container_width=True)
    p_cav = preset_cols[1].button("⚠️ Severe Cavitation Anomaly", key="pmp_cav", use_container_width=True)
    p_vibe = preset_cols[2].button("🔴 ISO Bearing Vibration Anomaly", key="pmp_vibe", use_container_width=True)

    def_flow = 250.0 if not (p_cav or p_vibe) else (180.0 if p_cav else 220.0)
    def_suct = 2.5 if not (p_cav or p_vibe) else (0.9 if p_cav else 2.2)
    def_disc = 12.0 if not (p_cav or p_vibe) else (19.5 if p_vibe else 11.5)
    def_vibe = 2.0 if not (p_cav or p_vibe) else (8.4 if p_vibe else 4.8)
    def_temp = 55.0 if not (p_cav or p_vibe) else (89.0 if p_vibe else 60.0)
    def_cav = 2.0 if not (p_cav or p_vibe) else (0.6 if p_cav else 1.8)

    c1, c2 = st.columns(2)
    with c1:
        pump_flow = st.number_input("Flow Rate [L/min]", min_value=10.0, max_value=1000.0, value=float(def_flow))
        pump_suct = st.number_input("Suction Pressure [bar]", min_value=0.1, max_value=25.0, value=float(def_suct))
        pump_disc = st.number_input("Discharge Pressure [bar]", min_value=1.0, max_value=100.0, value=float(def_disc))
        pump_cav_idx = st.number_input("Cavitation Index", min_value=0.1, max_value=10.0, value=float(def_cav))
    with c2:
        pump_vibe_val = st.number_input("Vibration RMS [mm/s]", min_value=0.1, max_value=50.0, value=float(def_vibe))
        pump_bearing_temp = st.number_input("Bearing Temperature [°C]", min_value=0.0, max_value=200.0, value=float(def_temp))
        pump_power = st.number_input("Motor Power [kW]", min_value=1.0, max_value=1000.0, value=60.0)
        pump_fluid_temp = st.number_input("Fluid Temperature [°C]", min_value=-10.0, max_value=150.0, value=40.0)

    st.divider()
    if st.button("Run Industrial Pump Assessment", type="primary", use_container_width=True, key="btn_run_pump"):
        input_data = pd.DataFrame([[
            pump_flow, pump_suct, pump_disc, pump_vibe_val,
            pump_bearing_temp, pump_power, pump_fluid_temp, pump_cav_idx
        ]], columns=pump_features)

        prediction = pump_model.predict(input_data)[0]
        failure_prob = float(pump_model.predict_proba(input_data)[0][1]) if hasattr(pump_model, "predict_proba") else (1.0 if prediction == 1 else 0.0)
        fail_pct = failure_prob * 100.0
        health = "GOOD 🟢" if fail_pct < 35.0 else ("ATTENTION 🟡" if fail_pct < 70.0 else "CRITICAL 🔴")

        st.session_state.latest_pump_result = {
            "prediction": int(prediction),
            "failure_probability": fail_pct,
            "health": health
        }

        st.session_state.pump_prediction_history.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Flow Rate": pump_flow,
            "Discharge Pressure": pump_disc,
            "Vibration RMS": pump_vibe_val,
            "Bearing Temp": pump_bearing_temp,
            "Failure Probability %": round(fail_pct, 2),
            "Health": health
        })

    if st.session_state.latest_pump_result is not None:
        res = st.session_state.latest_pump_result
        col1, col2, col3 = st.columns(3)
        col1.metric("Failure Prediction", "FAILURE DETECTED ⚠️" if res["prediction"] == 1 else "NO FAILURE DETECTED ✅")
        col2.metric("Machine Health", res["health"])
        col3.metric("Failure Probability", f"{res['failure_probability']:.2f}%")
        st.progress(int(min(max(res["failure_probability"], 0.0), 100.0)))

        pump_telemetry = {
            "Flow_Rate": pump_flow,
            "Suction_Pressure": pump_suct,
            "Discharge_Pressure": pump_disc,
            "Vibration_RMS": pump_vibe_val,
            "Bearing_Temperature": pump_bearing_temp,
            "Motor_Power": pump_power,
            "Fluid_Temperature": pump_fluid_temp,
            "Cavitation_Index": pump_cav_idx
        }
        HumanoidAssistant.render_humanoid_section(
            machine_name="Industrial Pump",
            telemetry=pump_telemetry,
            is_failure=(res["prediction"] == 1),
            failure_prob=res["failure_probability"] / 100.0,
            key_suffix="pump"
        )

    if st.session_state.pump_prediction_history:
        st.divider()
        st.subheader("Pump Analysis History")
        st.dataframe(pd.DataFrame(st.session_state.pump_prediction_history), use_container_width=True, hide_index=True)


# ==================================================
# 6. CNC MACHINE WORKSPACE
# ==================================================

elif selected_machine == "CNC Machine":
    st.markdown('<div class="section-title">🔧 CNC Machining Center Diagnostics</div>', unsafe_allow_html=True)
    st.write("Machining dynamics analysis monitoring dynamic spindle chatter, cutting force, tool wear, and servo feed error.")

    if cnc_model is None or not cnc_features:
        st.error("CNC Machine model is offline. Train models using `python train_all_models.py`.")
        st.stop()

    preset_cols = st.columns(3)
    p_norm = preset_cols[0].button("🟢 Normal Machining Preset", key="cnc_norm", use_container_width=True)
    p_chat = preset_cols[1].button("⚠️ Regenerative Chatter Anomaly", key="cnc_chat", use_container_width=True)
    p_wear = preset_cols[2].button("🔴 Critical Tool Wear & Overload", key="cnc_wear", use_container_width=True)

    def_spindle = 6000.0 if not (p_chat or p_wear) else (7800.0 if p_chat else 5500.0)
    def_force = 850.0 if not (p_chat or p_wear) else (1650.0 if p_wear else 950.0)
    def_wear = 45.0 if not (p_chat or p_wear) else (210.0 if p_wear else 50.0)
    def_vibe = 1.8 if not (p_chat or p_wear) else (7.2 if p_chat else 3.5)
    def_cool = 22.0 if not (p_chat or p_wear) else (8.0 if p_wear else 20.0)

    c1, c2 = st.columns(2)
    with c1:
        cnc_spindle = st.number_input("Spindle Speed [RPM]", min_value=100.0, max_value=30000.0, value=float(def_spindle))
        cnc_feed = st.number_input("Feed Rate [mm/min]", min_value=10.0, max_value=5000.0, value=800.0)
        cnc_force = st.number_input("Cutting Force [N]", min_value=10.0, max_value=10000.0, value=float(def_force))
        cnc_wear_idx = st.number_input("Tool Wear Index [min]", min_value=0.0, max_value=500.0, value=float(def_wear))
    with c2:
        cnc_vibe_val = st.number_input("Spindle Vibration [mm/s]", min_value=0.1, max_value=50.0, value=float(def_vibe))
        cnc_err = st.number_input("Axis Feed Error [µm]", min_value=0.0, max_value=100.0, value=4.0)
        cnc_coolant = st.number_input("Coolant Pressure [bar]", min_value=0.0, max_value=100.0, value=float(def_cool))
        cnc_temp = st.number_input("Motor Temperature [°C]", min_value=0.0, max_value=150.0, value=48.0)

    st.divider()
    if st.button("Run CNC Machining Assessment", type="primary", use_container_width=True, key="btn_run_cnc"):
        input_data = pd.DataFrame([[
            cnc_spindle, cnc_feed, cnc_force, cnc_wear_idx,
            cnc_vibe_val, cnc_err, cnc_coolant, cnc_temp
        ]], columns=cnc_features)

        prediction = cnc_model.predict(input_data)[0]
        failure_prob = float(cnc_model.predict_proba(input_data)[0][1]) if hasattr(cnc_model, "predict_proba") else (1.0 if prediction == 1 else 0.0)
        fail_pct = failure_prob * 100.0
        health = "GOOD 🟢" if fail_pct < 35.0 else ("ATTENTION 🟡" if fail_pct < 70.0 else "CRITICAL 🔴")

        st.session_state.latest_cnc_result = {
            "prediction": int(prediction),
            "failure_probability": fail_pct,
            "health": health
        }

        st.session_state.cnc_prediction_history.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Spindle Speed": cnc_spindle,
            "Cutting Force": cnc_force,
            "Tool Wear": cnc_wear_idx,
            "Vibration": cnc_vibe_val,
            "Failure Probability %": round(fail_pct, 2),
            "Health": health
        })

    if st.session_state.latest_cnc_result is not None:
        res = st.session_state.latest_cnc_result
        col1, col2, col3 = st.columns(3)
        col1.metric("Failure Prediction", "FAILURE DETECTED ⚠️" if res["prediction"] == 1 else "NO FAILURE DETECTED ✅")
        col2.metric("Machine Health", res["health"])
        col3.metric("Failure Probability", f"{res['failure_probability']:.2f}%")
        st.progress(int(min(max(res["failure_probability"], 0.0), 100.0)))

        cnc_telemetry = {
            "Spindle_Speed": cnc_spindle,
            "Feed_Rate": cnc_feed,
            "Cutting_Force": cnc_force,
            "Tool_Wear_Index": cnc_wear_idx,
            "Spindle_Vibration": cnc_vibe_val,
            "Axis_Feed_Error": cnc_err,
            "Coolant_Pressure": cnc_coolant,
            "Motor_Temperature": cnc_temp
        }
        HumanoidAssistant.render_humanoid_section(
            machine_name="CNC Machine",
            telemetry=cnc_telemetry,
            is_failure=(res["prediction"] == 1),
            failure_prob=res["failure_probability"] / 100.0,
            key_suffix="cnc"
        )

    if st.session_state.cnc_prediction_history:
        st.divider()
        st.subheader("CNC Analysis History")
        st.dataframe(pd.DataFrame(st.session_state.cnc_prediction_history), use_container_width=True, hide_index=True)


# ==================================================
# 7. CONVEYOR SYSTEM WORKSPACE
# ==================================================

elif selected_machine == "Conveyor System":
    st.markdown('<div class="section-title">📦 Industrial Conveyor System Diagnostics</div>', unsafe_allow_html=True)
    st.write("Material handling dynamics evaluating drive pulley belt slip, idler roll seizure, and tension overload.")

    if conveyor_model is None or not conveyor_features:
        st.error("Conveyor System model is offline. Train models using `python train_all_models.py`.")
        st.stop()

    preset_cols = st.columns(3)
    p_norm = preset_cols[0].button("🟢 Normal Transit Preset", key="cvr_norm", use_container_width=True)
    p_slip = preset_cols[1].button("⚠️ Drive Pulley Slip Anomaly", key="cvr_slip", use_container_width=True)
    p_seize = preset_cols[2].button("🔴 Idler Bearing Seizure Anomaly", key="cvr_seize", use_container_width=True)

    def_speed = 2.2 if not (p_slip or p_seize) else (1.8 if p_slip else 2.1)
    def_tension = 22.0 if not (p_slip or p_seize) else (38.5 if p_slip else 24.0)
    def_curr = 45.0 if not (p_slip or p_seize) else (85.0 if p_slip else 52.0)
    def_temp = 48.0 if not (p_slip or p_seize) else (86.0 if p_seize else 50.0)
    def_vibe = 1.8 if not (p_slip or p_seize) else (7.1 if p_seize else 2.0)
    def_slip = 1.2 if not (p_slip or p_seize) else (7.8 if p_slip else 1.5)

    c1, c2 = st.columns(2)
    with c1:
        cvr_speed = st.number_input("Belt Speed [m/s]", min_value=0.1, max_value=10.0, value=float(def_speed))
        cvr_tension = st.number_input("Belt Tension [kN]", min_value=1.0, max_value=100.0, value=float(def_tension))
        cvr_current = st.number_input("Motor Current [A]", min_value=1.0, max_value=500.0, value=float(def_curr))
        cvr_bearing_temp = st.number_input("Roller Bearing Temp [°C]", min_value=0.0, max_value=150.0, value=float(def_temp))
    with c2:
        cvr_vibe_val = st.number_input("Idler Vibration [mm/s]", min_value=0.1, max_value=30.0, value=float(def_vibe))
        cvr_slip_pct = st.number_input("Belt Slip Percentage [%]", min_value=0.0, max_value=50.0, value=float(def_slip))
        cvr_load = st.number_input("Load Weight [t/h]", min_value=0.0, max_value=5000.0, value=500.0)
        cvr_amb = st.number_input("Ambient Temperature [°C]", min_value=-20.0, max_value=60.0, value=28.0)

    st.divider()
    if st.button("Run Conveyor System Assessment", type="primary", use_container_width=True, key="btn_run_cvr"):
        input_data = pd.DataFrame([[
            cvr_speed, cvr_tension, cvr_current, cvr_bearing_temp,
            cvr_vibe_val, cvr_slip_pct, cvr_load, cvr_amb
        ]], columns=conveyor_features)

        prediction = conveyor_model.predict(input_data)[0]
        failure_prob = float(conveyor_model.predict_proba(input_data)[0][1]) if hasattr(conveyor_model, "predict_proba") else (1.0 if prediction == 1 else 0.0)
        fail_pct = failure_prob * 100.0
        health = "GOOD 🟢" if fail_pct < 35.0 else ("ATTENTION 🟡" if fail_pct < 70.0 else "CRITICAL 🔴")

        st.session_state.latest_conveyor_result = {
            "prediction": int(prediction),
            "failure_probability": fail_pct,
            "health": health
        }

        st.session_state.conveyor_prediction_history.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Belt Speed": cvr_speed,
            "Tension": cvr_tension,
            "Motor Current": cvr_current,
            "Roller Temp": cvr_bearing_temp,
            "Failure Probability %": round(fail_pct, 2),
            "Health": health
        })

    if st.session_state.latest_conveyor_result is not None:
        res = st.session_state.latest_conveyor_result
        col1, col2, col3 = st.columns(3)
        col1.metric("Failure Prediction", "FAILURE DETECTED ⚠️" if res["prediction"] == 1 else "NO FAILURE DETECTED ✅")
        col2.metric("Machine Health", res["health"])
        col3.metric("Failure Probability", f"{res['failure_probability']:.2f}%")
        st.progress(int(min(max(res["failure_probability"], 0.0), 100.0)))

        conveyor_telemetry = {
            "Belt_Speed": cvr_speed,
            "Belt_Tension": cvr_tension,
            "Motor_Current": cvr_current,
            "Roller_Bearing_Temperature": cvr_bearing_temp,
            "Idler_Vibration": cvr_vibe_val,
            "Belt_Slip_Percentage": cvr_slip_pct,
            "Load_Weight": cvr_load,
            "Ambient_Temperature": cvr_amb
        }
        HumanoidAssistant.render_humanoid_section(
            machine_name="Conveyor System",
            telemetry=conveyor_telemetry,
            is_failure=(res["prediction"] == 1),
            failure_prob=res["failure_probability"] / 100.0,
            key_suffix="conveyor"
        )

    if st.session_state.conveyor_prediction_history:
        st.divider()
        st.subheader("Conveyor Analysis History")
        st.dataframe(pd.DataFrame(st.session_state.conveyor_prediction_history), use_container_width=True, hide_index=True)


# ==================================================
# FOOTER
# ==================================================

st.divider()
st.caption(
    "Predictive Maintenance AI · Dr. Nova Humanoid Reliability Platform | "
    "Autonomous Diagnostics | Root Cause Analysis | Voice Dispatch | ISO-14224 Work Orders"
)