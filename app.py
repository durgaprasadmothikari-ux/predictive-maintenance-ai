import streamlit as st
import pandas as pd
import joblib
from datetime import datetime


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="PM",
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
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 1480px;
    }

    .main-title {
        text-align: center;
        font-size: 44px;
        font-weight: 760;
        line-height: 1.08;
        margin-bottom: 8px;
        letter-spacing: -1.2px;
    }

    .subtitle {
        text-align: center;
        font-size: 16px;
        color: var(--muted);
        margin-top: 0;
        margin-bottom: 30px;
        letter-spacing: 0.2px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 700;
        margin-top: 8px;
        margin-bottom: 8px;
        letter-spacing: -0.35px;
    }

    .page-kicker {
        color: #94A3B8;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.11em;
        margin-bottom: 7px;
        font-weight: 700;
    }

    .dashboard-card, .module-card {
        padding: 22px;
        border-radius: 16px;
        border: 1px solid var(--border);
        background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.02));
        margin-bottom: 15px;
        min-height: 180px;
    }

    .module-card {
        min-height: 190px;
    }

    .hero-panel {
        padding: 30px;
        border-radius: 20px;
        border: 1px solid rgba(96,165,250,0.22);
        background: linear-gradient(135deg, rgba(30,58,95,0.72), rgba(30,41,59,0.65));
        margin-bottom: 24px;
    }

    .hero-title {
        font-size: 30px;
        font-weight: 720;
        margin-bottom: 7px;
        letter-spacing: -0.4px;
    }

    .hero-copy, .small-label {
        color: var(--muted);
        font-size: 15px;
        line-height: 1.55;
    }

    .status-good, .status-warning, .status-critical {
        padding: 18px 20px;
        border-radius: 14px;
        margin-top: 12px;
        line-height: 1.55;
    }

    .status-good {
        background: rgba(34,197,94,0.10);
        border: 1px solid rgba(34,197,94,0.24);
        border-left: 4px solid var(--green);
    }

    .status-warning {
        background: rgba(245,158,11,0.10);
        border: 1px solid rgba(245,158,11,0.24);
        border-left: 4px solid var(--amber);
    }

    .status-critical {
        background: rgba(239,68,68,0.10);
        border: 1px solid rgba(239,68,68,0.24);
        border-left: 4px solid var(--red);
    }

    .insight-box {
        padding: 18px 20px;
        border-radius: 14px;
        border: 1px solid var(--border);
        background: var(--panel-soft);
        margin: 8px 0 12px 0;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem;
    }

    div[data-testid="stMetric"] {
        padding: 4px 0;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    .stButton > button {
        border-radius: 10px;
        min-height: 44px;
        font-weight: 650;
    }

    .stDownloadButton > button {
        border-radius: 10px;
        min-height: 42px;
        font-weight: 650;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# SESSION STATE
# ==================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


if "ai4i_prediction_history" not in st.session_state:
    st.session_state.ai4i_prediction_history = []


if "latest_motor_result" not in st.session_state:
    st.session_state.latest_motor_result = None


if "latest_ai4i_result" not in st.session_state:
    st.session_state.latest_ai4i_result = None


if "sensor_machine_prediction_history" not in st.session_state:
    st.session_state.sensor_machine_prediction_history = []


if "latest_sensor_machine_result" not in st.session_state:
    st.session_state.latest_sensor_machine_result = None


# ==================================================
# LOAD ELECTRIC MOTOR MODEL
# ==================================================

@st.cache_resource
def load_electric_motor_model():

    try:

        model = joblib.load(
            "models/electric_motor_rul_model.pkl"
        )

        return model

    except FileNotFoundError:

        return None


    except Exception:

        return None


# ==================================================
# LOAD AI4I MODEL
# ==================================================

@st.cache_resource
def load_ai4i_model():

    try:

        model = joblib.load(
            "models/predictive_maintenance_model.pkl"
        )

        return model

    except FileNotFoundError:

        return None


    except Exception:

        return None


# ==================================================
# LOAD SENSOR MACHINE MODEL
# ==================================================

@st.cache_resource
def load_sensor_machine_assets():

    try:

        sensor_model = joblib.load(
            "models/sensor_machine_model.pkl"
        )

        sensor_features = joblib.load(
            "models/sensor_machine_features.pkl"
        )

        sensor_label_encoder = joblib.load(
            "models/sensor_machine_label_encoder.pkl"
        )

        return sensor_model, list(sensor_features), sensor_label_encoder

    except Exception:

        return None, [], None


@st.cache_data
def load_sensor_baseline_values(features):

    """Load one real dataset row as sensible default sensor readings."""

    defaults = {feature: 0.0 for feature in features}

    try:

        sample = pd.read_csv(
            "data/sensor.csv",
            nrows=1
        )

        for feature in features:

            if feature in sample.columns:

                value = sample.iloc[0][feature]

                try:

                    defaults[feature] = float(value)

                except Exception:

                    defaults[feature] = 0.0

    except Exception:

        pass

    return defaults


# ==================================================
# LOAD MODELS
# ==================================================

electric_motor_model = load_electric_motor_model()

ai4i_model = load_ai4i_model()

sensor_machine_model, sensor_machine_features, sensor_machine_label_encoder = (
    load_sensor_machine_assets()
)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("Predictive Maintenance")

    st.caption(
        "Machine health monitoring and maintenance intelligence"
    )

    st.divider()


    st.subheader("Model availability")


    if electric_motor_model is not None:

        st.success(
            "Electric Motor Model — ONLINE"
        )

    else:

        st.error(
            "Electric Motor Model — OFFLINE"
        )


    if ai4i_model is not None:

        st.success(
            "AI4I Machine Model — ONLINE"
        )

    else:

        st.error(
            "AI4I Machine Model — OFFLINE"
        )


    if sensor_machine_model is not None and sensor_machine_features:

        st.success(
            "Sensor Machine Model — ONLINE"
        )

    else:

        st.error(
            "Sensor Machine Model — OFFLINE"
        )


    st.divider()


    st.subheader("Session overview")


    total_motor_predictions = len(
        st.session_state.prediction_history
    )


    total_ai4i_predictions = len(
        st.session_state.ai4i_prediction_history
    )


    total_sensor_machine_predictions = len(
        st.session_state.sensor_machine_prediction_history
    )


    total_predictions = (
        total_motor_predictions
        +
        total_ai4i_predictions
        +
        total_sensor_machine_predictions
    )


    st.metric(
        "Total Predictions",
        total_predictions
    )


    st.metric(
        "Electric Motor Predictions",
        total_motor_predictions
    )


    st.metric(
        "AI4I Predictions",
        total_ai4i_predictions
    )


    st.metric(
        "Sensor Machine Predictions",
        total_sensor_machine_predictions
    )


    st.divider()


    st.caption(
        "Predictive maintenance workspace"
    )


# ==================================================
# INPUT VALIDATION
# ==================================================

def validate_range(label, value, minimum, maximum):
    if value < minimum or value > maximum:
        st.error(f"{label} must be between {minimum} and {maximum}. Current value: {value}")
        return False
    return True


# ==================================================
# APPLICATION HEADER
# ==================================================

st.markdown(
    """
    <div class="main-title">
        Predictive Maintenance
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="subtitle">
        Machine health monitoring • Failure prediction • Remaining useful life • Maintenance decisions
    </div>
    """,
    unsafe_allow_html=True
)


st.divider()


# ==================================================
# MACHINE SELECTION
# ==================================================

st.subheader(
    "Select a workspace"
)


selected_machine = st.selectbox(
    "Choose a machine or return to the command center",
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
# PROFESSIONAL DASHBOARD / HOME
# ==================================================

if selected_machine == "Command Center":

    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-title">Predictive Maintenance Command Center</div>
            <div class="small-label">A single place to review machine health, recent analyses and the next maintenance priorities.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    total_motor = len(st.session_state.prediction_history)
    total_ai4i = len(st.session_state.ai4i_prediction_history)
    total_sensor = len(st.session_state.sensor_machine_prediction_history)
    total_predictions = total_motor + total_ai4i + total_sensor
    online_models = (
        int(electric_motor_model is not None)
        + int(ai4i_model is not None)
        + int(sensor_machine_model is not None and bool(sensor_machine_features))
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Predictions", total_predictions)
    m2.metric("AI Models Online", f"{online_models}/3")
    m3.metric("AI4I Analyses", total_ai4i)
    m4.metric("Motor + Sensor Analyses", total_motor + total_sensor)

    st.divider()
    st.subheader("Latest machine intelligence")
    left, center, right = st.columns(3)

    with left:
        st.markdown('<div class="module-card">', unsafe_allow_html=True)
        st.markdown("### AI4I Industrial Machine")
        if st.session_state.latest_ai4i_result is not None:
            r = st.session_state.latest_ai4i_result
            st.metric("Machine Health", r.get("health", "Not available"))
            st.metric("Failure Probability", f'{r.get("failure_probability", 0):.2f}%')
            st.caption("Latest AI-based failure analysis")
        else:
            st.info("No analysis has been run for this machine during the current session.")
        st.markdown('</div>', unsafe_allow_html=True)

    with center:
        st.markdown('<div class="module-card">', unsafe_allow_html=True)
        st.markdown("### Electric Motor")
        if st.session_state.latest_motor_result is not None:
            r = st.session_state.latest_motor_result
            st.metric("Machine Health", r.get("health", "Not available"))
            st.metric("Predicted RUL", f'{r.get("prediction", 0):.2f} Hours')
            st.caption("Latest Remaining Useful Life analysis")
        else:
            st.info("No analysis has been run for this machine during the current session.")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="module-card">', unsafe_allow_html=True)
        st.markdown("### Sensor Machine")
        if st.session_state.latest_sensor_machine_result is not None:
            r = st.session_state.latest_sensor_machine_result
            st.metric("Detected Status", r.get("status", "Not available"))
            st.metric("Prediction Confidence", f'{r.get("confidence", 0):.2f}%')
            st.caption("Latest sensor-pattern classification")
        else:
            st.info("No analysis has been run for this machine during the current session.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("Available machine modules")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.success("🏭 AI4I Industrial Machine\n\nAI-enabled failure prediction and maintenance recommendation.")
    with c2:
        st.success("⚙️ Electric Motor\n\nAI-enabled Remaining Useful Life and risk analysis.")
    with c3:
        st.success("📡 Sensor Machine\n\nAI-enabled machine-status classification from multi-sensor readings.")
    with c4:
        st.info("🚧 Pump / CNC / Conveyor\n\nReserved for future machine-specific AI models.")

    st.divider()
    st.subheader("Recent activity")
    if total_predictions == 0:
        st.info("Run an analysis from any available machine module to begin building the session history.")
    else:
        activity = []
        for row in st.session_state.ai4i_prediction_history[-10:]:
            activity.append({"Time": row.get("Time"), "Machine": "AI4I Industrial Machine", "Result": row.get("Machine Health"), "Key Metric": f'{row.get("Failure Probability %", 0):.2f}% failure risk'})
        for row in st.session_state.prediction_history[-10:]:
            activity.append({"Time": row.get("Time"), "Machine": "Electric Motor", "Result": row.get("Machine Health"), "Key Metric": f'{row.get("Predicted RUL Hours", 0):.2f} h RUL'})
        for row in st.session_state.sensor_machine_prediction_history[-10:]:
            activity.append({"Time": row.get("Time"), "Machine": "Sensor Machine", "Result": row.get("Detected Machine Status"), "Key Metric": f'{row.get("Prediction Confidence %", 0):.2f}% confidence'})
        activity_df = pd.DataFrame(activity)
        if not activity_df.empty:
            activity_df = activity_df.sort_values("Time", ascending=False)
            st.dataframe(activity_df, use_container_width=True, hide_index=True)


# ==================================================
# AI4I INDUSTRIAL MACHINE
# ==================================================

elif selected_machine == "AI4I Industrial Machine":


    # ==============================================
    # PAGE HEADER
    # ==============================================

    st.markdown(
        '<div class="section-title">'
        'AI4I Industrial Machine'
        '</div>',
        unsafe_allow_html=True
    )


    st.write(
        "Enter the current machine operating parameters "
        "to analyze machine failure probability and "
        "receive AI-based maintenance recommendations."
    )


    # ==============================================
    # MODEL CHECK
    # ==============================================

    if ai4i_model is None:


        st.error(
            """
### ⚠️ AI4I Machine Model Not Available

The application could not load:

`models/predictive_maintenance_model.pkl`

Please make sure the trained AI4I model exists
inside your `models` folder.
"""
        )


    else:


        # ==========================================
        # MACHINE INPUTS
        # ==========================================

        st.subheader(
            "Current operating conditions"
        )


        col1, col2 = st.columns(2)


        # ==========================================
        # COLUMN 1
        # ==========================================

        with col1:


            machine_type = st.selectbox(
                "Machine Type",
                [
                    "L",
                    "M",
                    "H"
                ],
                key="ai4i_machine_type"
            )


            air_temperature = st.number_input(
                "Air Temperature (K)",
                min_value=0.0,
                value=300.0,
                step=0.1,
                key="ai4i_air_temperature"
            )


            process_temperature = st.number_input(
                "Process Temperature (K)",
                min_value=0.0,
                value=310.0,
                step=0.1,
                key="ai4i_process_temperature"
            )


        # ==========================================
        # COLUMN 2
        # ==========================================

        with col2:


            rotational_speed = st.number_input(
                "Rotational Speed (RPM)",
                min_value=0.0,
                value=1500.0,
                step=1.0,
                key="ai4i_rotational_speed"
            )


            torque = st.number_input(
                "Torque (Nm)",
                min_value=0.0,
                value=40.0,
                step=0.1,
                key="ai4i_torque"
            )


            tool_wear = st.number_input(
                "Tool Wear (minutes)",
                min_value=0.0,
                value=100.0,
                step=1.0,
                key="ai4i_tool_wear"
            )


        # ==========================================
        # CURRENT MACHINE PARAMETERS
        # ==========================================

        st.divider()


        st.subheader(
            "Current machine snapshot"
        )


        ai4i_sensor_data = pd.DataFrame(
            {
                "Parameter": [
                    "Air Temperature",
                    "Process Temperature",
                    "Rotational Speed",
                    "Torque",
                    "Tool Wear"
                ],

                "Value": [
                    air_temperature,
                    process_temperature,
                    rotational_speed,
                    torque,
                    tool_wear
                ]
            }
        )


        chart_col1, chart_col2 = st.columns(
            [2, 1]
        )


        with chart_col1:


            st.bar_chart(
                ai4i_sensor_data.set_index(
                    "Parameter"
                )
            )


        with chart_col2:


            st.dataframe(
                ai4i_sensor_data,
                use_container_width=True,
                hide_index=True
            )


        st.divider()


        # ==========================================
        # PREDICT BUTTON
        # ==========================================

        predict_ai4i = st.button(
            "Run machine assessment",
            use_container_width=True,
            type="primary",
            key="predict_ai4i_button"
        )


        # ==========================================
        # MAKE AI4I PREDICTION
        # ==========================================

        if predict_ai4i:

            valid_inputs = all([
                validate_range("Air Temperature (K)", air_temperature, 250.0, 400.0),
                validate_range("Process Temperature (K)", process_temperature, 250.0, 450.0),
                validate_range("Rotational Speed (RPM)", rotational_speed, 100.0, 20000.0),
                validate_range("Torque (Nm)", torque, 0.1, 500.0),
                validate_range("Tool Wear (minutes)", tool_wear, 0.0, 10000.0)
            ])

            if not valid_inputs:
                st.stop()

            input_data = pd.DataFrame(
                {
                    "Air temperature [K]": [
                        air_temperature
                    ],

                    "Process temperature [K]": [
                        process_temperature
                    ],

                    "Rotational speed [rpm]": [
                        rotational_speed
                    ],

                    "Torque [Nm]": [
                        torque
                    ],

                    "Tool wear [min]": [
                        tool_wear
                    ],

                    "Type_L": [
                        1 if machine_type == "L"
                        else 0
                    ],

                    "Type_M": [
                        1 if machine_type == "M"
                        else 0
                    ]
                }
            )


            try:


                prediction = ai4i_model.predict(
                    input_data
                )[0]


                # ==================================
                # FAILURE PROBABILITY
                # ==================================

                if hasattr(
                    ai4i_model,
                    "predict_proba"
                ):


                    probability = (
                        ai4i_model.predict_proba(
                            input_data
                        )[0]
                    )


                    failure_probability = (
                        float(probability[1])
                        *
                        100
                    )


                else:


                    if prediction == 1:

                        failure_probability = 100.0


                    else:

                        failure_probability = 0.0


                # ==================================
                # RISK LEVEL
                # ==================================

                if failure_probability < 30:


                    health = "GOOD 🟢"

                    risk_level = (
                        "LOW FAILURE RISK 🟢"
                    )


                elif failure_probability < 60:


                    health = "ATTENTION 🟡"

                    risk_level = (
                        "MEDIUM FAILURE RISK 🟡"
                    )


                else:


                    health = "CRITICAL 🔴"

                    risk_level = (
                        "HIGH FAILURE RISK 🔴"
                    )


                # ==================================
                # FAILURE REASONS
                # ==================================

                possible_reasons = []


                if air_temperature > 310:

                    possible_reasons.append(
                        "High air temperature may increase "
                        "thermal stress on the machine."
                    )


                if process_temperature > 320:

                    possible_reasons.append(
                        "High process temperature may indicate "
                        "overheating or excessive operating stress."
                    )


                if rotational_speed > 2500:

                    possible_reasons.append(
                        "High rotational speed may increase "
                        "mechanical wear and vibration."
                    )


                if rotational_speed < 1000:

                    possible_reasons.append(
                        "Low rotational speed may indicate "
                        "reduced performance or mechanical resistance."
                    )


                if torque > 60:

                    possible_reasons.append(
                        "High torque may indicate excessive "
                        "mechanical load."
                    )


                if tool_wear > 180:

                    possible_reasons.append(
                        "High tool wear may increase the "
                        "probability of machine failure."
                    )


                if prediction == 1:

                    possible_reasons.append(
                        "The trained AI model detected an operating "
                        "pattern associated with machine failure."
                    )


                if not possible_reasons:

                    possible_reasons.append(
                        "No major abnormal operating condition "
                        "was detected from the entered parameters."
                    )


                # ==================================
                # RECOMMENDATIONS
                # ==================================

                recommendations = []


                if air_temperature > 310:

                    recommendations.append(
                        "Inspect ventilation and improve cooling "
                        "around the machine."
                    )


                if process_temperature > 320:

                    recommendations.append(
                        "Inspect the process cooling system and "
                        "reduce excessive thermal load."
                    )


                if rotational_speed > 2500:

                    recommendations.append(
                        "Inspect bearings and rotating components "
                        "for excessive vibration and wear."
                    )


                if rotational_speed < 1000:

                    recommendations.append(
                        "Inspect mechanical components for "
                        "resistance or reduced operating performance."
                    )


                if torque > 60:

                    recommendations.append(
                        "Reduce excessive mechanical load and "
                        "inspect the driven components."
                    )


                if tool_wear > 180:

                    recommendations.append(
                        "Inspect and replace the worn tool or "
                        "component if necessary."
                    )


                if health == "GOOD 🟢":

                    recommendations.append(
                        "Continue normal operation and scheduled "
                        "preventive maintenance."
                    )


                elif health == "ATTENTION 🟡":

                    recommendations.append(
                        "Schedule a maintenance inspection before "
                        "the failure risk increases."
                    )


                else:

                    recommendations.append(
                        "Perform an immediate inspection and "
                        "maintenance before continuing heavy operation."
                    )


                # ==================================
                # SAVE RESULT
                # ==================================

                result = {

                    "prediction": prediction,

                    "failure_probability": (
                        failure_probability
                    ),

                    "health": health,

                    "risk_level": risk_level,

                    "possible_reasons": (
                        possible_reasons
                    ),

                    "recommendations": (
                        recommendations
                    )
                }


                st.session_state.latest_ai4i_result = (
                    result
                )


                # ==================================
                # SAVE HISTORY
                # ==================================

                ai4i_record = {

                    "Time": (
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    ),

                    "Machine Type": (
                        machine_type
                    ),

                    "Air Temperature": (
                        air_temperature
                    ),

                    "Process Temperature": (
                        process_temperature
                    ),

                    "Rotational Speed": (
                        rotational_speed
                    ),

                    "Torque": (
                        torque
                    ),

                    "Tool Wear": (
                        tool_wear
                    ),

                    "Failure Probability %": (
                        round(
                            failure_probability,
                            2
                        )
                    ),

                    "Machine Health": (
                        health
                    )
                }


                st.session_state.ai4i_prediction_history.append(
                    ai4i_record
                )


            except Exception as error:


                st.error(
                    f"Prediction error: {error}"
                )


        # ==========================================
        # DISPLAY LATEST AI4I RESULT
        # ==========================================

        if (
            st.session_state.latest_ai4i_result
            is not None
        ):


            result = (
                st.session_state.latest_ai4i_result
            )


            prediction = result["prediction"]

            failure_probability = (
                result["failure_probability"]
            )

            health = result["health"]

            risk_level = result["risk_level"]

            possible_reasons = (
                result["possible_reasons"]
            )

            recommendations = (
                result["recommendations"]
            )


            st.divider()


            st.subheader(
                "Machine assessment"
            )


            result_col1, result_col2, result_col3 = (
                st.columns(3)
            )


            with result_col1:


                if prediction == 1:

                    st.metric(
                        "Failure Prediction",
                        "FAILURE DETECTED ⚠️"
                    )


                else:

                    st.metric(
                        "Failure Prediction",
                        "NO FAILURE DETECTED ✅"
                    )


            with result_col2:


                st.metric(
                    "Machine Health",
                    health
                )


            with result_col3:


                st.metric(
                    "Failure Probability",
                    f"{failure_probability:.2f}%"
                )


            st.subheader(
                "Current operating risk"
            )


            risk_value = int(
                max(
                    0,
                    min(
                        100,
                        failure_probability
                    )
                )
            )


            st.progress(
                risk_value
            )


            st.write(
                f"### {risk_level}"
            )


            # ======================================
            # STATUS BOX
            # ======================================

            if failure_probability < 30:


                st.markdown(
                    """
                    <div class="status-good">
                    🟢 <b>LOW FAILURE RISK</b><br>
                    The machine is currently operating
                    within a generally safe condition.
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            elif failure_probability < 60:


                st.markdown(
                    """
                    <div class="status-warning">
                    🟡 <b>ATTENTION REQUIRED</b><br>
                    Machine operating conditions should
                    be monitored closely.
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            else:


                st.markdown(
                    """
                    <div class="status-critical">
                    🔴 <b>HIGH FAILURE RISK</b><br>
                    Immediate inspection is recommended.
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # ======================================
            # REASONS
            # ======================================

            st.subheader(
                "What needs attention"
            )


            for reason in possible_reasons:

                st.write(
                    f"• {reason}"
                )


            # ======================================
            # RECOMMENDATIONS
            # ======================================

            st.subheader(
                "Recommended next steps"
            )


            for recommendation in recommendations:

                st.write(
                    f"• {recommendation}"
                )


        # ==========================================
        # AI4I HISTORY
        # ==========================================

        if len(
            st.session_state.ai4i_prediction_history
        ) > 0:


            st.divider()


            history_col1, history_col2 = (
                st.columns(
                    [4, 1]
                )
            )


            with history_col1:


                st.subheader(
                    "AI4I analysis history"
                )


            with history_col2:


                if st.button(
                    "🗑️ Clear AI4I History",
                    use_container_width=True,
                    key="clear_ai4i_history"
                ):


                    st.session_state.ai4i_prediction_history = []

                    st.session_state.latest_ai4i_result = None

                    st.rerun()


            ai4i_history_df = pd.DataFrame(
                st.session_state.ai4i_prediction_history
            )


            st.dataframe(
                ai4i_history_df,
                use_container_width=True,
                hide_index=True
            )


            st.subheader(
                "Failure probability trend"
            )


            failure_chart_data = (
                ai4i_history_df[
                    [
                        "Time",
                        "Failure Probability %"
                    ]
                ]
                .copy()
                .set_index(
                    "Time"
                )
            )


            st.line_chart(
                failure_chart_data,
                use_container_width=True
            )


            st.subheader(
                "Export analysis history"
            )


            ai4i_csv_data = (
                ai4i_history_df.to_csv(
                    index=False
                ).encode(
                    "utf-8"
                )
            )


            st.download_button(
                label=(
                    "Download AI4I history as CSV"
                ),
                data=ai4i_csv_data,
                file_name=(
                    "ai4i_machine_prediction_history.csv"
                ),
                mime="text/csv",
                use_container_width=True,
                key="download_ai4i_csv"
            )


# ==================================================
# SENSOR MACHINE
# ==================================================

elif selected_machine == "Sensor Machine":

    st.markdown(
        '<div class="section-title">'
        'Sensor Machine Intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "This model reads the combined pattern of multiple machine sensors and "
        "classifies the current machine status. The table below is intentionally "
        "editable, so you can type values directly instead of using plus/minus controls."
    )

    if sensor_machine_model is None or not sensor_machine_features:

        st.error(
            """
### ⚠️ Sensor Machine Model Not Available

The application could not load all Sensor Machine model files.

Please confirm that these three files are present inside your `models` folder:

- `sensor_machine_model.pkl`
- `sensor_machine_features.pkl`
- `sensor_machine_label_encoder.pkl`
"""
        )

    else:

        baseline_values = load_sensor_baseline_values(
            tuple(sensor_machine_features)
        )

        st.subheader("Current sensor readings")
        st.caption(
            "The initial values come from one real row in your `data/sensor.csv` file. You can directly edit any reading before running the AI analysis."
        )

        sensor_input_table = pd.DataFrame(
            {
                "Sensor / Feature": sensor_machine_features,
                "Current Reading": [
                    baseline_values.get(feature, 0.0)
                    for feature in sensor_machine_features
                ]
            }
        )

        edited_sensor_table = st.data_editor(
            sensor_input_table,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "Sensor / Feature": st.column_config.TextColumn(
                    "Sensor / Feature",
                    disabled=True
                ),
                "Current Reading": st.column_config.NumberColumn(
                    "Current Reading",
                    help="Type the current value directly"
                )
            },
            key="sensor_machine_input_table"
        )

        st.divider()

        if st.button(
            "Run sensor assessment",
            use_container_width=True,
            key="predict_sensor_machine"
        ):

            try:

                sensor_values = pd.to_numeric(
                    edited_sensor_table[
                        "Current Reading"
                    ],
                    errors="coerce"
                )

                if sensor_values.isna().any():

                    st.error(
                        "⚠️ Every Sensor Machine feature must contain a valid numeric value."
                    )

                else:

                    sensor_input_data = pd.DataFrame(
                        [sensor_values.tolist()],
                        columns=sensor_machine_features
                    )

                    encoded_prediction = sensor_machine_model.predict(
                        sensor_input_data
                    )[0]

                    if sensor_machine_label_encoder is not None:

                        try:

                            detected_status = str(
                                sensor_machine_label_encoder.inverse_transform(
                                    [int(encoded_prediction)]
                                )[0]
                            )

                        except Exception:

                            detected_status = str(encoded_prediction)

                    else:

                        detected_status = str(encoded_prediction)

                    confidence = None

                    if hasattr(
                        sensor_machine_model,
                        "predict_proba"
                    ):

                        probabilities = sensor_machine_model.predict_proba(
                            sensor_input_data
                        )[0]

                        confidence = float(
                            max(probabilities) * 100
                        )

                    status_lower = detected_status.lower()

                    if any(
                        word in status_lower
                        for word in [
                            "normal",
                            "healthy",
                            "good",
                            "stable",
                            "running"
                        ]
                    ):

                        machine_health = "GOOD 🟢"
                        maintenance_priority = "ROUTINE"

                    elif any(
                        word in status_lower
                        for word in [
                            "warning",
                            "attention",
                            "degraded",
                            "maintenance",
                            "suspect"
                        ]
                    ):

                        machine_health = "ATTENTION 🟡"
                        maintenance_priority = "PLAN INSPECTION"

                    else:

                        machine_health = "CRITICAL / ABNORMAL 🔴"
                        maintenance_priority = "URGENT REVIEW"

                    result_time = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                    history_row = {
                        "Time": result_time,
                        "Detected Machine Status": detected_status,
                        "Machine Health": machine_health,
                        "Prediction Confidence %": (
                            confidence
                            if confidence is not None
                            else 0.0
                        ),
                        "Maintenance Priority": maintenance_priority
                    }

                    st.session_state.sensor_machine_prediction_history.append(
                        history_row
                    )

                    st.session_state.latest_sensor_machine_result = {
                        "status": detected_status,
                        "health": machine_health,
                        "confidence": (
                            confidence
                            if confidence is not None
                            else 0.0
                        ),
                        "priority": maintenance_priority
                    }

                    st.divider()
                    st.subheader("Sensor assessment")

                    r1, r2, r3 = st.columns(3)

                    with r1:

                        st.metric(
                            "Detected Machine Status",
                            detected_status
                        )

                    with r2:

                        st.metric(
                            "Machine Health",
                            machine_health
                        )

                    with r3:

                        if confidence is not None:

                            st.metric(
                                "Prediction Confidence",
                                f"{confidence:.2f}%"
                            )

                        else:

                            st.metric(
                                "Prediction Confidence",
                                "Not available"
                            )

                    st.subheader("Assessment summary")

                    if machine_health.startswith("GOOD"):

                        st.success(
                            f"""
### 🟢 Current sensor pattern looks stable

The AI classified the current sensor pattern as **{detected_status}**.

This means the combined sensor readings are currently most similar to the **{detected_status}** condition learned during training.

**Recommended action:** Continue normal operation, keep monitoring the sensor trend, and follow the regular preventive-maintenance schedule.
"""
                        )

                    elif machine_health.startswith("ATTENTION"):

                        st.warning(
                            f"""
### 🟡 The sensor pattern deserves attention

The AI classified the current machine condition as **{detected_status}**.

The machine may still be operating, but its combined sensor pattern is not fully consistent with a clearly healthy condition.

**Recommended action:** Schedule an inspection, compare these readings with recent historical values, and check for any developing trend before the condition worsens.
"""
                        )

                    else:

                        st.error(
                            f"""
### 🔴 The sensor pattern requires review

The AI classified the current machine condition as **{detected_status}**.

The current sensor combination is associated with a non-normal or abnormal status in the model's learned dataset.

**Recommended action:** Perform a technician review, verify the physical sensors and machine condition, and inspect the equipment before continuing long-duration operation.
"""
                        )

                    st.subheader("Recommended next steps")

                    recommendation_col1, recommendation_col2 = st.columns(2)

                    with recommendation_col1:

                        st.info(
                            f"**Priority:** {maintenance_priority}"
                        )

                    with recommendation_col2:

                        if confidence is not None:

                            st.info(
                                f"**AI confidence:** {confidence:.2f}%"
                            )

                        else:

                            st.info(
                                "**AI confidence:** Not provided by this model"
                            )

                    st.caption(
                        "The recommendation is generated from the model's predicted machine-status class and should be used as decision support together with physical inspection and operating procedures."
                    )

            except Exception as error:

                st.error(
                    f"⚠️ Sensor Machine prediction could not be completed: {error}"
                )

        st.divider()
        st.subheader("Sensor analysis history")

        if len(
            st.session_state.sensor_machine_prediction_history
        ) == 0:

            st.info(
                "No analysis has been run for this machine during the current session."
            )

        else:

            sensor_history_df = pd.DataFrame(
                st.session_state.sensor_machine_prediction_history
            )

            st.dataframe(
                sensor_history_df,
                use_container_width=True,
                hide_index=True
            )

            if len(sensor_history_df) > 1:

                st.subheader("Prediction confidence trend")

                chart_df = sensor_history_df.copy()
                chart_df["Time"] = pd.to_datetime(
                    chart_df["Time"]
                )
                chart_df = chart_df.set_index(
                    "Time"
                )

                st.line_chart(
                    chart_df[
                        ["Prediction Confidence %"]
                    ]
                )

            sensor_csv_data = sensor_history_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="📥 Download Sensor Machine History as CSV",
                data=sensor_csv_data,
                file_name="sensor_machine_prediction_history.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_sensor_machine_csv"
            )


# ==================================================
# ELECTRIC MOTOR
# ==================================================

elif selected_machine == "Electric Motor":


    # ==============================================
    # PAGE HEADER
    # ==============================================

    st.markdown(
        '<div class="section-title">'
        '⚙️ Electric Motor Health Analysis'
        '</div>',
        unsafe_allow_html=True
    )


    st.write(
        "Enter the current electric motor sensor readings "
        "to predict Remaining Useful Life, machine health "
        "and maintenance risk."
    )


    # ==============================================
    # MODEL CHECK
    # ==============================================

    if electric_motor_model is None:


        st.error(
            """
### ⚠️ Electric Motor Model Not Available

The application could not load:

`models/electric_motor_rul_model.pkl`

Please make sure the trained Electric Motor model
exists inside your `models` folder.
"""
        )


    else:


        # ==========================================
        # INPUT COLUMNS
        # ==========================================

        col1, col2 = st.columns(2)


        # ==========================================
        # COLUMN 1
        # ==========================================

        with col1:


            dc_bus_voltage = st.number_input(
                "DC Bus Voltage",
                min_value=0.0,
                value=315.0,
                step=0.1,
                key="motor_dc_bus_voltage"
            )


            frequency = st.number_input(
                "Frequency",
                min_value=0.0,
                value=50.0,
                step=0.1,
                key="motor_frequency"
            )


            high_resolution_output_current = (
                st.number_input(
                    "High Resolution Output Current",
                    min_value=0.0,
                    value=50.0,
                    step=0.1,
                    key="motor_high_resolution_current"
                )
            )


            output_current = st.number_input(
                "Output Current",
                min_value=0.0,
                value=5.0,
                step=0.1,
                key="motor_output_current"
            )


            output_voltage = st.number_input(
                "Output Voltage",
                min_value=0.0,
                value=220.0,
                step=0.1,
                key="motor_output_voltage"
            )


        # ==========================================
        # COLUMN 2
        # ==========================================

        with col2:


            speed = st.number_input(
                "Speed",
                min_value=0.0,
                value=1500.0,
                step=1.0,
                key="motor_speed"
            )


            temperature = st.number_input(
                "Temperature",
                min_value=0.0,
                value=35.0,
                step=0.1,
                key="motor_temperature"
            )


            load_index = st.number_input(
                "Load Index",
                min_value=0.0,
                value=50.0,
                step=0.1,
                key="motor_load_index"
            )


            power = st.number_input(
                "Power",
                min_value=0.0,
                value=500.0,
                step=1.0,
                key="motor_power"
            )


            thermal_load = st.number_input(
                "Thermal Load",
                min_value=0.0,
                value=300.0,
                step=1.0,
                key="motor_thermal_load"
            )


        # ==========================================
        # CURRENT SENSOR DASHBOARD
        # ==========================================

        st.divider()


        st.subheader(
            "📊 Current Sensor Dashboard"
        )


        sensor_data = pd.DataFrame(
            {
                "Sensor": [
                    "DC Bus Voltage",
                    "Frequency",
                    "High Resolution Current",
                    "Output Current",
                    "Output Voltage",
                    "Speed",
                    "Temperature",
                    "Load Index",
                    "Power",
                    "Thermal Load"
                ],

                "Value": [
                    dc_bus_voltage,
                    frequency,
                    high_resolution_output_current,
                    output_current,
                    output_voltage,
                    speed,
                    temperature,
                    load_index,
                    power,
                    thermal_load
                ]
            }
        )


        dashboard_col1, dashboard_col2 = (
            st.columns(
                [2, 1]
            )
        )


        with dashboard_col1:


            st.bar_chart(
                sensor_data.set_index(
                    "Sensor"
                )
            )


        with dashboard_col2:


            st.dataframe(
                sensor_data,
                use_container_width=True,
                hide_index=True
            )


        st.divider()


        # ==========================================
        # ANALYZE BUTTON
        # ==========================================

        analyze_motor = st.button(
            "Run motor assessment",
            use_container_width=True,
            type="primary",
            key="analyze_motor_button"
        )


        # ==========================================
        # MOTOR PREDICTION
        # ==========================================

        if analyze_motor:

            valid_inputs = all([
                validate_range("DC Bus Voltage", dc_bus_voltage, 0.0, 1000.0),
                validate_range("Frequency", frequency, 0.0, 500.0),
                validate_range("High Resolution Output Current", high_resolution_output_current, 0.0, 10000.0),
                validate_range("Output Current", output_current, 0.0, 1000.0),
                validate_range("Output Voltage", output_voltage, 0.0, 2000.0),
                validate_range("Speed", speed, 0.0, 20000.0),
                validate_range("Temperature", temperature, 0.0, 250.0),
                validate_range("Load Index", load_index, 0.0, 1000.0),
                validate_range("Power", power, 0.0, 1000000.0),
                validate_range("Thermal Load", thermal_load, 0.0, 1000000.0)
            ])

            if not valid_inputs:
                st.stop()

            input_data = pd.DataFrame(
                [[
                    dc_bus_voltage,
                    frequency,
                    high_resolution_output_current,
                    output_current,
                    output_voltage,
                    speed,
                    temperature,
                    load_index,
                    power,
                    thermal_load
                ]],

                columns=[
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
            )


            try:


                prediction = (
                    electric_motor_model.predict(
                        input_data
                    )[0]
                )


                prediction = max(
                    float(prediction),
                    0
                )


                # ==================================
                # RISK SCORE
                # ==================================

                risk_score = (
                    100
                    -
                    (
                        prediction
                        /
                        20000
                        *
                        100
                    )
                )


                risk_score = max(
                    0,
                    min(
                        100,
                        risk_score
                    )
                )


                # ==================================
                # HEALTH
                # ==================================

                if risk_score < 30:


                    health = "GOOD 🟢"

                    risk_level = "LOW RISK 🟢"


                elif risk_score < 60:


                    health = "ATTENTION 🟡"

                    risk_level = "MEDIUM RISK 🟡"


                else:


                    health = "CRITICAL 🔴"

                    risk_level = "HIGH RISK 🔴"


                # ==================================
                # FAILURE ANALYSIS
                # ==================================

                possible_reasons = []


                if temperature > 70:

                    possible_reasons.append(
                        "High motor temperature may indicate overheating."
                    )


                if load_index > 80:

                    possible_reasons.append(
                        "High load index may indicate excessive mechanical load."
                    )


                if thermal_load > 1000:

                    possible_reasons.append(
                        "High thermal load may cause insulation or winding damage."
                    )


                if output_current > 20:

                    possible_reasons.append(
                        "High output current may indicate overload or electrical problems."
                    )


                if speed < 1000:

                    possible_reasons.append(
                        "Low speed may indicate mechanical resistance or motor performance issues."
                    )


                if not possible_reasons:

                    possible_reasons.append(
                        "No major abnormal sensor condition was detected from the entered values."
                    )


                # ==================================
                # RECOMMENDATIONS
                # ==================================

                recommendations = []


                if temperature > 70:

                    recommendations.append(
                        "Inspect the cooling system, ventilation and motor bearings."
                    )


                if load_index > 80:

                    recommendations.append(
                        "Reduce excessive mechanical load and inspect the driven equipment."
                    )


                if thermal_load > 1000:

                    recommendations.append(
                        "Inspect insulation and reduce prolonged thermal stress."
                    )


                if output_current > 20:

                    recommendations.append(
                        "Inspect electrical connections, windings and motor load."
                    )


                if speed < 1000:

                    recommendations.append(
                        "Inspect bearings, shafts and mechanical components for resistance."
                    )


                if health == "GOOD 🟢":

                    recommendations.append(
                        "Continue normal operation and perform scheduled preventive maintenance."
                    )


                elif health == "ATTENTION 🟡":

                    recommendations.append(
                        "Schedule a maintenance inspection before machine health becomes critical."
                    )


                else:

                    recommendations.append(
                        "Perform immediate inspection and maintenance before continuing heavy operation."
                    )


                # ==================================
                # SAVE RESULT
                # ==================================

                result = {

                    "prediction": prediction,

                    "risk_score": risk_score,

                    "health": health,

                    "risk_level": risk_level,

                    "possible_reasons": (
                        possible_reasons
                    ),

                    "recommendations": (
                        recommendations
                    )
                }


                st.session_state.latest_motor_result = (
                    result
                )


                # ==================================
                # SAVE HISTORY
                # ==================================

                prediction_record = {

                    "Time": (
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    ),

                    "Temperature": (
                        temperature
                    ),

                    "Speed": (
                        speed
                    ),

                    "Load Index": (
                        load_index
                    ),

                    "Predicted RUL Hours": (
                        round(
                            prediction,
                            2
                        )
                    ),

                    "Risk Score %": (
                        round(
                            risk_score,
                            2
                        )
                    ),

                    "Machine Health": (
                        health
                    )
                }


                st.session_state.prediction_history.append(
                    prediction_record
                )


            except Exception as error:


                st.error(
                    f"Prediction error: {error}"
                )


        # ==========================================
        # DISPLAY LATEST MOTOR RESULT
        # ==========================================

        if (
            st.session_state.latest_motor_result
            is not None
        ):


            result = (
                st.session_state.latest_motor_result
            )


            prediction = (
                result["prediction"]
            )

            risk_score = (
                result["risk_score"]
            )

            health = (
                result["health"]
            )

            risk_level = (
                result["risk_level"]
            )

            possible_reasons = (
                result["possible_reasons"]
            )

            recommendations = (
                result["recommendations"]
            )


            st.divider()


            st.subheader(
                "Motor assessment"
            )


            result_col1, result_col2, result_col3 = (
                st.columns(3)
            )


            with result_col1:


                st.metric(
                    "Remaining Useful Life",
                    f"{prediction:.2f} Hours"
                )


            with result_col2:


                st.metric(
                    "Machine Health",
                    health
                )


            with result_col3:


                st.metric(
                    "Risk Score",
                    f"{risk_score:.2f}%"
                )


            st.subheader(
                "Current operating risk"
            )


            st.progress(
                int(risk_score)
            )


            st.write(
                f"### {risk_level}"
            )


            # ======================================
            # STATUS CARD
            # ======================================

            if health == "GOOD 🟢":


                st.markdown(
                    f"""
                    <div class="status-good">
                    🟢 <b>MACHINE HEALTH: GOOD</b><br><br>
                    Predicted Remaining Useful Life:
                    <b>{prediction:.2f} Hours</b><br><br>
                    Continue normal operation and scheduled
                    preventive maintenance.
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            elif health == "ATTENTION 🟡":


                st.markdown(
                    f"""
                    <div class="status-warning">
                    🟡 <b>MACHINE HEALTH: ATTENTION REQUIRED</b><br><br>
                    Predicted Remaining Useful Life:
                    <b>{prediction:.2f} Hours</b><br><br>
                    Schedule a maintenance inspection and monitor
                    the machine operating conditions closely.
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            else:


                st.markdown(
                    f"""
                    <div class="status-critical">
                    🔴 <b>MACHINE HEALTH: CRITICAL</b><br><br>
                    Predicted Remaining Useful Life:
                    <b>{prediction:.2f} Hours</b><br><br>
                    Immediate inspection and maintenance
                    are recommended.
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # ======================================
            # POSSIBLE REASONS
            # ======================================

            st.subheader(
                "What needs attention"
            )


            for reason in possible_reasons:

                st.write(
                    f"• {reason}"
                )


            # ======================================
            # RECOMMENDATIONS
            # ======================================

            st.subheader(
                "Recommended next steps"
            )


            for recommendation in recommendations:

                st.write(
                    f"• {recommendation}"
                )


        # ==========================================
        # PREDICTION HISTORY
        # ==========================================

        if len(
            st.session_state.prediction_history
        ) > 0:


            st.divider()


            history_col1, history_col2 = (
                st.columns(
                    [4, 1]
                )
            )


            with history_col1:


                st.subheader(
                    "Motor analysis history"
                )


            with history_col2:


                if st.button(
                    "🗑️ Clear Prediction History",
                    use_container_width=True,
                    key="clear_motor_history"
                ):


                    st.session_state.prediction_history = []

                    st.session_state.latest_motor_result = None

                    st.rerun()


            history_df = pd.DataFrame(
                st.session_state.prediction_history
            )


            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True
            )


            # ======================================
            # RUL HISTORY
            # ======================================

            st.subheader(
                "Remaining useful life trend"
            )


            chart_data = (
                history_df[
                    [
                        "Time",
                        "Predicted RUL Hours"
                    ]
                ]
                .copy()
                .set_index(
                    "Time"
                )
            )


            st.line_chart(
                chart_data,
                use_container_width=True
            )


            # ======================================
            # RISK HISTORY
            # ======================================

            st.subheader(
                "Risk score trend"
            )


            risk_chart_data = (
                history_df[
                    [
                        "Time",
                        "Risk Score %"
                    ]
                ]
                .copy()
                .set_index(
                    "Time"
                )
            )


            st.line_chart(
                risk_chart_data,
                use_container_width=True
            )


            # ======================================
            # DOWNLOAD CSV
            # ======================================

            st.subheader(
                "Export analysis history"
            )


            csv_data = (
                history_df.to_csv(
                    index=False
                ).encode(
                    "utf-8"
                )
            )


            st.download_button(
                label=(
                    "Download history as CSV"
                ),
                data=csv_data,
                file_name=(
                    "electric_motor_prediction_history.csv"
                ),
                mime="text/csv",
                use_container_width=True,
                key="download_motor_csv"
            )


# ==================================================
# OTHER MACHINES
# ==================================================

else:


    st.markdown(
        '<div class="section-title">'
        f'🏭 {selected_machine}'
        '</div>',
        unsafe_allow_html=True
    )


    st.info(
        f"""
### {selected_machine}

This workspace is reserved for a machine-specific model and is not connected to a trained dataset yet.

When you add a dataset later, the workflow will be:

1. Inspect and clean the machine data.
2. Define the prediction target.
3. Train and evaluate the model.
4. Save the trained model in the `models` folder.
5. Connect the result to this workspace.
"""
    )


# ==================================================
# APPLICATION FOOTER
# ==================================================

st.divider()


st.caption(
    "Predictive Maintenance | "
    "Machine Health Monitoring | "
    "Failure Prediction | "
    "Remaining Useful Life | "
    "Sensor Pattern Classification | "
    "Maintenance Recommendation System"
)