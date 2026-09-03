import streamlit as st
import pandas as pd
import joblib

# -----------------------------------
# Page configuration
# -----------------------------------

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="🤖"
)

# -----------------------------------
# Load trained model
# -----------------------------------

model = joblib.load("models/predictive_maintenance_model.pkl")

# -----------------------------------
# App title
# -----------------------------------

st.title("🤖 Predictive Maintenance AI")

st.write(
    "Enter machine parameters to predict whether machine failure may occur."
)

# -----------------------------------
# User inputs
# -----------------------------------

machine_type = st.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)

air_temperature = st.number_input(
    "Air Temperature (K)",
    value=300.00
)

process_temperature = st.number_input(
    "Process Temperature (K)",
    value=310.00
)

rotational_speed = st.number_input(
    "Rotational Speed (RPM)",
    value=1500
)

torque = st.number_input(
    "Torque (Nm)",
    value=40.00
)

tool_wear = st.number_input(
    "Tool Wear (minutes)",
    value=100
)

# -----------------------------------
# Prediction button
# -----------------------------------

if st.button("Predict Machine Failure"):

    # Create input data
    input_data = pd.DataFrame({
        "Air temperature [K]": [air_temperature],
        "Process temperature [K]": [process_temperature],
        "Rotational speed [rpm]": [rotational_speed],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear],
        "Type_L": [1 if machine_type == "L" else 0],
        "Type_M": [1 if machine_type == "M" else 0]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display raw prediction
    st.write("Raw prediction:", prediction)

    # Display prediction probability
    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(input_data)[0]

        st.write(
            "Prediction probabilities:",
            probability
        )

        # Show failure probability more clearly
        failure_probability = probability[1] * 100

        st.write(
            f"Machine Failure Probability: {failure_probability:.2f}%"
        )

    # Display final result
    if prediction == 1:

        st.error(
            "⚠️ Machine Failure Predicted!"
        )

    else:

        st.success(
            "✅ No Machine Failure Predicted"
        )