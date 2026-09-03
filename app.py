import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="🤖",
    layout="centered"
)

# Load trained model
model = joblib.load("models/predictive_maintenance_model.pkl")

# Title
st.title("🤖 Predictive Maintenance AI")
st.write("Enter machine parameters to predict whether machine failure may occur.")

# Machine type
machine_type = st.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)

# Input values
air_temperature = st.number_input(
    "Air Temperature (K)",
    value=300.0
)

process_temperature = st.number_input(
    "Process Temperature (K)",
    value=310.0
)

rotational_speed = st.number_input(
    "Rotational Speed (RPM)",
    value=1500
)

torque = st.number_input(
    "Torque (Nm)",
    value=40.0
)

tool_wear = st.number_input(
    "Tool Wear (minutes)",
    value=100
)

# Predict button
if st.button("Predict Machine Failure"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "Air temperature [K]": [air_temperature],
        "Process temperature [K]": [process_temperature],
        "Rotational speed [rpm]": [rotational_speed],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear],
        "Type_L": [machine_type == "L"],
        "Type_M": [machine_type == "M"]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display result
    if prediction == 1:
        st.error("⚠️ Machine Failure Predicted!")
    else:
        st.success("✅ No Machine Failure Predicted")