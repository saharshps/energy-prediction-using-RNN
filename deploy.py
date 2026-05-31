import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pickle

model = load_model('energy_consumption_model.keras')
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("⚡ Energy Consumption Prediction")

n_steps = 30

default_data = "1250,1325,1180,1400,1285,1350,1420,1385,1450,1500,1480,1525,1580,1600,1555,1620,1680,1655,1700,1750,1725,1800,1850,1820,1885,1900,1950,1980,2025,2050"

input_data = st.text_area(
    f"Enter the last {n_steps} days of energy consumption",
    value=default_data,
    height=120
)

if st.button("Predict"):
    try:
        input_values = list(map(float, input_data.split(',')))

        if len(input_values) != n_steps:
            st.error(f"Please enter exactly {n_steps} values.")
        else:
            input_array = np.array(input_values).reshape(-1, 1)

            input_scaled = scaler.transform(input_array)

            input_scaled = input_scaled.reshape(1, n_steps, 1)

            predicted_scaled = model.predict(input_scaled)

            predicted_consumption = scaler.inverse_transform(
                predicted_scaled
            )

            st.success(
                f"Predicted energy consumption for the next day: "
                f"{predicted_consumption[0][0]:.2f} kWh"
            )

            st.write("Raw Model Output:", predicted_scaled)

    except ValueError:
        st.error("Please enter valid numeric values separated by commas.")