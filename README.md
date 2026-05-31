# Energy Consumption Prediction using RNN

## Project Overview

This project predicts the next day's energy consumption using a Recurrent Neural Network (RNN). The model is trained on historical energy consumption data and uses the previous 30 days of consumption values to forecast the next day's usage.

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Streamlit
* Scikit-learn

## Features

* Predicts next-day energy consumption
* Uses RNN for time-series forecasting
* Interactive Streamlit web application
* Data scaling using MinMaxScaler

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run deploy.py
```

## Project Files

* `model.py` - Model training script
* `deploy.py` - Streamlit deployment file
* `energy_consumption_model.keras` - Trained RNN model
* `scaler.pkl` - Saved MinMaxScaler
* `requirements.txt` - Required libraries

## Input

The application accepts the last 30 days of energy consumption values.

Example:

1250,1325,1180,1400,1285,1350,1420,1385,1450,1500,1480,1525,1580,1600,1555,1620,1680,1655,1700,1750,1725,1800,1850,1820,1885,1900,1950,1980,2025,2050

## Output

The model predicts the energy consumption for the next day in kWh.


P Sai Saharsh
