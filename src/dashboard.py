import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import schedule
import time
import subprocess
import threading
import os
from real_time_predict import predict_next_10_days


st.title("📈 Real-Time 10-Day S&P 500 Forecasting Dashboard")

def run_prediction():
    try:
        predict_next_10_days()  # Run function instead of subprocess
        st.success("✅ Prediction updated successfully!")
        st.session_state["prediction_updated"] = True  # Refresh Streamlit UI

    except Exception as e:
        st.error(f"❌ Error: {e}")


# Button to manually trigger prediction update
if st.button("🔄 Run Prediction Now"):
    run_prediction()

# Load latest 10-day prediction
try:
    st.cache_data.clear()  # Ensure Streamlit loads fresh data
    prediction_file = os.path.join(os.path.dirname(__file__), "real_time_predictions.csv")
    future_predictions = pd.read_csv(prediction_file, index_col="Date", parse_dates=True)
    
    st.write("### 🔮 Predicted S&P 500 Closing Prices for the Next 10 Days")
    st.dataframe(future_predictions)

    # Plot the forecast
    plt.figure(figsize=(10,5))
    plt.plot(future_predictions.index, future_predictions["LSTM_Forecast"], marker="o", linestyle="dashed", color="green", label="LSTM Prediction")
    plt.xlabel("Date")
    plt.ylabel("Predicted Close Price (USD)")
    plt.title("S&P 500 Forecast for the Next 10 Days")
    plt.legend()
    plt.grid()
    st.pyplot(plt)

except FileNotFoundError:
    st.error("No prediction file found. Run `real_time_predict.py` first.")