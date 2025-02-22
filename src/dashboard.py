import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import schedule
import time
import subprocess
import threading
import os

st.title("📈 Real-Time 10-Day S&P 500 Forecasting Dashboard")

def run_prediction_script():
    try:
        # Dynamically locate real_time_predict.py
        script_path = os.path.join(os.path.dirname(__file__), "real_time_predict.py")
        st.text(f"🔍 Looking for script at: {script_path}")

        # Check if the script exists
        if not os.path.exists(script_path):
            st.error(f"❌ Error: Script not found at {script_path}")
            return
        
        # Run the script and capture error logs
         # Run the script (suppress TensorFlow logs)
        with open(os.devnull, "w") as devnull:
            result = subprocess.run(
                ["python3", script_path], 
                check=True, 
                stdout=devnull,  # Suppress standard output
                stderr=devnull   # Suppress error output
            )

        st.success("✅ Prediction updated successfully!")
        st.text(result.stdout)  # Show script output
        st.session_state["prediction_updated"] = True  # Refresh Streamlit UI

    except subprocess.CalledProcessError as e:
        st.error(f"❌ Error running script: {e}")
        st.text(e.stderr)  # Show full error details


# Schedule the script to run every day at 6 AM (Only one scheduler runs)
if "scheduler_initialized" not in st.session_state:
    st.session_state.scheduler_initialized = True  # Prevent multiple schedulers
    schedule.every().day.at("06:00").do(run_prediction_script)

    def schedule_runner():
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every 60 seconds

    # Start the scheduler in a separate thread
    thread = threading.Thread(target=schedule_runner, daemon=True)
    thread.start()

# Button to manually trigger prediction update
if st.button("🔄 Run Prediction Now"):
    run_prediction_script()

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

st.write("🚀 This prediction updates automatically every day at 6 AM.")
