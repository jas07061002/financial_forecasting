import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import os

# Load trained LSTM model
model = tf.keras.models.load_model("advanced_fine_tuned_lstm.h5")

# Function to fetch real-time S&P 500 data
def fetch_real_time_data():
    sp500 = yf.Ticker("^GSPC")
    df = sp500.history(period="150d")  # Get last 150 days of stock data
    df = df[["Close"]]
    df.index = pd.to_datetime(df.index)
    return df

# Function to predict the next 10 days' prices
def predict_next_10_days():
    df = fetch_real_time_data()

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0,1))
    df_scaled = scaler.fit_transform(df)

    # Use last 150 days as input
    input_seq = df_scaled[-150:].reshape(1, 150, 1)

    # Predict the next 10 days
    future_forecast = []
    for _ in range(10):
        prediction_scaled = model.predict(input_seq)[0][0]  # Predict next value
        future_forecast.append(prediction_scaled)

        # Append prediction to input sequence and remove the oldest value
        input_seq = np.append(input_seq[:, 1:, :], [[[prediction_scaled]]], axis=1)

    # Convert predictions back to original scale
    future_forecast = scaler.inverse_transform(np.array(future_forecast).reshape(-1,1)).flatten()

    return future_forecast

# Get real-time 10-day prediction
future_prices = predict_next_10_days()

# Generate future dates for the next 10 days
future_dates = [datetime.now() + timedelta(days=i) for i in range(1, 11)]

# Save predictions dynamically
prediction_df = pd.DataFrame({
    "Date": future_dates,
    "LSTM_Forecast": future_prices
})
prediction_df.set_index("Date", inplace=True)
prediction_file = os.path.join(os.path.dirname(__file__), "real_time_predictions.csv")
prediction_df.to_csv(prediction_file)

# Print the predictions
print("🔮 **10-Day S&P 500 Forecast:**")
print(prediction_df)
