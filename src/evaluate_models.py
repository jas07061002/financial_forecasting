import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import timedelta
# Load cleaned data
df = pd.read_csv("../S&P500_cleaned.csv", index_col="Date", parse_dates=True)
test_data = df["Close"].iloc[-30:]

# Load Advanced Fine-Tuned LSTM Model
model = tf.keras.models.load_model("../advanced_fine_tuned_lstm.h5")

# Prepare Data for Prediction
scaler = MinMaxScaler(feature_range=(0,1))
df_scaled = scaler.fit_transform(df[["Close"]])
X_test = df_scaled[-150:].reshape(1,150,1)  # Increased to 150 days

# Predict & Convert Back to Original Scale
lstm_forecast = []
X_input = df_scaled[-150:].reshape(1, 150, 1)  # Use last 150 days

for _ in range(30):  # Predict next 30 days
    next_pred = model.predict(X_input).reshape(1,1,1)  # Ensure 3D shape
    lstm_forecast.append(next_pred[0, 0, 0])  # Extract scalar
    
    # Update input: remove first value, append new prediction
    X_input = np.append(X_input[:, 1:, :], next_pred, axis=1)

# Convert back to original scale
lstm_forecast = scaler.inverse_transform(np.array(lstm_forecast).reshape(-1,1)).flatten()

# Predict Next 30 Days
future_forecast = []
input_seq = X_test

for _ in range(30):  # Predict 30 days into the future
    prediction = model.predict(input_seq)[0][0]  # Get next predicted value
    future_forecast.append(prediction)

    # Append prediction to input sequence and remove oldest value
    input_seq = np.append(input_seq[:,1:,:], [[[prediction]]], axis=1)
    
# Convert predictions back to original scale
future_forecast = scaler.inverse_transform(np.array(future_forecast).reshape(-1,1)).flatten()

# Generate future dates for the next 30 days
last_date = df.index[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]

# Create a DataFrame for Future Predictions
future_predictions = pd.DataFrame({
    "Date": future_dates,
    "LSTM_Forecast": future_forecast
})
future_predictions.set_index("Date", inplace=True)

# Save Future Forecasts to CSV
future_predictions.to_csv("../lstm_predictions.csv")

# Print Confirmation
print("Future 30-Day Predictions Saved in `lstm_predictions.csv`!")
# Create a DataFrame for Predictions
lstm_predictions = pd.DataFrame({
    "Date": test_data.index,
    "Actual_Close": test_data.values,
    "LSTM_Forecast": lstm_forecast
})
lstm_predictions.set_index("Date", inplace=True)

# Calculate Metrics
lstm_rmse = np.sqrt(mean_squared_error(test_data, lstm_forecast))
lstm_mae = mean_absolute_error(test_data, lstm_forecast)
lstm_r2 = r2_score(test_data, lstm_forecast)

# Print Results
print("**Advanced Fine-Tuned LSTM Model Performance**")
print(f"LSTM RMSE: {lstm_rmse:.4f}, MAE: {lstm_mae:.4f}, R²: {lstm_r2:.4f}")
print("LSTM Predictions Saved as `lstm_predictions.csv`.")

# Plot Predictions
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(test_data.index, test_data, label="Actual S&P 500", color="blue")
plt.plot(test_data.index, lstm_forecast, label="Advanced Fine-Tuned LSTM Forecast", color="green", linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("S&P 500 Price Prediction (Advanced LSTM)")
plt.legend()
plt.grid(True)
plt.show()
