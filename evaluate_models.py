import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load cleaned data
df = pd.read_csv("S&P500_cleaned.csv", index_col="Date", parse_dates=True)
test_data = df["Close"].iloc[-30:]

# Load Advanced Fine-Tuned LSTM Model
model = tf.keras.models.load_model("advanced_fine_tuned_lstm.h5")

# Prepare Data for Prediction
scaler = MinMaxScaler(feature_range=(0,1))
df_scaled = scaler.fit_transform(df[["Close"]])
X_test = df_scaled[-150:].reshape(1,150,1)  # Increased to 150 days

# Predict & Convert Back to Original Scale
lstm_forecast = scaler.inverse_transform(model.predict(X_test)).flatten()[-30:]

# Calculate Metrics
lstm_rmse = np.sqrt(mean_squared_error(test_data, lstm_forecast))
lstm_mae = mean_absolute_error(test_data, lstm_forecast)
lstm_r2 = r2_score(test_data, lstm_forecast)

# Print Results
print("📊 **Advanced Fine-Tuned LSTM Model Performance**")
print(f"LSTM RMSE: {lstm_rmse:.4f}, MAE: {lstm_mae:.4f}, R²: {lstm_r2:.4f}")

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
