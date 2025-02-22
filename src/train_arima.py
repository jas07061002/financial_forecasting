import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import joblib

df = pd.read_csv("S&P500_cleaned.csv", index_col="Date", parse_dates=True)

# Auto-select best ARIMA parameters
auto_arima_model = auto_arima(df["Close"], seasonal=False)
best_p, best_d, best_q = auto_arima_model.order

# Train ARIMA model
arima_model = ARIMA(df["Close"], order=(best_p, best_d, best_q))
arima_fitted = arima_model.fit()

# Save model

joblib.dump(arima_fitted, "arima_model.pkl")
print("ARIMA Model Trained & Saved!")
