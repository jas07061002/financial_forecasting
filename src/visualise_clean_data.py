import matplotlib.pyplot as plt
import pandas as pd

# Load CSV file (skip first extra row)
df = pd.read_csv("financial_forecasting/S&P500_cleaned.csv", index_col="Date", parse_dates=True)

# Display first few rows
print(df.head())

plt.figure(figsize=(12,6))
plt.plot(df.index, df["Close"], label="S&P 500 Close Price", color="blue")
plt.plot(df.index, df["SMA_10"], label="10-Day SMA", color="orange", linestyle="dashed")
plt.plot(df.index, df["SMA_50"], label="50-Day SMA", color="green", linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("S&P 500 Stock Price with Moving Averages")
plt.legend()
plt.grid(True)
plt.show()
