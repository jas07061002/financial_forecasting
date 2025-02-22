import yfinance as yf

# Fetch S&P 500 index data (^GSPC)
sp500 = yf.download("^GSPC", start="2000-01-01", end="2025-01-01")

# Save to CSV
sp500.to_csv("S&P500_data.csv")

# Display first few rows
print(sp500.head())
