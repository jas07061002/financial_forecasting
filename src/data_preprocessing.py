import pandas as pd

# Load CSV file (skip first extra row)
df = pd.read_csv("../S&P500_data.csv", skiprows=2, index_col="Date", parse_dates=True)

# Rename columns (Remove Ticker row)
df.columns = ["Close", "High", "Low", "Open", "Volume"]

# Convert numeric columns to float
df = df.astype(float)

# Display first few rows
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Fill missing values using forward fill
df.ffill(inplace=True)

# Drop rows if still any missing values
df.dropna(inplace=True)

# Moving Averages
df["SMA_10"] = df["Close"].rolling(window=10).mean()
df["SMA_50"] = df["Close"].rolling(window=50).mean()

# Volatility (Rolling Standard Deviation)
df["Volatility"] = df["Close"].rolling(window=10).std()

# Relative Strength Index (RSI)
delta = df["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))

# Drop NaN values created during rolling calculations
df.dropna(inplace=True)

# Save cleaned data
df.to_csv("../S&P500_cleaned.csv")
print("Cleaned and preprocessed S&P 500 data saved!")
