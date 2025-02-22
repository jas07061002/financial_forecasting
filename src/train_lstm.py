import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.preprocessing import MinMaxScaler

# Load cleaned S&P 500 data
df = pd.read_csv("../S&P500_cleaned.csv", index_col="Date", parse_dates=True)

# Scale data
scaler = MinMaxScaler(feature_range=(0,1))
df_scaled = scaler.fit_transform(df[["Close"]])

# 📌 Increase Sequence Length (100 → 150 Days)
SEQ_LENGTH = 150  # Increased from 100

# Prepare data for LSTM
X_train, y_train = [], []
for i in range(SEQ_LENGTH, len(df_scaled)):
    X_train.append(df_scaled[i-SEQ_LENGTH:i, 0])
    y_train.append(df_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # Reshape for LSTM

# Define Improved LSTM Model
model = tf.keras.Sequential([
    Bidirectional(LSTM(units=150, return_sequences=True, input_shape=(X_train.shape[1], 1))),  # More LSTM units
    Dropout(0.2),
    Bidirectional(LSTM(units=100, return_sequences=True)),
    Dropout(0.2),
    Bidirectional(LSTM(units=50, return_sequences=False)),
    Dropout(0.2),
    Dense(units=50),
    Dense(units=1)  # Final output layer
])

# Optimizer: Try RMSprop instead of Adam
optimizer = RMSprop(learning_rate=0.001)

# Learning Rate Scheduler (Reduces LR when loss plateaus)
lr_scheduler = ReduceLROnPlateau(monitor="loss", patience=5, factor=0.5, min_lr=1e-5)

# Early Stopping: Stop training when loss stops improving
early_stopping = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)

# Compile & Train Model with More Epochs
model.compile(optimizer=optimizer, loss="mean_squared_error")

# Increase Epochs (from 100 → 150) & Batch Size (from 16 → 32)
model.fit(X_train, y_train, batch_size=32, epochs=150, callbacks=[lr_scheduler, early_stopping])

# Save Fine-Tuned Model
model.save("../advanced_fine_tuned_lstm.h5")
print("Advanced Fine-Tuned LSTM Model Trained & Saved!")
