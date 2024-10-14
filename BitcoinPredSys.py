import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Load the dataset
data = pd.read_csv('BTC-USD-cleaned-without-leakage.csv')
print(data.head())


# Prepare the data
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by='Date')

train_data = data[data['Set'] == 'Train']
test_data = data[data['Set'] == 'Test']

features = ['Open', 'High', 'Low', 'Close', 'Volume']

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_features = scaler.fit_transform(train_data[features])
scaled_test_features = scaler.transform(test_data[features])


window_size = 60
X_train, y_train = [], []
for i in range(window_size, len(scaled_train_features)):
    X_train.append(scaled_train_features[i - window_size:i])
    y_train.append(scaled_train_features[i, 3])

X_test, y_test = [], []
for i in range(window_size, len(scaled_test_features)):
    X_test.append(scaled_test_features[i - window_size:i])
    y_test.append(scaled_test_features[i, 3]) 

X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)


# Build the LSTM model
model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=100, return_sequences=True))  # Increased units and layers
model.add(Dropout(0.2))
model.add(LSTM(units=100, return_sequences=True))  # Additional LSTM layer
model.add(Dropout(0.2))
model.add(LSTM(units=100, return_sequences=False))  # Additional LSTM layer
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Set early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


# Train the model with increased epochs and batch size
model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stop])


# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

predicted_prices = model.predict(X_test)

# Inverse transform predicted prices
predicted_prices = scaler.inverse_transform(
    np.concatenate((np.zeros((len(predicted_prices), 4)), predicted_prices), axis=1)
)[:, -1]

actual_prices = scaler.inverse_transform(
    np.concatenate((np.zeros((len(y_test), 4)), y_test.reshape(-1, 1)), axis=1)
)[:, -1]

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(test_data['Date'].iloc[window_size:], actual_prices, color='blue', label='Actual Price')
plt.plot(test_data['Date'].iloc[window_size:], predicted_prices, color='red', label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Bitcoin Price')
plt.legend()
plt.show()

# Calculate MAPE and R² Score
mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
print("Mean Absolute Percentage Error (MAPE): {:.2f}%".format(mape))

r2 = r2_score(actual_prices, predicted_prices)
print("R-squared (R²) Score: {:.2f}".format(r2 * 100))


# Use the last available 60-day data point as the starting point for prediction
last_window = scaled_test_features[-window_size:]

predicted_next_prices = []

for i in range(7):
    next_pred = model.predict(last_window.reshape(1, window_size, len(features)))  # Predict the next day's price
    predicted_next_prices.append(next_pred[0, 0])  # Store the predicted 'Close' price
    
    # Shift the window: keep the most recent 59 days + the predicted value as the 60th day
    new_entry = np.array([[0, 0, 0, next_pred[0, 0], 0]])  # The new predicted price as part of the window
    last_window = np.concatenate((last_window[1:], new_entry), axis=0)  # Shift the window forward

# Inverse transform the predicted next prices
predicted_next_prices = scaler.inverse_transform(
    np.concatenate((np.zeros((len(predicted_next_prices), 4)), np.array(predicted_next_prices).reshape(-1, 1)), axis=1)
)[:, -1]

# Divide the predicted prices by 100,000
predicted_next_prices /= 1000000

# Print the predicted prices for the next 7 days
print("Predicted Prices for the next 7 days:")
for i, price in enumerate(predicted_next_prices, 1):
    print(f"Day {i}: {price:.2f}")
