import pandas as pd
data = pd.read_csv('Bitcoin_cleaned.csv')
print(data.head())


data['Date'] = pd.to_datetime(data['Date'])

data = data.sort_values(by='Date')

features = data[['Open', 'High', 'Low', 'Close', 'Volume']]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)


import numpy as np

window_size = 60

X = []
y = []
for i in range(window_size, len(scaled_features)):
    X.append(scaled_features[i - window_size:i])
    y.append(scaled_features[i, 3])

X, y = np.array(X), np.array(y)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# Initialize the LSTM model
model = Sequential()

# Define the input layer
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))

# Add the first LSTM layer with Dropout
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

# Add a second LSTM layer with Dropout
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Add the output layer
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')


# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))


import matplotlib.pyplot as plt

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Predict the closing prices
predicted_prices = model.predict(X_test)

# Inverse transform to get actual values for the predicted prices
predicted_prices = scaler.inverse_transform(np.concatenate((np.zeros((len(predicted_prices), 4)), predicted_prices), axis=1))[:, -1]

# Inverse transform to get actual values for the test labels
actual_prices = scaler.inverse_transform(np.concatenate((np.zeros((len(y_test), 4)), y_test.reshape(-1, 1)), axis=1))[:, -1]

# Plot the results, adjusting the date range to match the length of actual prices
plt.figure(figsize=(10, 6))
plt.plot(data['Date'][split:split + len(actual_prices)], actual_prices, color='blue', label='Actual Price')
plt.plot(data['Date'][split:split + len(predicted_prices)], predicted_prices, color='red', label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Bitcoin Price')
plt.legend()
plt.show()


# Calculate Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
print("Mean Absolute Percentage Error (MAPE): {:.2f}%".format(mape))

from sklearn.metrics import r2_score

# Calculate R-squared (R²) score
r2 = r2_score(actual_prices, predicted_prices)
print("R-squared (R²) Score: {:.2f}".format(r2 * 100))

# Calculate MAPE
mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
print("Mean Absolute Percentage Error (MAPE): {:.2f}%".format(mape))

# Calculate R-squared score
r2 = r2_score(actual_prices, predicted_prices)
print("R-squared (R²) Score: {:.2f}%".format(r2 * 100))


model.save('bitcoin_price_lstm_model.h5')
