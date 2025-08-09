import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

# 1. Download historical stock data
ticker = 'AAPL'
df = yf.download(ticker, start="2020-01-01", end="2024-01-01")

# 2. Extract dates and closing prices
df = df.reset_index()
df['Date_Ordinal'] = pd.to_datetime(df['Date']).map(pd.Timestamp.toordinal)
dates = df['Date_Ordinal'].values.reshape(-1, 1)
prices = df['Close'].values.reshape(-1, 1)

# 3. Handle missing values
prices[np.isnan(prices)] = np.median(prices[~np.isnan(prices)])

# 4. Train/test split
x_train, x_test, y_train, y_test = train_test_split(dates, prices, test_size=0.2, random_state=42)

# 5. Train linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# 6. Evaluate the model
accuracy = model.score(x_test, y_test)
print(f"Model Accuracy (R² score): {accuracy:.4f}")

# 7. Save model
with open('aapl_model.pickle', 'wb') as f:
    pickle.dump(model, f)

# 8. Predict using the model
predicted = model.predict(x_test)

# 9. Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(x_test, y_test, color='green', label='Actual Price', s=10)
plt.plot(x_test, predicted, color='blue', linewidth=2, label='Predicted Price')
plt.xlabel("Date")
plt.ylabel("Stock Price (USD)")
plt.title(f"{ticker} Stock Price Prediction using Linear Regression")
plt.legend()
plt.show()
