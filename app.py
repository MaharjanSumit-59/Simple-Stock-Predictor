import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime

st.set_page_config(page_title="Stock Price Predictor", layout="centered")

# App title
st.title("📈 Stock Price Predictor using Linear Regression")

# User input: stock ticker
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, TSLA):", value='AAPL').upper()

# Predict when button is clicked
if st.button("Predict"):
    try:
        with st.spinner("Fetching data..."):
            # Download historical stock data
            df = yf.download(ticker, start="2020-01-01", end="2024-01-01")
            df = df.reset_index()

        if df.empty:
            st.error("No data found. Please try another ticker.")
        else:
            # Use actual datetime for x-axis but convert to ordinal for training
            df['Date_Ordinal'] = df['Date'].map(datetime.toordinal)
            x = df['Date_Ordinal'].values.reshape(-1, 1)
            y = df['Close'].values.reshape(-1, 1)

            # Handle NaNs
            y[np.isnan(y)] = np.median(y[~np.isnan(y)])

            # Split the data
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            # Train the model
            model = LinearRegression().fit(x_train, y_train)
            predictions = model.predict(x_test)

            # Calculate error metrics
            r2 = model.score(x_test, y_test)
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)

            # Display metrics
            st.success(f"✅ R² Score: {r2:.4f}")
            st.info(f"📌 MAE: {mae:.4f} USD")
            st.info(f"📌 MSE: {mse:.4f}")

            # Convert x_test to dates for plotting
            x_test_dates = [datetime.fromordinal(int(d)) for d in x_test.flatten()]

            # Plot actual vs predicted with readable dates
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(x_test_dates, y_test, color='green', s=10, label='Actual')
            ax.plot(x_test_dates, predictions, color='blue', linewidth=2, label='Predicted')
            ax.set_title(f"{ticker} Stock Price Prediction")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (USD)")
            ax.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # Show sample predictions
            sample_df = pd.DataFrame({
                'Date': x_test_dates,
                'Actual Price': y_test.flatten(),
                'Predicted Price': predictions.flatten()
            }).sort_values(by='Date')

            st.subheader("📊 Sample Predictions")
            st.write(sample_df.tail())

    except Exception as e:
        st.error(f"An error occurred: {e}")
