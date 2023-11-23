from flask import Flask, render_template
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.metrics import mean_squared_error
import numpy as np
from prophet.plot import plot_plotly, plot_components_plotly

app = Flask(__name__,template_folder='template')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    # Load historical Bitcoin price data
    symbol = "BTC-USD"
    start_date = "2019-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')
    bitcoin_data = yf.download(symbol, start=start_date, end=end_date)

    # Prepare the data for Prophet
    df = pd.DataFrame()
    df['ds'] = bitcoin_data.index
    df['y'] = bitcoin_data['Close'].values

    # Feature Engineering
    df['returns'] = df['y'].pct_change()
    df['ma7'] = df['y'].rolling(window=7).mean()
    df['ma30'] = df['y'].rolling(window=30).mean()
    df['volatility'] = df['returns'].rolling(window=7).std()
    df = df.dropna()

    # Instantiate the Prophet model
    model3 = Prophet(changepoint_prior_scale=0.05, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)

    # Add additional regressors
    model3.add_regressor('ma7')
    model3.add_regressor('ma30')
    model3.add_regressor('volatility')

    # Fit the model
    model3.fit(df)

    # Create a dataframe for the next day
    future = model3.make_future_dataframe(periods=1, freq='D', include_history=False)

    # Add regressor values for the next day
    # Adjust these values based on the actual values for the next day
    future['ma7'] = df['ma7'].values[-1]
    future['ma30'] = df['ma30'].values[-1]
    future['volatility'] = df['volatility'].values[-1]

    # Make predictions for the next day
    forecast = model3.predict(future)

    # Extract the predicted closing price for the next day
    predicted_price = forecast['yhat'].iloc[0]

    # Get the actual closing price for the next day
    actual_price = yf.download(symbol, start=end_date, end=(datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d'))['Close'].iloc[0]

    return render_template('result.html', actual_price=actual_price, predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
