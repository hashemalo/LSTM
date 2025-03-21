#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow info and warnings

import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import io
import contextlib

def load_data(csv_file):
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print("Error reading CSV file:", e, file=sys.stderr)
        sys.exit(1)
    required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            print(f"CSV file missing required column: {col}", file=sys.stderr)
            sys.exit(1)
    try:
        df['date'] = pd.to_datetime(df['date'])
    except Exception as e:
        print("Error parsing dates:", e, file=sys.stderr)
        sys.exit(1)
    df = df.sort_values('date')
    df.reset_index(drop=True, inplace=True)
    return df

def preprocess_data(df, sequence_length=60, forecast_horizon=5):
    df['daily_return'] = (df['close'] - df['open']) / df['open']
    df['volatility'] = (df['high'] - df['low']) / df['open']
    df['ma10'] = df['close'].rolling(window=10).mean()
    df = df.dropna()  

    features = ['open', 'high', 'low', 'close', 'volume', 'daily_return', 'volatility', 'ma10']
    data = df[features].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []

    for i in range(sequence_length, len(scaled_data) - forecast_horizon + 1):
        X.append(scaled_data[i - sequence_length:i])
        y.append(scaled_data[i:i + forecast_horizon, 3])
    X = np.array(X)
    y = np.array(y)
    return X, y, scaler, scaled_data, df

def build_model(input_shape, forecast_horizon):
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=32))
    model.add(Dropout(0.2))
    model.add(Dense(units=forecast_horizon))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    return model


def forecast_future(model, scaled_data, scaler, sequence_length, forecast_horizon):
    last_sequence = scaled_data[-sequence_length:]
    with io.StringIO() as buf, contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        prediction = model.predict(last_sequence.reshape(1, sequence_length, last_sequence.shape[1]), verbose=0)

    num_features = scaled_data.shape[1]
    dummy = np.zeros((forecast_horizon, num_features))
    dummy[:, 3] = prediction[0]  
    inverted = scaler.inverse_transform(dummy)
    predicted_close = inverted[:, 3]
    return predicted_close

def generate_forecast_json(future_dates, predictions, df):
    forecast_data = [
        {"date": d.strftime('%Y-%m-%d'), "predicted_close": float(pred.item())}
        for d, pred in zip(future_dates, predictions)
    ]
    # Generate historical data based on CSV columns: date, open, high, low, close, volume.
    # Ensure the date is formatted as a string.
    historical_df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
    historical_df['date'] = historical_df['date'].apply(
        lambda d: d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else d
    )
    historical_data = historical_df.to_dict(orient='records')
    return json.dumps({"historical": historical_data, "forecast": forecast_data}, indent=2)

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <csv_file>")
        sys.exit(1)
    csv_file = sys.argv[1]
    df = load_data(csv_file)
    sequence_length = 60
    forecast_horizon = 5

    X, y, scaler, scaled_data, df = preprocess_data(df, sequence_length, forecast_horizon)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = build_model((X_train.shape[1], X_train.shape[2]), forecast_horizon)

    with io.StringIO() as buf, contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    predictions = forecast_future(model, scaled_data, scaler, sequence_length, forecast_horizon)

    last_date = df['date'].iloc[-1]
    future_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(forecast_horizon)]

    print(generate_forecast_json(future_dates, predictions, df))
    

if __name__ == '__main__':
    main()
