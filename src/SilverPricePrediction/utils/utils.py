import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.SilverPricePrediction.logger import logging
from src.SilverPricePrediction.exception import customexception

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def save_object(file_path, obj):
    """Save a Python object to a pickle file."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customexception(e, sys)


def load_object(file_path):
    """Load a Python object from a pickle file."""
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occurred in load_object function utils')
        raise customexception(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models):
    """
    Evaluate multiple models and return their R2 scores.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        models: Dictionary of model_name: model_object
    
    Returns:
        Dictionary of model_name: R2 score
    """
    try:
        report = {}
        for model_name, model in models.items():
            # Train model
            model.fit(X_train, y_train)

            # Predict on test data
            y_test_pred = model.predict(X_test)

            # Get R2 score for test data
            test_model_score = r2_score(y_test, y_test_pred)
            
            # Get additional metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            mae = mean_absolute_error(y_test, y_test_pred)
            
            logging.info(f"{model_name}: R2={test_model_score:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

            report[model_name] = test_model_score

        return report

    except Exception as e:
        logging.info('Exception occurred during model training')
        raise customexception(e, sys)


def create_features(df, target_col='Close', lags=[1, 2, 3, 5, 7, 14, 21]):
    """
    Create technical indicators and lag features for time-series prediction.
    
    Args:
        df: DataFrame with OHLCV data
        target_col: Target column name (default: 'Close')
        lags: List of lag periods
    
    Returns:
        DataFrame with additional features
    """
    df = df.copy()
    
    # Ensure we have the required columns
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    # Create lag features
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    # Moving Averages
    df['MA_5'] = df[target_col].rolling(window=5).mean()
    df['MA_10'] = df[target_col].rolling(window=10).mean()
    df['MA_20'] = df[target_col].rolling(window=20).mean()
    df['MA_50'] = df[target_col].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df['EMA_12'] = df[target_col].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df[target_col].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Price change and returns
    df['Price_Change'] = df[target_col].diff()
    df['Returns'] = df[target_col].pct_change()
    
    # Volatility (Rolling Standard Deviation)
    df['Volatility_5'] = df['Returns'].rolling(window=5).std()
    df['Volatility_20'] = df['Returns'].rolling(window=20).std()
    
    # RSI (Relative Strength Index)
    delta = df[target_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df[target_col].rolling(window=20).mean()
    df['BB_Std'] = df[target_col].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    # Price position relative to Bollinger Bands
    df['BB_Position'] = (df[target_col] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # High-Low Spread (if available)
    if 'High' in df.columns and 'Low' in df.columns:
        df['HL_Spread'] = df['High'] - df['Low']
        df['HL_Spread_Pct'] = df['HL_Spread'] / df[target_col]
    
    # Volume features (if available)
    if 'Volume' in df.columns:
        df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
    
    # Day of week and month (cyclical features)
    if isinstance(df.index, pd.DatetimeIndex):
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['DayOfMonth'] = df.index.day
        # Cyclical encoding
        df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    return df


def prepare_target(df, target_col='Close', horizon=1):
    """
    Create target variable for prediction (next day's price).
    
    Args:
        df: DataFrame with price data
        target_col: Column to predict
        horizon: Prediction horizon (default: 1 = next day)
    
    Returns:
        DataFrame with target column added
    """
    df = df.copy()
    df['Target'] = df[target_col].shift(-horizon)
    return df
