"""
Prediction Pipeline for Silver Price Prediction - Indian Market

This module provides:
1. PredictPipeline - Load model and preprocessor for predictions
2. CustomData - Handle user input data (for API/web integration)
3. SilverDataFetcher - Fetch latest data for predictions
4. IndianMarketConverter - Convert USD prices to INR with GST calculations
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.SilverPricePrediction.exception import customexception
from src.SilverPricePrediction.logger import logging
from src.SilverPricePrediction.utils.utils import load_object, create_features


# Indian Market Constants
GST_RATE = 0.03  # 3% GST on silver in India
TROY_OUNCE_TO_GRAMS = 31.1035  # 1 troy ounce = 31.1035 grams

# Import Duty and Premium adjustments (Updated Jan 2026)
# Silver Import Duty in India: 6% (Basic Customs Duty + AIDC)
# Reference: India Budget 2024-25 reduced import duty on silver
IMPORT_DUTY_RATE = 0.06  # 6% import duty

# Local premium includes: dealer margins, logistics, storage, refining, 
# MCX vs COMEX spread, and other market factors
# Calibrated to match actual Indian market prices (~₹3,350-3,500/10g range)
# Note: Despite high USD/INR (91.5), market remains competitive
LOCAL_PREMIUM_RATE = 0.10  # ~10% local premium


class IndianMarketConverter:
    """
    Convert silver prices to Indian market standards.
    
    Indian silver prices include:
    1. Base international price (SI=F in USD)
    2. USD to INR conversion
    3. Import Duty (~12.5%)
    4. Local Premium (~5% for logistics, dealer margins)
    5. GST (3%)
    """
    
    def __init__(self):
        self.usd_to_inr_rate = None
        self.last_rate_fetch = None
    
    def get_usd_to_inr_rate(self):
        """
        Fetch current USD to INR exchange rate.
        Uses Yahoo Finance for the exchange rate.
        """
        try:
            import yfinance as yf
            
            # Check if we have a recent rate (within 1 hour)
            if self.usd_to_inr_rate and self.last_rate_fetch:
                time_diff = datetime.now() - self.last_rate_fetch
                if time_diff.total_seconds() < 3600:  # 1 hour
                    return self.usd_to_inr_rate
            
            # Fetch fresh rate
            ticker = yf.Ticker("USDINR=X")
            data = ticker.history(period="1d")
            
            if not data.empty:
                self.usd_to_inr_rate = data['Close'].iloc[-1]
                self.last_rate_fetch = datetime.now()
                logging.info(f"USD to INR rate: {self.usd_to_inr_rate:.2f}")
                return self.usd_to_inr_rate
            else:
                # Fallback rate if API fails
                logging.warning("Could not fetch exchange rate, using fallback rate of 83")
                return 83.0
                
        except Exception as e:
            logging.error(f"Error fetching exchange rate: {e}")
            return 83.0  # Fallback rate
    
    def convert_to_indian_market(self, usd_price_per_ounce):
        """
        Convert USD per troy ounce to Indian market prices.
        
        Calculation:
        1. Convert USD to INR
        2. Convert per ounce to per gram
        3. Add Import Duty (12.5%)
        4. Add Local Premium (5%)
        5. Add GST (3%)
        
        Args:
            usd_price_per_ounce: Silver price in USD per troy ounce (COMEX price)
        
        Returns:
            Dictionary with Indian market prices
        """
        try:
            usd_to_inr = self.get_usd_to_inr_rate()
            
            # Step 1: Convert to INR per troy ounce
            inr_per_ounce_base = usd_price_per_ounce * usd_to_inr
            
            # Step 2: Convert to INR per gram
            inr_per_gram_base = inr_per_ounce_base / TROY_OUNCE_TO_GRAMS
            
            # Step 3: Add Import Duty (12.5%)
            inr_per_gram_with_duty = inr_per_gram_base * (1 + IMPORT_DUTY_RATE)
            
            # Step 4: Add Local Premium (5%)
            inr_per_gram_landed = inr_per_gram_with_duty * (1 + LOCAL_PREMIUM_RATE)
            
            # This is the base price WITHOUT GST (what jewellers quote as "without GST")
            inr_per_gram = inr_per_gram_landed
            inr_per_10_grams = inr_per_gram * 10
            inr_per_kg = inr_per_gram * 1000
            
            # Step 5: Calculate GST (3%)
            gst_per_gram = inr_per_gram * GST_RATE
            gst_per_10_grams = inr_per_10_grams * GST_RATE
            gst_per_kg = inr_per_kg * GST_RATE
            
            # Price with GST (final consumer price)
            inr_per_gram_with_gst = inr_per_gram + gst_per_gram
            inr_per_10_grams_with_gst = inr_per_10_grams + gst_per_10_grams
            inr_per_kg_with_gst = inr_per_kg + gst_per_kg
            
            # Log the calculation breakdown
            logging.info(f"Price Conversion: USD ${usd_price_per_ounce:.2f}/oz -> "
                        f"INR ₹{inr_per_10_grams:.0f}/10g (without GST), "
                        f"₹{inr_per_10_grams_with_gst:.0f}/10g (with GST)")
            
            return {
                'usd_price': round(usd_price_per_ounce, 2),
                'usd_to_inr_rate': round(usd_to_inr, 2),
                
                # Prices without GST (jeweller base price)
                'inr_per_gram': round(inr_per_gram, 2),
                'inr_per_10_grams': round(inr_per_10_grams, 2),
                'inr_per_kg': round(inr_per_kg, 2),
                
                # GST amounts
                'gst_rate': GST_RATE * 100,  # 3%
                'gst_per_gram': round(gst_per_gram, 2),
                'gst_per_10_grams': round(gst_per_10_grams, 2),
                'gst_per_kg': round(gst_per_kg, 2),
                
                # Prices with GST (final consumer price)
                'inr_per_gram_with_gst': round(inr_per_gram_with_gst, 2),
                'inr_per_10_grams_with_gst': round(inr_per_10_grams_with_gst, 2),
                'inr_per_kg_with_gst': round(inr_per_kg_with_gst, 2),
                
                # Additional info
                'import_duty_rate': IMPORT_DUTY_RATE * 100,  # 12.5%
                'local_premium_rate': LOCAL_PREMIUM_RATE * 100,  # 5%
            }
            
        except Exception as e:
            logging.error(f"Error converting to Indian market: {e}")
            raise customexception(e, sys)




class PredictPipeline:
    """
    Pipeline for making silver price predictions using trained model.
    """
    def __init__(self):
        self.preprocessor_path = os.path.join("Artifacts", "preprocessor.pkl")
        self.model_path = os.path.join("Artifacts", "model.pkl")
        self.feature_columns_path = os.path.join("Artifacts", "feature_columns.pkl")
        
        self.preprocessor = None
        self.model = None
        self.feature_columns = None
    
    def load_artifacts(self):
        """Load trained model and preprocessor."""
        try:
            if self.preprocessor is None:
                self.preprocessor = load_object(self.preprocessor_path)
                logging.info("Preprocessor loaded successfully")
            
            if self.model is None:
                self.model = load_object(self.model_path)
                logging.info(f"Model loaded successfully: {type(self.model).__name__}")
            
            if self.feature_columns is None:
                self.feature_columns = load_object(self.feature_columns_path)
                logging.info(f"Feature columns loaded: {len(self.feature_columns)} features")
                
        except Exception as e:
            logging.error("Error loading model artifacts")
            raise customexception(e, sys)
    
    def predict(self, features):
        """
        Make prediction using preprocessed features.
        
        Args:
            features: DataFrame with feature columns
        
        Returns:
            Predicted silver price
        """
        try:
            self.load_artifacts()
            
            # Ensure features are in correct order
            if isinstance(features, pd.DataFrame):
                features = features[self.feature_columns]
            
            # Scale features
            scaled_data = self.preprocessor.transform(features)
            
            # Make prediction
            pred = self.model.predict(scaled_data)
            
            logging.info(f"Prediction made: ${pred[0]:.2f}")
            
            return pred
            
        except Exception as e:
            logging.error("Error during prediction")
            raise customexception(e, sys)
    
    def predict_from_history(self, historical_data):
        """
        Make prediction from historical OHLCV data.
        Automatically creates features and makes prediction.
        
        Args:
            historical_data: DataFrame with columns ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                            Must have at least 50 rows for feature calculation
        
        Returns:
            Predicted next day's silver price
        """
        try:
            self.load_artifacts()
            
            # Set Date as index
            if 'Date' in historical_data.columns:
                historical_data = historical_data.set_index('Date')
            
            # Create features
            df_with_features = create_features(historical_data, target_col='Close')
            
            # Get the latest complete row (after feature creation)
            latest_features = df_with_features.dropna().iloc[[-1]]
            
            # Select only the required feature columns
            original_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                           'DayOfWeek', 'Month', 'DayOfMonth']
            drop_cols = [col for col in original_cols if col in latest_features.columns]
            features_for_prediction = latest_features.drop(columns=drop_cols, errors='ignore')
            
            # Ensure we have all required features
            features_for_prediction = features_for_prediction[self.feature_columns]
            
            # Make prediction
            return self.predict(features_for_prediction)
            
        except Exception as e:
            logging.error("Error during prediction from history")
            raise customexception(e, sys)


class SilverDataFetcher:
    """
    Fetch latest silver price data for making predictions.
    Uses multiple methods to ensure data availability on cloud platforms.
    """
    def __init__(self, symbol="SI=F"):
        self.symbol = symbol
        self.fallback_data_path = os.path.join("Artifacts", "raw_data.csv")
        # Alternative symbols to try
        self.alt_symbols = ["SI=F", "SIL", "XAGUSD=X"]
    
    def _fetch_with_yfinance(self, symbol, period="1d"):
        """Try fetching with yfinance."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            if not data.empty:
                return data
        except Exception as e:
            logging.warning(f"yfinance failed for {symbol}: {e}")
        return None
    
    def _fetch_with_download(self, symbol, period="1d"):
        """Try fetching with yfinance download method."""
        try:
            import yfinance as yf
            data = yf.download(symbol, period=period, progress=False)
            if not data.empty:
                return data
        except Exception as e:
            logging.warning(f"yfinance download failed for {symbol}: {e}")
        return None
    
    def _get_fallback_data(self):
        """Load fallback data from local CSV file."""
        try:
            if os.path.exists(self.fallback_data_path):
                data = pd.read_csv(self.fallback_data_path)
                logging.info(f"Loaded fallback data: {len(data)} rows")
                return data
            else:
                logging.warning(f"Fallback data not found at {self.fallback_data_path}")
                return None
        except Exception as e:
            logging.error(f"Error loading fallback data: {e}")
            return None
    
    def fetch_latest_data(self, days=100):
        """
        Fetch recent silver price data.
        Tries multiple methods and symbols.
        
        Args:
            days: Number of historical days to fetch (need ~50+ for features)
        
        Returns:
            DataFrame with OHLCV data
        """
        # Try multiple symbols with yfinance
        for symbol in self.alt_symbols:
            logging.info(f"Trying to fetch data with symbol: {symbol}")
            
            # Method 1: yfinance Ticker
            data = self._fetch_with_yfinance(symbol, period=f"{days}d")
            if data is not None and not data.empty:
                data = data.reset_index()
                # Handle different column structures
                if 'Datetime' in data.columns:
                    data = data.rename(columns={'Datetime': 'Date'})
                cols = [c for c in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'] if c in data.columns]
                data = data[cols]
                logging.info(f"Successfully fetched {len(data)} days with {symbol} (Ticker method)")
                return data
            
            # Method 2: yfinance download
            data = self._fetch_with_download(symbol, period=f"{days}d")
            if data is not None and not data.empty:
                data = data.reset_index()
                if 'Datetime' in data.columns:
                    data = data.rename(columns={'Datetime': 'Date'})
                cols = [c for c in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'] if c in data.columns]
                data = data[cols]
                logging.info(f"Successfully fetched {len(data)} days with {symbol} (download method)")
                return data
        
        # Fallback to local data
        logging.warning("All live data sources failed, using fallback CSV")
        fallback = self._get_fallback_data()
        if fallback is not None:
            return fallback.tail(days)
        
        raise ValueError("Could not fetch data from any source")
    
    def get_current_price(self):
        """
        Get the current silver price.
        Tries multiple methods to ensure availability.
        """
        # Try multiple symbols
        for symbol in self.alt_symbols:
            logging.info(f"Trying to get current price with symbol: {symbol}")
            
            # Method 1: yfinance Ticker
            data = self._fetch_with_yfinance(symbol, period="5d")
            if data is not None and not data.empty:
                price = data['Close'].iloc[-1]
                logging.info(f"Live price from {symbol}: ${price:.2f}")
                return float(price)
            
            # Method 2: yfinance download
            data = self._fetch_with_download(symbol, period="5d")
            if data is not None and not data.empty:
                price = data['Close'].iloc[-1]
                logging.info(f"Live price from {symbol} (download): ${price:.2f}")
                return float(price)
        
        # Final fallback - use local data (with warning)
        logging.error("WARNING: Using fallback price - all live sources failed!")
        fallback = self._get_fallback_data()
        if fallback is not None and 'Close' in fallback.columns:
            price = fallback['Close'].iloc[-1]
            logging.warning(f"Fallback price (may be outdated): ${price:.2f}")
            return float(price)
        
        return None


class CustomData:
    """
    Handle custom input data for web form predictions.
    Allows manual input of recent price data for prediction.
    """
    def __init__(self,
                 current_price: float,
                 prev_price_1: float,    # Yesterday's price
                 prev_price_2: float,    # 2 days ago
                 prev_price_3: float,    # 3 days ago
                 prev_price_5: float,    # 5 days ago
                 prev_price_7: float,    # 7 days ago
                 volume: float = 0):
        """
        Initialize with recent prices for feature calculation.
        """
        self.current_price = current_price
        self.prev_price_1 = prev_price_1
        self.prev_price_2 = prev_price_2
        self.prev_price_3 = prev_price_3
        self.prev_price_5 = prev_price_5
        self.prev_price_7 = prev_price_7
        self.volume = volume
    
    def get_data_as_dataframe(self):
        """
        Create a DataFrame with manually calculated features.
        
        This is a simplified version - for production, use historical OHLCV data.
        """
        try:
            # Calculate basic features manually
            prices = [self.current_price, self.prev_price_1, self.prev_price_2, 
                     self.prev_price_3, self.prev_price_5, self.prev_price_7]
            
            custom_data_input_dict = {
                'lag_1': [self.prev_price_1],
                'lag_2': [self.prev_price_2],
                'lag_3': [self.prev_price_3],
                'lag_5': [self.prev_price_5],
                'lag_7': [self.prev_price_7],
                'MA_5': [np.mean(prices[:5])],
                'Price_Change': [self.current_price - self.prev_price_1],
                'Returns': [(self.current_price - self.prev_price_1) / self.prev_price_1] if self.prev_price_1 != 0 else [0],
            }
            
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Custom DataFrame Created')
            return df
            
        except Exception as e:
            logging.error('Exception Occurred in CustomData')
            raise customexception(e, sys)


def make_prediction():
    """
    Quick function to make a prediction using latest market data.
    """
    try:
        # Fetch latest data
        fetcher = SilverDataFetcher()
        latest_data = fetcher.fetch_latest_data(days=100)
        current_price = fetcher.get_current_price()
        
        # Make prediction
        pipeline = PredictPipeline()
        predicted_price = pipeline.predict_from_history(latest_data)
        
        print("\n" + "="*50)
        print(" SILVER PRICE PREDICTION")
        print("="*50)
        print(f" Current Price: ${current_price:.2f}")
        print(f" Predicted Next Day Price: ${predicted_price[0]:.2f}")
        
        price_change = predicted_price[0] - current_price
        pct_change = (price_change / current_price) * 100
        
        if price_change > 0:
            print(f" Expected Change: +${price_change:.2f} (+{pct_change:.2f}%)")
        else:
            print(f" Expected Change: ${price_change:.2f} ({pct_change:.2f}%)")
        print("="*50 + "\n")
        
        return predicted_price[0], current_price
        
    except Exception as e:
        logging.error(f"Error in make_prediction: {e}")
        raise


if __name__ == "__main__":
    make_prediction()
