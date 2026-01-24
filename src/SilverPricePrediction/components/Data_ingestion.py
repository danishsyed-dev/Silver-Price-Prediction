import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from src.SilverPricePrediction.logger import logging
from src.SilverPricePrediction.exception import customexception
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
    """Configuration for data paths."""
    raw_data_path: str = os.path.join("Artifacts", "raw_data.csv")
    train_data_path: str = os.path.join("Artifacts", "train_data.csv")
    test_data_path: str = os.path.join("Artifacts", "test_data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def fetch_silver_data(self, 
                          symbol: str = "SI=F",  # Silver Futures
                          start_date: str = None, 
                          end_date: str = None,
                          period: str = "5y") -> pd.DataFrame:
        """
        Fetch silver price data from Yahoo Finance.
        
        Args:
            symbol: Yahoo Finance symbol for silver
                    "SI=F" - Silver Futures
                    "SLV" - iShares Silver Trust ETF
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            period: If dates not specified, use period (e.g., "1y", "5y", "max")
        
        Returns:
            DataFrame with OHLCV data
        """
        logging.info(f"Fetching silver data for symbol: {symbol}")
        
        try:
            ticker = yf.Ticker(symbol)
            
            if start_date and end_date:
                data = ticker.history(start=start_date, end=end_date)
            else:
                data = ticker.history(period=period)
            
            if data.empty:
                raise ValueError(f"No data retrieved for symbol {symbol}")
            
            logging.info(f"Retrieved {len(data)} records from {data.index[0]} to {data.index[-1]}")
            
            return data
            
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            raise customexception(e, sys)

    def initiate_data_ingestion(self, 
                                 use_existing: bool = False,
                                 existing_file: str = None) -> tuple:
        """
        Main data ingestion method.
        
        Args:
            use_existing: If True, use existing CSV file instead of fetching
            existing_file: Path to existing CSV file
        
        Returns:
            Tuple of (test_data_path, train_data_path)
        """
        logging.info("Data ingestion started")
        
        try:
            # Create directories
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            if use_existing and existing_file:
                logging.info(f"Reading existing data from {existing_file}")
                data = pd.read_csv(existing_file, index_col=0, parse_dates=True)
            else:
                # Fetch fresh data from Yahoo Finance
                data = self.fetch_silver_data(
                    symbol="SI=F",  # Silver Futures
                    period="5y"     # Last 5 years
                )
            
            # Reset index to have Date as a column
            data = data.reset_index()
            data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
            
            # Keep only relevant columns
            data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Save raw data
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Saved raw data to {self.ingestion_config.raw_data_path}")
            logging.info(f"Data shape: {data.shape}")
            logging.info(f"Data columns: {data.columns.tolist()}")
            logging.info(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
            
            # For time-series, we use chronological split (not random!)
            # 80% training, 20% testing - maintaining temporal order
            split_idx = int(len(data) * 0.8)
            train_data = data.iloc[:split_idx]
            test_data = data.iloc[split_idx:]
            
            logging.info(f"Train data: {len(train_data)} samples ({train_data['Date'].min()} to {train_data['Date'].max()})")
            logging.info(f"Test data: {len(test_data)} samples ({test_data['Date'].min()} to {test_data['Date'].max()})")
            
            # Save train and test data
            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Created train and test data files")
            logging.info("Data ingestion completed successfully")
            
            return (
                self.ingestion_config.test_data_path,
                self.ingestion_config.train_data_path
            )
            
        except Exception as e:
            logging.error("Exception occurred while ingesting data")
            raise customexception(e, sys)


if __name__ == "__main__":
    # Test data ingestion
    obj = DataIngestion()
    test_path, train_path = obj.initiate_data_ingestion()
    print(f"Training data saved to: {train_path}")
    print(f"Test data saved to: {test_path}")
