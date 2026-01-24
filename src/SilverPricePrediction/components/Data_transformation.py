import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from src.SilverPricePrediction.exception import customexception
from src.SilverPricePrediction.logger import logging

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.SilverPricePrediction.utils.utils import save_object, create_features, prepare_target


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('Artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
        # Define feature columns (will be set during transformation)
        self.feature_columns = None
        self.target_column = 'Target'
    
    def get_data_transformation(self, feature_columns):
        """
        Create preprocessing pipeline for numerical features.
        
        Silver price prediction uses only numerical features (price data, technical indicators).
        """
        try:
            logging.info('Creating data transformation pipeline')
            
            # All features are numerical for time-series data
            numerical_cols = feature_columns
            
            logging.info(f'Numerical columns: {numerical_cols}')
            logging.info('Building preprocessing pipeline')
            
            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            
            # Column Transformer (only numerical pipeline needed)
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols)
            ])
            
            logging.info('Preprocessing pipeline created successfully')
            
            return preprocessor
            
        except Exception as e:
            logging.error("Exception occurred in get_data_transformation")
            raise customexception(e, sys)
    
    def initialize_data_transformation(self, train_path, test_path, prediction_horizon=1):
        """
        Transform training and test data with feature engineering.
        
        Args:
            train_path: Path to training data CSV
            test_path: Path to test data CSV
            prediction_horizon: Number of days ahead to predict (default: 1)
        
        Returns:
            Tuple of (train_array, test_array)
        """
        try:
            # Read data
            train_df = pd.read_csv(train_path, parse_dates=['Date'])
            test_df = pd.read_csv(test_path, parse_dates=['Date'])
            
            logging.info("Read train and test data successfully")
            logging.info(f'Train DataFrame shape: {train_df.shape}')
            logging.info(f'Test DataFrame shape: {test_df.shape}')
            
            # Set Date as index for time-series operations
            train_df.set_index('Date', inplace=True)
            test_df.set_index('Date', inplace=True)
            
            # Create features for both datasets
            logging.info("Creating technical indicator features...")
            train_df = create_features(train_df, target_col='Close')
            test_df = create_features(test_df, target_col='Close')
            
            # Create target variable (next day's closing price)
            logging.info(f"Creating target variable with horizon={prediction_horizon}")
            train_df = prepare_target(train_df, target_col='Close', horizon=prediction_horizon)
            test_df = prepare_target(test_df, target_col='Close', horizon=prediction_horizon)
            
            # Drop rows with NaN (from feature creation and target shifting)
            initial_train_len = len(train_df)
            initial_test_len = len(test_df)
            
            train_df = train_df.dropna()
            test_df = test_df.dropna()
            
            logging.info(f"Dropped {initial_train_len - len(train_df)} NaN rows from training data")
            logging.info(f"Dropped {initial_test_len - len(test_df)} NaN rows from test data")
            
            logging.info(f'Train DataFrame Head:\n{train_df.head().to_string()}')
            logging.info(f'Test DataFrame Head:\n{test_df.head().to_string()}')
            
            # Define feature columns (all except Target)
            drop_columns = [self.target_column]
            
            # Exclude original OHLCV columns that are now represented by features
            # Keep Close as it's used in lag features, but we'll use lag_1 instead
            original_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            drop_columns.extend([col for col in original_cols if col in train_df.columns])
            
            # Also drop non-numeric columns if any
            drop_columns.extend(['DayOfWeek', 'Month', 'DayOfMonth'])
            drop_columns = [col for col in drop_columns if col in train_df.columns]
            
            self.feature_columns = [col for col in train_df.columns if col not in drop_columns]
            
            logging.info(f"Feature columns ({len(self.feature_columns)}): {self.feature_columns}")
            
            # Separate features and target
            input_feature_train_df = train_df[self.feature_columns]
            target_feature_train_df = train_df[self.target_column]
            
            input_feature_test_df = test_df[self.feature_columns]
            target_feature_test_df = test_df[self.target_column]
            
            # Get preprocessing object
            preprocessing_obj = self.get_data_transformation(self.feature_columns)
            
            # Fit and transform training data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            
            # Transform test data
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            logging.info("Applied preprocessing object on training and testing datasets")
            
            # Combine features and target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info(f"Final train array shape: {train_arr.shape}")
            logging.info(f"Final test array shape: {test_arr.shape}")
            
            # Save preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            # Also save feature columns for prediction pipeline
            feature_cols_path = os.path.join('Artifacts', 'feature_columns.pkl')
            save_object(file_path=feature_cols_path, obj=self.feature_columns)
            
            logging.info("Preprocessing pickle file saved")
            logging.info("Feature columns pickle file saved")
            
            return (train_arr, test_arr)
            
        except Exception as e:
            logging.error("Exception occurred in initialize_data_transformation")
            raise customexception(e, sys)


if __name__ == "__main__":
    # Test data transformation
    from src.SilverPricePrediction.components.Data_ingestion import DataIngestion
    
    # First run data ingestion
    ingestion = DataIngestion()
    test_path, train_path = ingestion.initiate_data_ingestion()
    
    # Then run transformation
    transformation = DataTransformation()
    train_arr, test_arr = transformation.initialize_data_transformation(train_path, test_path)
    
    print(f"Training array shape: {train_arr.shape}")
    print(f"Test array shape: {test_arr.shape}")
