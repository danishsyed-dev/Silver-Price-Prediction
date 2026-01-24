import pandas as pd
import numpy as np
import os
import sys
from src.SilverPricePrediction.logger import logging
from src.SilverPricePrediction.exception import customexception
from dataclasses import dataclass
from src.SilverPricePrediction.utils.utils import save_object, evaluate_model

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# Try to import XGBoost and LightGBM (optional dependencies)
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.info("XGBoost not available. Install with: pip install xgboost")

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.info("LightGBM not available. Install with: pip install lightgbm")


@dataclass 
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('Artifacts', 'model.pkl')
    

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def get_models(self):
        """
        Returns a dictionary of models to evaluate.
        Includes both linear and ensemble methods suitable for time-series regression.
        """
        models = {
            # Linear Models
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.01),
            'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5),
            
            # Tree-based Models
            'DecisionTree': DecisionTreeRegressor(max_depth=10, random_state=42),
            'RandomForest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        return models
    
    def initiate_model_training(self, train_array, test_array):
        """
        Train multiple models and select the best one based on R2 score.
        
        Args:
            train_array: Training data (features + target as last column)
            test_array: Test data (features + target as last column)
        """
        try:
            logging.info('Splitting features and target from train and test data')
            
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            logging.info(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
            logging.info(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

            models = self.get_models()
            logging.info(f"Models to evaluate: {list(models.keys())}")
            
            # Evaluate all models
            model_report = evaluate_model(X_train, y_train, X_test, y_test, models)
            
            print("\n" + "="*80)
            print("MODEL EVALUATION REPORT")
            print("="*80)
            for model_name, score in sorted(model_report.items(), key=lambda x: x[1], reverse=True):
                print(f"{model_name:20} | R² Score: {score:.6f}")
            print("="*80 + "\n")
            
            logging.info(f'Model Report: {model_report}')

            # Get best model
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            print(f'🏆 Best Model: {best_model_name}')
            print(f'   R² Score: {best_model_score:.6f}')
            print("="*80 + "\n")
            
            logging.info(f'Best Model Found - Name: {best_model_name}, R² Score: {best_model_score}')

            # Check if model performance is acceptable
            if best_model_score < 0.5:
                logging.warning(f"Best model R² score ({best_model_score:.4f}) is below 0.5. Consider feature engineering or hyperparameter tuning.")
            
            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Best model saved to {self.model_trainer_config.trained_model_file_path}")
            
            return best_model_name, best_model_score, model_report

        except Exception as e:
            logging.error('Exception occurred during model training')
            raise customexception(e, sys)


if __name__ == "__main__":
    # Test model training
    from src.SilverPricePrediction.components.Data_ingestion import DataIngestion
    from src.SilverPricePrediction.components.Data_transformation import DataTransformation
    
    # Data ingestion
    ingestion = DataIngestion()
    test_path, train_path = ingestion.initiate_data_ingestion()
    
    # Data transformation
    transformation = DataTransformation()
    train_arr, test_arr = transformation.initialize_data_transformation(train_path, test_path)
    
    # Model training
    trainer = ModelTrainer()
    best_model, best_score, report = trainer.initiate_model_training(train_arr, test_arr)
    
    print(f"\nBest Model: {best_model} with R² = {best_score:.4f}")
