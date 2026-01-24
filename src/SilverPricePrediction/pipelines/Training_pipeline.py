"""
Training Pipeline for Silver Price Prediction

This script orchestrates the complete training workflow:
1. Data Ingestion - Fetch silver price data
2. Data Transformation - Feature engineering and preprocessing
3. Model Training - Train multiple models and select best
4. Model Evaluation - Evaluate final model performance
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.SilverPricePrediction.components.Data_ingestion import DataIngestion
from src.SilverPricePrediction.components.Data_transformation import DataTransformation
from src.SilverPricePrediction.components.Model_trainer import ModelTrainer
from src.SilverPricePrediction.components.Model_evaluation import ModelEvaluation
from src.SilverPricePrediction.logger import logging


def run_training_pipeline():
    """Execute the complete training pipeline."""
    
    print("\n" + "="*70)
    print("🚀 SILVER PRICE PREDICTION - TRAINING PIPELINE")
    print("="*70 + "\n")
    
    # Step 1: Data Ingestion
    print("📥 Step 1: Data Ingestion")
    print("-" * 40)
    logging.info("Starting Data Ingestion")
    
    data_ingestion = DataIngestion()
    test_data_path, train_data_path = data_ingestion.initiate_data_ingestion()
    
    print(f"   ✅ Training data: {train_data_path}")
    print(f"   ✅ Test data: {test_data_path}")
    print()
    
    # Step 2: Data Transformation
    print("🔧 Step 2: Data Transformation & Feature Engineering")
    print("-" * 40)
    logging.info("Starting Data Transformation")
    
    data_transformation = DataTransformation()
    train_arr, test_arr = data_transformation.initialize_data_transformation(
        train_data_path, 
        test_data_path
    )
    
    print(f"   ✅ Training samples: {train_arr.shape[0]}")
    print(f"   ✅ Features created: {train_arr.shape[1] - 1}")
    print(f"   ✅ Test samples: {test_arr.shape[0]}")
    print()
    
    # Step 3: Model Training
    print("🤖 Step 3: Model Training & Selection")
    print("-" * 40)
    logging.info("Starting Model Training")
    
    model_trainer = ModelTrainer()
    best_model_name, best_score, model_report = model_trainer.initiate_model_training(
        train_arr, 
        test_arr
    )
    
    # Step 4: Model Evaluation
    print("📊 Step 4: Model Evaluation")
    print("-" * 40)
    logging.info("Starting Model Evaluation")
    
    model_evaluation = ModelEvaluation()
    metrics = model_evaluation.initiate_model_evaluation(train_arr, test_arr)
    
    # Summary
    print("\n" + "="*70)
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\n📁 Artifacts saved in: Artifacts/")
    print(f"   - model.pkl")
    print(f"   - preprocessor.pkl")
    print(f"   - feature_columns.pkl")
    print(f"   - raw_data.csv")
    print(f"   - train_data.csv")
    print(f"   - test_data.csv")
    print("\n" + "="*70 + "\n")
    
    return best_model_name, best_score, metrics


if __name__ == "__main__":
    run_training_pipeline()
