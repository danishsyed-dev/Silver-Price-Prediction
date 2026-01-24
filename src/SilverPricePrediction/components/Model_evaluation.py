import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.SilverPricePrediction.utils.utils import load_object
from src.SilverPricePrediction.logger import logging
from src.SilverPricePrediction.exception import customexception


class ModelEvaluation:
    def __init__(self):
        self.metrics = {}
    
    def eval_metrics(self, actual, pred):
        """
        Calculate regression metrics.
        
        Returns:
            rmse, mae, r2, mape
        """
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((actual - pred) / actual)) * 100
        
        return rmse, mae, r2, mape

    def initiate_model_evaluation(self, train_array, test_array):
        """
        Evaluate the trained model on test data.
        
        Args:
            train_array: Training data (not used for evaluation, kept for compatibility)
            test_array: Test data (features + target as last column)
        """
        try:
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            model_path = os.path.join("Artifacts", "model.pkl")
            model = load_object(model_path)
            
            logging.info(f"Loaded model from {model_path}")
            logging.info(f"Model type: {type(model).__name__}")

            # Make predictions
            predicted_prices = model.predict(X_test)

            # Calculate metrics
            rmse, mae, r2, mape = self.eval_metrics(y_test, predicted_prices)
            
            self.metrics = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape
            }

            # Print evaluation results
            print("\n" + "="*60)
            print("MODEL EVALUATION RESULTS")
            print("="*60)
            print(f"📊 R² Score:                 {r2:.6f}")
            print(f"📉 Root Mean Squared Error:  ${rmse:.4f}")
            print(f"📉 Mean Absolute Error:      ${mae:.4f}")
            print(f"📉 Mean Abs Percentage Error: {mape:.2f}%")
            print("="*60)
            
            # Additional analysis
            print("\n📈 Prediction Analysis:")
            print(f"   Actual Price Range:    ${y_test.min():.2f} - ${y_test.max():.2f}")
            print(f"   Predicted Price Range: ${predicted_prices.min():.2f} - ${predicted_prices.max():.2f}")
            print(f"   Mean Actual Price:     ${y_test.mean():.2f}")
            print(f"   Mean Predicted Price:  ${predicted_prices.mean():.2f}")
            print("="*60 + "\n")

            logging.info(f"Evaluation metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, MAPE: {mape:.2f}%")
            
            return self.metrics
            
        except Exception as e:
            logging.error("Exception occurred during model evaluation")
            raise customexception(e, sys)
    
    def plot_predictions(self, test_array, save_path=None):
        """
        Create visualization of actual vs predicted prices.
        
        Args:
            test_array: Test data
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            model_path = os.path.join("Artifacts", "model.pkl")
            model = load_object(model_path)
            
            predicted_prices = model.predict(X_test)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Plot 1: Actual vs Predicted over time
            ax1 = axes[0, 0]
            ax1.plot(range(len(y_test)), y_test, label='Actual', alpha=0.7)
            ax1.plot(range(len(predicted_prices)), predicted_prices, label='Predicted', alpha=0.7)
            ax1.set_xlabel('Time Index')
            ax1.set_ylabel('Silver Price ($)')
            ax1.set_title('Actual vs Predicted Silver Prices Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Scatter plot
            ax2 = axes[0, 1]
            ax2.scatter(y_test, predicted_prices, alpha=0.5, edgecolors='none')
            ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax2.set_xlabel('Actual Price ($)')
            ax2.set_ylabel('Predicted Price ($)')
            ax2.set_title('Actual vs Predicted (Scatter)')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Residuals
            ax3 = axes[1, 0]
            residuals = y_test - predicted_prices
            ax3.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
            ax3.axvline(x=0, color='r', linestyle='--')
            ax3.set_xlabel('Residual (Actual - Predicted) ($)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Residual Distribution')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Residuals over time
            ax4 = axes[1, 1]
            ax4.plot(range(len(residuals)), residuals, alpha=0.7)
            ax4.axhline(y=0, color='r', linestyle='--')
            ax4.set_xlabel('Time Index')
            ax4.set_ylabel('Residual ($)')
            ax4.set_title('Residuals Over Time')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logging.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logging.warning("Matplotlib not available for plotting")
        except Exception as e:
            logging.error(f"Error creating plot: {e}")


if __name__ == "__main__":
    # Test model evaluation
    from src.SilverPricePrediction.components.Data_ingestion import DataIngestion
    from src.SilverPricePrediction.components.Data_transformation import DataTransformation
    from src.SilverPricePrediction.components.Model_trainer import ModelTrainer
    
    # Run full pipeline
    ingestion = DataIngestion()
    test_path, train_path = ingestion.initiate_data_ingestion()
    
    transformation = DataTransformation()
    train_arr, test_arr = transformation.initialize_data_transformation(train_path, test_path)
    
    trainer = ModelTrainer()
    trainer.initiate_model_training(train_arr, test_arr)
    
    # Evaluate
    evaluator = ModelEvaluation()
    metrics = evaluator.initiate_model_evaluation(train_arr, test_arr)
    
    print(f"\nFinal Metrics: {metrics}")
