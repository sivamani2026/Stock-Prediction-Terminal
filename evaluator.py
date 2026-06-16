from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
from typing import Dict, Any

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes regression performance metrics: MAE, RMSE, R-squared.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }

def compute_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, y_prev: np.ndarray) -> float:
    """
    Computes Directional Accuracy (DA), which measures the percentage of times
    the model correctly predicted whether the price would go up or down.
    
    DA = Mean( sign(y_true - y_prev) == sign(y_pred - y_prev) )
    """
    if len(y_true) < 1 or len(y_prev) != len(y_true):
        return 0.0
        
    actual_direction = np.sign(y_true - y_prev)
    predicted_direction = np.sign(y_pred - y_prev)
    
    # 0 return represents no change. We treat matching signs (including zero) as correct.
    correct_direction = (actual_direction == predicted_direction)
    return float(np.mean(correct_direction))

def compare_models(model_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Consolidates performance metrics of multiple models into a single DataFrame.
    Expected structure of model_results:
    {
        'Linear Regression': {'MAE': 1.2, 'RMSE': 1.5, 'R2': 0.85, 'Directional Accuracy': 0.55},
        'Random Forest': ...
    }
    """
    df_compare = pd.DataFrame(model_results).T
    return df_compare
