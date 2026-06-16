import argparse
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

from src.data_loader import fetch_stock_data
from src.preprocessor import add_technical_indicators, prepare_ml_data, prepare_lstm_data
from src.models import train_linear_regression, train_random_forest, train_lstm_model, HAS_TENSORFLOW
from src.evaluator import compute_metrics, compute_directional_accuracy, compare_models
from src.visualizer import plot_actual_vs_predicted_static

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Stock Price Prediction using Machine Learning")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol (e.g. AAPL, MSFT, TSLA)")
    parser.add_argument("--start", type=str, default=(datetime.now() - timedelta(days=365*2)).strftime("%Y-%m-%d"), 
                        help="Start date for fetching data (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=datetime.now().strftime("%Y-%m-%d"), 
                        help="End date for fetching data (YYYY-MM-DD)")
    parser.add_argument("--lag", type=str, default="5", help="Number of lag days for ML features")
    parser.add_argument("--lstm-steps", type=str, default="10", help="Number of time steps for LSTM sequences")
    parser.add_argument("--epochs", type=str, default="15", help="Number of training epochs for LSTM")
    parser.add_argument("--batch-size", type=str, default="32", help="Batch size for LSTM training")
    parser.add_argument("--test-ratio", type=str, default="0.2", help="Train-test split ratio (0.0 to 1.0)")
    parser.add_argument("--plot-path", type=str, default="predictions_comparison.png", help="Path to save the output comparison plot")
    
    args = parser.parse_args()
    
    # Parse numbers from args
    lag = int(args.lag)
    lstm_steps = int(args.lstm_steps)
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    test_ratio = float(args.test_ratio)
    
    logger.info("=== Stock Price Prediction Pipeline ===")
    logger.info(f"Ticker: {args.ticker}")
    logger.info(f"Date Range: {args.start} to {args.end}")
    
    # 1. Fetch data
    df = fetch_stock_data(args.ticker, args.start, args.end)
    logger.info(f"Loaded dataset with {len(df)} days of stock history.")
    
    # 2. Add indicators
    df_indicators = add_technical_indicators(df)
    
    # 3. Prepare data for classical ML models (LR, RF)
    X_train_ml, X_test_ml, y_train_ml, y_test_ml, dates_train_ml, dates_test_ml = prepare_ml_data(
        df_indicators, lag_days=lag, test_ratio=test_ratio
    )
    
    # 4. Prepare data for LSTM model
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, scaler, dates_test_lstm = prepare_lstm_data(
        df_indicators, time_steps=lstm_steps, test_ratio=test_ratio
    )
    
    # 5. Train and predict with Linear Regression
    model_lr = train_linear_regression(X_train_ml, y_train_ml)
    y_pred_lr = model_lr.predict(X_test_ml)
    
    # 6. Train and predict with Random Forest
    model_rf = train_random_forest(X_train_ml, y_train_ml)
    y_pred_rf = model_rf.predict(X_test_ml)
    
    # 7. Train and predict with LSTM (if available)
    lstm_available = HAS_TENSORFLOW
    y_pred_lstm = None
    if lstm_available:
        try:
            model_lstm = train_lstm_model(X_train_lstm, y_train_lstm, epochs=epochs, batch_size=batch_size)
            # Predict
            y_pred_lstm_scaled = model_lstm.predict(X_test_lstm, verbose=0)
            # Inverse scale predictions
            y_pred_lstm = scaler.inverse_transform(y_pred_lstm_scaled).flatten()
        except Exception as e:
            logger.error(f"Error training/predicting with LSTM: {e}")
            lstm_available = False
            
    # 8. Align predictions on common dates for fair evaluation
    dates_ml_series = pd.Index(dates_test_ml)
    dates_lstm_series = pd.Index(dates_test_lstm)
    common_dates = dates_ml_series.intersection(dates_lstm_series)
    
    if len(common_dates) == 0:
        logger.error("Error: No overlapping test dates to align model predictions.")
        return
        
    logger.info(f"Aligning evaluation on {len(common_dates)} test days.")
    
    # Convert predictions to Series with date index for easy alignment
    lr_series = pd.Series(y_pred_lr, index=dates_test_ml).loc[common_dates]
    rf_series = pd.Series(y_pred_rf, index=dates_test_ml).loc[common_dates]
    actual_series = pd.Series(y_test_ml, index=dates_test_ml).loc[common_dates]
    
    # Get y_prev for directional accuracy (actual price on previous business day)
    # y_prev is close price shifted by 1 relative to y_true
    # Let's map back to the indicators df to get actual close of previous day
    y_prev_list = []
    for date in common_dates:
        # Find index in original df and get previous row's Close
        idx = df_indicators.index.get_loc(date)
        y_prev_list.append(df_indicators.iloc[idx - 1]['Close'])
    y_prev_aligned = np.array(y_prev_list)
    
    y_true_aligned = actual_series.values
    y_pred_lr_aligned = lr_series.values
    y_pred_rf_aligned = rf_series.values
    
    # Calculate metrics
    model_results = {}
    
    # Linear Regression metrics
    metrics_lr = compute_metrics(y_true_aligned, y_pred_lr_aligned)
    metrics_lr['Directional Accuracy'] = compute_directional_accuracy(y_true_aligned, y_pred_lr_aligned, y_prev_aligned)
    model_results['Linear Regression'] = metrics_lr
    
    # Random Forest metrics
    metrics_rf = compute_metrics(y_true_aligned, y_pred_rf_aligned)
    metrics_rf['Directional Accuracy'] = compute_directional_accuracy(y_true_aligned, y_pred_rf_aligned, y_prev_aligned)
    model_results['Random Forest'] = metrics_rf
    
    # LSTM metrics
    predictions_dict = {
        'Linear Regression': y_pred_lr_aligned,
        'Random Forest': y_pred_rf_aligned
    }
    
    if lstm_available and y_pred_lstm is not None:
        lstm_series = pd.Series(y_pred_lstm, index=dates_test_lstm).loc[common_dates]
        y_pred_lstm_aligned = lstm_series.values
        
        metrics_lstm = compute_metrics(y_true_aligned, y_pred_lstm_aligned)
        metrics_lstm['Directional Accuracy'] = compute_directional_accuracy(y_true_aligned, y_pred_lstm_aligned, y_prev_aligned)
        model_results['LSTM'] = metrics_lstm
        
        predictions_dict['LSTM'] = y_pred_lstm_aligned
    else:
        logger.warning("LSTM results are omitted due to missing libraries or errors.")
        
    # 9. Print comparison report
    df_compare = compare_models(model_results)
    print("\n=== Model Performance Comparison ===")
    print(df_compare.to_string())
    print("=====================================\n")
    
    # 10. Save comparison plot
    plot_actual_vs_predicted_static(
        common_dates, 
        y_true_aligned, 
        predictions_dict, 
        args.ticker, 
        args.plot_path
    )
    logger.info(f"Comparison plot saved successfully to {args.plot_path}")

if __name__ == "__main__":
    main()
