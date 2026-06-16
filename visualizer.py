import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from typing import Dict, Any

# Optional import for Plotly (dashboard)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

def plot_actual_vs_predicted_static(
    dates: pd.Index, 
    y_true: np.ndarray, 
    predictions: Dict[str, np.ndarray], 
    ticker: str, 
    save_path: str = 'predictions_comparison.png'
) -> None:
    """
    Generates a static matplotlib chart comparing actual and predicted prices.
    Saves the image to the specified path.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_true, label='Actual Price', color='black', linewidth=1.5)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for idx, (model_name, y_pred) in enumerate(predictions.items()):
        color = colors[idx % len(colors)]
        plt.plot(dates, y_pred, label=f'{model_name} Prediction', linestyle='--', color=color, alpha=0.8)
        
    plt.title(f'{ticker.upper()} Stock Price Prediction Comparison')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_actual_vs_predicted_interactive(
    dates: pd.Index, 
    y_true: np.ndarray, 
    predictions: Dict[str, np.ndarray], 
    ticker: str,
    theme: str = "dark"
) -> Any:
    """
    Generates a beautiful, interactive Plotly line chart comparing actual and predicted prices.
    Suitable for Streamlit rendering.
    """
    if not HAS_PLOTLY:
        raise ImportError("Plotly is not installed. Cannot generate interactive charts.")
        
    is_dark = theme.lower() == "dark"
    template = 'plotly_dark' if is_dark else 'plotly_white'
    bg_color = 'rgba(0,0,0,0)' if is_dark else 'rgba(255,255,255,1)'
    plot_bg = 'rgba(0,0,0,0)' if is_dark else 'rgba(249,250,251,1)'
    text_color = '#F3F4F6' if is_dark else '#111827'
    grid_color = '#374151' if is_dark else '#E5E7EB'
    actual_color = '#E5E7EB' if is_dark else '#374151'
    
    fig = go.Figure()
    
    # Actual price
    fig.add_trace(go.Scatter(
        x=dates, 
        y=y_true, 
        mode='lines', 
        name='Actual Price', 
        line=dict(color=actual_color, width=2.5)
    ))
    
    # Model predictions
    color_palette = ['#60A5FA', '#F59E0B', '#10B981', '#EC4899']  # Modern pastel colors
    for idx, (model_name, y_pred) in enumerate(predictions.items()):
        color = color_palette[idx % len(color_palette)]
        fig.add_trace(go.Scatter(
            x=dates, 
            y=y_pred, 
            mode='lines', 
            name=model_name, 
            line=dict(color=color, width=2.0, dash='dash')
        ))
        
    fig.update_layout(
        title=dict(
            text=f'{ticker.upper()} - Actual vs. Predicted Prices',
            font=dict(size=18, color=text_color)
        ),
        xaxis_title='Date',
        yaxis_title='Stock Price ($)',
        template=template,
        hovermode='x unified',
        paper_bgcolor=bg_color,
        plot_bgcolor=plot_bg,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(gridcolor=grid_color),
        yaxis=dict(gridcolor=grid_color)
    )
    
    return fig

def plot_stock_history_candlestick(df: pd.DataFrame, ticker: str, predictions: Dict[str, np.ndarray] = None, test_dates: pd.Index = None, theme: str = "dark") -> Any:
    """
    Generates an interactive Plotly Candlestick chart showing open, high, low, close with optional predictions overlay.
    """
    if not HAS_PLOTLY:
        raise ImportError("Plotly is not installed. Cannot generate candlestick charts.")
        
    is_dark = theme.lower() == "dark"
    template = 'plotly_dark' if is_dark else 'plotly_white'
    bg_color = 'rgba(0,0,0,0)' if is_dark else 'rgba(255,255,255,1)'
    plot_bg = 'rgba(0,0,0,0)' if is_dark else 'rgba(249,250,251,1)'
    text_color = '#F3F4F6' if is_dark else '#111827'
    grid_color = '#374151' if is_dark else '#E5E7EB'
    
    fig = go.Figure()
    
    # Add Candlestick chart trace
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='#10B981',
        decreasing_line_color='#EF4444',
        name=ticker.upper()
    ))
    
    # Add support and resistance horizontal zones
    if not df.empty:
        support_val = df['Close'].min()
        resistance_val = df['Close'].max()
        fig.add_hline(
            y=support_val, 
            line_dash="dash", 
            line_color="rgba(239, 68, 68, 0.5)",
            line_width=1.5,
            annotation_text="Support Zone", 
            annotation_position="bottom left"
        )
        fig.add_hline(
            y=resistance_val, 
            line_dash="dash", 
            line_color="rgba(16, 185, 129, 0.5)",
            line_width=1.5,
            annotation_text="Resistance Zone", 
            annotation_position="top left"
        )
        
    # Overlay AI predictions if provided
    if predictions is not None and test_dates is not None:
        colors = {'LSTM': '#00F0FE', 'Random Forest': '#A855F7', 'Linear Regression': '#3B82F6'}
        for model_name, y_pred in predictions.items():
            color = colors.get(model_name, '#E5E7EB')
            fig.add_trace(go.Scatter(
                x=test_dates,
                y=y_pred,
                mode='lines',
                name=f'AI {model_name}',
                line=dict(color=color, width=2.5 if model_name == 'LSTM' else 2.0, dash='solid' if model_name == 'LSTM' else 'dot')
            ))
            
    fig.update_layout(
        title=dict(
            text=f'{ticker.upper()} AI Forecasting & Technical Dashboard',
            font=dict(size=18, color=text_color)
        ),
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template=template,
        xaxis_rangeslider_visible=False,
        paper_bgcolor=bg_color,
        plot_bgcolor=plot_bg,
        xaxis=dict(gridcolor=grid_color),
        yaxis=dict(gridcolor=grid_color),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    return fig

def plot_metrics_comparison_bar(df_compare: pd.DataFrame, theme: str = "dark") -> Any:
    """
    Generates an interactive bar chart comparing error metrics (MAE and RMSE) across models.
    """
    if not HAS_PLOTLY:
        raise ImportError("Plotly is not installed. Cannot generate comparison bar charts.")
    
    # We want to plot MAE and RMSE
    metrics_to_plot = [col for col in ['MAE', 'RMSE'] if col in df_compare.columns]
    
    if not metrics_to_plot:
        return None
        
    is_dark = theme.lower() == "dark"
    template = 'plotly_dark' if is_dark else 'plotly_white'
    bg_color = 'rgba(0,0,0,0)' if is_dark else 'rgba(255,255,255,1)'
    plot_bg = 'rgba(0,0,0,0)' if is_dark else 'rgba(249,250,251,1)'
    text_color = '#F3F4F6' if is_dark else '#111827'
    grid_color = '#374151' if is_dark else '#E5E7EB'
    
    df_melted = df_compare.reset_index().rename(columns={'index': 'Model'})
    df_melted = pd.melt(df_melted, id_vars=['Model'], value_vars=metrics_to_plot, var_name='Metric', value_name='Value')
    
    fig = px.bar(
        df_melted, 
        x='Model', 
        y='Value', 
        color='Metric', 
        barmode='group',
        color_discrete_map={'MAE': '#3B82F6', 'RMSE': '#EF4444'}
    )
    
    fig.update_layout(
        title=dict(
            text='Model Error Metric Comparison (Lower is Better)',
            font=dict(size=16, color=text_color)
        ),
        xaxis_title='Model',
        yaxis_title='Error Value ($)',
        template=template,
        paper_bgcolor=bg_color,
        plot_bgcolor=plot_bg,
        xaxis=dict(gridcolor=grid_color),
        yaxis=dict(gridcolor=grid_color),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    return fig
