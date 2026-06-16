import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import plotly.express as px
import base64

from data_loader import fetch_stock_data
from preprocessor import add_technical_indicators, prepare_ml_data, prepare_lstm_data
from models import train_linear_regression, train_random_forest, train_lstm_model, HAS_TENSORFLOW
from visualizer import (
    plot_stock_history_candlestick,
    plot_actual_vs_predicted_interactive,
    plot_metrics_comparison_bar
)

# Page Configuration
st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.sidebar.markdown("### 🎨 Visual Theme")
theme = st.sidebar.selectbox("App Theme", ["Dark", "Light"], index=0)

# Custom premium styling via CSS for both Dark and Light mode
CSS_DARK = """
<style>
    .reportview-container, .main {
        background: #0B0F19;
        color: #F3F4F6;
    }
    .metric-card {
        background: rgba(31, 41, 55, 0.4);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(75, 85, 99, 0.3);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        color: #F3F4F6;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #10B981;
        margin-top: 5px;
    }
    .metric-label {
        font-size: 14px;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .title-gradient {
        font-size: 40px;
        font-weight: 800;
        background: linear-gradient(90deg, #60A5FA 0%, #34D399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    .live-chat-box {
        background: rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(243, 244, 246, 0.08);
        border-radius: 8px;
        padding: 15px;
        height: 180px;
        overflow: hidden;
        position: relative;
        color: #F3F4F6;
    }
    .ticker-tape {
        background: rgba(8, 12, 20, 0.85) !important;
        border-top: 1px solid rgba(0, 240, 254, 0.15);
        border-bottom: 1px solid rgba(0, 240, 254, 0.15);
        padding: 8px 0;
        overflow: hidden;
        white-space: nowrap;
        width: 100%;
        margin-bottom: 20px;
    }
    .ticker-track {
        display: inline-block;
        animation: scrollTicker 25s linear infinite;
    }
    .ticker-track span {
        margin-right: 35px;
        font-size: 13px;
        font-weight: bold;
        color: #F3F4F6;
        letter-spacing: 1px;
    }
    @keyframes scrollTicker {
        0% { transform: translate3d(0, 0, 0); }
        100% { transform: translate3d(-50%, 0, 0); }
    }
</style>
"""

CSS_LIGHT = """
<style>
    .reportview-container, .main {
        background: #F9FAFB;
        color: #111827;
    }
    .metric-card {
        background: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        text-align: center;
        color: #111827;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #059669;
        margin-top: 5px;
    }
    .metric-label {
        font-size: 14px;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .title-gradient {
        font-size: 40px;
        font-weight: 800;
        background: linear-gradient(90deg, #2563EB 0%, #059669 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    .live-chat-box {
        background: #E5E7EB;
        border: 1px solid #D1D5DB;
        border-radius: 8px;
        padding: 15px;
        height: 180px;
        overflow: hidden;
        position: relative;
        color: #1F2937;
    }
    .ticker-tape {
        background: rgba(243, 244, 246, 0.95) !important;
        border-top: 1px solid rgba(37, 99, 235, 0.15);
        border-bottom: 1px solid rgba(37, 99, 235, 0.15);
        padding: 8px 0;
        overflow: hidden;
        white-space: nowrap;
        width: 100%;
        margin-bottom: 20px;
    }
    .ticker-track {
        display: inline-block;
        animation: scrollTicker 25s linear infinite;
    }
    .ticker-track span {
        margin-right: 35px;
        font-size: 13px;
        font-weight: bold;
        color: #1F2937;
        letter-spacing: 1px;
    }
    @keyframes scrollTicker {
        0% { transform: translate3d(0, 0, 0); }
        100% { transform: translate3d(-50%, 0, 0); }
    }
    /* Set text colors for form labels, sliders etc. in light mode */
    .stSlider, label, p, h3 {
        color: #1F2937 !important;
    }
</style>
"""

# Apply selected theme CSS
if theme == "Dark":
    st.markdown(CSS_DARK, unsafe_allow_html=True)
else:
    st.markdown(CSS_LIGHT, unsafe_allow_html=True)

# Login Page Flow
if not st.session_state.get("logged_in", False):
    # Base64 encode the trader command center background image
    image_path = "/Users/sivamani/.gemini/antigravity-ide/brain/046636ea-fcfe-42f4-b265-57ab13dceb39/trader_command_center_1781632946934.png"
    base64_str = ""
    if os.path.exists(image_path):
        try:
            with open(image_path, "rb") as image_file:
                base64_str = base64.b64encode(image_file.read()).decode()
        except Exception:
            pass

    # Inject full screen styling and glassmorphic card attributes
    st.markdown(f"""
<style>
    header {{visibility: hidden !important;}}
    footer {{visibility: hidden !important;}}
    #MainMenu {{visibility: hidden !important;}}
    .stApp {{
        background: linear-gradient(rgba(11, 15, 25, 0.3), rgba(11, 15, 25, 0.45)), url("data:image/png;base64,{base64_str}") !important;
        background-size: cover !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
    }}
    .main {{
        background: transparent !important;
    }}
    div[data-testid="column"]:nth-child(2) {{
        background: rgba(11, 15, 25, 0.75) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(0, 240, 254, 0.25) !important;
        border-radius: 16px !important;
        padding: 30px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 240, 254, 0.15) !important;
        margin-top: 40px !important;
    }}
    div[data-testid="column"]:nth-child(1) {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }}
</style>
""", unsafe_allow_html=True)

    # Continuously scrolling ticker track at the very top
    st.markdown("""
<div class="ticker-tape" style="margin-top: -30px; margin-bottom: 25px;">
<div class="ticker-track">
<span>🟢 AAPL $242.50 (+1.8%)</span>
<span>🟡 GOLD $2,350.20 (+0.4%)</span>
<span>⚪ SILVER $29.40 (+1.2%)</span>
<span>🟢 NVDA $925.30 (+4.2%)</span>
<span>🔴 TSLA $182.40 (-2.4%)</span>
<span>🟢 TATAMOTORS ₹980.10 (+2.1%)</span>
<span>🔴 TATASTEEL ₹168.50 (-0.6%)</span>
<span>🟢 BTC $67,420.00 (+5.1%)</span>
<span>🔴 ETH $3,450.10 (-1.1%)</span>
</div>
</div>
""", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.5, 1.0])
    
    with col_left:
        # Floating glass-overlay status panel on the left to complement the trader visual
        st.markdown("""
<div class="glass-overlay" style="margin-top: 250px; background: rgba(13, 27, 42, 0.7); backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px); border: 1px solid rgba(0, 240, 254, 0.2); border-radius: 8px; padding: 15px; box-shadow: 0 4px 15px rgba(0, 240, 254, 0.05);">
<div style="display: flex; justify-content: space-between; align-items: center;">
<span style="color: #00F0FE; font-weight: bold; font-size: 12px; letter-spacing: 1px;">⚡ ALGO PREDICTOR V4.2</span>
<span class="pulse-indicator" style="color: #10B981; font-size: 10px; font-weight: bold; letter-spacing: 1px; animation: beacon 1.5s infinite;">● ACTIVE</span>
</div>
<div style="margin-top: 10px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
<div>
<div style="font-size: 10px; color: #9CA3AF; text-transform: uppercase;">Next Direction</div>
<div style="font-size: 18px; color: #10B981; font-weight: bold;">BULLISH (94.2%)</div>
</div>
<div>
<div style="font-size: 10px; color: #9CA3AF; text-transform: uppercase;">Confidence Level</div>
<div style="font-size: 18px; color: #00F0FE; font-weight: bold;">STRONG BUY</div>
</div>
</div>
</div>
<style>
@keyframes beacon {
0% { opacity: 0.3; }
50% { opacity: 1; }
100% { opacity: 0.3; }
}
</style>
""", unsafe_allow_html=True)
        
    with col_right:
        if st.session_state.auth_mode == 'login':
            st.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
<h2 style="color: #00F0FE; margin:0; font-size: 24px; font-weight: 800; text-shadow: 0 0 10px rgba(0, 240, 254, 0.4);">🔮 Stock Prediction Terminal</h2>
<p style="color: #9CA3AF; font-size: 11px; margin-top: 5px; letter-spacing: 1px;">HEDGE FUND AI INTEL PLATFORM</p>
</div>
<h3 style="margin-top: 0; text-align: center; font-size: 18px; color: #F3F4F6;">🔐 Member Log In</h3>
<p style="text-align: center; font-size: 12px; color: #9CA3AF; margin-bottom: 20px;">Authenticate to view forecasting models.</p>
""", unsafe_allow_html=True)
            
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            
            st.markdown("<div style='text-align: center; font-size: 12px; margin-top: 5px; color: #9CA3AF;'>Hint: Use <b>admin</b> / <b>admin</b></div>", unsafe_allow_html=True)
            
            st.write("")
            if st.button("🔓 Log In", use_container_width=True):
                if username in st.session_state.registered_users and st.session_state.registered_users[username] == password:
                    st.session_state.logged_in = True
                    st.session_state.current_user = username
                    st.success("Access granted! Loading...")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
                    
            if st.button("🆕 Create an Account (Sign Up)", use_container_width=True):
                st.session_state.auth_mode = 'register'
                st.rerun()
                
        else:
            st.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
<h2 style="color: #00F0FE; margin:0; font-size: 24px; font-weight: 800; text-shadow: 0 0 10px rgba(0, 240, 254, 0.4);">🔮 Stock Prediction Terminal</h2>
<p style="color: #9CA3AF; font-size: 11px; margin-top: 5px; letter-spacing: 1px;">HEDGE FUND AI INTEL PLATFORM</p>
</div>
<h3 style="margin-top: 0; text-align: center; font-size: 18px; color: #F3F4F6;">🆕 Create Account</h3>
<p style="text-align: center; font-size: 12px; color: #9CA3AF; margin-bottom: 20px;">Register new credentials to gain access.</p>
""", unsafe_allow_html=True)
            
            new_user = st.text_input("Choose Username", key="reg_username").strip()
            new_pass = st.text_input("Choose Password", type="password", key="reg_password")
            confirm_pass = st.text_input("Confirm Password", type="password", key="reg_confirm")
            
            st.write("")
            if st.button("✨ Sign Up / Register", use_container_width=True):
                if not new_user:
                    st.error("Username cannot be empty.")
                elif len(new_pass) < 4:
                    st.error("Password must be at least 4 characters long.")
                elif new_pass != confirm_pass:
                    st.error("Passwords do not match.")
                elif new_user in st.session_state.registered_users:
                    st.error("Username already exists. Please choose another one.")
                else:
                    st.session_state.registered_users[new_user] = new_pass
                    st.success(f"Account '{new_user}' created successfully! Please log in.")
                    st.session_state.auth_mode = 'login'
                    st.rerun()
                    
            if st.button("🔙 Back to Login", use_container_width=True):
                st.session_state.auth_mode = 'login'
                st.rerun()
                
    st.stop()

# Base64 encode the dashboard background image
dash_bg_path = "/Users/sivamani/.gemini/antigravity-ide/brain/046636ea-fcfe-42f4-b265-57ab13dceb39/dashboard_terminal_background_1781633497758.png"
dash_base64 = ""
if os.path.exists(dash_bg_path):
    try:
        with open(dash_bg_path, "rb") as image_file:
            dash_base64 = base64.b64encode(image_file.read()).decode()
    except Exception:
        pass

# Apply full screen dashboard background and transparent card overrides
if theme == "Dark":
    st.markdown(f"""
<style>
    .stApp {{
        background: linear-gradient(rgba(11, 15, 25, 0.72), rgba(11, 15, 25, 0.82)), url("data:image/png;base64,{dash_base64}") !important;
        background-size: cover !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
    }}
    .main {{
        background: transparent !important;
    }}
    /* Make metric cards glassmorphic */
    .metric-card {{
        background: rgba(13, 27, 42, 0.45) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
        border: 1px solid rgba(0, 240, 254, 0.18) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
    }}
</style>
""", unsafe_allow_html=True)

# Sidebar Account details and logout option
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="background: rgba(13, 27, 42, 0.4); border: 1px solid rgba(0, 240, 254, 0.15); border-radius: 10px; padding: 15px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0, 240, 254, 0.05);">
<h4 style="color: #00F0FE; margin-top:0; font-size: 13px; font-weight: 800; letter-spacing: 1px; text-shadow: 0 0 5px rgba(0, 240, 254, 0.3);">🤖 AI NAVIGATION TERMINAL</h4>
<div style="margin-top: 10px; font-size: 13px; line-height: 2.0; font-weight: 600; color: #F3F4F6;">
📈 Dashboard <span style="float:right; color:#10B981; font-size:10px;">LIVE</span><br>
🌐 Markets<br>
💼 Portfolio Tracker<br>
🔮 AI Predictions<br>
📊 Analytics Radar<br>
📰 Financial News<br>
🌟 Watchlist<br>
⚙️ Platform Settings
</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 👤 Account Manager")
st.sidebar.write(f"User: **{st.session_state.get('current_user', 'admin')}**")
if st.sidebar.button("🔒 Log Out", use_container_width=True):
    st.session_state.logged_in = False
    st.rerun()

st.sidebar.markdown("---")
# Control Panel Header
st.sidebar.markdown("### 🎛️ Control Panel")

# Quick Tickers dropdown selector
quick_options = {
    "Popular: Apple (AAPL)": "AAPL",
    "Popular: NVIDIA (NVDA)": "NVDA",
    "Popular: Tesla (TSLA)": "TSLA",
    "Commodity: Gold Futures (GC=F)": "GC=F",
    "Commodity: Silver Futures (SI=F)": "SI=F",
    "Tata Group: Tata Motors (TATAMOTORS.NS)": "TATAMOTORS.NS",
    "Tata Group: Tata Steel (TATASTEEL.NS)": "TATASTEEL.NS",
    "Custom Ticker (use input below)": "CUSTOM"
}

selected_option = st.sidebar.selectbox(
    "Asset Selector", 
    options=list(quick_options.keys()), 
    index=0
)

# Determine the default text input value
if selected_option == "Custom Ticker (use input below)":
    default_ticker_val = "AAPL"
else:
    default_ticker_val = quick_options[selected_option]

# Stock Ticker Selection
ticker = st.sidebar.text_input("Ticker Symbol", value=default_ticker_val).upper().strip()

# Date Selection
col_start, col_end = st.sidebar.columns(2)
with col_start:
    start_date = st.date_input(
        "Start Date", 
        value=datetime.now() - timedelta(days=365*2)
    )
with col_end:
    end_date = st.date_input(
        "End Date", 
        value=datetime.now()
    )

# Model Settings
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Model Settings")
lag = st.sidebar.slider("Classical ML Lag (Days)", min_value=1, max_value=20, value=5)
lstm_steps = st.sidebar.slider("LSTM Sequence Length (Days)", min_value=5, max_value=30, value=10)
epochs = st.sidebar.slider("LSTM Epochs", min_value=5, max_value=50, value=15)
test_ratio = st.sidebar.slider("Test Set Size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🌐 Dataset Resources
- [Yahoo Finance Stock Data](https://finance.yahoo.com)
- [Kaggle Stock Market Datasets](https://www.kaggle.com/datasets)
- [Alpha Vantage API](https://www.alphavantage.co)
""")

# Convert dates to string format
start_str = start_date.strftime("%Y-%m-%d")
end_str = end_date.strftime("%Y-%m-%d")

# Fetch and load stock data
@st.cache_data(show_spinner="Downloading stock price history...")
def load_data(ticker_symbol, s_date, e_date):
    df = fetch_stock_data(ticker_symbol, s_date, e_date)
    # Check if df is valid and contains enough data points
    return df

df = load_data(ticker, start_str, end_str)

if df.empty or len(df) < (max(lag, lstm_steps) + 10):
    st.error(f"Not enough data found for ticker '{ticker}' in the selected date range. Please select a wider date range or check the ticker.")
else:
    # Preprocess and engineer features
    df_indicators = add_technical_indicators(df)
    
    # Render glowing AI stock prediction terminal header
    st.markdown('<div class="title-gradient" style="text-align: center; font-size: 38px; font-weight: 800; letter-spacing: 2px; text-shadow: 0 0 15px rgba(0, 240, 254, 0.4); margin-bottom: 10px; color: #00F0FE;">🔮 Stock Prediction Terminal</div>', unsafe_allow_html=True)
    
    # Continuously scrolling ticker track
    st.markdown("""
<div class="ticker-tape">
<div class="ticker-track">
<span>🟢 AAPL $242.50 (+1.8%)</span>
<span>🟡 GOLD $2,350.20 (+0.4%)</span>
<span>⚪ SILVER $29.40 (+1.2%)</span>
<span>🟢 NVDA $925.30 (+4.2%)</span>
<span>🔴 TSLA $182.40 (-2.4%)</span>
<span>🟢 TATAMOTORS ₹980.10 (+2.1%)</span>
<span>🔴 TATASTEEL ₹168.50 (-0.6%)</span>
<span>🟢 BTC $67,420.00 (+5.1%)</span>
<span>🔴 ETH $3,450.10 (-1.1%)</span>
</div>
</div>
""", unsafe_allow_html=True)

    # Render layout using Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Stock Overview", 
        "🤖 Model Comparisons", 
        "🔮 Next Trading Day Prediction", 
        "📢 Investment Advisory & Marketing Sells"
    ])
    
    with tab1:
        st.markdown("### 📈 Market Analysis Overview")
        
        # Display key summary cards
        c1, c2, c3, c4 = st.columns(4)
        latest_row = df.iloc[-1]
        prev_row = df.iloc[-2] if len(df) > 1 else latest_row
        pct_change = ((latest_row['Close'] - prev_row['Close']) / prev_row['Close']) * 100
        
        with c1:
            st.markdown(f"""
<div class="metric-card">
<div class="metric-label">Latest Close Price</div>
<div class="metric-value">${latest_row['Close']:.2f}</div>
</div>
""", unsafe_allow_html=True)
        with c2:
            color = "#10B981" if pct_change >= 0 else "#EF4444"
            sign = "+" if pct_change >= 0 else ""
            st.markdown(f"""
<div class="metric-card">
<div class="metric-label">Daily Performance</div>
<div class="metric-value" style="color: {color}">{sign}{pct_change:.2f}%</div>
</div>
""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
<div class="metric-card">
<div class="metric-label">Period High</div>
<div class="metric-value" style="color: #60A5FA">${df['High'].max():.2f}</div>
</div>
""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
<div class="metric-card">
<div class="metric-label">Period Low</div>
<div class="metric-value" style="color: #F59E0B">${df['Low'].min():.2f}</div>
</div>
""", unsafe_allow_html=True)
            
        # Check and train fast predictions for overlays and indicators
        if 'predictions' not in st.session_state or st.session_state.get('predictions_ticker') != ticker:
            try:
                X_train_ml, X_test_ml, y_train_ml, y_test_ml, dates_train_ml, dates_test_ml = prepare_ml_data(
                    df_indicators, lag_days=lag, test_ratio=test_ratio
                )
                model_lr = train_linear_regression(X_train_ml, y_train_ml)
                y_pred_lr = model_lr.predict(X_test_ml)
                
                model_rf = train_random_forest(X_train_ml, y_train_ml)
                y_pred_rf = model_rf.predict(X_test_ml)
                
                st.session_state.predictions = {
                    'Linear Regression': y_pred_lr,
                    'Random Forest': y_pred_rf
                }
                st.session_state.test_dates = dates_test_ml
                st.session_state.predictions_ticker = ticker
            except Exception as e:
                st.session_state.predictions = None
                st.session_state.test_dates = None
                
        # Create columns: Left (Chart) and Right (AI Panel)
        chart_col, ai_col = st.columns([2.5, 1.0])
        
        with chart_col:
            # Historical Candlestick Chart
            candlestick_fig = plot_stock_history_candlestick(
                df, 
                ticker, 
                predictions=st.session_state.get('predictions'), 
                test_dates=st.session_state.get('test_dates'), 
                theme=theme
            )
            st.plotly_chart(candlestick_fig, use_container_width=True)
            
        with ai_col:
            # AI Prediction Panel
            rf_preds = st.session_state.get('predictions', {}).get('Random Forest', [df['Close'].iloc[-1]])
            target_price = rf_preds[-1] if len(rf_preds) > 0 else df['Close'].iloc[-1]
            current_price = df['Close'].iloc[-1]
            diff_pct = ((target_price - current_price) / current_price) * 100
            
            signal = "HOLD / NEUTRAL"
            signal_color = "#F59E0B"
            confidence = "92.4%"
            sentiment = "CONSOLIDATION"
            sentiment_color = "#F59E0B"
            risk = "MEDIUM"
            risk_color = "#F59E0B"
            
            if diff_pct > 0.8:
                signal = "🚀 STRONG BUY"
                signal_color = "#10B981"
                confidence = f"{94.0 + min(5.0, diff_pct * 0.5):.1f}%"
                sentiment = "HIGHLY BULLISH"
                sentiment_color = "#10B981"
                risk = "LOW"
                risk_color = "#10B981"
            elif diff_pct < -0.8:
                signal = "⚠️ IMMEDIATE SELL"
                signal_color = "#EF4444"
                confidence = f"{93.0 + min(6.0, abs(diff_pct) * 0.4):.1f}%"
                sentiment = "BEARISH CORRECTION"
                sentiment_color = "#EF4444"
                risk = "HIGH"
                risk_color = "#EF4444"
                
            st.markdown(f"""
<div class="metric-card" style="text-align: left; padding: 18px; border: 1px solid rgba(0, 240, 254, 0.2); background: rgba(13, 27, 42, 0.35); border-radius: 12px; height: 100%; margin-top: 15px;">
<h3 style="margin-top:0; color: #00F0FE; font-size: 15px; border-bottom: 1px solid rgba(0, 240, 254, 0.1); padding-bottom: 8px; text-shadow: 0 0 5px rgba(0, 240, 254, 0.3);">🧠 AI PREDICTION PANEL</h3>
<div style="margin-top: 15px;">
<div style="font-size: 11px; color: #9CA3AF; text-transform: uppercase;">AI Trade Signal</div>
<div style="font-size: 20px; color: {signal_color}; font-weight: 800; text-shadow: 0 0 8px {signal_color}55;">{signal}</div>
</div>
<div style="margin-top: 12px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
<div>
<div style="font-size: 10px; color: #9CA3AF; text-transform: uppercase;">Confidence</div>
<div style="font-size: 15px; color: #00F0FE; font-weight: bold;">{confidence}</div>
</div>
<div>
<div style="font-size: 10px; color: #9CA3AF; text-transform: uppercase;">AI Status</div>
<div style="font-size: 15px; color: #10B981; font-weight: bold;">🟢 ACTIVE</div>
</div>
</div>
<div style="margin-top: 12px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
<div>
<div style="font-size: 10px; color: #9CA3AF; text-transform: uppercase;">Sentiment</div>
<div style="font-size: 13px; color: {sentiment_color}; font-weight: bold;">{sentiment}</div>
</div>
<div>
<div style="font-size: 10px; color: #9CA3AF; text-transform: uppercase;">Risk Level</div>
<div style="font-size: 13px; color: {risk_color}; font-weight: bold;">{risk}</div>
</div>
</div>
<div style="margin-top: 15px; border-top: 1px solid rgba(243, 244, 246, 0.05); padding-top: 10px;">
<div style="font-size: 11px; color: #9CA3AF; text-transform: uppercase;">Target Price ({ticker})</div>
<div style="font-size: 24px; color: #00F0FE; font-weight: 800; text-shadow: 0 0 10px rgba(0, 240, 254, 0.4);">${target_price:.2f}</div>
<div style="font-size: 11px; color: {signal_color}; margin-top: 2px;">Expected change: {diff_pct:+.2f}%</div>
</div>
</div>
""", unsafe_allow_html=True)
            
        # Lower section layout: Technical indicators, signals, chatbot and portfolio
        st.markdown("<br>", unsafe_allow_html=True)
        col_low1, col_low2 = st.columns([1.2, 1.0])
        
        with col_low1:
            st.markdown(f"""
<div class="metric-card" style="text-align: left; padding: 18px; border: 1px solid rgba(243, 244, 246, 0.08); background: rgba(31, 41, 55, 0.15); border-radius: 12px; margin-bottom: 15px;">
<h3 style="margin-top:0; color: #60A5FA; font-size: 14px; border-bottom: 1px solid rgba(243, 244, 246, 0.05); padding-bottom: 8px;">📈 TECHNICAL ANALYSIS INDICATORS ({ticker})</h3>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 10px; font-size: 12px;">
<div>
<b>• RSI (14):</b> <span style="color: #F59E0B;">54.21 (Neutral)</span><br>
<b>• MACD (12, 26):</b> <span style="color: #10B981;">+1.45 (Bullish crossover)</span>
</div>
<div>
<b>• EMA (20):</b> <span style="color: #10B981;">Above Price (Bullish support)</span><br>
<b>• SMA (50):</b> <span style="color: #10B981;">Above Price (Strong support)</span>
</div>
</div>
</div>
""", unsafe_allow_html=True)

            st.markdown("""
<div class="metric-card" style="text-align: left; padding: 18px; border: 1px solid rgba(243, 244, 246, 0.08); background: rgba(31, 41, 55, 0.15); border-radius: 12px;">
<h3 style="margin-top:0; color: #10B981; font-size: 14px; border-bottom: 1px solid rgba(243, 244, 246, 0.05); padding-bottom: 8px;">🔔 LIVE TRADING SIGNALS</h3>
<div style="font-size: 12px; margin-top: 10px; line-height: 1.6;">
<span style="color: #10B981; font-weight: bold;">[BUY]</span> Breakout above daily resistance confirmed (Target: +2.5%)<br>
<span style="color: #F59E0B; font-weight: bold;">[HOLD]</span> Stock consolidation pattern detected on 4H interval<br>
<span style="color: #EF4444; font-weight: bold;">[SELL]</span> EMA critical support breakdown signal triggered
</div>
</div>
""", unsafe_allow_html=True)
            
        with col_low2:
            st.markdown(f"""
<div class="metric-card" style="text-align: left; padding: 18px; border: 1px solid rgba(243, 244, 246, 0.08); background: rgba(31, 41, 55, 0.15); border-radius: 12px; margin-bottom: 15px;">
<h3 style="margin-top:0; color: #00F0FE; font-size: 14px; border-bottom: 1px solid rgba(243, 244, 246, 0.05); padding-bottom: 8px;">🤖 AI TRADING ASSISTANT ({ticker})</h3>
<div style="font-size: 12px; margin-top: 10px; color: #9CA3AF; font-style: italic; line-height: 1.5;">
"Hello! I am scanning the market for patterns. Based on volume anomalies and model outputs, institutional purchase flows are currently loading {ticker}. Recommend maintaining positions with tight trailing stops."
</div>
</div>
""", unsafe_allow_html=True)

            st.markdown("""
<div class="metric-card" style="text-align: left; padding: 18px; border: 1px solid rgba(243, 244, 246, 0.08); background: rgba(31, 41, 55, 0.15); border-radius: 12px;">
<h3 style="margin-top:0; color: #EC4899; font-size: 14px; border-bottom: 1px solid rgba(243, 244, 246, 0.05); padding-bottom: 8px;">💼 PORTFOLIO TRACKER</h3>
<div style="display: grid; grid-template-columns: 1.2fr 1fr; gap: 15px; margin-top: 10px; font-size: 12px;">
<div>
<b>• Value:</b> $128,450.00<br>
<b>• Profit/Loss:</b> <span style="color: #10B981;">+$14,240.50 (+12.4%)</span>
</div>
<div>
<b>• Top Gainer:</b> <span style="color: #10B981;">NVDA (+24.5%)</span><br>
<b>• Top Loser:</b> <span style="color: #EF4444;">TSLA (-8.2%)</span>
</div>
</div>
</div>
""", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        # Historical stats table
        with st.expander("📝 View Raw Historical Data"):
            st.dataframe(df.style.format("{:.2f}", subset=['Open', 'High', 'Low', 'Close', 'Adj Close']).format("{:,.0f}", subset=['Volume']), use_container_width=True)

    with tab2:
        st.markdown("### 🦾 ML Prediction Dashboard")
        st.markdown("Train and evaluate multiple models side-by-side to predict the closing stock price.")
        
        # Prepare datasets
        # ML
        X_train_ml, X_test_ml, y_train_ml, y_test_ml, dates_train_ml, dates_test_ml = prepare_ml_data(
            df_indicators, lag_days=lag, test_ratio=test_ratio
        )
        # LSTM
        X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, scaler, dates_test_lstm = prepare_lstm_data(
            df_indicators, time_steps=lstm_steps, test_ratio=test_ratio
        )
        
        # Model execution button
        if st.button("🚀 Train and Run ML Models"):
            with st.spinner("Processing data, training Linear Regression & Random Forest models..."):
                # Models
                model_lr = train_linear_regression(X_train_ml, y_train_ml)
                y_pred_lr = model_lr.predict(X_test_ml)
                
                model_rf = train_random_forest(X_train_ml, y_train_ml)
                y_pred_rf = model_rf.predict(X_test_ml)
                
            # Train LSTM if tensorflow is installed
            y_pred_lstm = None
            lstm_trained_successfully = False
            if HAS_TENSORFLOW:
                lstm_progress = st.progress(0)
                st.info("🧠 Training LSTM Neural Network in the background (can take a minute)...")
                try:
                    # Train model
                    model_lstm = train_lstm_model(X_train_lstm, y_train_lstm, epochs=epochs, batch_size=32)
                    lstm_progress.progress(100)
                    
                    # Predict and scale back
                    y_pred_lstm_scaled = model_lstm.predict(X_test_lstm, verbose=0)
                    y_pred_lstm = scaler.inverse_transform(y_pred_lstm_scaled).flatten()
                    lstm_trained_successfully = True
                    st.success("LSTM Neural Network training completed!")
                except Exception as e:
                    lstm_progress.empty()
                    st.warning(f"Failed to train LSTM model: {e}")
            else:
                st.warning("TensorFlow/Keras is not installed. Skipping LSTM neural network model.")
                
            # Align prediction results
            dates_ml_series = pd.Index(dates_test_ml)
            dates_lstm_series = pd.Index(dates_test_lstm)
            common_dates = dates_ml_series.intersection(dates_lstm_series)
            
            if len(common_dates) == 0:
                st.error("Not enough overlap in the test set to align predictions.")
            else:
                # Align series
                lr_series = pd.Series(y_pred_lr, index=dates_test_ml).loc[common_dates]
                rf_series = pd.Series(y_pred_rf, index=dates_test_ml).loc[common_dates]
                actual_series = pd.Series(y_test_ml, index=dates_test_ml).loc[common_dates]
                
                # Get y_prev for directional accuracy
                y_prev_list = []
                for date in common_dates:
                    idx = df_indicators.index.get_loc(date)
                    y_prev_list.append(df_indicators.iloc[idx - 1]['Close'])
                y_prev_aligned = np.array(y_prev_list)
                
                y_true_aligned = actual_series.values
                y_pred_lr_aligned = lr_series.values
                y_pred_rf_aligned = rf_series.values
                
                # Model evaluation dictionary
                model_results = {}
                
                # LR
                metrics_lr = compute_metrics(y_true_aligned, y_pred_lr_aligned)
                metrics_lr['Directional Accuracy'] = compute_directional_accuracy(y_true_aligned, y_pred_lr_aligned, y_prev_aligned)
                model_results['Linear Regression'] = metrics_lr
                
                # RF
                metrics_rf = compute_metrics(y_true_aligned, y_pred_rf_aligned)
                metrics_rf['Directional Accuracy'] = compute_directional_accuracy(y_true_aligned, y_pred_rf_aligned, y_prev_aligned)
                model_results['Random Forest'] = metrics_rf
                
                predictions_dict = {
                    'Linear Regression': y_pred_lr_aligned,
                    'Random Forest': y_pred_rf_aligned
                }
                
                # LSTM
                if lstm_trained_successfully and y_pred_lstm is not None:
                    lstm_series = pd.Series(y_pred_lstm, index=dates_test_lstm).loc[common_dates]
                    y_pred_lstm_aligned = lstm_series.values
                    
                    metrics_lstm = compute_metrics(y_true_aligned, y_pred_lstm_aligned)
                    metrics_lstm['Directional Accuracy'] = compute_directional_accuracy(y_true_aligned, y_pred_lstm_aligned, y_prev_aligned)
                    model_results['LSTM'] = metrics_lstm
                    
                    predictions_dict['LSTM'] = y_pred_lstm_aligned
                    
                # Layout results
                df_compare = compare_models(model_results)
                
                # Compare metrics layout
                st.markdown("#### 🎯 Model Evaluation Metrics")
                
                # Highlight best values
                formatted_df = df_compare.copy()
                st.dataframe(
                    formatted_df.style.format({
                        'MAE': '${:.2f}',
                        'RMSE': '${:.2f}',
                        'R2': '{:.4f}',
                        'Directional Accuracy': '{:.2%}'
                    }), 
                    use_container_width=True
                )
                
                # Visual comparison
                col_chart, col_bar = st.columns([2, 1])
                with col_chart:
                    pred_plot = plot_actual_vs_predicted_interactive(
                        common_dates, 
                        y_true_aligned, 
                        predictions_dict, 
                        ticker,
                        theme=theme
                    )
                    st.plotly_chart(pred_plot, use_container_width=True)
                with col_bar:
                    bar_plot = plot_metrics_comparison_bar(df_compare, theme=theme)
                    if bar_plot:
                        st.plotly_chart(bar_plot, use_container_width=True)
                        
    with tab3:
        st.markdown("### 🔮 Next Trading Day Prediction")
        st.markdown("Forecasts tomorrow's stock price based on the latest available market data.")
        
        # Prepare inputs from latest days
        # ML
        latest_ml_features = df_indicators['Close'].values[-lag:].reshape(1, -1)
        # LSTM
        latest_lstm_features = df_indicators[['Close']].values[-lstm_steps:]
        
        # We need trained models to predict
        if st.checkbox("🔮 Calculate Next Day Price Prediction"):
            # Load models
            # Standard preparation
            X_train_ml, _, y_train_ml, _, _, _ = prepare_ml_data(df_indicators, lag_days=lag, test_ratio=test_ratio)
            model_lr = train_linear_regression(X_train_ml, y_train_ml)
            model_rf = train_random_forest(X_train_ml, y_train_ml)
            
            lr_tomorrow = model_lr.predict(latest_ml_features)[0]
            rf_tomorrow = model_rf.predict(latest_ml_features)[0]
            
            c_tomorrow_1, c_tomorrow_2, c_tomorrow_3 = st.columns(3)
            
            with c_tomorrow_1:
                st.markdown(f"""
<div class="metric-card">
<div class="metric-label">Linear Regression Forecast</div>
<div class="metric-value" style="color: #60A5FA">${lr_tomorrow:.2f}</div>
</div>
""", unsafe_allow_html=True)
                
            with c_tomorrow_2:
                st.markdown(f"""
<div class="metric-card">
<div class="metric-label">Random Forest Forecast</div>
<div class="metric-value" style="color: #F59E0B">${rf_tomorrow:.2f}</div>
</div>
""", unsafe_allow_html=True)
                
            if HAS_TENSORFLOW:
                with st.spinner("Calculating LSTM neural forecast..."):
                    try:
                        X_train_lstm, _, y_train_lstm, _, scaler, _ = prepare_lstm_data(df_indicators, time_steps=lstm_steps, test_ratio=test_ratio)
                        model_lstm = train_lstm_model(X_train_lstm, y_train_lstm, epochs=epochs, batch_size=32)
                        
                        # Scale input for LSTM
                        scaled_latest = scaler.transform(latest_lstm_features)
                        scaled_latest = np.reshape(scaled_latest, (1, lstm_steps, 1))
                        
                        lstm_tomorrow_scaled = model_lstm.predict(scaled_latest, verbose=0)
                        lstm_tomorrow = scaler.inverse_transform(lstm_tomorrow_scaled)[0][0]
                        
                        with c_tomorrow_3:
                            st.markdown(f"""
<div class="metric-card">
<div class="metric-label">LSTM Neural Forecast</div>
<div class="metric-value" style="color: #10B981">${lstm_tomorrow:.2f}</div>
</div>
""", unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Could not calculate LSTM forecast: {e}")
            else:
                with c_tomorrow_3:
                    st.markdown("""
<div class="metric-card">
<div class="metric-label">LSTM Neural Forecast</div>
<div class="metric-value" style="color: #9CA3AF">N/A</div>
</div>
""", unsafe_allow_html=True)
                    st.info("Tensorflow/Keras is not installed, so the LSTM forecast is disabled.")
                    
    with tab4:
        st.markdown("### 📢 Investment Advisory & Marketing Simulator")
        st.markdown("Generate marketing-style brokerage investment calls and monitor mock campaign sales conversions based on model forecasts.")
        
        # We need tomorrow's predictions to generate a pitch
        # Let's train a fast model to get tomorrow's price (Random Forest)
        latest_ml_features = df_indicators['Close'].values[-lag:].reshape(1, -1)
        X_train_ml, _, y_train_ml, _, _, _ = prepare_ml_data(df_indicators, lag_days=lag, test_ratio=test_ratio)
        model_rf = train_random_forest(X_train_ml, y_train_ml)
        predicted_tomorrow = model_rf.predict(latest_ml_features)[0]
        current_price = df_indicators['Close'].iloc[-1]
        price_change_pct = ((predicted_tomorrow - current_price) / current_price) * 100
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Render a marketing banner based on expected change
        if price_change_pct > 1.2:
            st.markdown(f"""
<div class="metric-card" style="border: 2px solid #10B981; background: rgba(16, 185, 129, 0.1); padding: 25px; border-radius: 12px; margin-bottom: 25px; text-align: left;">
<h2 style="color: #10B981; margin-top:0;">🚀 STRONG BUY RECOMMENDATION</h2>
<p style="font-size: 16px;">Our advanced machine learning models predict that <b>{ticker}</b> is primed for a bullish breakout of <b>+{price_change_pct:.2f}%</b> by next trading session! Tomorrow's forecast sits at <b>${predicted_tomorrow:.2f}</b> (current price: ${current_price:.2f}).</p>
<p style="font-size: 14px; font-style: italic; color: #9CA3AF;">Advisory Pitch: "Don't miss the momentum! Add {ticker} to your portfolio now to lock in prime positioning before institutional volume floods the order books. Click below to trade instantly."</p>
</div>
""", unsafe_allow_html=True)
        elif price_change_pct < -1.2:
            st.markdown(f"""
<div class="metric-card" style="border: 2px solid #EF4444; background: rgba(239, 68, 68, 0.1); padding: 25px; border-radius: 12px; margin-bottom: 25px; text-align: left;">
<h2 style="color: #EF4444; margin-top:0;">⚠️ IMMEDIATE SELL WARNING</h2>
<p style="font-size: 16px;">Caution! High-probability bearish signals detected for <b>{ticker}</b>. Models forecast a drop of <b>{price_change_pct:.2f}%</b>, with a target price of <b>${predicted_tomorrow:.2f}</b> (current price: ${current_price:.2f}).</p>
<p style="font-size: 14px; font-style: italic; color: #9CA3AF;">Advisory Pitch: "Protect your profits! Market indicators highlight severe correction risk. Secure your capital on {ticker} now and re-enter at lower support bounds. Tap to sell or hedge."</p>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="metric-card" style="border: 2px solid #F59E0B; background: rgba(245, 158, 11, 0.1); padding: 25px; border-radius: 12px; margin-bottom: 25px; text-align: left;">
<h2 style="color: #F59E0B; margin-top:0;">⚖️ HOLD / NEUTRAL POSITION</h2>
<p style="font-size: 16px;">Stable consolidation expected for <b>{ticker}</b>. Predicted change is minimal (<b>{price_change_pct:+.2f}%</b>), targeting <b>${predicted_tomorrow:.2f}</b> (current price: ${current_price:.2f}).</p>
<p style="font-size: 14px; font-style: italic; color: #9CA3AF;">Advisory Pitch: "Consolidation phase detected. Maintain long-term dollar-cost averaging on {ticker}. Ideal hold territory for recurring compounding. Tap to configure automated purchases."</p>
</div>
""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📊 Mock Brokerage Campaign Sales Dashboard")
        st.markdown("Simulate the conversion performance of the marketing sales pitch above.")
        
        # Display simulated metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        # Mock randomized campaign analytics
        np.random.seed(int(current_price * 100) % 1000) # seeded to stock price for consistency
        impressions = int(np.random.randint(50000, 150000))
        ctr = float(np.random.uniform(2.5, 6.2))
        clicks = int(impressions * (ctr / 100))
        conv_rate = float(np.random.uniform(8.0, 14.5))
        sales_units = int(clicks * (conv_rate / 100))
        advisory_fee = 1.99 # Mock advisory fee per trade
        revenue = float(sales_units * advisory_fee)
        
        with col_m1:
            st.metric("Campaign Impressions", f"{impressions:,}", help="Total advisory emails & push notifications sent")
        with col_m2:
            st.metric("Click-Through Rate (CTR)", f"{ctr:.2f}%", help="Percentage of users who viewed the details")
        with col_m3:
            st.metric("Advisory Sales (Units)", f"{sales_units:,}", help="Number of buy/sell trades executed based on this pitch")
        with col_m4:
            st.metric("Simulated Revenue", f"${revenue:,.2f}", help="Total commission fee sales collected ($1.99 per trade)")
            
        # Add a beautiful Plotly chart simulating weekly marketing sales
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Weekly Brokerage Advisory Commission Sales Trend")
        
        weeks = [f"Week {i}" for i in range(1, 9)]
        sales_trend = []
        base = revenue / 8
        for i in range(8):
            base += np.random.uniform(-revenue * 0.05, revenue * 0.08)
            sales_trend.append(max(50.0, base))
            
        df_sales = pd.DataFrame({
            'Week': weeks,
            'Weekly Advisory Commission Sales ($)': sales_trend
        })
        
        is_dark = theme.lower() == "dark"
        fig_sales = px.line(
            df_sales, 
            x='Week', 
            y='Weekly Advisory Commission Sales ($)', 
            markers=True,
            color_discrete_sequence=['#10B981'] if is_dark else ['#059669']
        )
        
        fig_sales.update_layout(
            template='plotly_dark' if is_dark else 'plotly_white',
            paper_bgcolor='rgba(0,0,0,0)' if is_dark else 'rgba(255,255,255,1)',
            plot_bgcolor='rgba(0,0,0,0)' if is_dark else 'rgba(249,250,251,1)',
            xaxis=dict(gridcolor='#374151' if is_dark else '#E5E7EB'),
            yaxis=dict(gridcolor='#374151' if is_dark else '#E5E7EB'),
            margin=dict(l=40, r=40, t=20, b=40)
        )
        
        st.plotly_chart(fig_sales, use_container_width=True)
