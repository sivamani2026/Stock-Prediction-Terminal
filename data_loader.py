import logging
import pandas as pd
import numpy as np
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def generate_mock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Generates high-fidelity mock stock price data for testing/fallback.
    Simulates a geometric Brownian motion (random walk) with drift.
    """
    logger.info(f"Generating mock stock data for {ticker} from {start_date} to {end_date}")
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Generate business days index
    dates = pd.date_range(start=start, end=end, freq='B')
    n_days = len(dates)
    
    if n_days == 0:
        raise ValueError("No business days in the selected date range.")
    
    # Set starting price based on ticker
    ticker_upper = ticker.upper()
    base_prices = {
        'AAPL': 150.0, 'MSFT': 250.0, 'GOOG': 100.0, 'AMZN': 120.0, 'TSLA': 200.0,
        'GC=F': 2300.0, 'SI=F': 30.0, 'TATASTEEL.NS': 170.0, 'TATAMOTORS.NS': 950.0
    }
    start_price = base_prices.get(ticker_upper, 100.0)
    
    # Geometric Brownian Motion parameters
    mu = 0.0005  # Average daily return (drift)
    sigma = 0.02  # Daily volatility
    
    # Daily returns
    daily_returns = np.random.normal(loc=mu, scale=sigma, size=n_days)
    
    # Cumulative price multiplier
    price_multipliers = np.exp(np.cumsum(daily_returns))
    close_prices = start_price * price_multipliers
    
    # Generate OHLC and Volume
    opens = close_prices * (1 + np.random.normal(0, 0.005, n_days))
    highs = np.maximum(opens, close_prices) * (1 + np.abs(np.random.normal(0, 0.003, n_days)))
    lows = np.minimum(opens, close_prices) * (1 - np.abs(np.random.normal(0, 0.003, n_days)))
    volumes = np.random.randint(1000000, 10000000, size=n_days).astype(float)
    
    df = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': close_prices,
        'Adj Close': close_prices,  # Mock Adj Close is identical to Close
        'Volume': volumes
    }, index=dates)
    
    df.index.name = 'Date'
    return df

def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical stock data from Yahoo Finance via yfinance.
    Falls back to mock data if download fails or if there are no internet connections.
    """
    logger.info(f"Attempting to fetch stock data for {ticker} from {start_date} to {end_date}")
    
    try:
        import yfinance as yf
        # Fetch data using yfinance
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            logger.warning(f"No data returned for ticker {ticker}. Falling back to mock data.")
            return generate_mock_data(ticker, start_date, end_date)
            
        logger.info(f"Successfully fetched {len(df)} records for {ticker}")
        
        # Ensure correct column headers if they are multi-index or have specific names
        # yfinance returns single index for single ticker, but sometimes can have multi-index columns.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Ensure the index is named Date
        df.index.name = 'Date'
        
        # Verify required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column {col} in fetched data.")
                
        if 'Adj Close' not in df.columns:
            df['Adj Close'] = df['Close']
            
        return df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        
    except Exception as e:
        logger.error(f"Error fetching data via yfinance: {e}. Falling back to mock data.")
        return generate_mock_data(ticker, start_date, end_date)
