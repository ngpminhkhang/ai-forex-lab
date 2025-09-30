from fastapi import FastAPI
import MetaTrader5 as mt5
import pandas as pd
from textblob import TextBlob
from fastapi.middleware.cors import CORSMiddleware
import logging
from datetime import datetime, timedelta
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Cho phép frontend connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Biến để kiểm tra MT5 connection
mt5_connected = False

@app.on_event("startup")
async def startup_event():
    """Khởi tạo MT5 khi app start"""
    global mt5_connected
    try:
        # Initialize MT5
        if not mt5.initialize():
            logger.error(f"MT5 initialize() failed, error code = {mt5.last_error()}")
            return
            
        # Login to demo account - THAY BẰNG THÔNG TIN THẬT CỦA BẠN
        account = 590930856  # Thay bằng số tài khoản demo của bạn
        password = "!wiXP011zY"  # Thay bằng mật khẩu demo
        server = "FxPro-MT5 Demo"  # Thay bằng server broker của bạn
        
        if not mt5.login(account, password=password, server=server):
            logger.error(f"MT5 login() failed, error code = {mt5.last_error()}")
            mt5.shutdown()
            return
            
        mt5_connected = True
        logger.info("MT5 connected successfully")
        logger.info(f"Account: {mt5.account_info().login}")
        logger.info(f"Balance: {mt5.account_info().balance}")
        
    except Exception as e:
        logger.error(f"MT5 connection failed: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "AI Forex Lab API is running", "mt5_connected": mt5_connected}

@app.get("/fetch_data")
def fetch_data(pair: str = "EURUSD", period: str = "1y", rebalance: str = "monthly", cost: float = 0.05):
    try:
        # Nếu MT5 không kết nối được, trả về dữ liệu giả
        if not mt5_connected:
            logger.info("Using mock data (MT5 not connected)")
            return get_mock_data(pair, period, cost)
            
        # Lấy dữ liệu thật từ MT5
        return get_real_mt5_data(pair, period, cost)
        
    except Exception as e:
        logger.error(f"Error in fetch_data: {str(e)}", exc_info=True)
        return {"error": str(e)}

def get_real_mt5_data(pair: str, period: str, cost: float):
    """Lấy dữ liệu thật từ MT5"""
    
    # Map period to days và timeframe
    period_map = {
        "1y": {"days": 365, "timeframe": mt5.TIMEFRAME_D1},
        "3y": {"days": 1095, "timeframe": mt5.TIMEFRAME_D1},
        "5y": {"days": 1825, "timeframe": mt5.TIMEFRAME_W1}
    }
    
    if period not in period_map:
        period = "1y"
    
    config = period_map[period]
    days = config["days"]
    timeframe = config["timeframe"]
    
    # Tính thời gian bắt đầu
    utc_from = datetime.now() - timedelta(days=days)
    utc_from_timestamp = int(utc_from.timestamp())
    
    # Lấy dữ liệu từ MT5
    rates = mt5.copy_rates_from(pair, timeframe, utc_from_timestamp, days)
    
    if rates is None or len(rates) == 0:
        logger.error(f"No data fetched for {pair}, error code = {mt5.last_error()}")
        return get_mock_data(pair, period, cost)  # Fallback to mock data
    
    # Convert to DataFrame
    data = pd.DataFrame(rates)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    
    # Chọn và đổi tên columns
    data = data[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
    data = data.rename(columns={'close': 'Close', 'tick_volume': 'Volume'})
    
    logger.info(f"Real MT5 data shape: {data.shape}")
    logger.info(f"Date range: {data['time'].min()} to {data['time'].max()}")
    
    # Tính toán sentiment dựa trên price movement và volume
    data['price_change'] = data['Close'].pct_change()
    data['volume_change'] = data['Volume'].pct_change()
    
    # Sentiment đơn giản: positive nếu price tăng và volume tăng
    data['sentiment'] = np.where(
        (data['price_change'] > 0) & (data['volume_change'] > 0), 0.3,
        np.where((data['price_change'] < 0) & (data['volume_change'] > 0), -0.3, 0.1)
    )
    
    sentiment_mean = float(data['sentiment'].mean())
    
    # Tính portfolio value
    initial_investment = 10000  # $10,000
    data['portfolio_value'] = initial_investment * (1 + data['Close'].pct_change().cumsum()) * (1 + sentiment_mean - cost/100)
    data['portfolio_value'] = data['portfolio_value'].fillna(initial_investment)
    
    # Weights và insight
    weights = {
        pair: 0.4 + min(max(sentiment_mean, -0.3), 0.3),
        "GBPUSD": 0.3,
        "USDJPY": 0.3
    }
    
    # Normalize weights để tổng = 1
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}
    
    if sentiment_mean < -0.1:
        insight = f"Reduce {pair} exposure due to negative sentiment"
    elif sentiment_mean > 0.1:
        insight = f"Increase {pair} allocation for potential gains"
    else:
        insight = "Maintain current portfolio allocation"
    
    # Tính metrics
    returns = data['portfolio_value'].pct_change().dropna()
    
    if len(returns) == 0:
        metrics = get_default_metrics()
    else:
        # Sharpe Ratio
        returns_mean = returns.mean()
        returns_std = returns.std()
        sharpe = returns_mean / returns_std * (252 ** 0.5) if returns_std != 0 else 0
        
        # Max Drawdown
        data['cummax'] = data['portfolio_value'].cummax()
        data['drawdown'] = (data['cummax'] - data['portfolio_value']) / data['cummax']
        max_drawdown = data['drawdown'].max() * 100
        
        # Win Rate
        win_rate = (returns > 0).mean() * 100
        
        # Calmar Ratio
        calmar = sharpe / (max_drawdown / 100) if max_drawdown > 0 else float('inf')
        
        metrics = {
            "Sharpe Ratio": round(float(sharpe), 3),
            "Max Drawdown": f"{max_drawdown:.2f}%",
            "Win Rate": f"{win_rate:.2f}%",
            "Calmar Ratio": f"{calmar:.2f}" if calmar != float('inf') else "Infinity",
            "Total Return": f"{(data['portfolio_value'].iloc[-1] - initial_investment) / initial_investment * 100:.2f}%"
        }
    
    # Chuẩn bị dữ liệu trả về
    result_data = {
        'time': data['time'].dt.strftime('%Y-%m-%d').tolist(),
        'Close': data['Close'].tolist(),
        'portfolio_value': data['portfolio_value'].tolist(),
        'sentiment': data['sentiment'].tolist(),
        'Volume': data['Volume'].tolist()
    }
    
    return {
        "data": result_data,
        "weights": weights,
        "insight": insight,
        "metrics": metrics,
        "cost": cost,
        "data_source": "MT5 Real Data"
    }

def get_mock_data(pair: str, period: str, cost: float):
    """Tạo dữ liệu giả để test"""
    # Tạo dates
    days = 252 if period == "1y" else 756 if period == "3y" else 1260
    dates = [datetime.now() - timedelta(days=x) for x in range(days, 0, -1)]
    
    # Tạo price data giả với random walk
    prices = [1.0]
    for i in range(1, days):
        change = np.random.normal(0, 0.01)
        prices.append(max(0.1, prices[-1] * (1 + change)))  # Đảm bảo price > 0
    
    # Tạo sentiment giả
    sentiments = np.random.normal(-0.1, 0.2, days)
    
    # Tính portfolio value
    portfolio_values = [10000]  # $10,000 initial
    for i in range(1, days):
        ret = (prices[i] - prices[i-1]) / prices[i-1] if prices[i-1] != 0 else 0
        port_val = portfolio_values[-1] * (1 + ret + sentiments[i] - cost/100)
        portfolio_values.append(port_val)
    
    data = {
        'time': [d.strftime('%Y-%m-%d') for d in dates],
        'Close': prices,
        'portfolio_value': portfolio_values,
        'sentiment': sentiments.tolist(),
        'Volume': np.random.randint(1000, 10000, days).tolist()
    }
    
    weights = {"EURUSD": 0.4, "GBPUSD": 0.3, "USDJPY": 0.3}
    insight = "Market conditions stable - using mock data"
    
    metrics = {
        "Sharpe Ratio": 1.2,
        "Max Drawdown": "5.2%",
        "Win Rate": "55.8%", 
        "Calmar Ratio": "0.85",
        "Total Return": "12.5%"
    }
    
    return {
        "data": data,
        "weights": weights,
        "insight": insight,
        "metrics": metrics,
        "cost": cost,
        "data_source": "Mock Data (MT5 not connected)"
    }

def get_default_metrics():
    """Metrics mặc định khi không tính được"""
    return {
        "Sharpe Ratio": 0.0,
        "Max Drawdown": "0.0%",
        "Win Rate": "0.0%",
        "Calmar Ratio": "0.0",
        "Total Return": "0.0%"
    }

# Các endpoint khác cho MVP
@app.get("/backtest")
def backtest_studio():
    return {"page": "Backtesting Studio", "features": ["Strategy testing", "Performance analytics", "Optimization"]}

@app.get("/community")
def community_voting():
    return {"page": "Community Voting", "features": ["Strategy voting", "User rankings", "Discussion forums"]}

@app.get("/academy")
def academy():
    return {"page": "Trading Academy", "content": "Educational resources and tutorials"}

@app.get("/profile")
def user_profile():
    return {"page": "User Profile", "features": ["Portfolio overview", "Settings", "History"]}

@app.get("/about")
def about():
    return {"page": "About & Methodology", "content": "AI Forex Lab methodology and team information"}

# Shutdown MT5 khi app kết thúc
import atexit
@atexit.register
def shutdown_mt5():
    if mt5_connected:
        mt5.shutdown()
        logger.info("MT5 shutdown")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)