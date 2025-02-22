# AI-Trading-Agent.

## AI-Powered Stock Trading Agent

# 📌 Overview

This is a Streamlit-based AI-powered stock trading agent that analyzes stock data and provides recommendations to BUY, SELL, or HOLD using a Deep Q-Network (DQN). The model considers various stock indicators, such as moving averages and RSI, to predict optimal actions.

# 🚀 Features

Fetches real-time stock data using Yahoo Finance (yfinance).

Computes Simple Moving Averages (SMA) and Relative Strength Index (RSI).

Uses a Deep Q-Network (DQN) model for trading predictions.

Provides confidence scores for AI recommendations.

Displays interactive stock charts with price trends and RSI levels.

Fully interactive Streamlit UI for easy usage.

# 📦 Requirements

Ensure you have Python installed, then install the required dependencies:
```
pip install streamlit yfinance numpy torch pandas plotly
```

# 🎯 How to Run

Clone the repository or copy the script.

Place the trained model (trading_model.pth) in the same directory.

Run the Streamlit app:

```
streamlit run app.py
```

# 🛠 Troubleshooting

Model Not Found Error

If you see: ⚠️ Model file 'trading_model.pth' not found. Running with an untrained model.

Ensure trading_model.pth is present in the project directory.

If missing, retrain the model or use an alternative approach.

No Data Found for Stock Symbol

If you see: ❌ No data found for the given stock symbol.

Check if the stock symbol is correct (e.g., use AAPL, GOOGL).

Ensure you have an active internet connection.

Try manually fetching data using:

```
import yfinance as yf
print(yf.download("AAPL", period="3mo", interval="1d").head())
```
If the issue persists, Yahoo Finance servers may be down.

# 🏗 Future Improvements

Improve model accuracy with additional technical indicators.

Add support for multiple stocks in a portfolio.

Implement reinforcement learning for enhanced predictions.

# 👨‍💻 Author

Developed by [Your Name].

# 📜 License

This project is open-source under the MIT License.





