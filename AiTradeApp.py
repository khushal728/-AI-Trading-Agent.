import streamlit as st
import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
import os
import sys
import pandas as pd
import plotly.graph_objects as go

# Streamlit Page Config (must be the first command)
st.set_page_config(page_title="AI-Powered Stock Trading Agent", layout="centered")

# Define AI Model
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Load trained model with error handling
@st.cache_resource
def load_model():
    model = DQN(state_size=4, action_size=3)
    torch.save(model.state_dict(), "trading_model.pth")
    if os.path.exists("trading_model.pth"):
        model.load_state_dict(torch.load("trading_model.pth", map_location=torch.device('cpu')))
        model.eval()
        return model
    else:
        st.warning("⚠️ Model file 'trading_model.pth' not found. Running with an untrained model.")
        return model

model = load_model()

# Compute RSI (Relative Strength Index)
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Define function to get stock data
def get_stock_data(symbol):
    try:
        data = yf.download(symbol, period="3mo", interval="1d")
        
        if data.empty:
            st.error(f"❌ No data found for '{symbol}'. Please check the symbol and try again.")
            return None
        
        data['SMA_5'] = data['Close'].rolling(window=5).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['Returns'] = data['Close'].pct_change()
        data['RSI'] = compute_rsi(data['Close'])
        data.dropna(inplace=True)
        
        if data.empty:
            st.error("❌ Not enough historical data for analysis.")
            return None
        
        return data
    except Exception as e:
        st.error(f"⚠️ Error fetching data: {str(e)}")
        return None


# Normalize input features
def normalize_features(values):
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return values  # Avoid division by zero
    return (values - mean) / std

# Define function to predict action
def predict_action(data):
    try:
        close_price = float(data['Close'].iloc[-1])
        sma_5 = float(data['SMA_5'].iloc[-1])
        sma_20 = float(data['SMA_20'].iloc[-1])
        returns = float(data['Returns'].iloc[-1])

        features = np.array([close_price, sma_5, sma_20, returns], dtype=np.float32)
        normalized_features = normalize_features(features)
        state_tensor = torch.FloatTensor(normalized_features).unsqueeze(0)
        
        with torch.no_grad():
            q_values = model(state_tensor)
        action = torch.argmax(q_values).item()
        confidence = torch.softmax(q_values, dim=1).max().item() * 100  # Confidence score in %
        return ["HOLD", "BUY", "SELL"][action], confidence
    except Exception as e:
        st.error(f"⚠️ Error in prediction: {str(e)}")
        return "HOLD", 0.0

# Streamlit UI
st.title("📈 AI-Powered Stock Trading Agent")

symbol = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
if st.button("Get AI Prediction"):
    data = get_stock_data(symbol)
    if data is not None:
        action, confidence = predict_action(data)
        st.success(f"✅ AI Recommendation: **{action}** (Confidence: {confidence:.2f}%)")

        # Plot stock price with moving averages
        st.subheader("Stock Price with Moving Averages")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_5'], mode='lines', name='SMA 5', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], mode='lines', name='SMA 20', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='SMA 50', line=dict(dash='longdash')))
        fig.update_layout(title=f"{symbol} Stock Price with Moving Averages", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)
        
        # Plot RSI Indicator
        st.subheader("Relative Strength Index (RSI)")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI'))
        fig_rsi.add_hline(y=70, line=dict(color='red', dash='dash'), annotation_text='Overbought')
        fig_rsi.add_hline(y=30, line=dict(color='green', dash='dash'), annotation_text='Oversold')
        st.plotly_chart(fig_rsi)
