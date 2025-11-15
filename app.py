# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="옵션 달성 확률 분석기", layout="centered")
st.title("미국 주식 옵션 달성 가능성 분석기")
st.caption("by @joseunghye42513 | Black-Scholes + Monte Carlo 시뮬레이션")

with st.sidebar:
    st.header("입력 정보")
    ticker = st.text_input("종목 티커 (예: AAPL, TSLA)", "TSLA").upper()
    target_strike = st.number_input("목표 행사가 (Strike Price)", min_value=0.01, value=300.0)
    option_type = st.selectbox("옵션 타입", ["call", "put"])
    manual_expiry = st.text_input("만기일 (YYYY-MM-DD, 비워두면 자동)", "")
    simulations = st.slider("몬테카를로 시뮬레이션 수", 100, 3000, 1000, 100)

@st.cache_data(ttl=300)
def get_market_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if hist.empty:
            raise ValueError("No history data")
        price = hist['Close'].iloc[-1]
        expirations = stock.options
        if not expirations:
            raise ValueError("No options data")
        return price, expirations, stock
    except Exception as e:
        st.error(f"데이터 로드 오류: {str(e)}. 인터넷 연결이나 티커 확인하세요.")
        raise

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1)

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return 1 - norm.cdf(d1)

def monte_carlo_simulation(S, K, T, r, sigma, option_type, simulations=1000):
    dt = 1/365
    N = max(int(T * 365), 1)
    paths = np.zeros((simulations, N+1))
    paths[:, 0] = S
    for t in range(1, N+1):
        z = np.random.standard_normal(simulations)
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
    final = paths[:, -1]
    itm = final > K if option_type == "call" else final < K
    return paths, np.mean(itm)

if st.button("분석 시작", type="primary"):
    with st.spinner("시장 데이터 가져오는 중..."):
        try:
            current_price, expirations, stock = get_market_data(ticker)
        except:
            st.error("종목을 찾을 수 없습니다. 티커를 확인하세요.")
            st.stop()

    if manual_expiry and manual_expiry in expirations:
        expiry = manual_expiry
    elif expirations:
        expiry = expirations[0]
    else:
        st.error("옵션 데이터가 없습니다.")
        st.stop()

    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
    days_left = (expiry_dt - datetime.now()).days
    T = days_left / 365.0
    if T <= 0:
        st.error("만기일이 지났습니다.")
        st.stop()

    opt = stock.option_chain(expiry)
    chain = opt.calls if option_type == "call" else opt.puts
    near_strike = chain.iloc[(chain['strike'] - target_strike).abs().argsort()[:1]]
    iv = near_strike['impliedVolatility'].iloc[0] if not near_strike.empty else chain['impliedVolatility'].mean()
    iv = max(iv, 0.05)
    r = 0.04

    prob_delta = black_scholes_call(current_price, target_strike, T, r, iv) if option_type == "call" else black_scholes_put(current_price, target_strike, T, r, iv)

    with st.spinner("몬테카를로 시뮬레이션 중..."):
        paths, prob_mc = monte_carlo_simulation(current_price, target_strike, T, r, iv, option_type, simulations)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("현재 주가", f"${current_price:.2f}")
        st.metric("행사가", f"${target_strike:.2f}")
        st.metric("남은 일수", f"{days_left}일")
    with col2:
        st.metric("시장 IV", f"{iv*100:.2f}%")
        st.metric("Delta 확률", f"{prob_delta*100:.2f}%")
        st.metric("몬테카를로 확률", f"{prob_mc*100:.2f}%", delta=f"{(prob_mc-prob_delta)*100:+.2f}p.p.")

    fig = go.Figure()
    for i in range(min(100, simulations)):
        fig.add_trace(go.Scatter(y=paths[i], mode='lines', line=dict(color='lightgray', width=1), showlegend=False))
    fig.add_hline(y=target_strike, line_dash="dash", line_color="red", annotation_text=f"행사가 ${target_strike}")
    fig.add_hline(y=current_price, line_color="blue", annotation_text=f"현재가 ${current_price:.2f}")
    fig.update_layout(
        title=f"{ticker} {option_type.upper()} - 주가 경로 시뮬레이션",
        xaxis_title="일수", yaxis_title="주가 ($)",
        template="plotly_white", height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    st.success(f"**{option_type.upper()} 옵션 달성 확률: {prob_delta*100:.1f}%** (시장 IV 기반)")

else:
    st.info("입력 후 '분석 시작' 버튼을 눌러주세요.")
