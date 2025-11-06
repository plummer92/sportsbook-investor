import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import yfinance as yf

st.title("🔬 Simple Backtest: SMA Crossover")

ticker = st.text_input("Ticker", "SPY").upper().strip()
start = st.date_input("Start", value=pd.to_datetime("2015-01-01"))
fast = st.number_input("Fast SMA", 5, step=1, value=20)
slow = st.number_input("Slow SMA", 10, step=1, value=50)

if st.button("Run backtest"):
    try:
        pxs = yf.download(ticker, start=str(start), auto_adjust=True, progress=False)["Close"].dropna()
    except Exception as e:
        st.error(f"Price download failed: {e}")
        st.stop()

    df = pd.DataFrame({"close": pxs})
    df["fast"] = df["close"].rolling(fast).mean()
    df["slow"] = df["close"].rolling(slow).mean()
    df["signal"] = np.where(df["fast"] > df["slow"], 1, 0)
    df["signal_shift"] = df["signal"].shift(1).fillna(0)

    # Strategy return: hold when signal=1
    ret = df["close"].pct_change().fillna(0)
    strat = (1 + ret * df["signal_shift"]).cumprod()
    buyhold = (1 + ret).cumprod()

    chart_df = pd.DataFrame({
        "date": df.index,
        "Strategy": strat.values,
        "Buy & Hold": buyhold.values
    })

    fig = px.line(chart_df, x="date", y=["Strategy", "Buy & Hold"],
                  title=f"{ticker} SMA({fast},{slow}) Strategy vs Buy & Hold",
                  labels={"value":"Growth (start=1.0)"})
    st.plotly_chart(fig, use_container_width=True)

    # Simple stats
    def cagr(series):
        if len(series) < 2:
            return np.nan
        years = (series.index[-1] - series.index[0]).days / 365.25
        return series.iloc[-1] ** (1/years) - 1 if years > 0 else np.nan

    strat_cagr = cagr(strat)
    bh_cagr = cagr(buyhold)
    st.metric("Strategy CAGR", f"{(strat_cagr*100):.2f}%")
    st.metric("Buy & Hold CAGR", f"{(bh_cagr*100):.2f}%")
