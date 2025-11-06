python
from __future__ import annotations

import io
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Sportsbook Investor", layout="wide")

# ---------------------- Helpers ---------------------- #
def init_state():
    if "trades" not in st.session_state:
        st.session_state.trades = pd.DataFrame(
            columns=["date", "ticker", "side", "qty", "price", "fees", "notes"]
        )
    if "core_value_history" not in st.session_state:
        st.session_state.core_value_history = pd.DataFrame(
            columns=["date", "value"]
        )

def parse_date(x):
    if isinstance(x, (datetime, date)):
        return pd.to_datetime(x)
    try:
        return pd.to_datetime(str(x))
    except Exception:
        return pd.NaT

def load_prices(tickers: list[str], start: str = "2020-01-01"):
    tickers = sorted(set(tickers))
    if not tickers:
        return pd.DataFrame()
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    # If single ticker, yfinance returns Series-like column; normalize:
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:
        close = data[["Close"]].rename(columns={"Close": tickers[0]})
    return close.ffill().dropna()

def compute_equity(trades: pd.DataFrame, price_df: pd.DataFrame):
    """
    Mark-to-market equity curve from trades and closing prices.
    Assumes all trades are market orders on 'date' close.
    """
    if trades.empty or price_df.empty:
        return pd.DataFrame(columns=["date", "equity"])

    # Normalize
    t = trades.copy()
    t["date"] = t["date"].apply(parse_date)
    t = t.dropna(subset=["date"])
    t["qty"] = pd.to_numeric(t["qty"], errors="coerce").fillna(0.0)
    t["price"] = pd.to_numeric(t["price"], errors="coerce").fillna(0.0)
    t["fees"] = pd.to_numeric(t["fees"], errors="coerce").fillna(0.0)
    t["side"] = t["side"].str.upper().str.strip()

    # Portfolio positions over time (simple running net position per ticker)
    tickers = sorted(t["ticker"].dropna().unique())
    calendar = price_df.index
    pos = pd.DataFrame(0.0, index=calendar, columns=tickers)

    for _, r in t.iterrows():
        d = parse_date(r["date"])
        if pd.isna(d):
            continue
        if d not in pos.index:
            # align to next available trading day
            d = calendar[calendar.get_indexer([d], method="bfill")[0]]
        qty = r["qty"] * (1 if r["side"] == "BUY" else -1)
        pos.loc[d:, r["ticker"]] += qty

    # Value = sum(position * price) for all tickers
    aligned_prices = price_df.reindex(pos.index).ffill()
    value = (pos * aligned_prices).sum(axis=1)

    # Cash from trades (cash decreases on BUY, increases on SELL) minus fees
    # We assume initial cash = 0 and show mark-to-market equity (positions only).
    # If you want total account, add starting_cash.
    cash = 0.0
    # Build realized cash series (optional extension)

    equity = value + cash
    out = equity.reset_index()
    out.columns = ["date", "equity"]
    return out

def target_growth_curve(start_value: float, start_date, end_date, target_cagr=0.07):
    """
    Deterministic 7% CAGR target curve, daily compounding from start_date to end_date.
    """
    idx = pd.date_range(start=start_date, end=end_date, freq="D")
    days = (idx - idx[0]).days.values
    daily_rate = (1 + target_cagr) ** (1/365.0) - 1
    curve = start_value * (1 + daily_rate) ** days
    return pd.DataFrame({"date": idx, "target": curve})

def df_to_csv_download(df: pd.DataFrame, filename="trades.csv"):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return csv_bytes, filename

# ---------------------- UI ---------------------- #
init_state()
st.title("🏈 Sportsbook Investor — get the rush, keep the edge")

with st.sidebar:
    st.subheader("Portfolio Setup")
    starting_core = st.number_input("Starting Core ($)", 10000, step=500, value=10000)
    starting_plays = st.number_input("Starting Plays ($)", 1000, step=100, value=1000)
    target_cagr = st.slider("Target CAGR (Core)", 0.00, 0.20, 0.07, step=0.005)
    start_date = st.date_input("Start Date", value=date(2025, 1, 1))
    st.caption("Tip: keep 'Plays' small for the dopamine without the damage.")

tab1, tab2, tab3 = st.tabs(["📈 Dashboard", "🧾 Trade Journal", "🧪 Notes & Tips"])

# ---------------------- Dashboard ---------------------- #
with tab1:
    st.subheader("Equity vs. 7% Target")

    # Upload journal (optional)
    uploaded = st.file_uploader("Upload trade journal CSV (optional)", type=["csv"])
    if uploaded:
        st.session_state.trades = pd.read_csv(uploaded)

    colA, colB = st.columns([2, 1])

    with colA:
        # Build price list from journal
        tickers = sorted(st.session_state.trades["ticker"].dropna().unique().tolist())
        price_df = load_prices(tickers, start=str(start_date)) if tickers else pd.DataFrame()

        equity_df = compute_equity(st.session_state.trades, price_df)
        if equity_df.empty:
            st.info("Add trades in the Trade Journal to generate an equity curve.")
        else:
            equity_df = equity_df[~equity_df["date"].duplicated()].sort_values("date")
            equity_df["equity_total"] = equity_df["equity"] + starting_core + starting_plays
            tgt = target_growth_curve(
                start_value=starting_core, start_date=start_date,
                end_date=equity_df["date"].iloc[-1], target_cagr=target_cagr
            )
            # Merge for plotting
            plot_df = pd.merge(
                equity_df[["date", "equity_total"]],
                tgt, on="date", how="outer"
            ).sort_values("date")

            fig = px.line(
                plot_df,
                x="date",
                y=["equity_total", "target"],
                labels={"value": "Portfolio Value ($)", "date": "Date"},
                title="Total Equity vs 7% Target (Core only target)"
            )
            st.plotly_chart(fig, use_container_width=True)

    with colB:
        st.metric("Starting Core", f"${starting_core:,.0f}")
        st.metric("Starting Plays", f"${starting_plays:,.0f}")
        if not equity_df.empty:
            current_val = equity_df["equity_total"].iloc[-1]
            core_target_today = tgt["target"].iloc[-1] + starting_plays  # apples-ish
            diff = current_val - core_target_today
            st.metric("Δ vs Target", f"${diff:,.0f}")

    st.divider()
    st.subheader("Positions Snapshot")
    if len(tickers) == 0:
        st.write("No tickers yet.")
    else:
        last_prices = {}
        for tkr in tickers:
            try:
                last_prices[tkr] = yf.Ticker(tkr).history(period="5d")["Close"].iloc[-1]
            except Exception:
                last_prices[tkr] = np.nan
        st.write(pd.DataFrame({"ticker": list(last_prices.keys()),
                               "last_close": list(last_prices.values())}))

# ---------------------- Trade Journal ---------------------- #
with tab2:
    st.subheader("Add Trades")

    with st.form("add_trade"):
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            dt = st.date_input("Date", value=date.today())
            ticker = st.text_input("Ticker", value="SPY").upper().strip()
        with c2:
            side = st.selectbox("Side", ["BUY", "SELL"])
            qty = st.number_input("Qty", min_value=0.0, step=1.0, value=1.0)
        with c3:
            price = st.number_input("Price", min_value=0.0, step=0.01, value=100.00)
            fees = st.number_input("Fees", min_value=0.0, step=0.01, value=0.00)
        notes = st.text_input("Notes", value="")
        submitted = st.form_submit_button("Add trade")

        if submitted:
            new_row = {
                "date": pd.to_datetime(dt),
                "ticker": ticker,
                "side": side,
                "qty": qty,
                "price": price,
                "fees": fees,
                "notes": notes
            }
            st.session_state.trades = pd.concat(
                [st.session_state.trades, pd.DataFrame([new_row])],
                ignore_index=True
            )
            st.success(f"Added {side} {qty} {ticker} @ {price}")

    st.subheader("Journal")
    st.dataframe(st.session_state.trades, use_container_width=True, height=350)

    left, right = st.columns(2)
    with left:
        csv_bytes, filename = df_to_csv_download(st.session_state.trades)
        st.download_button("⬇️ Download CSV", data=csv_bytes, file_name=filename, mime="text/csv")
    with right:
        if st.button("🗑️ Clear all (session only)"):
            st.session_state.trades = st.session_state.trades.iloc[0:0]
            st.success("Cleared journal (this session).")

# ---------------------- Notes ---------------------- #
with tab3:
    st.markdown(
        """
        ### Tips
        - Keep **Core** boring, compounding toward ~7% CAGR.
        - Use **Plays** small for the rush; track **every** trade.
        - Favor **edge** over outcomes: improve your process, not your luck.
        """
    )

