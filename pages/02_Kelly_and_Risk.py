import streamlit as st

st.title("🎯 Kelly & Risk Sizing")

st.markdown("""Use this to size your **Plays** like bets:
- **Win rate (W):** your historical win probability
- **Payoff ratio (R):** average win size / average loss size
Kelly fraction f* = W - (1 - W)/R
""")

col1, col2 = st.columns(2)
with col1:
    W = st.slider("Win rate (W)", 0.0, 1.0, 0.55, step=0.01)
    R = st.slider("Payoff ratio (R)", 0.1, 5.0, 1.2, step=0.05)
with col2:
    bankroll = st.number_input("Plays Bankroll ($)", 1000.0, step=50.0, value=1000.0)
    risk_cap = st.slider("Cap per play (% of bankroll)", 0.0, 1.0, 0.05, step=0.01)

kelly = W - (1 - W) / R if R > 0 else 0.0
kelly = max(0.0, kelly)

st.metric("Kelly fraction", f"{kelly*100:.1f}%")

suggested = bankroll * min(kelly, risk_cap)
st.metric("Suggested $ per play", f"${suggested:,.2f}")

st.caption("Note: Full Kelly is aggressive; many pros use half-Kelly or less.")
