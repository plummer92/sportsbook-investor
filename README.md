# Sportsbook Investor (Streamlit)

A Streamlit app that scratches the “betting” itch while keeping you disciplined:
- Core portfolio targeting ~7% CAGR
- “Plays” account for small, fun trades
- Trade journal + equity curve
- Simple backtests (SMA cross)
- Kelly/Risk sizing helpers

## Quickstart

```bash
# 1) clone
git clone https://github.com/<you>/sportsbook-investor.git
cd sportsbook-investor

# 2) create venv (optional)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3) install deps
pip install -r requirements.txt

# 4) run
streamlit run app.py
