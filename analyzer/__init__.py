"""Stock Investability Analyzer — analysis logic.

Public API used by the Streamlit UI (app.py):
    resolve_ticker(user_input) -> str
    analyze_stock(ticker)      -> dict
"""

from analyzer.resolve import resolve_ticker
from analyzer.core import analyze_stock

__all__ = ["resolve_ticker", "analyze_stock"]
