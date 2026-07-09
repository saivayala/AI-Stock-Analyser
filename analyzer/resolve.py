"""Map a free-text company name to a Yahoo Finance ticker."""

# Common names -> Yahoo tickers (NSE symbols carry the .NS suffix).
_NAME_MAP = {
    "reliance": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "tata consultancy": "TCS.NS",
    "infosys": "INFY.NS",
    "infy": "INFY.NS",
    "hdfc bank": "HDFCBANK.NS",
    "hdfc": "HDFCBANK.NS",
    "icici": "ICICIBANK.NS",
    "icici bank": "ICICIBANK.NS",
    "sbi": "SBIN.NS",
    "state bank": "SBIN.NS",
    "wipro": "WIPRO.NS",
    "itc": "ITC.NS",
    "airtel": "BHARTIARTL.NS",
    "bharti airtel": "BHARTIARTL.NS",
    "l&t": "LT.NS",
    "larsen": "LT.NS",
    "tata motors": "TATAMOTORS.NS",
    "tata steel": "TATASTEEL.NS",
    "bajaj finance": "BAJFINANCE.NS",
    "asian paints": "ASIANPAINT.NS",
    "maruti": "MARUTI.NS",
    "nifty": "^NSEI",
    "nifty 50": "^NSEI",
    "sensex": "^BSESN",
    "bank nifty": "^NSEBANK",
    "banknifty": "^NSEBANK",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "meta": "META",
    "netflix": "NFLX",
}


def resolve_ticker(user_input: str) -> str:
    """Best-effort mapping of user input to a Yahoo Finance ticker.

    Known company names come from a small lookup table; anything else is
    treated as a ticker symbol as typed. `analyze_stock` retries with a
    .NS suffix if the plain symbol returns no data.
    """
    text = user_input.strip()
    mapped = _NAME_MAP.get(text.lower())
    if mapped:
        return mapped
    return text.upper()
