"""Composite investability analysis: fundamentals + trend + news + ML signal."""

import numpy as np
import pandas as pd
import yfinance as yf

from analyzer.ml import ml_signal, ml_score
from analyzer.news import fetch_news, news_score

WEIGHTS = {"Fundamentals": 0.35, "Trend": 0.30,
           "News sentiment": 0.20, "ML signal": 0.15}

FORECAST_DAYS = 30


def _fetch_history(ticker: str) -> tuple[str, pd.DataFrame]:
    """Download the last 5 years of OHLCV; retry with .NS for bare symbols."""
    candidates = [ticker]
    if "." not in ticker and not ticker.startswith("^"):
        candidates.append(f"{ticker}.NS")
    for t in candidates:
        df = yf.download(t, period="5y", auto_adjust=True, progress=False)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return t, df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    raise ValueError(f"No price data found for '{ticker}'. "
                     "Check the ticker symbol (e.g. INFY.NS, AAPL).")


def _score_fundamentals(info: dict) -> tuple[float | None, dict]:
    """Heuristic 0-100 score from valuation/quality ratios in Ticker.info."""
    scores, metrics = [], {}

    pe = info.get("trailingPE")
    if isinstance(pe, (int, float)) and pe > 0:
        metrics["P/E (trailing)"] = f"{pe:.1f}"
        scores.append(90 if pe < 15 else 75 if pe < 25 else 55 if pe < 40 else 30)

    roe = info.get("returnOnEquity")
    if isinstance(roe, (int, float)):
        metrics["Return on equity"] = f"{roe:.1%}"
        scores.append(90 if roe > 0.20 else 70 if roe > 0.12 else 45 if roe > 0.05 else 20)

    margin = info.get("profitMargins")
    if isinstance(margin, (int, float)):
        metrics["Profit margin"] = f"{margin:.1%}"
        scores.append(90 if margin > 0.20 else 70 if margin > 0.10 else 45 if margin > 0.03 else 20)

    growth = info.get("revenueGrowth")
    if isinstance(growth, (int, float)):
        metrics["Revenue growth (yoy)"] = f"{growth:.1%}"
        scores.append(90 if growth > 0.15 else 70 if growth > 0.07 else 45 if growth > 0 else 20)

    dte = info.get("debtToEquity")
    if isinstance(dte, (int, float)) and dte >= 0:
        metrics["Debt / equity"] = f"{dte / 100:.2f}"  # yfinance reports in %
        scores.append(90 if dte < 50 else 70 if dte < 100 else 45 if dte < 200 else 20)

    if not scores:
        return None, metrics
    return float(np.mean(scores)), metrics


def _score_trend(close: pd.Series) -> tuple[float, dict]:
    """Heuristic 0-100 score from moving averages, momentum and drawdown."""
    price = float(close.iloc[-1])
    metrics = {}
    pts, total = 0.0, 0.0

    def add(weight: float, frac: float):
        nonlocal pts, total
        total += weight
        pts += weight * max(0.0, min(1.0, frac))

    sma50 = float(close.rolling(50).mean().iloc[-1])
    metrics["Price vs 50d average"] = f"{price / sma50 - 1:+.1%}"
    add(25, 1.0 if price > sma50 else 0.0)

    if len(close) >= 200:
        sma200 = float(close.rolling(200).mean().iloc[-1])
        metrics["Price vs 200d average"] = f"{price / sma200 - 1:+.1%}"
        add(25, 1.0 if price > sma200 else 0.0)
        add(15, 1.0 if sma50 > sma200 else 0.0)

    if len(close) > 126:
        ret6m = price / float(close.iloc[-126]) - 1
        metrics["6-month return"] = f"{ret6m:+.1%}"
        add(20, 0.5 + ret6m / 0.4)          # -20% -> 0, +20% -> 1

    high52 = float(close.iloc[-252:].max())
    dd = price / high52 - 1
    metrics["Off 52-week high"] = f"{dd:+.1%}"
    add(15, 1 + dd / 0.30)                  # at high -> 1, -30% -> 0

    return 100 * pts / total, metrics


def _forecast(close: pd.Series, days: int = FORECAST_DAYS) -> pd.DataFrame:
    """Statistical price projection: drift + volatility cone from the
    last year of daily returns. `lower`/`upper` is an ~80% range."""
    rets = close.pct_change().dropna().iloc[-252:]
    mu, sigma = float(rets.mean()), float(rets.std())
    price = float(close.iloc[-1])

    t = np.arange(1, days + 1)
    expected = price * (1 + mu) ** t
    spread = 1.28 * sigma * np.sqrt(t)      # z=1.28 -> ~80% interval
    dates = pd.bdate_range(close.index[-1] + pd.Timedelta(days=1), periods=days)

    fc = pd.DataFrame({"expected": expected,
                       "lower": expected * np.exp(-spread),
                       "upper": expected * np.exp(spread)}, index=dates)
    fc.index.name = "Date"
    return fc


def analyze_stock(ticker: str) -> dict:
    """Run the full analysis; returns the dict rendered by app.py."""
    resolved, df = _fetch_history(ticker)
    close = df["Close"]

    try:
        info = yf.Ticker(resolved).info or {}
    except Exception:
        info = {}

    fund_score, fund_metrics = _score_fundamentals(info)
    trend_score, trend_metrics = _score_trend(close)
    news = fetch_news(resolved)

    try:
        ml = ml_signal(df)
    except Exception:
        ml = None

    pillars = {
        "Fundamentals": fund_score,
        "Trend": trend_score,
        "News sentiment": news_score(news),
        "ML signal": ml_score(ml) if ml else None,
    }

    # Composite = weighted mean over the pillars we actually have
    avail = {k: v for k, v in pillars.items() if v is not None}
    wsum = sum(WEIGHTS[k] for k in avail)
    composite = sum(WEIGHTS[k] * v for k, v in avail.items()) / wsum

    verdict = ("Favourable profile" if composite >= 65
               else "Mixed picture" if composite >= 45
               else "Weak profile")

    metrics = {**fund_metrics, **trend_metrics}
    mcap = info.get("marketCap")
    if isinstance(mcap, (int, float)) and mcap > 0:
        metrics["Market cap"] = f"{mcap / 1e9:,.1f}B {info.get('currency', '')}"

    history = pd.DataFrame({
        "Close": close,
        "SMA 50": close.rolling(50).mean(),
        "SMA 200": close.rolling(200).mean(),
    })
    history.index.name = "Date"

    return {
        "name": info.get("longName") or info.get("shortName") or resolved,
        "price": float(close.iloc[-1]),
        "currency": info.get("currency") or "",
        "composite": composite,
        "verdict": verdict,
        "pillars": pillars,
        "weights": WEIGHTS,
        "metrics": metrics,
        "ml": ml,
        "news": news,
        "history": history,
        "forecast": _forecast(close),
    }
