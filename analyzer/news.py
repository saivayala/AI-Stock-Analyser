"""News fetching and headline sentiment (VADER) for a ticker."""

from datetime import datetime

import numpy as np
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()

MAX_ITEMS = 10


def _normalize(item: dict) -> dict | None:
    """yfinance has shipped two news schemas; handle both."""
    content = item.get("content") or item
    title = content.get("title")
    if not title:
        return None

    link = ""
    url = content.get("canonicalUrl") or content.get("clickThroughUrl")
    if isinstance(url, dict):
        link = url.get("url", "")
    link = link or item.get("link", "")

    provider = content.get("provider")
    publisher = (provider.get("displayName") if isinstance(provider, dict)
                 else item.get("publisher", "")) or ""

    date = ""
    if content.get("pubDate"):
        date = str(content["pubDate"])[:10]
    elif item.get("providerPublishTime"):
        date = datetime.fromtimestamp(item["providerPublishTime"]).strftime("%Y-%m-%d")

    summary = content.get("summary") or ""
    return {"title": title, "summary": summary, "publisher": publisher,
            "date": date, "link": link}


def fetch_news(ticker: str) -> list[dict]:
    """Recent headlines with a per-item VADER sentiment in [-1, 1]."""
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []

    news = []
    for item in raw[:MAX_ITEMS]:
        norm = _normalize(item)
        if norm is None:
            continue
        text = f"{norm['title']}. {norm['summary']}".strip()
        norm["sentiment"] = _vader.polarity_scores(text)["compound"]
        news.append(norm)
    return news


def news_score(news: list[dict]) -> float | None:
    """0-100 pillar score from mean headline sentiment (50 = neutral)."""
    if not news:
        return None
    mean = float(np.mean([n["sentiment"] for n in news]))
    return float(np.clip(50 + mean * 50, 0, 100))
