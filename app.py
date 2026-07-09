"""Stock Investability Analyzer — UI layer (Streamlit).

All analysis logic lives in the `analyzer` package. This file only
handles input, calls `analyze_stock()`, and renders the result dict.
The result is kept in session state so widget interactions (e.g. the
chart range selector) don't clear the page.

Run:
    streamlit run app.py
"""

import altair as alt
import pandas as pd
import streamlit as st

from analyzer import resolve_ticker, analyze_stock

# Validated categorical palette (slots 1-3) + sequential step for the band
C_CLOSE, C_SMA50, C_SMA200, C_BAND = "#2a78d6", "#1baf7a", "#eda100", "#86b6ef"

# Chart range options -> lookback in calendar days (None = full history)
RANGES = {"1W": 7, "1M": 30, "6M": 182, "1Y": 365,
          "2Y": 730, "3Y": 1095, "Max": None}

st.set_page_config(page_title="Stock Investability Analyzer", page_icon="📊",
                   layout="centered")

st.title("📊 Stock Investability Analyzer")
st.caption("Fundamentals + trend + news sentiment + ML signal over the last "
           "5 years, combined into one score. Educational tool — not "
           "financial advice.")

user_input = st.text_input("Company name or ticker",
                           placeholder="e.g. Reliance, TCS, INFY.NS, AAPL")

if st.button("Analyze", type="primary") and user_input:
    ticker = resolve_ticker(user_input)
    with st.spinner(f"Analyzing {ticker}..."):
        try:
            st.session_state["result"] = analyze_stock(ticker)
        except ValueError as e:
            st.session_state.pop("result", None)
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.session_state.pop("result", None)
            st.error(f"Could not fetch data: {e}")
            st.stop()

result = st.session_state.get("result")
if result:
    # --- Header ---
    st.header(result["name"])
    st.metric("Current price", f"{result['price']:,.2f} {result['currency']}")

    # --- Price history chart with range selector ---
    st.subheader("Price history")
    choice = st.segmented_control("Chart range", list(RANGES), default="1Y",
                                  key="chart_range",
                                  label_visibility="collapsed") or "1Y"
    history = result["history"]
    days = RANGES[choice]
    if days is not None:
        cutoff = history.index.max() - pd.Timedelta(days=days)
        view = history[history.index >= cutoff]
    else:
        view = history

    # On short windows the long SMAs are flat lines that squash the
    # price scale — show the close price alone.
    series_colors = {"Close": C_CLOSE, "SMA 50": C_SMA50, "SMA 200": C_SMA200}
    if choice in ("1W", "1M"):
        series_colors = {"Close": C_CLOSE}
    hist = (view[list(series_colors)].reset_index()
            .melt("Date", var_name="Series", value_name="Price").dropna())
    price_chart = alt.Chart(hist).mark_line(strokeWidth=2).encode(
        x=alt.X("Date:T", title=None),
        y=alt.Y("Price:Q", title=None, scale=alt.Scale(zero=False)),
        color=alt.Color("Series:N",
                        scale=alt.Scale(domain=list(series_colors),
                                        range=list(series_colors.values())),
                        legend=(alt.Legend(orient="top", title=None)
                                if len(series_colors) > 1 else None)),
        tooltip=[alt.Tooltip("Date:T"), "Series:N",
                 alt.Tooltip("Price:Q", format=",.2f")],
    ).properties(height=320)
    st.altair_chart(price_chart, width="stretch")

    first, last = float(view["Close"].iloc[0]), float(view["Close"].iloc[-1])
    st.caption(f"{choice} change: {last / first - 1:+.1%} "
               f"({view.index.min().date()} → {view.index.max().date()})")

    # --- Verdict ---
    st.divider()
    score, verdict = result["composite"], result["verdict"]
    banner = {"Favourable profile": st.success,
              "Mixed picture": st.warning,
              "Weak profile": st.error}[verdict]
    banner(f"## Overall score: {score:.0f}/100 — {verdict}")

    # --- Pillar breakdown ---
    st.subheader("Score breakdown")
    for pillar, pscore in result["pillars"].items():
        weight = result["weights"][pillar]
        if pscore is None:
            st.write(f"**{pillar}** — data unavailable")
        else:
            st.write(f"**{pillar}** ({weight:.0%} weight)")
            st.progress(int(pscore), text=f"{pscore:.0f}/100")

    # --- 30-day projection ---
    st.divider()
    st.subheader("30-day price projection")
    fc = result["forecast"]
    exp_price = float(fc["expected"].iloc[-1])
    delta = exp_price / result["price"] - 1
    st.metric("Projected price in 30 trading days",
              f"{exp_price:,.2f} {result['currency']}", f"{delta:+.1%}")
    st.caption(f"~80% range: {fc['lower'].iloc[-1]:,.2f} – "
               f"{fc['upper'].iloc[-1]:,.2f} {result['currency']}. "
               "Statistical projection from the last year's drift and "
               "volatility — not a guarantee.")

    recent = (result["history"]["Close"].iloc[-126:]
              .rename("Price").reset_index())
    fc_reset = fc.reset_index()
    band = alt.Chart(fc_reset).mark_area(opacity=0.25, color=C_BAND).encode(
        x=alt.X("Date:T", title=None),
        y=alt.Y("lower:Q", title=None, scale=alt.Scale(zero=False)),
        y2="upper:Q",
    )
    expected = alt.Chart(fc_reset).mark_line(
        strokeWidth=2, strokeDash=[5, 4], color=C_CLOSE).encode(
        x="Date:T", y=alt.Y("expected:Q", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("Date:T"),
                 alt.Tooltip("expected:Q", format=",.2f", title="Expected"),
                 alt.Tooltip("lower:Q", format=",.2f", title="Low (80%)"),
                 alt.Tooltip("upper:Q", format=",.2f", title="High (80%)")],
    )
    actual = alt.Chart(recent).mark_line(strokeWidth=2, color=C_CLOSE).encode(
        x="Date:T", y=alt.Y("Price:Q", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Price:Q", format=",.2f")],
    )
    st.altair_chart((band + expected + actual).properties(height=280),
                    width="stretch")
    st.caption("Solid line: last 6 months. Dashed line: expected path. "
               "Shaded area: ~80% likely range.")

    # --- News & sentiment ---
    news = result["news"]
    if news:
        st.divider()
        st.subheader("Recent news & sentiment")
        nscore = result["pillars"]["News sentiment"]
        if nscore is not None:
            mood = ("positive" if nscore >= 60
                    else "negative" if nscore <= 40 else "neutral")
            st.write(f"Overall headline sentiment: **{mood}** "
                     f"({nscore:.0f}/100)")
        for item in news:
            s = item["sentiment"]
            badge = (f":green[▲ {s:+.2f}]" if s > 0.05
                     else f":red[▼ {s:+.2f}]" if s < -0.05
                     else f":gray[• {s:+.2f}]")
            title = (f"[{item['title']}]({item['link']})" if item["link"]
                     else item["title"])
            source = " — ".join(x for x in (item["publisher"], item["date"]) if x)
            st.markdown(f"{badge} {title}  \n:gray[{source}]")

    # --- Details ---
    with st.expander("Key metrics"):
        st.table(pd.DataFrame(list(result["metrics"].items()),
                              columns=["Metric", "Value"]))

    ml = result["ml"]
    if ml:
        with st.expander("ML model details"):
            st.write(f"Out-of-sample accuracy (last 6 months): "
                     f"**{ml['oos_acc']:.1%}** vs always-up baseline "
                     f"{ml['baseline']:.1%}")
            st.write(f"Probability next trading day closes up: "
                     f"**{ml['prob_up']:.1%}**")
            if ml["oos_acc"] <= ml["baseline"]:
                st.caption("⚠️ The model isn't beating the naive baseline for "
                           "this stock — treat its signal as noise.")

    st.divider()
    st.caption("⚠️ This is an automated screening tool for education. Scores "
               "are based on limited public data and simple heuristics. It "
               "does not know your goals, risk tolerance, or the full picture "
               "of the company. Do your own research or consult a registered "
               "investment advisor before investing.")
