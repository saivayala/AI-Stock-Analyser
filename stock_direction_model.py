"""
Stock Direction Prediction Pipeline
====================================
Predicts next-day direction (up/down) for a stock/index using
gradient boosting, with proper walk-forward validation and a
simple backtest including transaction costs.

Requirements:
    pip install yfinance pandas numpy scikit-learn lightgbm matplotlib

Usage:
    python stock_direction_model.py                 # defaults to ^NSEI (Nifty 50)
    python stock_direction_model.py --ticker RELIANCE.NS
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------
def download_data(ticker: str, start: str = "2015-01-01") -> pd.DataFrame:
    print(f"Downloading {ticker} from {start}...")
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    print(f"  {len(df)} rows, {df.index.min().date()} -> {df.index.max().date()}")
    return df


# ----------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ----------------------------------------------------------------------
def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-10)
    return 100 - 100 / (1 + rs)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

    # --- Lagged returns ---
    for lag in [1, 2, 3, 5, 10, 20]:
        out[f"ret_{lag}d"] = close.pct_change(lag)

    # --- Moving averages (as ratios, so they're stationary) ---
    for w in [5, 10, 20, 50]:
        out[f"close_vs_sma{w}"] = close / close.rolling(w).mean() - 1
    out["sma5_vs_sma20"] = close.rolling(5).mean() / close.rolling(20).mean() - 1
    out["sma20_vs_sma50"] = close.rolling(20).mean() / close.rolling(50).mean() - 1

    # --- Volatility ---
    daily_ret = close.pct_change()
    for w in [5, 10, 20]:
        out[f"vol_{w}d"] = daily_ret.rolling(w).std()
    out["vol_ratio"] = out["vol_5d"] / (out["vol_20d"] + 1e-10)

    # --- RSI ---
    out["rsi_14"] = rsi(close, 14)
    out["rsi_7"] = rsi(close, 7)

    # --- MACD ---
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    out["macd_hist"] = (macd - signal) / close  # normalized

    # --- Bollinger position ---
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    out["bb_position"] = (close - sma20) / (2 * std20 + 1e-10)

    # --- Range / candle features ---
    out["hl_range"] = (high - low) / close
    out["close_position"] = (close - low) / (high - low + 1e-10)

    # --- Volume ---
    out["vol_vs_avg20"] = vol / (vol.rolling(20).mean() + 1e-10) - 1
    out["vol_change"] = vol.pct_change()

    # --- Calendar ---
    out["day_of_week"] = df.index.dayofweek
    out["month"] = df.index.month

    # --- Target: next-day direction ---
    out["target"] = (close.shift(-1) > close).astype(int)
    out["next_ret"] = close.pct_change().shift(-1)  # for backtest, NOT a feature

    out = out.replace([np.inf, -np.inf], np.nan)
    return out.dropna()


# ----------------------------------------------------------------------
# 3. WALK-FORWARD VALIDATION
# ----------------------------------------------------------------------
def walk_forward(data: pd.DataFrame, feature_cols: list,
                 train_years: int = 4, test_months: int = 6):
    """
    Expanding-window walk-forward: train on all data up to a point,
    test on the next `test_months`, then roll forward.
    """
    results = []
    dates = data.index
    start_test = dates.min() + pd.DateOffset(years=train_years)

    fold = 0
    while start_test < dates.max():
        end_test = start_test + pd.DateOffset(months=test_months)
        train = data[data.index < start_test]
        test = data[(data.index >= start_test) & (data.index < end_test)]
        if len(test) < 20:
            break
        fold += 1

        X_tr, y_tr = train[feature_cols], train["target"]
        X_te, y_te = test[feature_cols], test["target"]

        # --- Baseline: logistic regression ---
        scaler = StandardScaler()
        lr = LogisticRegression(max_iter=1000, C=0.1)
        lr.fit(scaler.fit_transform(X_tr), y_tr)
        lr_pred = lr.predict_proba(scaler.transform(X_te))[:, 1]

        # --- Main model: LightGBM ---
        lgbm = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
        )
        lgbm.fit(X_tr, y_tr)
        lgbm_pred = lgbm.predict_proba(X_te)[:, 1]

        fold_res = test[["target", "next_ret"]].copy()
        fold_res["lr_prob"] = lr_pred
        fold_res["lgbm_prob"] = lgbm_pred
        results.append(fold_res)

        print(f"  Fold {fold}: test {start_test.date()} -> {end_test.date()} "
              f"| LR acc={accuracy_score(y_te, lr_pred > 0.5):.3f} "
              f"| LGBM acc={accuracy_score(y_te, lgbm_pred > 0.5):.3f}")

        start_test = end_test

    return pd.concat(results), lgbm, feature_cols


# ----------------------------------------------------------------------
# 4. EVALUATION + BACKTEST
# ----------------------------------------------------------------------
def evaluate(results: pd.DataFrame, cost_per_trade: float = 0.0005):
    """
    Simple long/flat strategy: hold the asset on days the model
    predicts 'up', stay in cash otherwise. Transaction cost charged
    whenever the position changes (default 5 bps per switch).
    """
    print("\n" + "=" * 60)
    print("OUT-OF-SAMPLE RESULTS (all walk-forward folds combined)")
    print("=" * 60)

    up_rate = results["target"].mean()
    print(f"\nBaseline 'always up' accuracy: {max(up_rate, 1 - up_rate):.3f}")

    for name in ["lr", "lgbm"]:
        prob = results[f"{name}_prob"]
        pred = (prob > 0.5).astype(int)
        acc = accuracy_score(results["target"], pred)
        auc = roc_auc_score(results["target"], prob)

        # Backtest
        position = pred  # 1 = long, 0 = cash
        trades = position.diff().abs().fillna(0)
        strat_ret = position * results["next_ret"] - trades * cost_per_trade
        bh_ret = results["next_ret"]

        def stats(r):
            cum = (1 + r).cumprod()
            total = cum.iloc[-1] - 1
            ann_ret = (1 + total) ** (252 / len(r)) - 1
            sharpe = r.mean() / (r.std() + 1e-10) * np.sqrt(252)
            dd = (cum / cum.cummax() - 1).min()
            return total, ann_ret, sharpe, dd

        s_tot, s_ann, s_sharpe, s_dd = stats(strat_ret)
        b_tot, b_ann, b_sharpe, b_dd = stats(bh_ret)

        label = "Logistic Regression" if name == "lr" else "LightGBM"
        print(f"\n--- {label} ---")
        print(f"  Accuracy: {acc:.3f} | AUC: {auc:.3f}")
        print(f"  Strategy : total {s_tot:+.1%} | ann {s_ann:+.1%} | "
              f"Sharpe {s_sharpe:.2f} | maxDD {s_dd:.1%}")
        print(f"  Buy&Hold : total {b_tot:+.1%} | ann {b_ann:+.1%} | "
              f"Sharpe {b_sharpe:.2f} | maxDD {b_dd:.1%}")
        print(f"  Trades: {int(trades.sum())} "
              f"(cost assumed {cost_per_trade:.2%} per switch)")


def show_feature_importance(model, feature_cols, top_n: int = 15):
    imp = pd.Series(model.feature_importances_, index=feature_cols)
    imp = imp.sort_values(ascending=False).head(top_n)
    print("\nTop features (last fold's model):")
    for feat, val in imp.items():
        print(f"  {feat:20s} {'█' * int(50 * val / imp.max())}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="^NSEI",
                        help="Yahoo Finance ticker (default: ^NSEI = Nifty 50). "
                             "Examples: RELIANCE.NS, TCS.NS, ^NSEBANK, AAPL")
    parser.add_argument("--start", default="2015-01-01")
    args = parser.parse_args()

    df = download_data(args.ticker, args.start)
    data = build_features(df)

    feature_cols = [c for c in data.columns if c not in ("target", "next_ret")]
    print(f"\n{len(feature_cols)} features, {len(data)} usable rows")
    print("\nRunning walk-forward validation...")

    results, last_model, feats = walk_forward(data, feature_cols)
    evaluate(results)
    show_feature_importance(last_model, feats)

    print("\nNOTE: If LightGBM doesn't clearly beat logistic regression and")
    print("the 'always up' baseline, the model has no real edge yet — improve")
    print("features (sentiment, macro, cross-asset signals) before anything else.")


if __name__ == "__main__":
    main()
