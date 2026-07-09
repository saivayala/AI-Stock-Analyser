"""ML signal: next-day direction model built on stock_direction_model features."""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

from stock_direction_model import build_features

# ~6 months of trading days held out for the out-of-sample check
TEST_DAYS = 126
MIN_ROWS = 500


def _make_model() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=200,
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


def ml_signal(df: pd.DataFrame) -> dict | None:
    """Train on history, report last-6-months OOS accuracy and the
    probability that the next trading day closes up.

    Returns None when there isn't enough history to train sensibly.
    """
    # build_features drops the final row (its target is unknown). Append a
    # dummy copy of the last bar so the real last day survives and we can
    # read its backward-looking features for the live prediction.
    dummy = df.tail(1).copy()
    dummy.index = [df.index[-1] + pd.Timedelta(days=1)]
    feats = build_features(pd.concat([df, dummy]))
    feats = feats[feats.index <= df.index[-1]]

    if len(feats) < MIN_ROWS + TEST_DAYS:
        return None

    feature_cols = [c for c in feats.columns if c not in ("target", "next_ret")]
    latest = feats.iloc[[-1]]          # live features (target row is bogus)
    hist = feats.iloc[:-1]             # rows with a real next-day target

    train, test = hist.iloc[:-TEST_DAYS], hist.iloc[-TEST_DAYS:]

    model = _make_model()
    model.fit(train[feature_cols], train["target"])
    test_prob = model.predict_proba(test[feature_cols])[:, 1]
    oos_acc = accuracy_score(test["target"], test_prob > 0.5)
    up_rate = test["target"].mean()
    baseline = max(up_rate, 1 - up_rate)

    # Refit on everything before predicting tomorrow
    model = _make_model()
    model.fit(hist[feature_cols], hist["target"])
    prob_up = float(model.predict_proba(latest[feature_cols])[0, 1])

    return {
        "oos_acc": float(oos_acc),
        "baseline": float(baseline),
        "prob_up": prob_up,
    }


def ml_score(ml: dict) -> float:
    """0-100 score for the ML pillar.

    An edge over the always-up baseline moves the score above 50; the
    predicted direction for tomorrow tilts it further. No edge => the
    signal is treated as roughly neutral.
    """
    edge = ml["oos_acc"] - ml["baseline"]
    tilt = (ml["prob_up"] - 0.5) * 100          # -50 .. +50
    if edge <= 0:
        score = 45 + 0.2 * tilt                 # near-neutral, slight tilt
    else:
        score = 50 + min(edge * 500, 35) + 0.3 * tilt
    return float(np.clip(score, 0, 100))
