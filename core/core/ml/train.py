# ml/train.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from core.data import get_prices, example_universe
from core.features import make_features, build_Xy


MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def timeseries_cv_scores(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Tuple[float, float]:
    """
    Walk-forward CV with AUC and Brier. Returns (mean_auc, mean_brier).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs, briers = [], []
    X_np = X.to_numpy(dtype=float)
    y_np = y.to_numpy(dtype=int)

    for train_idx, test_idx in tscv.split(X_np):
        Xtr, Xte = X_np[train_idx], X_np[test_idx]
        ytr, yte = y_np[train_idx], y_np[test_idx]

        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("clf", LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=None)),
            ]
        )
        pipe.fit(Xtr, ytr)

        # Calibrate on the train set to avoid lookahead (platt scaling on held-out chunk is better; this is simple)
        cal = CalibratedClassifierCV(pipe, method="sigmoid", cv="prefit")
        cal.fit(Xtr, ytr)

        prob = cal.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(yte, prob))
        briers.append(brier_score_loss(yte, prob))

    return float(np.mean(aucs)), float(np.mean(briers))


def train_and_save(
    tickers: Iterable[str],
    start: str = "2018-01-01",
    horizon: int = 5,
    model_name: str = "logreg",
) -> Path:
    """
    End-to-end: fetch -> features -> X,y -> CV -> fit full -> calibrate -> save joblib.
    Returns saved model path.
    """
    # 1) data & features
    prices = get_prices(tickers, start=start)
    feats = make_features(prices, horizons=(horizon,))
    X, y = build_Xy(feats, horizon=horizon)

    # Ensure time order
    X = X.sort_index()
    y = y.loc[X.index]

    # 2) CV diagnostics
    mean_auc, mean_brier = timeseries_cv_scores(X, y, n_splits=5)
    print(f"[CV] mean AUC: {mean_auc:.3f} | mean Brier: {mean_brier:.4f}")

    # 3) fit on all data
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(max_iter=300, class_weight="balanced")),
        ]
    )
    pipe.fit(X, y)

    # Calibrate probabilities
    cal = CalibratedClassifierCV(pipe, method="sigmoid", cv=5)
    cal.fit(X, y)

    # 4) save
    out_path = MODELS_DIR / f"{model_name}_H{horizon}.joblib"
    joblib.dump(
        {
            "model": cal,
            "horizon": horizon,
            "feature_columns": list(X.columns),
            "universe": list(tickers),
            "start": start,
            "cv_auc": mean_auc,
            "cv_brier": mean_brier,
        },
        out_path,
    )
    print(f"[OK] Saved model → {out_path}")
    return out_path


if __name__ == "__main__":
    # quick smoke test with default universe
    tickers = example_universe("core")
    train_and_save(tickers, start="2018-01-01", horizon=5, model_name="logreg_core")
