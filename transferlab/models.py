"""Restricted comparable baselines; fitted using training data only."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .data import FEATURES

MODELS = ("state_ridge", "cs_resid_ridge", "ret63_calibrated", "ret63_raw")

@dataclass
class PredictorBundle:
    state: object
    residual: object
    calibrated_return: object
    training_rows: int
    training_dates: int
    training_identities: int


def _ridge(alpha: float):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def fit_models(train: pd.DataFrame, alpha: float = 5.0) -> PredictorBundle:
    if len(train) < 30 or train.ticker.nunique() < 3 or train.date.nunique() < 3:
        raise ValueError("Insufficient training observations or diversification")
    x = train.loc[:, FEATURES].to_numpy(float)
    y = train.y.to_numpy(float)
    # Crucial: compute date mean only from *eligible training rows*, not holdout identities.
    train_means = train.groupby("date").y.transform("mean").to_numpy(float)
    state, resid, cal = _ridge(alpha), _ridge(alpha), _ridge(alpha)
    state.fit(x, y)
    resid.fit(x, y - train_means)
    cal.fit(train[["ret63"]].to_numpy(float), y)
    return PredictorBundle(state, resid, cal, len(train), train.date.nunique(), train.ticker.nunique())


def predict(bundle: PredictorBundle, test: pd.DataFrame) -> pd.DataFrame:
    out = test.loc[:, ["date", "ticker", "y", "ret63"]].copy()
    x = test.loc[:, FEATURES].to_numpy(float)
    out["state_ridge"] = bundle.state.predict(x)
    out["cs_resid_ridge"] = bundle.residual.predict(x)
    out["ret63_calibrated"] = bundle.calibrated_return.predict(test[["ret63"]].to_numpy(float))
    out["ret63_raw"] = test.ret63.to_numpy(float)
    return out


def calibration_line(bundle: PredictorBundle) -> dict:
    """Return raw-feature-space line yhat=intercept+slope*ret63.

    StandardScaler is internal; naively reporting Ridge.coef_ alone would
    report the slope for standardized, not raw, 63-day return.
    """
    scaler = bundle.calibrated_return.named_steps["standardscaler"]
    ridge = bundle.calibrated_return.named_steps["ridge"]
    slope = float(ridge.coef_[0] / scaler.scale_[0])
    intercept = float(ridge.intercept_ - ridge.coef_[0]*scaler.mean_[0]/scaler.scale_[0])
    return {"raw_ret63_slope": slope,"raw_ret63_intercept": intercept,
            "direction": "reversal" if slope < 0 else "momentum" if slope > 0 else "flat",
            "ret63_train_mean": float(scaler.mean_[0]),
            "ret63_train_std": float(scaler.scale_[0]),
            "ridge_alpha": float(ridge.alpha)}
