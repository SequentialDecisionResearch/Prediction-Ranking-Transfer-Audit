"""Date-level ranking and exact squared-error decomposition."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .models import MODELS


def _one_date(g: pd.DataFrame, model: str) -> dict | None:
    if len(g) < 5:
        return None
    y, p = g.y.to_numpy(float), g[model].to_numpy(float)
    cy, cp = y.mean(), p.mean()
    yrel, prel = y - cy, p - cp
    rho = spearmanr(p, y).statistic if np.std(p) > 1e-12 and np.std(y) > 1e-12 else np.nan
    k = max(1, int(np.ceil(len(g) * 0.1)))
    selected = np.argsort(-p, kind="stable")[:k]
    # All metrics computed on identical model/date/test-universe units.
    pooled_mse = np.mean((y - p)**2)
    between_mse = (cy-cp)**2
    within_mse = np.mean((yrel - prel)**2)
    if not np.isclose(pooled_mse, between_mse+within_mse, atol=1e-10):
        raise AssertionError("Between/within MSE identity failed")
    return {"date": g.date.iloc[0], "model": model, "n_assets": len(g),
            "mse": pooled_mse, "mae": np.mean(abs(y-p)), "spearman": rho,
            "top_decile_lift": float(y[selected].mean() - cy),
            "top1_regret": float(y.max() - y[np.argmax(p)]),
            "between_mse": between_mse, "within_mse": within_mse,
            "cs_mae": np.mean(abs(yrel-prel))}


def date_metrics(pred: pd.DataFrame, minimum_stocks: int = 5) -> pd.DataFrame:
    rows = []
    for _, g in pred.groupby("date", sort=True):
        if len(g) < minimum_stocks:
            continue
        for m in MODELS:
            rec = _one_date(g, m)
            if rec is not None:
                rows.append(rec)
    if not rows:
        raise ValueError("No eligible cross-sectional dates; relax window or inspect data")
    return pd.DataFrame(rows)


def summarize(m: pd.DataFrame) -> pd.DataFrame:
    keys = ["mse", "mae", "spearman", "top_decile_lift", "top1_regret", "between_mse", "within_mse", "cs_mae"]
    df = m.groupby("model", as_index=False)[keys].mean()
    df["dates"] = df.model.map(m.groupby("model").date.nunique())
    df["between_sse_share"] = df.between_mse/(df.between_mse+df.within_mse)
    return df


def metric_difference(m: pd.DataFrame, a: str, b: str, metric: str) -> pd.DataFrame:
    p = m.pivot_table(index="date", columns="model", values=metric, aggfunc="mean")
    return (p[a]-p[b]).dropna().rename("difference").reset_index()
