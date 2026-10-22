"""Schema validation, maturity boundaries and provenance checks."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ("ret1", "ret21", "ret63", "vol63", "dd252", "spy_ret63")
REQUIRED = ("date", "ticker", "y", *FEATURES)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_panel(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Missing panel: {p}. Extract the full ZIP before running.")
    df = pd.read_csv(p)
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["ticker"] = df["ticker"].astype(str)
    if df[["date", "ticker"]].duplicated().any():
        raise ValueError("Duplicate date/ticker rows detected")
    for c in ("y", *FEATURES):
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if not np.isfinite(df[c].to_numpy()).all():
            raise ValueError(f"Nonfinite/missing values in {c}; fix the source rather than silently imputing")
    df.sort_values(["date", "ticker"], inplace=True, kind="stable")
    return df.reset_index(drop=True)


def with_label_maturity(df: pd.DataFrame, sessions_file: str | Path, horizon: int) -> pd.DataFrame:
    """Map signal date to market session H steps later (rather than approximate calendar days)."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    sessions = pd.to_datetime(pd.read_csv(sessions_file)["date"])
    dates = pd.DatetimeIndex(sessions.unique()).sort_values()
    if len(dates) == 0 or dates.has_duplicates:
        raise ValueError("Invalid market session calendar")
    unique_dates = pd.DatetimeIndex(df["date"].unique())
    locs = dates.get_indexer(unique_dates)
    if (locs < 0).any():
        raise ValueError("Some panel dates absent from market session calendar")
    nxt = locs + horizon
    lookup = {d: (dates[i] if i < len(dates) else pd.NaT) for d, i in zip(unique_dates, nxt)}
    out = df.copy()
    out["maturity_date"] = out["date"].map(lookup)
    return out


def audit(df: pd.DataFrame) -> dict:
    g = df.groupby("date").size()
    return {"records": len(df), "identities": df.ticker.nunique(), "dates": df.date.nunique(),
            "first_date": str(df.date.min().date()), "last_date": str(df.date.max().date()),
            "min_assets_per_date": int(g.min()), "median_assets_per_date": float(g.median()),
            "nonmatured_labels": int(df.maturity_date.isna().sum()) if "maturity_date" in df else None,
            "features": list(FEATURES)}


def training_before(df: pd.DataFrame, end: str | pd.Timestamp, tickers: set[str]) -> pd.DataFrame:
    end = pd.Timestamp(end)
    # A label belongs to training only after its outcome has matured.
    return df.loc[df.ticker.isin(tickers) & (df.date <= end) & (df.maturity_date <= end)].copy()


def evaluate_window(df: pd.DataFrame, start: str, end: str, tickers: set[str]) -> pd.DataFrame:
    return df.loc[df.ticker.isin(tickers) & df.date.between(pd.Timestamp(start), pd.Timestamp(end))].copy()
