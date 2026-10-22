"""Calendar-month paired resampling, with explicitly qualified uncertainty.

This is exploratory block-bootstrap sensitivity, NOT independent confirmation,
not a finite-sample-valid test, and not an estimated population asset ESS.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd


def circular_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0 or block <= 0:
        raise ValueError("n and block must be positive")
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n


def _validate_series(frame: pd.DataFrame, column: str = "difference") -> np.ndarray:
    if not {"date", column}.issubset(frame.columns):
        raise ValueError(f"Need date and {column} columns")
    if frame.date.isna().any() or frame.date.duplicated().any():
        raise ValueError("Resampling requires one finite paired contrast per unique date")
    v = frame.sort_values("date")[column].to_numpy(float)
    if len(v) < 6 or not np.isfinite(v).all():
        raise ValueError("Require >=6 dates and finite paired differences")
    return v


def _draw_means(s: np.ndarray, block: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    draws = np.empty(reps, dtype=float)
    for k in range(reps):
        draws[k] = s[circular_block_indices(len(s), block, rng)].mean()
    return draws


def _summary(point: float, draws: np.ndarray, null_draws: np.ndarray,
             n: int, block: int, reps: int) -> dict:
    # Null-centering before resampling tests H0: E[difference] = 0 under
    # approximate calendar-month stationarity, rather than treating an
    # uncentered bootstrap percentile as an automatically calibrated p-value.
    if len(draws) != reps or len(null_draws) != reps:
        raise AssertionError("Bootstrap draws incomplete")
    blocks = int(math.ceil(n / block))
    # Approximate number of NONOVERLAPPING blocks is only a diagnostic,
    # not independent observations and definitely not a true ESS.
    independent_blocks_heuristic = n / block
    reliable = (block <= n and independent_blocks_heuristic >= 4)
    ci_lo, ci_hi = np.quantile(draws, [0.025, 0.975])
    count = int(np.count_nonzero(np.abs(null_draws) >= abs(point)))
    p = (count + 1) / (reps + 1) if reliable else np.nan
    return {
        "block_months": int(block),
        "unique_calendar_months": int(n),
        "nominal_nonoverlapping_blocks": float(independent_blocks_heuristic),
        "circular_blocks_drawn_per_rep": blocks,
        "bootstrap_replicates": int(reps),
        "mean_delta": float(point),
        "ci95_low": float(ci_lo),
        "ci95_high": float(ci_hi),
        "ci_method": "circular_calendar_block_bootstrap_percentile_95",
        "null_p_method": "mean_centered_circular_block_bootstrap_plus_one_two_sided",
        "exploratory_null_p_two_sided": float(p),
        "minimum_reportable_p": float(1 / (reps + 1)),
        "inference_tier": ("exploratory_block_sensitivity" if reliable
                           else "descriptive_only_too_few_blocks"),
        "effective_resampling_unit": "calendar month with all repeated identity splits aggregated",
        "caveat": "nominal intervals and exploratory p-values; time stationarity/coverage unverified"
    }


def block_bootstrap(differences: pd.DataFrame, blocks: list[int], replicates: int,
                    seed: int) -> pd.DataFrame:
    """Paired split/date contrast already averaged across overlapping splits.

    For each block length: a separate RNG, percentile interval, null-centered
    block bootstrap Monte Carlo p, and explicit count of calendar-month units.
    """
    s = _validate_series(differences)
    if replicates < 100:
        raise ValueError("Need >=100 replicates")
    rows=[]
    for block in blocks:
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(block), 101]))
        null_rng = np.random.default_rng(np.random.SeedSequence([seed, int(block), 202]))
        draws = _draw_means(s, block, replicates, rng)
        null = _draw_means(s - s.mean(), block, replicates, null_rng)
        rows.append(_summary(float(s.mean()), draws, null, len(s), block, replicates))
    return pd.DataFrame(rows)


def paired_date_difference(all_metrics: pd.DataFrame, model_a: str, model_b: str,
                           metric: str) -> pd.DataFrame:
    # First within (split, date); then average across splits on calendar date.
    piv = all_metrics.pivot_table(index=["split", "date"], columns="model",
                                  values=metric, aggfunc="mean")
    d = (piv[model_a] - piv[model_b]).dropna().rename("difference").reset_index()
    return d.groupby("date", as_index=False).difference.mean()


def factorial_paired_contrasts(all_metrics: pd.DataFrame, model_a="cs_resid_ridge",
                               model_b="state_ridge", metric="spearman") -> pd.DataFrame:
    """Within-date *paired* identity-gap: (A-B)_unseen - (A-B)_seen.

    Uses a fixed identity roster, different seen/unseen asset populations, and
    does not identify a causal identity or market-regime effect.
    """
    index = ["time_regime", "identity_regime", "date"]
    if all_metrics.duplicated(index + ["model"]).any():
        raise ValueError("Duplicate factorial date/model rows")
    p = all_metrics.pivot(index=index, columns="model", values=metric)
    if model_a not in p or model_b not in p:
        raise ValueError("Missing model in factorial data")
    d = (p[model_a] - p[model_b]).rename("model_delta").reset_index()
    d = d.pivot(index=["time_regime", "date"], columns="identity_regime", values="model_delta")
    if not {"seen", "unseen"}.issubset(set(d.columns)):
        raise ValueError("Need both seen and unseen groups")
    if d[["seen", "unseen"]].isna().any().any():
        raise ValueError("Every date must contain both identity groups and defined ranks")
    out = d.reset_index()
    out["identity_gap"] = out["unseen"] - out["seen"]
    if set(out.time_regime.unique()) != {"near", "future"}:
        raise ValueError("Need exactly near and future regimes")
    return out.sort_values(["time_regime", "date"]).reset_index(drop=True)


def factorial_interaction_bootstrap(gaps: pd.DataFrame, blocks: list[int],
                                    replicates: int, seed: int) -> pd.DataFrame:
    """Difference-in-differences OF DESCRIPTIVE paired model contrasts.

    Parameter is mean_t(Delta_unseen - Delta_seen)_future minus the analogous
    near mean. Within each regime, draw entire date blocks so the within-date
    seen/unseen pairing remains intact. Two regimes are resampled separately;
    assumption of negligible cross-boundary serial dependence is UNVERIFIED.
    """
    if replicates < 100:
        raise ValueError("Need >=100 replicates")
    near = _validate_series(gaps[gaps.time_regime == "near"], "identity_gap")
    future = _validate_series(gaps[gaps.time_regime == "future"], "identity_gap")
    point = float(future.mean() - near.mean())
    rows=[]
    for block in blocks:
        rnf = np.random.default_rng(np.random.SeedSequence([seed, block, 301]))
        rnn = np.random.default_rng(np.random.SeedSequence([seed, block, 302]))
        rrnf = np.random.default_rng(np.random.SeedSequence([seed, block, 303]))
        rrnn = np.random.default_rng(np.random.SeedSequence([seed, block, 304]))
        future_means = _draw_means(future, block, replicates, rnf)
        near_means = _draw_means(near, block, replicates, rnn)
        observed = future_means - near_means
        # Under H0 equal regime means: center each regime independently.
        centered = (_draw_means(future-future.mean(), block, replicates, rrnf)
                    - _draw_means(near-near.mean(), block, replicates, rrnn))
        # Shorter series (near) is the binding reliability constraint.
        res = _summary(point, observed, centered, min(len(near), len(future)),block,replicates)
        res["unique_calendar_months"] = int(len(near) + len(future))
        res["limiting_regime_months"] = int(min(len(near), len(future)))
        res["n_near_months"] = int(len(near))
        res["n_future_months"] = int(len(future))
        res["near_circular_blocks_drawn"] = int(math.ceil(len(near)/block))
        res["future_circular_blocks_drawn"] = int(math.ceil(len(future)/block))
        res["near_nominal_blocks"] = float(len(near) / block)
        res["future_nominal_blocks"] = float(len(future) / block)
        res["effective_resampling_unit"] = "paired seen/unseen calendar date, stratified by time regime"
        res["contrast"] = "[(CS-State)_future_unseen - (CS-State)_future_seen] - [(CS-State)_near_unseen - (CS-State)_near_seen]"
        res["estimand_class"] = "associational_fixed_roster_interaction_not_causal_DiD"
        res["caveat"] = ("nominal exploratory; separate regime blocks assume weak boundary dependence; "
                         "different seen/unseen constituents; only one frozen identity split")
        rows.append(res)
    return pd.DataFrame(rows)
