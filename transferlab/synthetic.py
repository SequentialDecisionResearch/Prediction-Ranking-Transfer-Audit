"""Mechanism falsification: high pooled predictive fit without ranking information."""
from __future__ import annotations
import numpy as np
import pandas as pd

def generate(seed:int=20261020, dates:int=80,assets:int=40)->pd.DataFrame:
    rng=np.random.default_rng(seed)
    rows=[]
    for t in range(dates):
        common=rng.normal(scale=2.)
        relative=rng.normal(scale=0.03,size=assets)
        tiny=rng.normal(scale=0.001,size=assets)
        for i in range(assets):
            rows.append({"date":pd.Timestamp("2018-01-01")+pd.offsets.MonthBegin(t),
                         "ticker":f"SIM{i:03d}","y":common+relative[i],
                         "predict_common_only":common+tiny[i],"predict_oracle":common+relative[i]})
    return pd.DataFrame(rows)

def sanity(seed:int=20261020):
    from scipy.stats import spearmanr
    a=generate(seed)
    mae_common=np.mean(abs(a.y-a.predict_common_only))
    rho_common=np.mean([spearmanr(g.y,g.predict_common_only).statistic for _,g in a.groupby("date")])
    rho_oracle=np.mean([spearmanr(g.y,g.predict_oracle).statistic for _,g in a.groupby("date")])
    return {"case":"date common signal + independent small ranking noise",
            "common_only_mae":float(mae_common),"common_only_spearman":float(rho_common),
            "oracle_spearman":float(rho_oracle),"interpretation":"low level error does not imply cross-sectional rank information"}
