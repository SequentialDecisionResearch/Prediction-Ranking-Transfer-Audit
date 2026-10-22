import numpy as np
import pandas as pd
from transferlab.metrics import date_metrics
from transferlab.models import MODELS

def test_exact_decomposition_and_ranking(panel):
    p=panel[['date','ticker','y','ret63']].copy()
    for m in MODELS: p[m]=panel.y.to_numpy()*.5
    m=date_metrics(p)
    assert np.allclose(m.mse,m.between_mse+m.within_mse)
    assert np.allclose(m.spearman,1.)

def test_cross_section_constant_scores_are_nan_not_zero(panel):
    p=panel[['date','ticker','y','ret63']].copy()
    for m in MODELS:p[m]=0.
    m=date_metrics(p)
    assert m.spearman.isna().all()
