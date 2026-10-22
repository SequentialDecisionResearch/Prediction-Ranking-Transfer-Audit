import numpy as np
import pandas as pd
import pytest
from transferlab.models import fit_models, calibration_line, predict
from transferlab.inference import (block_bootstrap, factorial_paired_contrasts,
                                   factorial_interaction_bootstrap)


def test_calibration_unscaled_slope_exactly_reproduces_predictions(panel):
    train=panel[panel.ticker.isin(['T0','T1','T2','T3','T4','T5'])]
    b=fit_models(train)
    line=calibration_line(b)
    out=predict(b,train)
    expected=line['raw_ret63_intercept'] + line['raw_ret63_slope']*train.ret63.to_numpy(float)
    assert np.allclose(expected, out.ret63_calibrated.to_numpy(float), atol=1e-12)
    assert line['direction'] in {'reversal','momentum','flat'}


def test_block_report_counts_and_nonzero_p_resolution():
    d=pd.DataFrame({'date':pd.date_range('2020-01-01',periods=24,freq='MS'),
                    'difference':np.ones(24)*0.25})
    a=block_bootstrap(d,[3,6,12],300,33)
    assert list(a.unique_calendar_months)==[24]*3
    assert list(a.nominal_nonoverlapping_blocks)==[8.0,4.0,2.0]
    assert all(a.bootstrap_replicates==300)
    assert a.exploratory_null_p_two_sided.iloc[0] == pytest.approx(1/301)
    assert a.exploratory_null_p_two_sided.iloc[2] != a.exploratory_null_p_two_sided.iloc[2] # NaN
    assert a.inference_tier.iloc[2] == 'descriptive_only_too_few_blocks'


def _rows(value_by_cell):
    rows=[]
    for regime,n in [('near',12),('future',24)]:
        for month in pd.date_range('2020-01-01' if regime=='near' else '2021-01-01',periods=n,freq='MS'):
            for ident in ('seen','unseen'):
                diff=value_by_cell[(regime,ident)]
                rows.extend([
                    {'date':month,'time_regime':regime,'identity_regime':ident,'model':'state_ridge','spearman':0.02},
                    {'date':month,'time_regime':regime,'identity_regime':ident,'model':'cs_resid_ridge','spearman':0.02+diff}
                ])
    return pd.DataFrame(rows)


def test_factorial_interaction_exact_known_value_and_pairing():
    cells={('near','seen'):0.01,('near','unseen'):0.03,
           ('future','seen'):0.02,('future','unseen'):0.08}
    contrasts=factorial_paired_contrasts(_rows(cells))
    assert len(contrasts)==36
    assert np.allclose(contrasts.query("time_regime == 'near'").identity_gap,0.02)
    assert np.allclose(contrasts.query("time_regime == 'future'").identity_gap,0.06)
    infer=factorial_interaction_bootstrap(contrasts,[3,6,12],300,2026)
    assert np.allclose(infer.mean_delta,0.04)
    assert list(infer.n_near_months)==[12]*3
    assert list(infer.n_future_months)==[24]*3
    assert list(infer.unique_calendar_months)==[36]*3
    assert infer.inference_tier.iloc[0]=='exploratory_block_sensitivity'
    assert infer.inference_tier.iloc[1]=='descriptive_only_too_few_blocks'
    assert infer.exploratory_null_p_two_sided.iloc[1] != infer.exploratory_null_p_two_sided.iloc[1]


def test_factorial_missing_group_raises():
    data=_rows({('near','seen'):0,('near','unseen'):0,
                ('future','seen'):0,('future','unseen'):0})
    data=data[~((data.time_regime=='future') & (data.identity_regime=='unseen')
                & (data.date==pd.Timestamp('2021-01-01')))]
    with pytest.raises(ValueError,match='Every date'):
        factorial_paired_contrasts(data)


def test_duplicate_calendar_dates_rejected():
    d=pd.DataFrame({'date':[pd.Timestamp('2022-01-01')]*8,'difference':range(8)})
    with pytest.raises(ValueError,match='one finite paired contrast'):
        block_bootstrap(d,[3],300,1)
