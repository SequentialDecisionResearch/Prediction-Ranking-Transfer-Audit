import pandas as pd
import pytest
from transferlab.data import load_panel,with_label_maturity,training_before

def test_maturity_boundary_and_heldout(panel,tmp_path):
    f=tmp_path/'data.csv';panel.to_csv(f,index=False)
    sess=tmp_path/'sessions.csv';pd.DataFrame({'date':sorted(panel.date.unique())}).to_csv(sess,index=False)
    d=with_label_maturity(load_panel(f),sess,21)
    train=training_before(d,'2020-03-31',{'T0','T1'})
    assert train.ticker.nunique()==2
    assert (train.maturity_date<=pd.Timestamp('2020-03-31')).all()
    assert (train.date <= pd.Timestamp('2020-03-31')).all()
    assert (train.ticker.isin({'T0','T1'})).all()

def test_reject_duplicate_and_missing(panel,tmp_path):
    f=tmp_path/'x.csv';pd.concat([panel,panel.iloc[[0]]]).to_csv(f,index=False)
    with pytest.raises(ValueError,match='Duplicate'):load_panel(f)
    panel.drop(columns=['vol63']).to_csv(f,index=False)
    with pytest.raises(ValueError,match='Missing required'):load_panel(f)
