import numpy as np
from transferlab.models import fit_models,predict,MODELS

def test_heldout_outcomes_do_not_change_fitted_predictions(panel):
    train=panel[panel.ticker.isin(['T0','T1','T2','T3','T4','T5'])]
    test=panel[~panel.ticker.isin(['T0','T1','T2','T3','T4','T5'])]
    b=fit_models(train)
    p1=predict(b,test)
    p2=predict(b,test.assign(y=test.y.to_numpy()+999.))
    for name in MODELS:
        assert np.allclose(p1[name],p2[name])

def test_training_residual_targets_use_train_only(panel):
    train=panel[panel.ticker.isin(['T0','T1','T2','T3','T4','T5'])]
    b1=fit_models(train)
    test=panel[panel.ticker=='T11']
    y1=predict(b1,test).cs_resid_ridge.to_numpy()
    tampered=panel.copy();tampered.loc[tampered.ticker=='T11','y']+=1000.
    y2=predict(fit_models(tampered[tampered.ticker.isin(['T0','T1','T2','T3','T4','T5'])]),test).cs_resid_ridge.to_numpy()
    assert np.allclose(y1,y2)
