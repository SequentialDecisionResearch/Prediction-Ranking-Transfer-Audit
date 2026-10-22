import numpy as np
import pandas as pd
from transferlab.inference import circular_block_indices,block_bootstrap,paired_date_difference

def test_block_indices_length_and_bounded():
    v=circular_block_indices(19,5,np.random.default_rng(2))
    assert len(v)==19 and v.min()>=0 and v.max()<19

def test_bootstrap_reproducible():
    d=pd.DataFrame({'date':pd.date_range('2020-01-01',periods=24,freq='MS'),'difference':np.linspace(-.01,.03,24)})
    a=block_bootstrap(d,[3,6],100,42);b=block_bootstrap(d,[3,6],100,42)
    assert a.equals(b)

def test_split_repetition_averaged_before_date_bootstrap():
    d=pd.DataFrame({'date':[pd.Timestamp('2020-01-01')]*6,'split':[0]*2+[1]*2+[2]*2,
      'model':['A','B']*3,'spearman':[1,0,3,0,5,0]})
    out=paired_date_difference(d,'A','B','spearman')
    assert len(out)==1 and np.isclose(out.difference.iloc[0],3.)
