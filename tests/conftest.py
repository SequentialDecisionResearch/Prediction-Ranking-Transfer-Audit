import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def panel():
    rng=np.random.default_rng(13);rows=[]
    dates=pd.bdate_range("2020-01-01",periods=160)
    for t in dates:
        for i in range(12):
            vals=rng.normal(size=6)
            rows.append({"date":t,"ticker":f"T{i}","y":float(vals[0]*.02+vals[1]*.01+rng.normal(scale=.01)),
                         **dict(zip(["ret1","ret21","ret63","vol63","dd252","spy_ret63"],vals))})
    return pd.DataFrame(rows)
