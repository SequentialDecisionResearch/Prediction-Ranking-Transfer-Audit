from transferlab.synthetic import sanity

def test_synthetic_falsification():
    r=sanity()
    assert r['common_only_mae']<.1
    assert abs(r['common_only_spearman'])<.1
    assert r['oracle_spearman']>.999
