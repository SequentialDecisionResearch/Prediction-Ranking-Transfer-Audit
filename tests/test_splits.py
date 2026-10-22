from transferlab.experiments import split_identities

def test_nonoverlap_reproducible():
    ids=[f'X{i}' for i in range(40)]
    a,b=split_identities(ids,.2,123,4)
    a2,b2=split_identities(ids,.2,123,4)
    assert a==a2 and b==b2 and not a.intersection(b) and a|b==set(ids)
