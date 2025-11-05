import numpy as np
from neurocampus.models.strategies.rbm_pura import RBM

def test_rbm_pura_fit_predict_shapes():
    X = np.random.rand(200, 12).astype(np.float32)
    m = RBM(n_hidden=8, epochs=2, batch_size=32, cd_k=1, lr=0.05, seed=7)
    m.fit(X)
    proba = m.predict_proba(X[:10])
    assert proba.shape == (10, 3)
    y = m.predict(X[:10])
    assert len(y) == 10
