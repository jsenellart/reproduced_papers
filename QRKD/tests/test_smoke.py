import numpy as np

from QRKD.implementation import HyperParams, QRKDModel


def test_smoke_shapes():
    hp = HyperParams(n_features=16, n_layers=1)
    model = QRKDModel(hp)
    x = np.random.randn(4, 3)
    y = (x.sum(axis=1) > 0).astype(int)
    model.fit(x, y)
    preds = model.predict(x)
    assert preds.shape[0] == x.shape[0]
