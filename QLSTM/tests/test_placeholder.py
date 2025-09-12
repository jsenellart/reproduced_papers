import torch
from QLSTM.implementation import BaselineLSTM, QLSTMPlaceholder

def test_baseline_forward():
    model = BaselineLSTM(input_dim=8, hidden_dim=16, num_layers=1, num_classes=2)
    x = torch.randn(4, 10, 8)
    y = model(x)
    assert y.shape == (4, 2)

def test_qlstm_forward():
    model = QLSTMPlaceholder(input_dim=8, hidden_dim=16, num_layers=1, num_classes=2)
    x = torch.randn(4, 10, 8)
    y = model(x)
    assert y.shape == (4, 2)
