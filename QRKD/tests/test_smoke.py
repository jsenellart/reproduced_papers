import torch

from QRKD.lib.models import StudentCNN, TeacherCNN


def test_smoke_shapes():
    model_t = TeacherCNN()
    model_s = StudentCNN()
    x = torch.randn(2, 1, 28, 28)
    logits_t, feat_t = model_t(x)
    logits_s, feat_s = model_s(x)
    assert logits_t.shape == (2, 10)
    assert logits_s.shape == (2, 10)
    assert feat_t.ndim == 2 and feat_s.ndim == 2
