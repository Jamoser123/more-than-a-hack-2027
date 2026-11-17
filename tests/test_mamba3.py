"""Tests for the Mamba-3 model and its training wrapper. Small synthetic inputs; no data/ needed."""

import numpy as np
import torch

from solution.common import A, D, H
from solution.mamba3 import Block, Mamba3, rotate_pairs
from solution.sequence_model import (
    STEP,
    WINDOW,
    Standardizer,
    TrainConfig,
    fit,
    predict_from_features,
)


def perturbed_layer(seed=0):
    torch.manual_seed(seed)
    layer = Mamba3(12, n_heads=2, head_dim=3, d_state=6).double()
    with torch.no_grad():
        for p in layer.parameters():
            p.add_(0.3 * torch.randn_like(p))
    return layer


def test_parallel_form_equals_the_complex_trapezoidal_recurrence():
    layer = perturbed_layer()
    u = torch.randn(2, 30, 12, dtype=torch.float64)
    torch.testing.assert_close(layer(u), layer.recurrent(u), rtol=1e-9, atol=1e-9)


def test_layer_is_causal():
    layer = perturbed_layer(1)
    u = torch.randn(1, 20, 12, dtype=torch.float64)
    v = u.clone()
    v[:, 15:] += 1.0
    torch.testing.assert_close(layer(u)[:, :15], layer(v)[:, :15])


def test_block_sees_both_past_and_future():
    torch.manual_seed(0)
    block = Block(8, n_heads=2, head_dim=4, d_state=4, dropout=0.0).double().eval()
    u = torch.randn(1, 20, 8, dtype=torch.float64)
    v = u.clone()
    v[:, 15] += 1.0
    changed = (block(u) - block(v)).abs().sum(-1)[0]
    assert changed[:15].max() > 0 and changed[16:].max() > 0


def test_rotation_preserves_norm_and_inverts():
    v = torch.randn(5, 8, dtype=torch.float64)
    angle = torch.randn(5, 4, dtype=torch.float64)
    r = rotate_pairs(v, angle)
    torch.testing.assert_close(r.norm(dim=-1), v.norm(dim=-1))
    torch.testing.assert_close(rotate_pairs(r, -angle), v)


def test_standardizer_zero_fills_and_flags_missing():
    x = np.array([[1.0, np.nan], [3.0, 2.0], [5.0, 4.0]], dtype=np.float32)
    std = Standardizer().fit([x])
    z = std.transform(x)
    assert z.shape == (3, 3) and std.n_out == 3
    assert not np.isnan(z).any()
    np.testing.assert_allclose(z[:, 2], [1, 0, 0])
    np.testing.assert_allclose(z[:, 0].mean(), 0, atol=1e-6)


def test_fit_and_predict_cover_every_frame_and_are_team_symmetric():
    rng = np.random.default_rng(0)
    n = WINDOW * STEP + 333
    x = rng.normal(size=(n, 4)).astype(np.float32)
    label = np.where(x[:, 0] > 0, H, A).astype(np.int8)
    label[:50] = D
    period = np.ones(n, dtype=np.int8)
    period[:10] = 0
    swapped = x.copy()
    swapped[:, 0] *= -1
    tracks = [
        (x, label, period),
        (swapped, np.where(label == H, A, np.where(label == A, H, D)), period),
    ]
    model = fit(tracks, TrainConfig(steps=3, batch=2, d_model=8, head_dim=4), log=None)
    probs = predict_from_features(model, x, swapped, period)
    assert probs.shape == (n, 3)
    np.testing.assert_allclose(probs.sum(1), 1.0, atol=1e-9)
    np.testing.assert_allclose(probs[:10], [[0, 0, 1]] * 10)
    mirror = predict_from_features(model, swapped, x, period)
    np.testing.assert_allclose(mirror[:, H], probs[:, A], atol=1e-6)
