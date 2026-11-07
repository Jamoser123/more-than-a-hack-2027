"""Tests for the label-changing logic of solution/. None of them needs data/."""

import numpy as np
import pytest

from solution import atd
from solution.common import A, D, H, X, Match, gt_with_half_breaks, nan_window_mean, run_bounds
from solution.decode import DecoderParams, decode, decode_half
from solution.degrade import burst_mask, degrade
from solution.features import attack_signs, build_features, interpolate_ball


def onehot(labels, p=0.9):
    probs = np.full((len(labels), 3), (1 - p) / 2)
    probs[np.arange(len(labels)), labels] = p
    return np.log(probs)


def test_decoder_removes_flicker_shorter_than_min_len():
    labels = np.array([H] * 100 + [A] * 3 + [H] * 100)
    out = decode_half(onehot(labels), DecoderParams(min_alive=10, cost_switch=1.0))
    assert (out == H).all()


def test_decoder_keeps_long_spells_and_exact_boundaries():
    labels = np.array([H] * 80 + [A] * 80 + [D] * 80 + [A] * 80)
    out = decode_half(onehot(labels), DecoderParams(min_alive=10, min_dead=10, cost_switch=2.0))
    np.testing.assert_array_equal(out, labels)


def test_decoder_respects_min_duration_everywhere_but_the_edges():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), 500)
    out = decode_half(
        np.log(probs),
        DecoderParams(min_alive=7, min_dead=9, cost_switch=0.0, cost_dead=0.0, cost_live=0.0),
    )
    starts, stops, values = run_bounds(out)
    lengths = (stops - starts)[1:-1]
    minimum = np.where(values[1:-1] == D, 9, 7)
    assert (lengths >= minimum).all()


def test_decode_writes_x_outside_periods():
    period = np.array([0] * 5 + [1] * 50 + [0] * 5 + [2] * 50)
    probs = np.tile([0.8, 0.1, 0.1], (len(period), 1))
    out = decode(probs, period, DecoderParams(min_alive=5))
    assert (out[period == 0] == X).all() and (out[period > 0] == H).all()


def test_gt_breaks_halves_for_the_scorer():
    period = np.array([1, 1, 1, 2, 2])
    gt = gt_with_half_breaks(np.array([H, H, H, H, H]), period)
    np.testing.assert_array_equal(gt, [X, H, H, X, H])


def test_nan_window_mean():
    values = np.array([1.0, np.nan, 3.0, 5.0])
    np.testing.assert_allclose(nan_window_mean(values, 1, 1), [1.0, 2.0, 4.0, 4.0])
    np.testing.assert_allclose(nan_window_mean(values, 1, 0), [1.0, 1.0, 3.0, 4.0])


def test_burst_mask_share():
    mask = burst_mask(np.random.default_rng(1), 200_000, 60.0, 40.0)
    assert len(mask) == 200_000 and abs(mask.mean() - 0.4) < 0.03


def toy_match(n=600, seed=0):
    rng = np.random.default_rng(seed)
    home = rng.uniform([10, 5], [50, 60], (n, 11, 2)).astype(np.float32)
    away = rng.uniform([55, 5], [95, 60], (n, 11, 2)).astype(np.float32)
    home[:, 0] = [3.0, 34.0]  # home keeper deep at -x
    away[:, 0] = [102.0, 34.0]
    ball = rng.uniform([0, 0], [105, 68], (n, 2)).astype(np.float32)
    period = np.array([1] * (n // 2) + [2] * (n - n // 2), dtype=np.int8)
    label = rng.integers(0, 3, n).astype(np.int8)
    return Match(home=home, away=away, ball=ball, period=period, pitch=(105.0, 68.0), label=label)


def test_degrade_is_deterministic_and_keeps_labels():
    match = toy_match()
    one, two = degrade(match, 7), degrade(match, 7)
    np.testing.assert_array_equal(one.ball, two.ball)
    np.testing.assert_array_equal(one.home, two.home)
    np.testing.assert_array_equal(one.label, match.label)


def test_mirror_swaps_teams_and_labels():
    match = toy_match()
    mirrored = match.mirrored()
    assert mirrored.home is match.away
    np.testing.assert_array_equal(mirrored.label == H, match.label == A)
    assert (attack_signs(match) == 1).all() and (attack_signs(mirrored) == -1).all()


def test_features_are_permutation_invariant_over_slots():
    match = toy_match()
    order = np.random.default_rng(3).permutation(11)
    shuffled = Match(
        home=match.home[:, order],
        away=match.away[:, order[::-1]],
        ball=match.ball,
        period=match.period,
        pitch=match.pitch,
    )
    a, names = build_features(match)
    b, _ = build_features(shuffled)
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4, equal_nan=True)
    assert len(names) == a.shape[1] == len(set(names))


def test_interpolation_bridges_short_gaps_only():
    ball = np.full((200, 2), np.nan, dtype=np.float32)
    ball[0], ball[10], ball[150] = [0, 0], [10, 0], [20, 0]
    out, interpolated, since, until = interpolate_ball(ball, np.ones(200, dtype=np.int8), 50)
    np.testing.assert_allclose(out[5], [5, 0])
    assert np.isnan(out[100, 0]) and interpolated[5] == 1 and since[5] == 5 and until[5] == 5


@pytest.fixture
def fake_bundle(tmp_path):
    """A full-length tactical ATD pair in the documented format, made of synthetic positions."""
    from games import GAMES, source_frames

    folder = tmp_path / "tactical"
    folder.mkdir()
    (folder / "BAR-ATH_tactical_atd.xml").write_text(
        "<Root><Teams><Team id='T1'><Name>Home</Name></Team><Team id='T2'><Name>Away</Name>"
        "</Team></Teams><Players><Player teamId='T1'/><Player teamId='T2'/>"
        "<Player teamId='T1'/></Players></Root>"
    )
    n = source_frames("tactical")
    rng = np.random.default_rng(0)
    slots = rng.uniform(0, 1, (n, 3, 2)).round(4)
    ball = rng.uniform(0, 1, (n, 2)).round(4)
    lines = []
    for i in range(n):
        groups = ["{},{},1.0".format(*slots[i, k]) for k in range(3)]
        groups[1] = "NaN,NaN,NaN" if i % 7 == 0 else groups[1]
        ball_group = "NaN,NaN,NaN" if i % 3 == 0 else "{},{},0.0".format(*ball[i])
        lines.append("{}:{}:{}\n".format(i + 1, ";".join(groups), ball_group))
    (folder / "BAR-ATH_tactical_atd.txt").write_text("".join(lines))
    return tmp_path, slots * np.array(GAMES["BAR-ATH"]["pitch"]), ball


def test_atd_loader_aligns_rows_teams_and_halves(fake_bundle):
    data_dir, slots, ball = fake_bundle
    match = atd.load(data_dir=str(data_dir))
    assert match.n_frames == len(slots) and match.home.shape[1] == 2 and match.away.shape[1] == 1
    np.testing.assert_allclose(match.home[5], slots[5, [0, 2]], atol=1e-3)
    np.testing.assert_allclose(match.away[8, 0], slots[8, 1], atol=1e-3)
    assert np.isnan(match.away[7, 0]).all() and np.isnan(match.ball[3]).all()
    assert match.period[11184] == 0 and match.period[11185] == 1 and match.period[90652] == 0
    assert match.period[110808] == 2 and match.period[190197] == 2 and match.period[190198] == 0
