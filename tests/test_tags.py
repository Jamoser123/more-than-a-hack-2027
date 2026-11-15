"""Tests for the Smart Tagging rules in solution/tags.py. Events are built inline; no data/ needed."""

import numpy as np

from solution.common import FPS, A, D, H
from solution.decode import DecoderParams, decode
from solution.tags import (
    ACC_MAX,
    RESTART_TAIL,
    SET_PIECE_CODE,
    agreement,
    build_prior,
    tag_rules,
    weights,
)

HOME = "Barcelona"


def event(code, start, stop, team=None, kind=None):
    labels = {}
    if team is not None:
        labels["Team"] = team
    if kind is not None:
        labels["Type"] = kind
    return {"code": code, "start_frame": start, "end_frame": stop, "tags": labels}


def test_rules_keep_possession_phases_and_map_teams():
    rules = tag_rules(
        [event("BUILD UP", 0, 10, "Barcelona"), event("Progression", 10, 20, "Atletic Club")],
        HOME,
    )
    assert rules == [("BUILD UP", H, 0, 10), ("PROGRESSION", A, 10, 20)]


def test_rules_ignore_mirrors_locations_and_teamless_restarts():
    rules = tag_rules(
        [
            event("DEFENDING IN MIDDLE THIRD", 0, 10, "Barcelona"),
            event("BALL IN THE BOX", 0, 10, "Barcelona"),
            event("SET PIECES", 0, 10, "N/A", "Throw-in"),
            event("SET PIECES", 0, 10, None, "Kick Off (After Goal)"),
            event("BUILD UP", 0, 10, "N/A"),
        ],
        HOME,
    )
    assert rules == []


def test_set_piece_window_extends_past_the_tag_and_goals_are_dead():
    rules = tag_rules(
        [event("SET PIECES", 100, 200, "Atletic Club", "Corner Kick"), event("GOALS", 0, 50)],
        HOME,
    )
    assert rules == [(SET_PIECE_CODE, A, 100, 200 + RESTART_TAIL), ("GOALS", D, 0, 50)]


def test_phases_can_be_left_out():
    events = [
        event("BUILD UP", 0, 10, "Barcelona"),
        event("SET PIECES", 20, 30, "Barcelona", "Goal Kick"),
        event("GOALS", 40, 50),
    ]
    assert [r[0] for r in tag_rules(events, HOME, phases=False)] == [SET_PIECE_CODE, "GOALS"]


def test_team_evidence_penalises_only_the_other_team():
    prior = build_prior([(SET_PIECE_CODE, H, 2, 5)], {SET_PIECE_CODE: 1.5}, 8)
    np.testing.assert_array_equal(prior[:, H], 0)
    np.testing.assert_array_equal(prior[:, D], 0)
    np.testing.assert_array_equal(prior[:, A], [0, 0, -1.5, -1.5, -1.5, 0, 0, 0])


def test_a_phase_counts_once_spread_over_its_frames():
    prior = build_prior([("BUILD UP", H, 0, 4)], {"BUILD UP": 2.0}, 6)
    np.testing.assert_allclose(prior[:, A], [-0.5, -0.5, -0.5, -0.5, 0, 0])
    assert np.isclose(prior[:, A].sum(), -2.0)


def test_dead_evidence_penalises_both_teams():
    prior = build_prior([("GOALS", D, 0, 3)], {"GOALS": 2.0}, 4)
    np.testing.assert_array_equal(prior[:3, H], -2.0)
    np.testing.assert_array_equal(prior[:3, A], -2.0)
    np.testing.assert_array_equal(prior[:, D], 0)


def test_overlapping_tags_take_the_strongest_penalty_not_the_sum():
    rules = [(SET_PIECE_CODE, H, 0, 10), ("GOALS", D, 5, 15)]
    prior = build_prior(rules, {SET_PIECE_CODE: 2.0, "GOALS": 1.0}, 20)
    assert prior[7, A] == -2.0
    assert prior[7, H] == -1.0
    assert prior[12, A] == -1.0


def test_agreement_counts_only_frames_the_classifier_calls_alive():
    probs = np.array([[0.8, 0.1, 0.1]] * 6 + [[0.1, 0.8, 0.1]] * 2 + [[0.1, 0.1, 0.8]] * 2)
    agree = agreement([("BUILD UP", H, 0, 10)], probs)
    assert agree["BUILD UP"] == (0.75, 8)


def test_weights_are_capped_log_odds_and_zero_at_chance_or_on_little_data():
    big = 10 * FPS
    w = weights({"a": (0.99, big), "b": (0.75, big), "c": (0.4, big), "d": (0.95, 5)})
    assert np.isclose(w["a"], np.log(ACC_MAX / (1 - ACC_MAX)))
    assert np.isclose(w["b"], np.log(3))
    assert w["c"] == 0.0
    assert w["d"] == 0.0


def test_set_piece_flips_an_uncertain_stretch_but_not_a_confident_one():
    n = 300
    probs = np.full((n, 3), [0.6, 0.35, 0.05])
    probs[100:200] = [0.45, 0.5, 0.05]  # classifier leans A, weakly
    period = np.ones(n, dtype=np.int8)
    params = DecoderParams(min_alive=10, cost_switch=5.0)
    assert (decode(probs, period, params)[100:200] == A).all()
    prior = build_prior([(SET_PIECE_CODE, H, 100, 200)], {SET_PIECE_CODE: 1.0}, n)
    assert (decode(probs, period, params, prior) == H).all()

    probs[100:200] = [0.02, 0.96, 0.02]  # classifier is sure it is A
    assert (decode(probs, period, params, prior)[100:200] == A).all()


def test_a_phase_keeps_a_short_confident_possession_of_the_other_team():
    n = 400
    probs = np.full((n, 3), [0.8, 0.15, 0.05])
    probs[180:240] = [0.15, 0.8, 0.05]  # a clear 2.4 s possession of A inside an H phase
    period = np.ones(n, dtype=np.int8)
    params = DecoderParams(min_alive=10, cost_switch=5.0)
    prior = build_prior([("PROGRESSION", H, 0, n)], {"PROGRESSION": 2.2}, n)
    assert (decode(probs, period, params, prior)[180:240] == A).all()


def test_a_phase_settles_a_near_tie():
    n = 400
    probs = np.full((n, 3), [0.8, 0.15, 0.05])
    # The classifier cannot tell: A leads by 60 * log(0.476 / 0.474) ~ 0.25 in total, less than
    # the phase's share over these frames (2.2 * 60 / 400 = 0.33).
    probs[180:240] = [0.474, 0.476, 0.05]
    period = np.ones(n, dtype=np.int8)
    params = DecoderParams(min_alive=10, cost_switch=0.01)
    assert (decode(probs, period, params)[180:240] == A).all()
    prior = build_prior([("PROGRESSION", H, 0, n)], {"PROGRESSION": 2.2}, n)
    assert (decode(probs, period, params, prior) == H).all()
