"""Make clean open-data tracking look like the competition's automatic tracking.

IDSSE tracking is corrected broadcast-grade TRACAB: every player on every frame, the ball on every
frame. The competition ATD is automatic single-camera output: the ball missing on ~43% of frames,
in bursts and disproportionately where play is crowded or the ball is out of the camera's view,
player slots that drop out, positional jitter, and the odd player put on the wrong team. A model
trained on clean data would lean on the ball and fall over on day two.

So the training and validation inputs are corrupted with those failure modes, with every rate drawn
from a range rather than fixed (domain randomisation). The ranges are set from the facts published
in DATA.md (ball visible 56.9% of frames, ~20.5 of 22 players tracked), never from any label.
Every draw comes from a seeded numpy Generator, so the same seed gives the same corruption.
"""

from dataclasses import dataclass, replace

import numpy as np

from solution.common import FPS, Match


@dataclass
class DegradeRanges:
    """Uniform ranges each corruption rate is drawn from, once per degraded copy."""

    ball_visible_run: tuple = (40.0, 160.0)  # mean frames of a visible ball burst
    ball_missing_run: tuple = (18.0, 90.0)  # mean frames of a missing ball burst
    crowd_drop: tuple = (0.3, 0.7)  # extra drop probability with >= 2 players within 2 m
    out_drop: tuple = (0.5, 0.9)  # extra drop probability with the ball off the pitch
    ball_noise: tuple = (0.1, 0.5)  # metres, temporally smooth
    player_visible_run: tuple = (300.0, 2000.0)
    player_missing_run: tuple = (10.0, 120.0)
    player_noise: tuple = (0.1, 0.4)
    team_swap_fraction: tuple = (0.0, 0.03)  # share of slot-frames carried on the wrong team


def _draw(rng, bounds):
    return float(rng.uniform(*bounds))


def burst_mask(rng, n, mean_on, mean_off):
    """Boolean [n], True during "off" bursts of an alternating on/off renewal process.

    Run lengths are geometric, so the expected off share is mean_off / (mean_on + mean_off).
    The phase at frame 0 is random rather than always "on".
    """
    n_pairs = int(2 * n / (mean_on + mean_off)) + 16
    while True:
        on = rng.geometric(1.0 / mean_on, n_pairs)
        off = rng.geometric(1.0 / mean_off, n_pairs)
        if on.sum() + off.sum() >= n + mean_on + mean_off:
            break
        n_pairs *= 2
    lengths = np.stack([on, off], axis=1).ravel()
    state = np.tile([False, True], n_pairs)
    mask = np.repeat(state, lengths)
    shift = int(rng.integers(0, int(mean_on + mean_off)))
    return mask[shift : shift + n]


def smooth_noise(rng, shape, sigma, window=FPS):
    """Gaussian noise low-passed over `window` frames along axis 0 and rescaled to `sigma`.

    Tracker jitter is correlated in time. White noise of the same size would add several m/s of
    fake speed after differencing, which the real tracking does not have.
    """
    white = rng.standard_normal((shape[0] + window,) + tuple(shape[1:]))
    total = np.cumsum(white, axis=0)
    low = (total[window:] - total[:-window]) / np.sqrt(window)
    return (low * sigma).astype(np.float32)


def degrade(match, seed, ranges=None):
    """A corrupted copy of `match`. Labels and periods are left exactly as they were."""
    ranges = ranges or DegradeRanges()
    rng = np.random.default_rng(seed)
    n = match.n_frames
    length, width = match.pitch

    # Ball: background bursts, plus extra drop-outs in crowds and off the pitch.
    ball = match.ball.astype(np.float32).copy()
    missing = burst_mask(
        rng, n, _draw(rng, ranges.ball_visible_run), _draw(rng, ranges.ball_missing_run)
    )
    players = np.concatenate([match.home, match.away], axis=1)
    with np.errstate(invalid="ignore"):
        dist = np.linalg.norm(players - ball[:, None, :], axis=-1)
        crowded = np.sum(dist < 2.0, axis=1) >= 2
        off_pitch = (
            (ball[:, 0] < 0) | (ball[:, 0] > length) | (ball[:, 1] < 0) | (ball[:, 1] > width)
        )
    # Condition-driven drops come in bursts too: a burst process gates where they can happen.
    crowd_gate = burst_mask(rng, n, 30.0, 30.0 / max(1e-3, 1.0 / _draw(rng, ranges.crowd_drop) - 1))
    out_gate = burst_mask(rng, n, 30.0, 30.0 / max(1e-3, 1.0 / _draw(rng, ranges.out_drop) - 1))
    missing |= (crowded & crowd_gate) | (off_pitch & out_gate)
    ball += smooth_noise(rng, ball.shape, _draw(rng, ranges.ball_noise))
    ball[missing] = np.nan

    # Players: per-slot drop-out bursts and jitter, and short spells on the wrong team.
    teams = []
    for side in (match.home, match.away):
        side = side.astype(np.float32).copy()
        side += smooth_noise(rng, side.shape, _draw(rng, ranges.player_noise))
        visible_run = _draw(rng, ranges.player_visible_run)
        missing_run = _draw(rng, ranges.player_missing_run)
        for slot in range(side.shape[1]):
            side[burst_mask(rng, n, visible_run, missing_run), slot] = np.nan
        teams.append(side)

    swap_share = _draw(rng, ranges.team_swap_fraction)
    home, away = teams
    if swap_share > 0:
        moved = []
        for side in (home, away):
            out = np.full_like(side, np.nan)
            for slot in range(side.shape[1]):
                wrong = burst_mask(rng, n, 250.0 / swap_share, 250.0)
                out[wrong, slot] = side[wrong, slot]
                side[wrong, slot] = np.nan
            moved.append(out)
        home, away = (
            np.concatenate([home, moved[1]], axis=1),
            np.concatenate([away, moved[0]], axis=1),
        )

    return replace(match, home=home, away=away, ball=ball)


def ball_visible_share(match: Match):
    """Share of in-period frames with a ball position, for checking a degraded copy."""
    inside = match.period > 0
    return float(np.mean(~np.isnan(match.ball[inside, 0])))
