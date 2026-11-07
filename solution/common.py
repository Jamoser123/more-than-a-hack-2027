"""The one data shape every part of the pipeline speaks, so open data and the match are interchangeable.

The model is trained on open matches from other games and applied to the competition match. Both
are converted into a `Match`: team-split player positions and a ball position in metres, one row
per frame, NaN where nothing was observed. Nothing downstream knows where a Match came from, which
is what lets the validation on open data say something about the competition match.

Label codes follow evaluation/possession_metrics.py (H=0, A=1, D=2, X=3) so arrays can be handed
to the official scorer unchanged.
"""

import os
import sys
from dataclasses import dataclass, replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

CACHE_DIR = os.path.join(ROOT, "cache")

H, A, D, X = 0, 1, 2, 3
LETTERS = np.array(["H", "A", "D", "X"])
FPS = 25


@dataclass
class Match:
    """Positions of one match. Arrays are frame-major; NaN marks a missing observation.

    home, away: [T, K, 2] player positions in metres, x along the pitch length. Slots carry no
        identity that the pipeline relies on: every feature is permutation invariant.
    ball: [T, 2] ball position in metres.
    period: [T] 1 or 2 inside a half, 0 outside the graded play (warm-up, halftime).
    pitch: (length, width) in metres.
    label: [T] H/A/D codes where known (open data), None for the competition match.
    """

    home: np.ndarray
    away: np.ndarray
    ball: np.ndarray
    period: np.ndarray
    pitch: tuple
    label: np.ndarray = None

    @property
    def n_frames(self):
        return len(self.period)

    def mirrored(self):
        """The same match with the two teams swapped, H and A labels swapped with them.

        Used as augmentation: the classifier should not learn that "home" is special, and the
        swap doubles the training data for free.
        """
        label = None
        if self.label is not None:
            label = self.label.copy()
            label[self.label == H] = A
            label[self.label == A] = H
        return replace(self, home=self.away, away=self.home, label=label)


def run_bounds(values):
    """(starts, stops, values) of the maximal constant runs of a 1-D array."""
    values = np.asarray(values)
    if not len(values):
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, values[:0]
    breaks = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [len(values)]))
    return starts, stops, values[starts]


def period_slices(period):
    """[(period, start, stop)] for each contiguous block of a non-zero period."""
    starts, stops, values = run_bounds(period)
    return [(int(v), int(a), int(b)) for a, b, v in zip(starts, stops, values) if v > 0]


def nan_window_mean(values, before, after):
    """Mean of `values` over frames [t - before, t + after], ignoring NaN. Columns independent.

    Cumulative sums make the cost independent of the window size. NaN where the window holds no
    observation at all.
    """
    values = np.asarray(values, dtype=np.float64)
    squeeze = values.ndim == 1
    if squeeze:
        values = values[:, None]
    seen = ~np.isnan(values)
    total = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(np.where(seen, values, 0), 0)])
    count = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(seen, 0)])
    index = np.arange(len(values))
    low = np.maximum(0, index - before)
    high = np.minimum(len(values), index + after + 1)
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (total[high] - total[low]) / n
    out[n == 0] = np.nan
    return out[:, 0] if squeeze else out


def gt_with_half_breaks(label, period):
    """Ground-truth codes for the scorer: X outside the halves and on the first frame of each.

    The official scorer finds the halves as runs of non-X frames. Open-data periods are stored
    back to back, so without a break the two halves would merge into one run and a possession
    could straddle halftime.
    """
    gt = np.where(period > 0, label, X).astype(np.int8)
    for _, start, _ in period_slices(period):
        gt[start] = X
    return gt
