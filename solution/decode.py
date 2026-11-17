"""Turn per-frame class probabilities into a few clean possessions: a duration-aware Viterbi.

spellF1 scores whole possessions, so a 3-frame flicker costs a full false possession and splits a
true one. A majority-vote window (the baseline) removes flicker but also smears every boundary.
Instead we decode the most likely label path under an explicit-duration model:

- each label is a chain of `min_len` sub-states that must be walked through before the label can
  end, so no H, A or D run shorter than its minimum is possible;
- leaving a label costs a fixed log-penalty per transition type (H<->A, alive->D, D->alive);
- emissions are the classifier's log-probabilities, plus a per-class log-bias.

That is a hidden semi-Markov model with a shifted-geometric duration per label. Decoding is exact,
runs per half, and is compiled with numba because Viterbi is inherently sequential over frames.
All of its parameters are chosen leave-one-match-out on the open IDSSE matches (solution.validate,
solution.compare_models), never on the competition match.
"""

from dataclasses import asdict, dataclass

import numba
import numpy as np

from solution.common import A, D, H, X, period_slices

LOG_FLOOR = np.log(1e-6)


@dataclass(frozen=True)
class DecoderParams:
    min_alive: int = 25  # frames
    min_dead: int = 25
    cost_switch: float = 10.0  # H <-> A
    cost_dead: float = 10.0  # alive -> D
    cost_live: float = 10.0  # D -> alive
    bias_dead: float = 0.0  # added to log p(D)

    def to_dict(self):
        return asdict(self)


@numba.njit(cache=True)
def _viterbi(scores, min_len, cost):
    """Best label path. scores [T, L] per-frame log-scores, min_len [L], cost [L, L]."""
    n_frames, n_labels = scores.shape
    offsets = np.zeros(n_labels + 1, dtype=np.int64)
    for label in range(n_labels):
        offsets[label + 1] = offsets[label] + min_len[label]
    n_states = offsets[n_labels]
    state_label = np.empty(n_states, dtype=np.int64)
    for label in range(n_labels):
        state_label[offsets[label] : offsets[label + 1]] = label

    neg = -1e18
    prev = np.full(n_states, neg)
    for label in range(n_labels):
        prev[offsets[label + 1] - 1] = scores[0, label]  # a half may open mid-spell
    back = np.empty((n_frames, n_states), dtype=np.int32)
    cur = np.empty(n_states)

    for t in range(1, n_frames):
        for label in range(n_labels):
            first = offsets[label]
            last = offsets[label + 1] - 1
            emit = scores[t, label]
            # Entering the chain: from the final sub-state of another label.
            best, arg = neg, -1
            for other in range(n_labels):
                if other == label:
                    continue
                value = prev[offsets[other + 1] - 1] - cost[other, label]
                if value > best:
                    best, arg = value, offsets[other + 1] - 1
            if first == last:
                if prev[last] >= best:
                    best, arg = prev[last], last
                cur[first] = best + emit
                back[t, first] = arg
                continue
            cur[first] = best + emit
            back[t, first] = arg
            for state in range(first + 1, last):
                cur[state] = prev[state - 1] + emit
                back[t, state] = state - 1
            if prev[last] >= prev[last - 1]:
                cur[last] = prev[last] + emit
                back[t, last] = last
            else:
                cur[last] = prev[last - 1] + emit
                back[t, last] = last - 1
        prev, cur = cur, prev

    path = np.empty(n_frames, dtype=np.int64)
    state = int(np.argmax(prev))
    for t in range(n_frames - 1, -1, -1):
        path[t] = state_label[state]
        if t > 0:
            state = back[t, state]
    return path


def decode_half(log_probs, params):
    """Labels for one contiguous half from [T, 3] log-probabilities over (H, A, D)."""
    scores = np.maximum(log_probs, LOG_FLOOR).astype(np.float64).copy()
    scores[:, D] += params.bias_dead
    min_len = np.array([params.min_alive, params.min_alive, params.min_dead], dtype=np.int64)
    cost = np.zeros((3, 3))
    cost[H, A] = cost[A, H] = params.cost_switch
    cost[H, D] = cost[A, D] = params.cost_dead
    cost[D, H] = cost[D, A] = params.cost_live
    return _viterbi(scores, min_len, cost).astype(np.int8)


def decode(probs, period, params):
    """Full-length label codes: decoded inside each period, X outside."""
    out = np.full(len(period), X, dtype=np.int8)
    log_probs = np.log(np.clip(probs, 1e-9, 1.0))
    for _, start, stop in period_slices(period):
        out[start:stop] = decode_half(log_probs[start:stop], params)
    return out
