"""Per-frame evidence for H / A / D that survives a missing ball and anonymous, swapping slots.

Three families, all permutation invariant over player slots (the tracking has no identities):

- ball: is it seen, where is it (in the home side's attacking direction), how fast, is it off the
  pitch, and how long since / until it was last seen. Short gaps are bridged by linear interpolation.
- proximity: distance of the nearest and second-nearest player of each team to the (interpolated)
  ball, players within 3 m, and the last / next proximity verdict seen before / after a ball gap.
- team shape: per team, centroid, spread, back and front line, speed and forward drift, measured
  in that team's own attacking direction. The team in possession spreads and pushes up; the other
  compacts and runs back. This is what carries the 41% of frames without a ball.

Each family is also averaged over several centred, past and future windows so the classifier sees
context. The decoder (decode.py) then turns per-frame probabilities into possessions.
"""

import numpy as np

from solution.common import FPS, nan_window_mean, period_slices

MAX_INTERP = 2 * FPS  # bridge ball gaps up to this many frames
GAP_CAP = 20 * FPS
SPEED_CAP = 12.0  # m/s; anything faster is a slot swap, not a sprint
NEAR = 3.0
CONTEXT_WINDOWS = ((12, 12), (37, 37), (100, 100), (250, 250))
DIRECTIONAL_WINDOWS = ((25, 0), (0, 25), (75, 0), (0, 75))


def attack_signs(match):
    """[T] +1 where the home side attacks towards +x, -1 where towards -x.

    Per half, the side whose player is most often the deepest on the -x end (its goalkeeper)
    defends that end. Read off positions only, so it needs no labels and no metadata.
    """
    sign = np.ones(match.n_frames, dtype=np.float32)
    for _, start, stop in period_slices(match.period):
        home_min = np.nanmin(
            np.where(np.isnan(match.home[start:stop, :, 0]), np.inf, match.home[start:stop, :, 0]),
            axis=1,
        )
        away_min = np.nanmin(
            np.where(np.isnan(match.away[start:stop, :, 0]), np.inf, match.away[start:stop, :, 0]),
            axis=1,
        )
        both = np.isfinite(home_min) & np.isfinite(away_min)
        home_deepest = np.mean(home_min[both] < away_min[both]) if both.any() else 0.5
        sign[start:stop] = 1.0 if home_deepest >= 0.5 else -1.0
    return sign


def interpolate_ball(ball, period, max_gap=MAX_INTERP):
    """(ball, interpolated_flag, frames_since_seen, frames_until_seen), all per frame.

    Interpolates linearly inside each period across gaps no longer than max_gap.
    """
    n = len(ball)
    out = np.full_like(ball, np.nan, dtype=np.float32)
    since = np.full(n, GAP_CAP, dtype=np.float32)
    until = np.full(n, GAP_CAP, dtype=np.float32)
    seen = ~np.isnan(ball[:, 0])
    for _, start, stop in period_slices(period):
        index = np.arange(start, stop)
        known = index[seen[start:stop]]
        if len(known) < 2:
            continue
        prev = np.maximum.accumulate(np.where(seen[start:stop], index, -1))
        nxt = np.minimum.accumulate(np.where(seen[start:stop], index, n)[::-1])[::-1]
        has_prev, has_next = prev >= 0, nxt < n
        since[start:stop] = np.where(has_prev, np.minimum(index - prev, GAP_CAP), GAP_CAP)
        until[start:stop] = np.where(has_next, np.minimum(nxt - index, GAP_CAP), GAP_CAP)
        bridge = has_prev & has_next & (nxt - prev <= max_gap)
        for axis in range(2):
            out[start:stop, axis] = np.where(
                bridge, np.interp(index, known, ball[known, axis]), np.nan
            )
    out[seen] = ball[seen]
    interpolated = (~seen & ~np.isnan(out[:, 0])).astype(np.float32)
    return out, interpolated, since, until


def slot_speeds(positions):
    """[T, K] speed of each slot in m/s from a 4-frame central difference, capped."""
    speed = np.full(positions.shape[:2], np.nan, dtype=np.float32)
    delta = positions[4:] - positions[:-4]
    with np.errstate(invalid="ignore"):
        speed[2:-2] = np.minimum(np.linalg.norm(delta, axis=-1) * FPS / 4.0, SPEED_CAP)
    return speed


def _nan_stat(fn, values, axis=1):
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        return fn(values, axis=axis)


def kth_smallest(values, k):
    """k-th smallest (0-based) along axis 1 ignoring NaN; NaN if fewer than k+1 values."""
    filled = np.where(np.isnan(values), np.inf, values)
    if filled.shape[1] <= k:
        return np.full(len(values), np.nan, dtype=np.float32)
    part = np.partition(filled, k, axis=1)[:, k]
    return np.where(np.isinf(part), np.nan, part).astype(np.float32)


def team_shape(positions, own_sign, length, width, prefix):
    """Shape and motion of one team in its own attacking direction. Dict of [T] arrays."""
    x = np.where(own_sign[:, None] > 0, positions[..., 0], length - positions[..., 0])
    y = positions[..., 1]
    count = np.sum(~np.isnan(x), axis=1).astype(np.float32)
    cx = _nan_stat(np.nanmean, x)
    cy = _nan_stat(np.nanmean, y)
    speed = slot_speeds(np.stack([x, y], axis=-1))
    cx_velocity = np.full_like(cx, np.nan)
    cx_velocity[FPS:] = cx[FPS:] - cx[:-FPS]
    return {
        prefix + "count": count,
        prefix + "cx": cx,
        prefix + "cy": np.abs(cy - width / 2.0),
        prefix + "sx": _nan_stat(np.nanstd, x),
        prefix + "sy": _nan_stat(np.nanstd, y),
        prefix + "back": kth_smallest(x, 1),  # second deepest: the back line, not the keeper
        prefix + "front": -kth_smallest(-x, 0),
        prefix + "speed": _nan_stat(np.nanmean, speed),
        prefix + "speed_max": _nan_stat(np.nanmax, speed),
        prefix + "fast": np.sum(speed > 4.0, axis=1).astype(np.float32),
        prefix + "drift": cx_velocity,
        prefix + "opp_half": np.sum(x > length / 2.0, axis=1).astype(np.float32),
    }


def build_features(match):
    """(features [T, F] float32, names). NaN is left in place; the classifier handles it."""
    length, width = match.pitch
    sign = attack_signs(match)
    ball, interpolated, since, until = interpolate_ball(match.ball, match.period)
    seen = (~np.isnan(match.ball[:, 0])).astype(np.float32)

    f = {}
    f["ball_seen"] = seen
    f["ball_interp"] = interpolated
    f["ball_since"] = since
    f["ball_until"] = until
    bx = np.where(sign > 0, ball[:, 0], length - ball[:, 0])
    f["ball_x"] = bx
    f["ball_yc"] = np.abs(ball[:, 1] - width / 2.0)
    with np.errstate(invalid="ignore"):
        f["ball_out"] = np.maximum.reduce(
            [-ball[:, 0], ball[:, 0] - length, -ball[:, 1], ball[:, 1] - width]
        )
    velocity = np.full(len(ball), np.nan, dtype=np.float32)
    velocity[2:-2] = np.linalg.norm(ball[4:] - ball[:-4], axis=1) * FPS / 4.0
    f["ball_speed"] = np.minimum(velocity, 40.0)

    for side, positions in (("h", match.home), ("a", match.away)):
        with np.errstate(invalid="ignore"):
            dist = np.linalg.norm(positions - ball[:, None, :], axis=-1)
        f["d1_" + side] = kth_smallest(dist, 0)
        f["d2_" + side] = kth_smallest(dist, 1)
        f["near_" + side] = np.sum(dist < NEAR, axis=1).astype(np.float32)
    with np.errstate(invalid="ignore"):
        prox = np.clip(f["d1_a"] - f["d1_h"], -15.0, 15.0)
    f["prox"] = prox
    f["prox_seen"] = np.where(seen > 0, prox, np.nan)
    # The last and next proximity verdict with the ball actually seen, carried across a gap.
    index = np.arange(len(prox))
    valid = ~np.isnan(f["prox_seen"])
    prev = np.maximum.accumulate(np.where(valid, index, 0))
    nxt = np.minimum.accumulate(np.where(valid, index, len(prox) - 1)[::-1])[::-1]
    f["prox_prev"] = np.where(valid[prev], f["prox_seen"][prev], np.nan)
    f["prox_next"] = np.where(valid[nxt], f["prox_seen"][nxt], np.nan)

    f.update(team_shape(match.home, sign, length, width, "h_"))
    f.update(team_shape(match.away, -sign, length, width, "a_"))
    for key in ("cx", "sx", "sy", "speed", "drift", "front", "back", "opp_half", "fast"):
        f["diff_" + key] = f["h_" + key] - f["a_" + key]
    f["all_speed"] = _nan_stat(np.nanmean, np.stack([f["h_speed"], f["a_speed"]]), axis=0)
    f["all_fast"] = f["h_fast"] + f["a_fast"]
    f["players"] = f["h_count"] + f["a_count"]
    f["h_cx_ball"] = f["h_cx"] - bx
    f["a_cx_ball"] = f["a_cx"] - (length - bx)

    context_keys = [
        "ball_seen",
        "prox_seen",
        "prox",
        "ball_out",
        "ball_speed",
        "all_speed",
        "all_fast",
        "diff_cx",
        "diff_sx",
        "diff_speed",
        "diff_drift",
        "diff_front",
        "diff_back",
        "h_drift",
        "a_drift",
        "near_h",
        "near_a",
        "ball_x",
        "diff_opp_half",
        "players",
    ]
    base = np.stack([f[key] for key in context_keys], axis=1)
    for before, after in CONTEXT_WINDOWS:
        means = nan_window_mean(base, before, after)
        for j, key in enumerate(context_keys):
            f["{}_c{}".format(key, before)] = means[:, j]
    directional_keys = ["ball_seen", "prox_seen", "all_speed", "diff_drift", "ball_out"]
    base = np.stack([f[key] for key in directional_keys], axis=1)
    for before, after in DIRECTIONAL_WINDOWS:
        means = nan_window_mean(base, before, after)
        tag = "p{}".format(before) if before else "f{}".format(after)
        for j, key in enumerate(directional_keys):
            f["{}_{}".format(key, tag)] = means[:, j]

    names = list(f)
    return np.stack([np.asarray(f[k], dtype=np.float32) for k in names], axis=1), names
