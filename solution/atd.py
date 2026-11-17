"""The competition match's tracking (ATD) as a `Match`, indexed by tactical video frame.

The raw file is one line per frame, "<frame_id>:<slot>;<slot>;...:<ball>", each slot "x,y,speed"
and the ball "x,y,..." in pitch-normalised 0-1 coordinates, NaN where unseen (the format the
reference baseline in starter/baseline.py reads). Instead of splitting 199k lines in Python we
turn both separators into commas and hand the whole file to pandas' C parser.

Alignment goes through the upstream helpers rather than being re-derived here: rows are placed
with starter.load_atd.video_frame_of, slot teams come from starter.baseline.team_of_each_track,
and the graded halves (period 1 / 2; everything else 0) from games.half_video_blocks.
"""

import io

import numpy as np
import pandas as pd

from games import GAME, GAMES, half_video_blocks, require, source_frames
from solution.common import Match
from starter.baseline import AWAY, HOME, team_of_each_track
from starter.load_atd import video_frame_of

SLOT_FIELDS = 3  # x, y, speed


def read_raw(raw_path, n_slots):
    """(frame_ids [N], slots [N, n_slots, 2], ball [N, 2]) in normalised coordinates."""
    with open(raw_path) as handle:
        text = handle.read()
    text = text.translate(str.maketrans(":;", ",,"))
    n_fields = max(line.count(",") for line in text.splitlines() if line) + 1
    n_ball = n_fields - 1 - SLOT_FIELDS * n_slots
    if n_ball < 2:
        raise ValueError(
            "{}: {} fields per line cannot hold {} slots and a ball".format(
                raw_path, n_fields, n_slots
            )
        )
    table = pd.read_csv(
        io.StringIO(text),
        header=None,
        names=range(n_fields),
        dtype=np.float64,
        na_values=["NaN", "nan"],
        engine="c",
    ).to_numpy()
    frame_ids = table[:, 0].astype(np.int64)
    slots = table[:, 1 : 1 + SLOT_FIELDS * n_slots].reshape(len(table), n_slots, SLOT_FIELDS)
    ball = table[:, 1 + SLOT_FIELDS * n_slots : 3 + SLOT_FIELDS * n_slots]
    return frame_ids, slots[..., :2], ball


def load(game=GAME, data_dir=None, source="tactical"):
    """The ATD of one recording as a Match in metres, one row per video frame, no labels."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, source)
    team = team_of_each_track(meta_path)
    frame_ids, slots, ball = read_raw(raw_path, len(team))

    n_frames = source_frames(source, game)
    rows = frame_ids + video_frame_of(0)  # video_frame_of is a fixed shift; vectorise it
    inside = (rows >= 0) & (rows < n_frames)
    length, width = GAMES[game]["pitch"]
    scale = np.array([length, width])
    players = np.full((n_frames, len(team), 2), np.nan, dtype=np.float32)
    players[rows[inside]] = slots[inside] * scale
    ball_m = np.full((n_frames, 2), np.nan, dtype=np.float32)
    ball_m[rows[inside]] = ball[inside] * scale

    period = np.zeros(n_frames, dtype=np.int8)
    if source == "tactical":
        for half, first, last in half_video_blocks(game):
            period[first : last + 1] = half
    return Match(
        home=players[:, team == HOME],
        away=players[:, team == AWAY],
        ball=ball_m,
        period=period,
        pitch=(length, width),
    )
