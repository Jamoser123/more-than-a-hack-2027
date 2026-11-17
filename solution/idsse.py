"""Open training and validation data: the seven IDSSE matches, converted to `Match` objects.

We may not tune on the competition's public ground truth, and the full-match one is private, so
every learned or chosen parameter in this pipeline comes from other games. IDSSE (Bassek et al.,
Scientific Data 2025, CC-BY 4.0) is seven 2022/23 Bundesliga matches with TRACAB tracking at
25 fps that carries a ball-alive flag and a ball-owning team on every frame. That is the same kind
of signal the challenge ground truth is derived from (corrected tracking marking the ball dead or
naming the owning team), so the labels can be taken as they are.

The files come from the public Hugging Face mirror (pysport/idsse-data) because the figshare
original sits behind a bot check. Conversion through kloppy takes about two minutes per match and
is cached as a compressed .npz under cache/idsse/.

    python -m solution.idsse            # download and convert all seven matches
"""

import argparse
import os
import urllib.request

import numpy as np

from solution.common import CACHE_DIR, D, A, H, Match

MATCH_IDS = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
COMPETITION = {"J03WMX": "000001", "J03WN1": "000001"}  # the rest are "000002" (2. Bundesliga)
MIRROR = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/"
IDSSE_DIR = os.path.join(CACHE_DIR, "idsse")


def file_names(match_id):
    """(meta, positions) file names as published."""
    stem = "DFL-COM-{}_DFL-MAT-{}.xml".format(COMPETITION.get(match_id, "000002"), match_id)
    return ("DFL_02_01_matchinformation_" + stem, "DFL_04_03_positions_raw_observed_" + stem)


def download(match_id, folder=IDSSE_DIR):
    """Local paths of the two raw files, fetched from the mirror if not already there."""
    os.makedirs(folder, exist_ok=True)
    paths = []
    for name in file_names(match_id):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            print("downloading {} ...".format(name))
            urllib.request.urlretrieve(MIRROR + name, path + ".part")
            os.replace(path + ".part", path)
        paths.append(path)
    return paths


def convert(match_id, folder=IDSSE_DIR):
    """Read one match with kloppy and store it as arrays. Returns the .npz path."""
    from kloppy import sportec

    out = os.path.join(folder, "{}.npz".format(match_id))
    if os.path.isfile(out):
        return out
    meta, raw = download(match_id, folder)
    print("converting {} ...".format(match_id))
    dataset = sportec.load_tracking(raw_data=raw, meta_data=meta, only_alive=False)
    length = dataset.metadata.pitch_dimensions.pitch_length
    width = dataset.metadata.pitch_dimensions.pitch_width
    frame = dataset.to_df()

    home_team, away_team = dataset.metadata.teams  # kloppy lists the home side first
    teams = []
    for team in (home_team, away_team):
        ids = [p.player_id for p in team.players if "{}_x".format(p.player_id) in frame.columns]
        xs = frame[["{}_x".format(i) for i in ids]].to_numpy(np.float32) * length
        ys = frame[["{}_y".format(i) for i in ids]].to_numpy(np.float32) * width
        teams.append(np.stack([xs, ys], axis=-1))

    ball = np.stack(
        [
            frame["ball_x"].to_numpy(np.float32) * length,
            frame["ball_y"].to_numpy(np.float32) * width,
        ],
        axis=-1,
    )
    dead = (frame["ball_state"].astype(str) == "dead").to_numpy()
    owner_home = (frame["ball_owning_team_id"].astype(str) == home_team.team_id).to_numpy()
    label = np.where(dead, D, np.where(owner_home, H, A)).astype(np.int8)
    period = frame["period_id"].to_numpy(np.int8)

    np.savez_compressed(
        out,
        home=teams[0],
        away=teams[1],
        ball=ball,
        period=period,
        label=label,
        pitch=np.array([length, width], dtype=np.float64),
    )
    return out


def load(match_id, folder=IDSSE_DIR):
    """One IDSSE match as a clean (undegraded) Match with labels."""
    with np.load(convert(match_id, folder)) as data:
        return Match(
            home=data["home"],
            away=data["away"],
            ball=data["ball"],
            period=data["period"],
            pitch=tuple(data["pitch"].tolist()),
            label=data["label"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matches", nargs="+", default=list(MATCH_IDS))
    args = parser.parse_args()
    for match_id in args.matches:
        print(convert(match_id))
