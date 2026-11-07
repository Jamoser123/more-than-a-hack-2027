"""Show our labels on the tactical video, so a human can see where the model makes sense.

Writes an ASS subtitle track from a submission: a coloured banner with the current label (H blue,
A red, D grey) and its segment number and time span, plus a per-frame line with the frame index
and, if available, the classifier's H / A / D probabilities. Subtitles instead of a re-encoded
video: the file is a few MB, it is ready in seconds, and any player can seek freely.

    python -m solution.overlay                         # -> submissions/zueri_giele.ass
    mpv data/.../BAR-ATH_tactical.mp4 --sub-file=submissions/zueri_giele.ass
    vlc data/.../BAR-ATH_tactical.mp4 --sub-file=submissions/zueri_giele.ass

Without such a player, --burn renders a downscaled clip with the overlay baked in via ffmpeg:

    python -m solution.overlay --burn clip.mp4 --start 1200 --duration 300   # seconds of video

Use it to understand failure modes and design general fixes. RULES.md forbids turning what you
see into labels, corrections, validation sets or rules tied to specific frames of this game.
"""

import argparse
import os
import subprocess

import numpy as np
import pandas as pd

from games import asset_paths
from solution.common import FPS, ROOT, run_bounds
from solution.make_submission import TEAM, probs_path

COLOURS = {"H": "&H00C86400", "A": "&H001E1EC8", "D": "&H00808080", "X": "&H00303030"}
NAMES = {"H": "HOME", "A": "AWAY", "D": "DEAD", "X": "--"}
HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, \
Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, \
Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Label,DejaVu Sans,64,&H00FFFFFF,&H00FFFFFF,&H00808080,&H80000000,1,0,0,0,100,100,0,0,\
3,12,0,7,40,40,40,1
Style: Frame,DejaVu Sans Mono,34,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,\
0,0,3,6,0,1,40,40,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def clock(frame):
    """ASS timestamp (h:mm:ss.cc) of the start of a video frame. 25 fps = 4 cs per frame."""
    centis = int(frame) * 100 // FPS
    return "{}:{:02d}:{:02d}.{:02d}".format(
        centis // 360000, centis // 6000 % 60, centis // 100 % 60, centis % 100
    )


def short_clock(frame):
    seconds = int(frame) // FPS
    return "{}:{:02d}:{:02d}".format(seconds // 3600, seconds // 60 % 60, seconds % 60)


def ass_events(letters, probs=None, first=0, last=None):
    """Dialogue lines for frames [first, last), timed relative to `first`."""
    last = len(letters) if last is None else last
    starts, stops, values = run_bounds(letters)
    lines = []
    for number, (a, b, v) in enumerate(zip(starts, stops, values)):
        lo, hi = max(a, first), min(b, last)
        if lo >= hi or v == "X":
            continue
        text = "{{\\3c{}}}{}  #{}  {} - {}  ({:.1f} s)".format(
            COLOURS[v], NAMES[v], number, short_clock(a), short_clock(b), (b - a) / FPS
        )
        lines.append(
            "Dialogue: 1,{},{},Label,,0,0,0,,{}".format(clock(lo - first), clock(hi - first), text)
        )
    for frame in range(first, last):
        text = "frame {}  {}".format(frame, short_clock(frame))
        if probs is not None:
            text += "   p  H {:.2f}  A {:.2f}  D {:.2f}".format(*probs[frame])
        lines.append(
            "Dialogue: 0,{},{},Frame,,0,0,0,,{}".format(
                clock(frame - first), clock(frame + 1 - first), text
            )
        )
    return lines


def write_ass(path, letters, probs=None, first=0, last=None):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(HEADER)
        handle.write("\n".join(ass_events(letters, probs, first, last)) + "\n")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default = os.path.join(ROOT, "submissions", TEAM + ".csv")
    parser.add_argument("--submission", default=default)
    parser.add_argument("--out", help="defaults to the submission path with .ass")
    parser.add_argument("--video", default=None, help="defaults to the tactical video in data/")
    parser.add_argument("--burn", help="also render an mp4 clip with the overlay baked in")
    parser.add_argument("--start", type=float, default=0.0, help="clip start, video seconds")
    parser.add_argument("--duration", type=float, default=300.0, help="clip length, seconds")
    args = parser.parse_args(argv)

    letters = pd.read_csv(args.submission, dtype={"label": str})["label"].str.strip().to_numpy()
    probs = (
        np.load(probs_path(args.submission))
        if os.path.isfile(probs_path(args.submission))
        else None
    )
    out = args.out or os.path.splitext(args.submission)[0] + ".ass"
    print("wrote {}".format(write_ass(out, letters, probs)))

    if args.burn:
        first = int(args.start * FPS)
        last = min(len(letters), first + int(args.duration * FPS))
        clip_ass = os.path.splitext(args.burn)[0] + ".ass"
        write_ass(clip_ass, letters, probs, first, last)
        video = args.video or asset_paths()["video"]
        command = [
            "ffmpeg",
            "-y",
            "-ss",
            "{:.2f}".format(first / FPS),
            "-i",
            video,
            "-t",
            "{:.2f}".format((last - first) / FPS),
            "-an",
            "-vf",
            "ass={},scale=1280:-2".format(clip_ass),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "26",
            args.burn,
        ]
        subprocess.check_call(command)
        print("wrote {}".format(args.burn))


if __name__ == "__main__":
    main()
