# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Our team's fork of the official More than a Hack 2027 Phase 1 challenge repo (upstream: `metrica-sports/more-than-a-hack-2027`). The task: for every frame of the tactical video (199,319 frames, 25 fps) output who has the ball as a `frame,label` CSV with labels `H` (home), `A` (away), `D` (dead ball), `X` (abstain/ungraded). Ranked on `spellF1` (whole possessions recovered, IoU 0.3–0.7); `macroF1` is the first tie-break. Baseline to beat on the full match: `spellF1` 0.2433 (`baseline_smooth50`). Deadline: November 17, 2026. Full spec: `CHALLENGE.md`, `DATA.md`, `EVALUATION.md`, `SUBMISSION.md`, `RULES.md`.

## Commands

```bash
uv venv && uv pip install -r requirements-starter.txt   # numpy+pandas, plus kloppy/matplotlib/notebook for starter/
python evaluation/run_evaluation.py                      # score every CSV in evaluation/results/ on the public 5 minutes
python evaluation/run_evaluation.py --bootstrap 1000     # confidence intervals; do this before believing any gap
python evaluation/run_evaluation.py --results some/dir   # score a different folder
python evaluation/validate_submission.py <file>.csv      # exactly the official checks; run on the exact file before submitting
python starter/baseline.py [--window 0 25 50 125]        # rebuild the baselines (needs data/)
ruff check . && ruff format --check .                    # lint + format check
pytest                                                   # all tests; single test: pytest path/to/test_x.py::test_name
```

Match data is not in git: unpack the download into `./data` or set `MTAH_DATA_DIR`. The evaluation itself needs no data, only CSVs.

## Architecture

- `games.py` (stdlib only) is the single source of per-match constants: frame counts, fps, pitch size, asset paths, and the **frame alignment**. Tracking (ATD) runs on its own `IdFrame` counter: `id_frame = video_frame + offsets[half]`, with a *different offset per half* (they differ by one frame). Submissions are indexed by **tactical video frame** only; the broadcast video is not aligned to it. Always go through `games.py` / `starter/load_atd.py` (`video_frame_of`, `frame_id_of`) rather than re-deriving offsets; a one-frame error shifts every scored transition.
- `starter/` are loaders and examples: `load_atd.py` (tracking → video frames), `load_smart_tagging.py` (tactical tags as video-frame intervals), `load_homography.py` (`project`/`unproject` pitch ↔ pixels), `baseline.py` (nearest player to ball + majority-vote smoothing, numpy only), `utils.py` (pitch plots), and three notebooks (`03_first_submission.ipynb` is the end-to-end path).
- `evaluation/` is the official scorer: `possession_metrics.py` defines every metric, `run_evaluation.py` ranks `evaluation/results/*.csv`, `validate_submission.py` mirrors its file checks. The CSV filename becomes the name on the board. `X` on a graded frame is a miss for `macroF1`; for `spellF1` it is filled with your last label in that half.

## Team coding standards

### Where code goes
- **Do not modify upstream files** (`evaluation/`, `games.py`, `starter/`, the top-level `*.md` docs). The scorer must stay byte-identical to the official one, and keeping upstream untouched lets us pull their fixes without conflicts. Import from them instead; if something there is wrong, open an issue upstream.
- Our own pipeline lives in `solution/`. Reuse the `starter/` loaders and `games.py` constants rather than copying them.
- Exploration notebooks are fine, but anything the final submission depends on must be a `.py` module or script, not a notebook.

### Python
- Python 3.11, managed with `uv`. Pin every dependency we add with an exact version (`pkg==x.y.z`) in our own requirements file; do not edit upstream `requirements*.txt`.
- `ruff` for linting and formatting, line length 100 (matching upstream). Code must pass `ruff check` and `ruff format --check` before it is pushed.
- Match the upstream style: a module docstring that explains *why* the module exists, `os.path`-based paths built from the repo root, `argparse` for every script with defaults that work from a fresh clone, frame indices as plain numpy arrays/ints.
- Vectorise with numpy/pandas; the pipeline runs over ~200k frames, so no per-frame Python loops in hot paths.
- Tests with `pytest` under `tests/`, for non-trivial logic (alignment, smoothing, segment/possession logic, anything that changes labels). Tests must not need `data/` unless marked so they can be skipped on a fresh clone.

### Reproducibility (a hard rule of the competition)
- Running the repo against the provided data must regenerate the **identical** submission file. Fix every random seed, avoid non-deterministic ops, and keep one documented command that produces the final CSV.
- Final submission file is named after the team (`<team_name>.csv`) and must pass `validate_submission.py` as the last step before upload.

### Competition rules that constrain code (see `RULES.md`)
- **Labels must come from an algorithm.** No hand labels on this game in any amount, not for training, validation, or correction. No constants or rules tied to specific frames of this game (e.g. "frames 18000–18200 are D").
- **Do not tune on the public 5-minute ground truth** (`evaluation/gt/`). It is for checking the pipeline is wired correctly (alignment, format, label meaning), not for choosing parameters; it ranks methods wrongly (see `EVALUATION.md`).
- No external human-produced data about this match (Opta, StatsBomb, event data, corrected tracking). Data about other games is fine.
- **Never commit match data or derived frame-level data** (`data/`, videos, tracking exports, clipped frames, our submission CSVs). The repo is public. `.gitignore` already covers `data/*`, `evaluation/results/*.csv` and `*.mp4`; put any new output folder in `.gitignore` before writing to it.
- Every external model or dataset we use must be publicly accessible, and gets a line in the technical document.

### Git
- Shared team branch is `zueri_giele`; `main` tracks upstream. Work on short-lived feature branches off `zueri_giele` and merge back via PR.
- Commit messages: short imperative summary line. No AI attribution lines in commits or PRs.
- Markdown: no hard wraps, one line per paragraph or list item.
