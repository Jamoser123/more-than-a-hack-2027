# Solution explanation

Possession is treated as temporal segmentation: a per-frame classifier produces noisy H / A / D evidence, and a duration-aware Viterbi decoder turns it into a small number of possessions with clean boundaries. Everything learned or chosen is fitted on open data from other matches (IDSSE). The competition match is only used as input at prediction time.

This implements items 1 and 3 of the build order in `docs/possession-approaches.md` (HSMM decoder and team-shape GBM trained on degraded IDSSE), with the dead-ball signals of item 2 folded into the classifier instead of a separate model.

## Reproduce

```bash
uv venv --python 3.11 && uv pip install -r solution/requirements.txt
python -m solution.make_submission          # -> submissions/zueri_giele.csv, then runs the official validator
python -m solution.validate                 # optional: leave-one-match-out validation on IDSSE
pytest tests
```

Optional variants (none of them is the default submission):

```bash
uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu    # only for the Mamba-3 variant
python -m solution.make_submission --classifier mamba --out submissions/zueri_giele_mamba.csv
python -m solution.make_submission --tags [--tag-phases] --out submissions/zueri_giele_tags.csv
python -m solution.compare_models [--skip-gbm]   # GBM vs Mamba-3, leave-one-match-out (~2 h CPU with --skip-gbm)
```

`make_submission` downloads and converts the seven IDSSE matches on the first run (about 2.6 GB, cached in `cache/`), trains the classifier (cached as `cache/final_model.pkl`), reads the match ATD from `data/` (or `MTAH_DATA_DIR`), and writes the CSV. Every random draw is seeded, so a rerun produces the same file. We checked this on a synthetic bundle (an IDSSE match written in the ATD format): retraining from scratch and rerunning gave a byte-identical CSV.

## Pipeline

1. **Open training data** (`solution/idsse.py`). The seven IDSSE Bundesliga matches (Bassek et al., 2025, CC-BY 4.0), downloaded from the public Hugging Face mirror `pysport/idsse-data` and read with kloppy. TRACAB tracking provides a ball-alive flag and a ball-owning team on every frame, which is the same kind of signal the challenge ground truth is derived from. Labels: D where the ball is dead, otherwise the owning team.
2. **Degradation** (`solution/degrade.py`). Clean TRACAB is corrupted to look like the automatic single-camera ATD: the ball missing in bursts (extra dropouts in crowds and off the pitch), player-slot dropouts, temporally smooth positional jitter, and short spells where a player is put on the wrong team. Each rate is drawn from a range for every copy (domain randomisation). The ranges come from the published facts in `DATA.md` (ball visible ~57%, ~20.5 of 22 players tracked) and give 45–65% ball visibility and ~20.5 visible players. No label was used to set them.
3. **Features** (`solution/features.py`). 156 per-frame features, all invariant to slot permutations because the ATD has no identities:
   - ball: seen / interpolated, frames since and until last seen, position in the home attacking direction, speed, distance off the pitch.
   - proximity: nearest and second-nearest distance per team to the (interpolated) ball, players within 3 m, and the last / next proximity verdict carried across ball gaps.
   - team shape, in each team's own attacking direction: centroid, spread, back and front line, mean / max speed, sprinters, forward drift, players in the opponent half. These carry the frames without a ball.
   - context: centred means over ±0.5 s, 1.5 s, 4 s and 10 s, plus past-only and future-only means.

   Attacking direction per half is read off positions: the team whose player is most often deepest on the −x end defends it.
4. **Classifier** (`solution/model.py`). scikit-learn `HistGradientBoostingClassifier` (handles NaN natively), trained on every 3rd frame of two degraded copies of each match. Each copy is also mirrored (teams and labels swapped). At prediction time the model runs on the match and on its mirror, and the two outputs are averaged, so the output is exactly symmetric between the teams.
5. **Decoder** (`solution/decode.py`). An exact Viterbi over an explicit-duration HMM. Each label is a chain of `min_len` sub-states, so no run can be shorter than its minimum. Each transition type has a fixed log-cost (H↔A, alive→D, D→alive), and emissions are the classifier's log-probabilities. It is compiled with numba and decoded per half. Frames outside the graded halves are written as `X` (never scored).
6. **Parameter choice** (`solution/validate.py`). Decoder settings are picked by grid search on out-of-fold IDSSE predictions. The result is stored in `solution/decoder_params.json` (`min_alive` 50 frames, `min_dead` 12, H↔A cost 5, alive↔dead cost 40, no D bias).

## Inspecting the output

`python -m solution.overlay` turns a submission into an ASS subtitle track. It shows a coloured banner with the label, segment number and time span, and a per-frame line with the frame index and the classifier's probabilities. On the frames the public ground truth grades (the 5-minute window, about 7:27 to 12:27 in the tactical video) a third line shows the true label and `ok` / `WRONG`; `--gt none` hides it. It is there to catch wiring errors (alignment, team direction), not to choose parameters. Each frame is one subtitle event, because VLC shows only one of several overlapping events. Play it over the tactical video in mpv or VLC (`--sub-file=submissions/zueri_giele.ass`). Alternatively, `--burn clip.mp4 --start <s> --duration <s>` renders a clip with ffmpeg. This is a debugging aid for finding failure modes. In line with `RULES.md`, nothing seen this way becomes a label, a validation set or a frame-specific rule.

## Validation (open data only)

Leave-one-match-out over the seven IDSSE matches, scored with the official `evaluation/possession_metrics.py`. For each held-out match:

- the classifier is trained on the other six;
- the held-out match is degraded with a seed no training copy uses;
- the decoder settings are the best on the other six matches.

The baseline is the reference method (nearest player to the seen ball, majority vote ±50 frames) run on the same degraded input.

| held-out match | baseline_smooth50 spellF1 | GBM argmax spellF1 | GBM + HSMM spellF1 | GBM + HSMM macroF1 |
|---|---|---|---|---|
| J03WMX | 0.196 | 0.279 | 0.488 | 0.901 |
| J03WN1 | 0.202 | 0.261 | 0.512 | 0.872 |
| J03WOH | 0.230 | 0.363 | 0.625 | 0.918 |
| J03WOY | 0.174 | 0.299 | 0.571 | 0.889 |
| J03WPY | 0.197 | 0.301 | 0.588 | 0.915 |
| J03WQQ | 0.216 | 0.287 | 0.615 | 0.906 |
| J03WR9 | 0.173 | 0.258 | 0.569 | 0.895 |
| **mean** | **0.198** | **0.292** | **0.567** | **0.900** |

The decoder beats the baseline on every held-out match by far more than the 0.04 noise band. On degraded IDSSE the baseline scores 0.198 spellF1, close to its published 0.243 on the competition match, which suggests the degradation is in a realistic range. The decoder does most of the work: the same probabilities read frame by frame score 0.29.

## Alternative classifier: Mamba-3 (not the default)

`solution/mamba3.py` is a small Mamba-3 (Lahoti et al., 2026) in plain PyTorch, so it runs on a CPU without the official CUDA kernels. It follows the paper's SISO layer: exponential-trapezoidal recurrence with data-dependent step, decay and trapezoid weight; complex state through data-dependent rotations of B and C (the "RoPE trick"); RMS-normalised B and C with learnable biases; no short convolution. It is computed in the parallel form `Y = (L o C B^T) X`, and a test checks it against the step-by-step complex recurrence of the paper (agreement to 1e-14). Two deliberate departures: every block runs forwards and backwards (labelling is offline), and there is no MIMO (a GPU decoding-speed device).

It is a drop-in replacement for the GBM (`solution/sequence_model.py`): the same 156 features (standardised, NaN set to 0 plus a missing flag), the same degraded and mirrored copies, the same team-swap averaging at prediction. It reads the match at 5 fps in windows of 256 steps (51 s); at prediction all five frame phases are run and overlapping windows are blended. Settings were fixed before any run, without a search: width 64, 2 bidirectional blocks, 4 heads of 32, state 16 (178k parameters, about the GBM's size), 500 steps of batch 32, AdamW 2e-3 with cosine decay, dropout 0.1. Everything is seeded and runs in deterministic mode.

`python -m solution.compare_models` runs it through exactly the GBM's protocol: same folds, same degraded inputs and seeds, decoder settings chosen per fold on the other six matches, official scorer. The GBM column below is the table above (fold J03WMX was re-run and matches it exactly).

| held-out match | GBM argmax | Mamba-3 argmax | GBM + HSMM | Mamba-3 + HSMM |
|---|---|---|---|---|
| J03WMX | 0.279 | 0.360 | 0.488 | 0.522 |
| J03WN1 | 0.261 | 0.291 | 0.512 | 0.530 |
| J03WOH | 0.363 | 0.425 | 0.625 | 0.655 |
| J03WOY | 0.299 | 0.364 | 0.571 | 0.619 |
| J03WPY | 0.301 | 0.396 | 0.588 | 0.598 |
| J03WQQ | 0.287 | 0.315 | 0.615 | 0.586 |
| J03WR9 | 0.258 | 0.352 | 0.569 | 0.592 |
| **mean spellF1** | **0.292** | **0.358** | **0.567** | **0.586** |

Frame by frame Mamba-3 is better on every match (+0.065): sequence modelling adds information beyond the hand-made window averages. After the decoder most of that is absorbed, leaving +0.019 (6 of 7 matches, below the 0.04 noise band); macroF1 is the same (0.903 vs 0.900). Its decoder settings, chosen the same way, are much lighter (switch cost 1 instead of 5, live/dead cost 20 instead of 40; `solution/decoder_params_mamba.json`), because its probabilities are already smooth. Training stopped at 500 steps for CPU time while the loss was still falling.

On the competition match the two submissions agree on 92.7% of graded frames. Most disagreements are about dead ball: Mamba-3 labels 45.8% of graded frames D against 48.8% for the GBM; only about 2% switch between H and A. Possession counts are almost equal (282 vs 283; median length 8.0 s vs 7.0 s). As a wiring check on the public 5 minutes both score alike (spellF1 0.729 vs 0.739, 17 and 16 possessions against 17 true, bootstrap ranges overlapping almost entirely); that window cannot rank them and was not used to choose anything. The GBM stays the default: the gain is within noise and the GBM is simpler and faster to retrain. An average of the two was not validated (it needs the GBM folds that `--skip-gbm` leaves out).

## Smart Tagging as decoder evidence (experimental, off by default)

`solution/tags.py` turns Metrica's Smart Tagging into extra log-scores for the decoder, by fixed rules about what each code means: possession phases (BUILD UP, PROGRESSION, ATTACKING TRANSITION, CREATING CHANCES) and set pieces with a team penalise the other team, GOALS penalise both teams (dead ball); DEFENDING codes are ignored because they mirror a possession phase exactly, as are location codes and set pieces tagged without a team (all throw-ins and kickoffs). Team evidence never argues against a dead ball. Each rule's weight is the log-odds of how often it agrees with the classifier on this match, capped at 0.9 because the tags come from the same tracking and fail together with our features. No labels are used.

A phase is one judgement about a stretch that can hold several possessions, so its weight is spread over its frames rather than applied to each. A first version that applied it per frame merged real short possessions (a Barcelona phase erased a 3 s Athletic possession): about 1.9% of frames changed and possessions dropped from 283 to 257. With the spread, `--tags --tag-phases` changes 0.39% of frames (mostly longer dead-ball stretches around set pieces and goals) and none in the public window, so there is no permitted way to tell whether it helps. It stays off.

## Caveats

- **Definition gap.** IDSSE possession comes from DFL's own ball-possession flag, while the challenge ground truth comes from Metrica's corrected tracking. If Metrica's "controlled possession" is coarser (the public 5 minutes average 14.2 s per possession against ~8 s in IDSSE), longer minimum durations would fit better. We have not tuned towards that: the public window is too small to rank settings and is not a tuning set.
- **Dead-ball gap.** In TRACAB the ball is tracked during stoppages; in the ATD it is probably lost more often. The degradation adds extra dropouts off the pitch, but the D class is the most likely to transfer badly.
- **Not used in the default submission:** Smart Tagging (implemented as an option, see above), restart-type rules, the video, and the broadcast feed.

## Compliance notes

- No hand labels, no frame-specific rules, no tuning on `evaluation/gt/`. All parameters come from IDSSE.
- No external data about this match. IDSSE covers other games and is public (CC-BY 4.0).
- The pipeline was developed with a hosted LLM coding assistant, which never had access to the competition data. All development and validation ran on IDSSE, and the competition ATD is processed only on the team's machine.

## External resources

- IDSSE dataset: Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data* 12, 195. https://doi.org/10.1038/s41597-025-04505-y. Mirror: https://huggingface.co/datasets/pysport/idsse-data
- kloppy 3.18.0 (reading IDSSE), scikit-learn 1.5.2, numba 0.60.0, numpy, pandas.
- Mamba-3 variant only: Lahoti, A., Li, K. Y., Chen, B., Wang, C., Bick, A., Kolter, J. Z., Dao, T., Gu, A. (2026). Mamba-3: Improved Sequence Modeling using State Space Principles. arXiv:2603.15569. Our own implementation from the paper (no code or weights taken); PyTorch 2.5.1 (CPU build).
