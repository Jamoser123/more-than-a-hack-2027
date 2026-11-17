# züri giele: technical document

More than a Hack 2027, Phase 1: per-frame ball possession (H / A / D) for FC Barcelona vs Athletic Club.

## 1. Approach

Possession is treated as temporal segmentation in two stages:

1. A **per-frame classifier** turns the automatic tracking into noisy probabilities for H, A and D.
2. A **duration-aware Viterbi decoder** (an explicit-duration hidden semi-Markov model) turns those probabilities into a small number of possessions with clean boundaries.

`spellF1` scores whole possessions, so a three-frame flicker costs a false possession and splits a true one. A smoothing window removes flicker but blurs every boundary; the decoder instead finds the single most likely label sequence under minimum durations and a cost per change, so boundaries stay where the evidence changes.

Everything learned or chosen is fitted on open data from **other matches** (IDSSE, Bundesliga). The competition match is only used as input at prediction time.

The submitted classifier is a small **Mamba-3** sequence model; a gradient-boosted tree classifier (GBM) with the same inputs is kept as an alternative (`--classifier gbm`). Mamba-3 was chosen because it scored better in leave-one-match-out validation on the open data (section 5).

## 2. Pipeline

1. **Open training data** (`solution/idsse.py`). The seven IDSSE matches (Bassek et al., 2025, CC-BY 4.0), from the public Hugging Face mirror `pysport/idsse-data`, read with kloppy. TRACAB tracking carries a ball-alive flag and a ball-owning team on every frame. Label: D where the ball is dead, otherwise the owning team.
2. **Degradation** (`solution/degrade.py`). Clean TRACAB is corrupted to resemble the automatic single-camera tracking of the competition: the ball missing in bursts (more often in crowds and off the pitch), player slots dropping out, smooth positional jitter, and short spells with a player on the wrong team. Every rate is drawn from a range per copy. The ranges come from the facts published in `DATA.md` (ball visible about 57% of frames, about 20.5 of 22 players tracked).
3. **Features** (`solution/features.py`). 156 per-frame features, all invariant to the order of player slots (the tracking has no identities): ball (seen, interpolated across short gaps, time since / until seen, position towards the home team's attacking goal, speed, distance off the pitch); proximity (nearest and second-nearest player of each team to the ball, players within 3 m, last / next proximity verdict across ball gaps); team shape in each team's attacking direction (centroid, spread, front and back line, speed, forward drift, players in the opponent half), which carries the frames without a ball; and centred, past-only and future-only means of these over 0.5 to 10 s. The attacking direction per half is read off positions.
4. **Classifier: Mamba-3** (`solution/mamba3.py`, `solution/sequence_model.py`). Our own PyTorch implementation of the SISO Mamba-3 layer from the paper (Lahoti et al., 2026): exponential-trapezoidal recurrence with data-dependent step, decay and trapezoid weight; complex-valued state through data-dependent rotations of B and C; RMS-normalised B and C with learnable biases; no short convolution. It is computed in the parallel form, and a unit test checks it against the paper's step-by-step complex recurrence (agreement to 1e-14). Each block runs forwards and backwards (labelling is offline, the future is informative), followed by a SwiGLU MLP. The model reads the features at 5 fps in windows of 256 steps (51 s); at prediction all five frame phases are run and overlapping windows are blended, so every frame gets its own output. Inputs are standardised on the training data, missing values set to 0 with a missing flag.
5. **Symmetry.** Every training copy is also used with the teams swapped (labels swapped too), and at prediction the model runs on the match and on its team-swapped view and the two outputs are averaged. The output therefore treats the two teams exactly alike.
6. **Decoder** (`solution/decode.py`). Exact Viterbi over an explicit-duration HMM: each label is a chain of sub-states, so no run is shorter than its minimum; each transition type has a fixed log-cost (H to A, live to dead, dead to live); emissions are the classifier's log-probabilities. Decoded per half; frames outside the halves are written as `X`.
7. **Parameter choice.** The decoder settings are chosen by grid search, leave-one-match-out, on out-of-fold IDSSE predictions (`solution/compare_models.py`): minimum possession 50 frames, minimum dead ball 12 frames, H/A switch cost 1, live/dead cost 20 (`solution/decoder_params_mamba.json`). The Mamba-3 hyperparameters were fixed before any run, without a search: width 64, 2 bidirectional blocks, 4 heads of 32, state size 16 (178k parameters), 500 training steps of 32 windows, AdamW at 2e-3 with cosine decay, dropout 0.1.

The alternative GBM (`solution/model.py`, scikit-learn `HistGradientBoostingClassifier`) uses the same features, copies and symmetry, with its own decoder settings from `solution/validate.py` (`solution/decoder_params.json`).

## 3. Assets: used and dropped

| asset | use |
|---|---|
| Tactical tracking (ATD) | the only input of the submitted pipeline |
| Smart Tagging | **tried, dropped.** We turned tags with a clear meaning (possession phases, set pieces with a team, goals) into extra decoder evidence, weighted by how often each tag type agrees with the classifier. The effect on the full match was small (0.4 to 0.6% of frames), and there is no labelled open data with Smart Tagging to validate it, so it is not used. |
| Homography, tactical and broadcast video, broadcast tracking | not used |
| Public 5-minute ground truth | used only to check that the pipeline is wired correctly (frame alignment, label orientation, file format). Nothing in the submitted pipeline (model, features, decoder settings, choice of classifier) was chosen with it. |

## 4. Training data

| dataset | source | labelled by | amount |
|---|---|---|---|
| IDSSE (DFL Bundesliga, 7 matches) | Bassek et al. 2025, CC-BY 4.0, https://huggingface.co/datasets/pysport/idsse-data | DFL / TRACAB: ball-alive flag and ball-owning team, delivered with the tracking | about 1 million in-play frames at 25 fps |

No data about the provided game is used for training or validation, and no frame of it is labelled by hand.

## 5. Validation (open data only)

Leave-one-match-out over the seven IDSSE matches, scored with the official `evaluation/possession_metrics.py`. For each held-out match the classifier is trained on the other six, the held-out match is degraded with a seed no training copy uses, and the decoder settings are the best on the other six matches. The baseline is the reference method (nearest player to the seen ball, majority vote over ±50 frames) on the same degraded input.

| held-out match | baseline spellF1 | GBM, per frame | Mamba-3, per frame | GBM + decoder | **Mamba-3 + decoder** |
|---|---|---|---|---|---|
| J03WMX | 0.196 | 0.279 | 0.360 | 0.488 | 0.522 |
| J03WN1 | 0.202 | 0.261 | 0.291 | 0.512 | 0.530 |
| J03WOH | 0.230 | 0.363 | 0.425 | 0.625 | 0.655 |
| J03WOY | 0.174 | 0.299 | 0.364 | 0.571 | 0.619 |
| J03WPY | 0.197 | 0.301 | 0.396 | 0.588 | 0.598 |
| J03WQQ | 0.216 | 0.287 | 0.315 | 0.615 | 0.586 |
| J03WR9 | 0.173 | 0.258 | 0.352 | 0.569 | 0.592 |
| **mean spellF1** | **0.198** | **0.292** | **0.358** | **0.567** | **0.586** |

Mean macroF1 after decoding: Mamba-3 0.903, GBM 0.900, baseline 0.486.

- The decoder does most of the work: the same probabilities read frame by frame score about 0.3.
- Mamba-3 is better than the GBM frame by frame on every match (+0.065); after decoding the gain is +0.019 (6 of 7 matches), smaller than the roughly 0.04 spread between matches. Its decoder settings are lighter than the GBM's (switch cost 1 against 5), because its probabilities are already smooth.
- On degraded IDSSE the baseline scores 0.198, close to its published 0.243 on the competition match, which suggests the degradation is realistic.

## 6. Possession rules

We do not hand-code possession rules; the line is the one in the training labels, plus what the decoder enforces.

- **Dead ball (D):** the IDSSE ball-alive flag is off (out of play, stoppages, until the restart).
- **Team possession:** the IDSSE ball-owning team on every live frame. Loose balls and challenges therefore follow the DFL convention as delivered; we did not alter it.
- **Short contacts:** a possession lasts at least 2 s (50 frames) and a dead ball at least 0.48 s (12 frames). Deflections, blocks and touches shorter than that are absorbed into the surrounding possession unless the evidence for a change is strong.
- **Restarts:** no explicit restart rules. A restart appears as a dead-ball stretch followed by the team the classifier sees in control.

## 7. Limitations

- **Definition gap.** IDSSE possession is DFL's ball-possession flag; the challenge ground truth comes from Metrica's corrected tracking. If Metrica counts coarser possessions, longer minimum durations would fit better; nothing about the competition's definition was tuned.
- **Too few possessions on open data.** The decoder predicts about 270 to 320 possessions per IDSSE match against 370 to 510 true ones: quick exchanges are merged.
- **Dead ball.** TRACAB tracks the ball through stoppages, the competition tracking probably loses it more often. The degradation adds dropouts off the pitch, but D is the class most likely to transfer badly.
- **Simulated input.** The degradation matches two published numbers (ball visibility, players tracked); failure modes of the real tracking it does not produce are not covered by the validation.
- **With more time:** validate an average of Mamba-3 and the GBM, give Mamba-3 the per-frame features only and let it learn the time context itself, train longer (the loss was still falling at 500 steps), and use Smart Tagging once labelled open data with similar tags is available.

## 8. Reproduction

Python 3.11 (we used 3.11.7) and `uv`. The match data goes into `./data` (or `MTAH_DATA_DIR`), as in the challenge README.

```bash
uv venv --python 3.11
uv pip install -r solution/requirements-lock.txt    # exact versions of every package we used
uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu   # CPU build of the pinned torch
python -m solution.make_submission        # -> "submissions/züri giele.csv", then the official validator
```

The first run downloads and converts the seven IDSSE matches (about 2.6 GB, cached in `cache/`), trains the classifier on CPU (about 15 minutes on a 12-thread laptop, cached), and writes the CSV. All random draws are seeded, PyTorch runs in deterministic mode with a fixed thread count (10). Retraining the classifier from scratch in a fresh environment (reusing the converted IDSSE data and features in `cache/`) gave bit-identical model weights and a byte-identical CSV (SHA-256 `48ad98c8b4f3881258da1c4d97b12c9e79952ac3b70c37208a6dde75ceb9504f`).

Optional:

```bash
python -m solution.make_submission --classifier gbm --out "submissions/züri giele gbm.csv"
python -m solution.validate               # GBM, leave-one-match-out on IDSSE
python -m solution.compare_models         # GBM vs Mamba-3, leave-one-match-out on IDSSE (hours on CPU)
pytest tests
```

## 9. External resources

- IDSSE: Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data* 12, 195. https://doi.org/10.1038/s41597-025-04505-y. Mirror: https://huggingface.co/datasets/pysport/idsse-data
- Mamba-3: Lahoti, A., Li, K. Y., Chen, B., Wang, C., Bick, A., Kolter, J. Z., Dao, T., Gu, A. (2026). Mamba-3: Improved Sequence Modeling using State Space Principles. arXiv:2603.15569. Our own implementation from the paper; no code or weights taken.
- Software: PyTorch 2.5.1 (CPU), scikit-learn 1.5.2, numba 0.60.0, kloppy 3.18.0, numpy 1.26.4, pandas 2.2.3.
