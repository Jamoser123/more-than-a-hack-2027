# Ideas: sophisticated approaches to frame-level possession

> Research notes for More than a Hack 2027 (FC Barcelona vs Athletic Club, `H` / `A` / `D` / `X` per frame).
> Check every idea against `RULES.md` and `EVALUATION.md` — the repository is authoritative.

## Core framing

We are ranked on `spellF1`, so this is **temporal segmentation**, not per-frame classification. The
published baselines show it: smoothing barely moves `macroF1` but moves `spellF1` fivefold.

Design principle — split the system in two:

1. **Emissions**: noisy per-frame evidence for `H` / `A` / `D` from as many sources as possible.
2. **Decoder**: a structured model that turns that evidence into a small number of plausible
   possessions with clean boundaries.

---

## 1. Decoder: HMM / semi-Markov / CRF instead of smoothing windows

- **HMM over {H, A, D}** with a transition matrix that makes switches expensive and forbids
  impossible ones (e.g. a goal must be followed by `D`, then kickoff by the conceding team).
  Decode with Viterbi.
- **Hidden semi-Markov model (HSMM)** with explicit duration distributions per state — directly
  suppresses 3-frame "possessions" that destroy `spellF1`. Estimate duration priors from open data
  of *other* matches (never from the competition ground truth).
- **CRF with forbidden transitions** — see PathCRF (Kim et al., KAIST, 2026): models possession as
  selecting one edge per timestep in a dynamic player graph, uses a CRF to block impossible
  transitions, decodes with Viterbi. Motivated by the observation that unconstrained ball-free
  methods produce unrealistically frequent possession changes — exactly what `spellF1` punishes.
  Code: https://github.com/hyunsungkim-ds/pathcrf
- **Adaptation for our data**: no player IDs and 46 slots, so work at team level — a
  possessor-team CRF (`H` / `A` / `D`) with a "ball in flight" sub-state inside each team so passes
  don't cause flicker.

## 2. Ball visible: rule-based possession zones

- Reference: Vidal-Codina et al. (2022), *Automatic event detection in football using tracking
  data*. Possession from ball–player distances, then football rules on top.
- Their ball-control states — **dead ball / no possession / possession / duel** — are worth copying.
  "No possession" (ball in flight) and "duel" should *not* trigger a team switch; only clean
  control by the other team should.
- Paper notes the method transfers to lower-quality single-camera data by enlarging the possession
  radius and retuning hyperparameters.
- Add **hysteresis**: delay the switch until control is established. Removes most
  deflection-induced false spells.
- TODO: read the exact possession definition in `EVALUATION.md` (how contested balls change hands).

## 3. Ball missing (41% of graded frames)

### 3a. Learned inference of ball state from player configurations
- **TranSPORTmer** (Capellera et al., ACCV 2024): set-attention transformer that classifies
  per-frame states (pass, possession, uncontrolled, out of play) from partial ball trajectories.
- **Multi-Modal Soccer Scene Analysis with Masked Pre-Training** (Peral, Capellera et al., WACV
  2026): tactical-camera input — player trajectories + player types + player image crops — for
  ball trajectory inference, ball state and possessor identification. Very close to our data.
- Key property: **permutation invariance** (Set Transformers / set-attention blocks) handles
  anonymous slots that appear, vanish and swap.

### 3b. Cheaper: hand-crafted team-shape features + GBM
The team in possession spreads out; the defending team compacts and moves toward the ball.
Candidate features per team:
- convex-hull area, width, depth
- centroid velocity along the attacking direction
- number of players converging on one point (pressure)
- back-line height

Feed a gradient-boosted classifier's probabilities into the HSMM as emissions.

### 3c. Physical gap-filling
Kalman filter/smoother with ground-roll vs airborne modes; constrain touches to occur near players.
Bridges short gaps and recovers passer/receiver team.

## 4. Use the tactical video directly

- **Own ball detector** on the 1920×1080 tactical feed (+ temporal tracking), projected through the
  provided homographies (valid on 100% of tactical frames). Goal: recover part of the missing 41%.
- **Ball-action spotting** for touch change-points:
  - SoccerNet Ball Action Spotting — locate the frame a pass leaves the foot / a touch controls the ball.
  - **T-DEED** (Xarles et al., SoccerNet 2024 winner): https://github.com/arturxe2/T-DEED
  - **SoccerNet 2025 Team Ball Action Spotting winner** (int8.io): single head predicting joint
    team+action classes — closest fit to possession. `pip install dude.k`
  - Caveat: trained on broadcast; expect a domain gap on the tactical wide shot.
- Simple win: team of the nearest (team-labelled) track at each detected touch = strong emission.
- "Out" / "Throw In" spotting classes feed dead-ball detection.

## 5. Dead-ball (`D`) model

`D` matters for both metrics (a class in `macroF1`, a spell-splitter in `spellF1`). Signals:
- ball outside the pitch in normalised coordinates, or last trajectory exiting it
- sharp drop in collective player speed
- restart configurations: corner clustering, free-kick wall, both teams in own halves (kickoff),
  single player at touchline (throw-in)
- Smart Tagging *Set Pieces* and *Goals* as approximate anchors

Bonus: restart type predicts **which team gets the next spell** — strong prior for the decoder.

## 6. Smart Tagging as weak supervision

Team-tagged phases imply possession:
- Build Up / Progression / Creating Chances / Attacking Transition → tagged team has the ball
- Defending in X Third → the *other* team has the ball

Use as soft emissions or labelling functions in a **Snorkel-style label model** that learns each
source's accuracy (proximity rules, shape classifier, video touches, tags) without ground truth.
Let the decoder place exact change-points; tag boundaries are likely imprecise.

## 7. Training and validation without hidden labels

- **IDSSE** (Bassek et al., *Scientific Data* 2025): 7 Bundesliga 2022/23 matches, TRACAB tracking
  of players + ball at 25 fps, synchronised events, CC-BY 4.0. Loads via kloppy:
  ```python
  from kloppy import sportec
  tracking = sportec.load_open_tracking_data(match_id="J03WPY")
  events = sportec.load_open_event_data(match_id="J03WPY")
  ```
  Match IDs: J03WPY, J03WMX, J03WN1, J03WOH, J03WOY, J03WPF, J03WQF
- **ELASTIC** (CIKM 2026, same KAIST group): benchmark re-annotating three IDSSE matches frame by
  frame for event–tracking sync → more precise timings for deriving possession labels.
- **Synthetic degradation** (key technique): corrupt clean TRACAB to look like our ATD —
  - drop the ball on ~41% of frames, in realistic bursts
  - strip player IDs
  - fragment tracks, inject identity swaps
  - add positional noise

  Train on corrupted input vs clean-derived labels; validate `spellF1` on held-out IDSSE matches
  using the repo's `evaluation/possession_metrics.py`.
- **Self-training** on the competition match (algorithmic, so should be allowed — confirm in
  `RULES.md`): train on IDSSE → predict on Barça–Athletic → keep high-confidence frames as
  pseudo-labels → retrain → repeat.
- The public 5 minutes are for pipeline sanity checks only — not training, not tuning.

## 8. Suggested build order

1. [ ] HSMM decoder (duration priors + restart logic) on top of existing baseline emissions — likely
       the largest single jump over `baseline_smooth50` (spellF1 0.2433)
2. [ ] Dead-ball classifier + restart → next-team rules
3. [ ] Team-shape GBM for ball-missing frames, trained on degraded IDSSE
4. [ ] Smart Tagging weak prior
5. [ ] Video ball detector + touch spotting
6. [ ] (Optional) set-transformer emission model (TranSPORTmer / PathCRF style), ensembled

## Open questions

- How are `X` frames scored in `EVALUATION.md`? If abstaining at uncertain boundaries is cheaper
  than being wrong, `X` on low-confidence transition frames could raise `spellF1`.
- Exact possession definition for contested balls / deflections.

## Data-handling reminder

The terms prohibit sharing competition data (video, tracking, CSVs, extracts) with any hosted /
cloud LLM. Methodology discussion is fine; keep work on the actual files local.

---

## References

- Vidal-Codina et al. (2022). Automatic event detection in football using tracking data. https://arxiv.org/abs/2202.00804
- Kim et al. (2026). PathCRF: Ball-Free Soccer Event Detection via Possession Path Inference from Player Trajectories. https://arxiv.org/abs/2602.12080
- Capellera et al. (2024). TranSPORTmer: A Holistic Approach to Trajectory Understanding in Multi-Agent Sports. ACCV. https://arxiv.org/abs/2410.17785
- Peral, Capellera et al. (2026). Multi-Modal Soccer Scene Analysis with Masked Pre-Training. WACV. https://arxiv.org/abs/2512.19528
- Xarles et al. T-DEED. https://github.com/arturxe2/T-DEED
- SoccerNet 2024 Challenges Results. https://arxiv.org/abs/2409.10587
- SoccerNet 2025 Challenges Results. https://arxiv.org/abs/2508.19182
- Bassek et al. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data* 12, 195. https://doi.org/10.1038/s41597-025-04505-y
- IDSSE mirror (kloppy). https://huggingface.co/datasets/pysport/idsse-data
