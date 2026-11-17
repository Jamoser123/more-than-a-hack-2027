"""Leave-one-match-out validation on open data, scored with the official scorer.

This is how we know whether a change helps without looking at the competition match. For each of
the seven IDSSE matches: train the classifier on the other six, degrade the held-out match with a
seed no training copy used, predict it, and decode it. Decoder parameters are also chosen
leave-one-match-out: the setting scored on a held-out match is the one that did best on the other
six matches' out-of-fold predictions. Every score is evaluation/possession_metrics.evaluate on
that held-out match, next to the reference baseline (nearest player + majority vote, +/- 50
frames) run on the same degraded input.

    python -m solution.validate                 # all seven folds, cached under cache/oof/
    python -m solution.validate --refresh       # retrain the folds
"""

import argparse
import itertools
import json
import os
import time

import numpy as np

from solution import idsse
from solution.common import CACHE_DIR, A, D, H, X, gt_with_half_breaks
import possession_metrics as pm  # noqa: E402  (on the path via solution.common)
from solution.decode import DecoderParams, decode
from solution.degrade import degrade
from solution.model import build_training_set, fit, predict_probs
from starter.baseline import AWAY, HOME, NOBODY, smooth

OOF_DIR = os.path.join(CACHE_DIR, "oof")
VALIDATION_SEED = 10_000
# Candidate decoder settings. Coarse on purpose: seven matches can separate big differences only.
GRID = dict(
    min_alive=(25, 50, 75, 100),
    min_dead=(12, 25, 50),
    cost_switch=(1.0, 2.5, 5.0, 10.0),
    cost_dead=(20.0, 40.0, 80.0),
    bias_dead=(-0.5, 0.0, 0.5),
)


def grid():
    keys = list(GRID)
    return [
        DecoderParams(**dict(zip(keys, values)), cost_live=values[keys.index("cost_dead")])
        for values in itertools.product(*(GRID[k] for k in keys))
    ]


def letters_to_codes(letters):
    return np.select([letters == "H", letters == "A"], [H, A], D).astype(np.int8)


def baseline_codes(match, window=50):
    """The reference baseline on a Match: nearest player to the seen ball, majority vote."""
    with np.errstate(invalid="ignore"):
        dh = np.nanmin(np.linalg.norm(match.home - match.ball[:, None], axis=-1), axis=1)
        da = np.nanmin(np.linalg.norm(match.away - match.ball[:, None], axis=-1), axis=1)
    raw = np.full(match.n_frames, NOBODY, dtype=np.int8)
    seen = ~np.isnan(match.ball[:, 0])
    raw[seen & (dh <= da)] = HOME
    raw[seen & (da < dh)] = AWAY
    raw[seen & np.isnan(dh) & ~np.isnan(da)] = AWAY
    codes = letters_to_codes(smooth(raw, window))
    return np.where(match.period > 0, codes, X).astype(np.int8)


def score(gt, pred):
    result = pm.evaluate(gt, pred)
    return {
        "spellF1": result["spell_f1_mean"],
        "macroF1": result["macro_f1"],
        "n_pred_spells": result["n_pred_spells"],
        "n_gt_spells": result["n_gt_spells"],
    }


def out_of_fold(refresh=False, copies=2, stride=5):
    """{match_id: (probs, gt, baseline)} for every match, training a model per held-out match."""
    os.makedirs(OOF_DIR, exist_ok=True)
    clean = {match_id: idsse.load(match_id) for match_id in idsse.MATCH_IDS}
    out = {}
    for index, match_id in enumerate(idsse.MATCH_IDS):
        path = os.path.join(OOF_DIR, "{}.npz".format(match_id))
        if refresh or not os.path.isfile(path):
            started = time.time()
            train = [(i, clean[m]) for i, m in enumerate(idsse.MATCH_IDS) if m != match_id]
            x, y, _ = build_training_set(train, copies=copies, stride=stride)
            model = fit(x, y)
            held_out = degrade(clean[match_id], VALIDATION_SEED + index)
            probs = predict_probs(model, held_out)
            np.savez_compressed(
                path,
                probs=probs,
                gt=gt_with_half_breaks(held_out.label, held_out.period),
                baseline=baseline_codes(held_out),
                period=held_out.period,
            )
            print(
                "fold {} trained on {} rows in {:.0f}s".format(
                    match_id, len(y), time.time() - started
                ),
                flush=True,
            )
        with np.load(path) as data:
            out[match_id] = (data["probs"], data["gt"], data["baseline"], data["period"])
    return out


def select_decoder(oof, candidates):
    """Leave-one-match-out choice of decoder settings. ({match: params}, table of scores)."""
    table = np.zeros((len(candidates), len(oof)))
    for j, (probs, gt, _, period) in enumerate(oof.values()):
        for i, params in enumerate(candidates):
            table[i, j] = pm.spell_scores(
                gt,
                pm.fill_abstentions(decode(probs, period, params), pm.valid_runs(gt)),
                pm.valid_runs(gt),
            )["mean_f1"]
    chosen = {}
    for j, match_id in enumerate(oof):
        others = np.delete(table, j, axis=1).mean(axis=1)
        chosen[match_id] = candidates[int(np.argmax(others))]
    return chosen, table


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="retrain every fold")
    parser.add_argument("--out", default=os.path.join(CACHE_DIR, "validation.json"))
    args = parser.parse_args()

    oof = out_of_fold(args.refresh)
    candidates = grid()
    chosen, table = select_decoder(oof, candidates)

    rows = {}
    for match_id, (probs, gt, baseline, period) in oof.items():
        argmax = np.where(period > 0, np.argmax(probs, axis=1), X).astype(np.int8)
        rows[match_id] = {
            "baseline_smooth50": score(gt, baseline),
            "gbm_argmax": score(gt, argmax),
            "gbm_hsmm": score(gt, decode(probs, period, chosen[match_id])),
            "params": chosen[match_id].to_dict(),
        }

    print(
        "\n{:<8} {:>22} {:>22} {:>22}".format(
            "match", "baseline_smooth50", "gbm_argmax", "gbm_hsmm"
        )
    )
    for match_id, row in rows.items():
        print(
            "{:<8} ".format(match_id)
            + " ".join(
                "{:>10.4f} {:>5.3f} {:>4d}".format(
                    row[k]["spellF1"], row[k]["macroF1"], row[k]["n_pred_spells"]
                )
                for k in ("baseline_smooth50", "gbm_argmax", "gbm_hsmm")
            )
            + "   (gt spells {})".format(row["gbm_hsmm"]["n_gt_spells"])
        )
    means = {
        k: {m: float(np.mean([r[k][m] for r in rows.values()])) for m in ("spellF1", "macroF1")}
        for k in ("baseline_smooth50", "gbm_argmax", "gbm_hsmm")
    }
    print("\nmean spellF1 / macroF1 over held-out matches")
    for k, v in means.items():
        print("  {:<20} {:.4f} / {:.4f}".format(k, v["spellF1"], v["macroF1"]))
    best = candidates[int(np.argmax(table.mean(axis=1)))]
    print("\ndecoder chosen on all seven matches: {}".format(best.to_dict()))

    with open(args.out, "w") as handle:
        json.dump({"folds": rows, "means": means, "best_params": best.to_dict()}, handle, indent=1)


if __name__ == "__main__":
    main()
