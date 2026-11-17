"""Mamba-3 against the GBM, leave-one-match-out on open data, everything else held equal.

    python -m solution.compare_models            # ~2-3 h on a 12-thread CPU; folds are cached

Both classifiers see the same 156 features of the same degraded IDSSE copies (solution.degrade,
same seeds as solution.validate), are scored on the same held-out degraded match, and go through
the same decoder whose settings are chosen per fold on the other six matches' out-of-fold
predictions (solution.validate.select_decoder). The GBM part reproduces solution.validate exactly
and writes its folds where solution.validate looks for them (cache/oof/). A third row averages
the two models' probabilities.

Features of every degraded copy are computed once and cached under cache/features/, because the
copies are the same in every fold (their seeds depend on the match, not on the fold).
"""

import argparse
import json
import os
import time

import numpy as np

from solution import idsse
from solution.common import CACHE_DIR, A, D, H, X, gt_with_half_breaks
from solution.decode import decode
from solution.degrade import degrade
from solution.features import build_features
from solution.model import COPIES, copy_seed, fit as fit_gbm
from solution.sequence_model import Standardizer, TrainConfig, predict_from_features
from solution.sequence_model import fit as fit_mamba
from solution.validate import (
    OOF_DIR,
    VALIDATION_SEED,
    baseline_codes,
    grid,
    score,
    select_decoder,
)

FEATURE_DIR = os.path.join(CACHE_DIR, "features")
MAMBA_OOF_DIR = os.path.join(CACHE_DIR, "oof_mamba")
GBM_STRIDE = 5  # as in solution.validate.out_of_fold


def swap_teams(probs):
    out = probs.copy()
    out[:, H], out[:, A] = probs[:, A], probs[:, H]
    return out


def cached_views(match, tag):
    """(features direct, features team-swapped, labels direct, period) for a degraded match."""
    path = os.path.join(FEATURE_DIR, tag + ".npz")
    if not os.path.isfile(path):
        os.makedirs(FEATURE_DIR, exist_ok=True)
        direct = build_features(match)[0]
        swapped = build_features(match.mirrored())[0]
        np.savez(path, direct=direct, swapped=swapped, label=match.label, period=match.period)
    with np.load(path) as data:
        return data["direct"], data["swapped"], data["label"], data["period"]


def mirrored_labels(label):
    out = label.copy()
    out[label == H], out[label == A] = A, H
    return out


def training_views(clean, match_ids):
    """Yield (direct, swapped, label, period) for every training copy, in solution.model order."""
    for index, match_id in enumerate(idsse.MATCH_IDS):
        if match_id not in match_ids:
            continue
        for copy in range(COPIES):
            tag = "{}_copy{}".format(match_id, copy)
            path = os.path.join(FEATURE_DIR, tag + ".npz")
            match = (
                None if os.path.isfile(path) else degrade(clean[match_id], copy_seed(index, copy))
            )
            yield cached_views(match, tag)


def gbm_fold(clean, train_ids, held):
    xs, ys = [], []
    for direct, swapped, label, period in training_views(clean, train_ids):
        keep = np.flatnonzero(period > 0)[::GBM_STRIDE]
        xs += [direct[keep], swapped[keep]]
        ys += [label[keep], mirrored_labels(label)[keep]]
    model = fit_gbm(np.concatenate(xs), np.concatenate(ys))
    direct, swapped, _, _ = held
    order = list(model.classes_)
    p_direct = model.predict_proba(direct)[:, [order.index(c) for c in (H, A, D)]]
    p_swapped = model.predict_proba(swapped)[:, [order.index(c) for c in (H, A, D)]]
    return 0.5 * (p_direct + swap_teams(p_swapped))


def mamba_fold(clean, train_ids, held, config):
    raw = list(training_views(clean, train_ids))
    std = Standardizer().fit([v for d, s, _, _ in raw for v in (d, s)])
    tracks = []
    for direct, swapped, label, period in raw:
        tracks.append((std.transform(direct).astype(np.float16), label, period))
        tracks.append((std.transform(swapped).astype(np.float16), mirrored_labels(label), period))
    del raw
    model = fit_mamba(tracks, config, log=lambda s: print(s, flush=True))
    direct, swapped, _, period = held
    return predict_from_features(model, std.transform(direct), std.transform(swapped), period)


def run_folds(config, refresh=False, skip_gbm=False):
    clean = {m: idsse.load(m) for m in idsse.MATCH_IDS}
    os.makedirs(OOF_DIR, exist_ok=True)
    os.makedirs(MAMBA_OOF_DIR, exist_ok=True)
    for index, match_id in enumerate(idsse.MATCH_IDS):
        train_ids = [m for m in idsse.MATCH_IDS if m != match_id]
        held_match = degrade(clean[match_id], VALIDATION_SEED + index)
        held = cached_views(held_match, "{}_heldout".format(match_id))
        extra = dict(
            gt=gt_with_half_breaks(held_match.label, held_match.period),
            baseline=baseline_codes(held_match),
            period=held_match.period,
        )
        for name, folder, runner in (
            ("gbm", OOF_DIR, lambda: gbm_fold(clean, train_ids, held)),
            ("mamba", MAMBA_OOF_DIR, lambda: mamba_fold(clean, train_ids, held, config)),
        ):
            path = os.path.join(folder, match_id + ".npz")
            if name == "gbm" and skip_gbm:
                continue
            if refresh or not os.path.isfile(path):
                started = time.time()
                print("fold {} {} ...".format(match_id, name), flush=True)
                np.savez_compressed(path, probs=runner(), **extra)
                print("fold {} {} done in {:.0f}s".format(match_id, name, time.time() - started))


def load_oof(folder):
    out = {}
    for match_id in idsse.MATCH_IDS:
        with np.load(os.path.join(folder, match_id + ".npz")) as data:
            out[match_id] = (data["probs"], data["gt"], data["baseline"], data["period"])
    return out


def evaluate():
    """Scores for Mamba-3, and for the GBM and the average when every GBM fold is available."""
    mamba = load_oof(MAMBA_OOF_DIR)
    methods = {"mamba": mamba}
    have_gbm = all(os.path.isfile(os.path.join(OOF_DIR, m + ".npz")) for m in idsse.MATCH_IDS)
    if have_gbm:
        gbm = load_oof(OOF_DIR)
        methods["gbm"] = gbm
        methods["ensemble"] = {
            m: (0.5 * (gbm[m][0] + mamba[m][0]),) + gbm[m][1:] for m in idsse.MATCH_IDS
        }
    else:
        print("GBM folds incomplete (--skip-gbm): reporting Mamba-3 and the baseline only")
    candidates = grid()
    rows = {m: {"baseline_smooth50": score(mamba[m][1], mamba[m][2])} for m in idsse.MATCH_IDS}
    best = {}
    for name, oof in methods.items():
        started = time.time()
        chosen, table = select_decoder(oof, candidates)
        best[name] = candidates[int(np.argmax(table.mean(axis=1)))].to_dict()
        for m, (probs, gt, _, period) in oof.items():
            argmax = np.where(period > 0, np.argmax(probs, axis=1), X).astype(np.int8)
            rows[m][name + "_argmax"] = score(gt, argmax)
            rows[m][name + "_hsmm"] = score(gt, decode(probs, period, chosen[m]))
        print("decoder selection for {} took {:.0f}s".format(name, time.time() - started))
    columns = ["baseline_smooth50"] + [n + s for n in methods for s in ("_argmax", "_hsmm")]
    print("\nspellF1 per held-out match")
    print("{:<8}".format("match") + "".join("{:>18}".format(c) for c in columns))
    for m, row in rows.items():
        print("{:<8}".format(m) + "".join("{:>18.4f}".format(row[c]["spellF1"]) for c in columns))
    means = {
        c: {k: float(np.mean([r[c][k] for r in rows.values()])) for k in ("spellF1", "macroF1")}
        for c in columns
    }
    print("\nmean spellF1 / macroF1")
    for c, v in means.items():
        print("  {:<20} {:.4f} / {:.4f}".format(c, v["spellF1"], v["macroF1"]))
    return {"folds": rows, "means": means, "best_params": best}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="retrain every fold")
    parser.add_argument("--steps", type=int, default=TrainConfig.steps)
    parser.add_argument("--skip-gbm", action="store_true", help="train only the Mamba-3 folds")
    parser.add_argument("--out", default=os.path.join(CACHE_DIR, "model_comparison.json"))
    args = parser.parse_args()
    config = TrainConfig(steps=args.steps)
    run_folds(config, args.refresh, args.skip_gbm)
    result = evaluate()
    result["mamba_config"] = config.__dict__
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=1)
    print("wrote {}".format(args.out))


if __name__ == "__main__":
    main()
