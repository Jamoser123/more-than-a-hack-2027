"""The one command that produces our submission from the data bundle, identically every time.

    python -m solution.make_submission                       # -> submissions/zueri_giele.csv

Steps: fetch and convert the seven open IDSSE matches (cached), train the classifier on degraded
copies of all of them (cached under cache/), read the competition ATD, predict per-frame H/A/D
probabilities, optionally add the Smart Tagging evidence (solution.tags: --tags for set pieces
and goals, plus --tag-phases for the possession phases; off by default), decode
per half with the duration-aware Viterbi using the parameters chosen by solution.validate
(solution/decoder_params.json), write X outside the graded halves, and run the official
validator on the written file.

Everything is seeded; rerunning it on the same data gives the same file byte for byte.
"""

import argparse
import json
import os
import pickle
import subprocess
import sys

import numpy as np

from solution import atd, idsse, tags
from solution.common import CACHE_DIR, HERE, LETTERS, ROOT
from solution.decode import DecoderParams, decode
from solution.features import build_features
from solution.model import build_training_set, fit, predict_probs

PARAMS_PATH = os.path.join(HERE, "decoder_params.json")
MODEL_PATH = os.path.join(CACHE_DIR, "final_model.pkl")
# Mamba-3 classifier (--classifier mamba): its own decoder settings, chosen leave-one-match-out on
# its IDSSE predictions by solution.compare_models (it needs far less smoothing than the GBM).
MAMBA_PARAMS_PATH = os.path.join(HERE, "decoder_params_mamba.json")
MAMBA_PATH = os.path.join(CACHE_DIR, "final_mamba.pkl")
TEAM = "zueri_giele"


def final_mamba(refresh=False, path=MAMBA_PATH):
    """(Mamba-3 model, standardizer) trained on all seven open matches, cached after first run."""
    from solution.compare_models import mirrored_labels, training_views
    from solution.sequence_model import Standardizer, TrainConfig
    from solution.sequence_model import fit as fit_mamba

    if not refresh and os.path.isfile(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)
    clean = {m: idsse.load(m) for m in idsse.MATCH_IDS}
    raw = list(training_views(clean, idsse.MATCH_IDS))
    std = Standardizer().fit([v for d, s, _, _ in raw for v in (d, s)])
    tracks = []
    for direct, swapped, label, period in raw:
        tracks.append((std.transform(direct).astype(np.float16), label, period))
        tracks.append((std.transform(swapped).astype(np.float16), mirrored_labels(label), period))
    del raw
    model = fit_mamba(tracks, TrainConfig(), log=lambda s: print(s, flush=True))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump((model, std), handle)
    return model, std


def mamba_probs(model, std, match):
    """[T, 3] Mamba-3 probabilities for a match, averaged with the team-swapped view."""
    from solution.sequence_model import predict_from_features

    direct = std.transform(build_features(match)[0])
    swapped = std.transform(build_features(match.mirrored())[0])
    return predict_from_features(model, direct, swapped, match.period)


def final_model(refresh=False, path=MODEL_PATH):
    """The classifier trained on all seven open matches, cached after the first run."""
    if not refresh and os.path.isfile(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)
    matches = [(i, idsse.load(m)) for i, m in enumerate(idsse.MATCH_IDS)]
    x, y, _ = build_training_set(matches)
    model = fit(x, y)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(model, handle)
    return model


def probs_path(submission_path):
    """Where the per-frame classifier probabilities of a submission are kept."""
    return os.path.splitext(submission_path)[0] + "_probs.npy"


def write_submission(path, codes):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, c) for i, c in enumerate(LETTERS[codes]))
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(ROOT, "submissions", TEAM + ".csv"))
    parser.add_argument("--data-dir", default=None, help="defaults to games.DATA_DIR")
    parser.add_argument(
        "--classifier", choices=("gbm", "mamba"), default="gbm", help="per-frame classifier"
    )
    parser.add_argument("--params", default=None, help="decoder settings; per-classifier default")
    parser.add_argument("--model", default=None, help="model cache; per-classifier default")
    parser.add_argument("--refresh-model", action="store_true")
    parser.add_argument(
        "--tags", action="store_true", help="add Smart Tagging set pieces and goals (experimental)"
    )
    parser.add_argument(
        "--tag-phases", action="store_true", help="with --tags, also the possession phases"
    )
    args = parser.parse_args(argv)

    mamba = args.classifier == "mamba"
    with open(args.params or (MAMBA_PARAMS_PATH if mamba else PARAMS_PATH)) as handle:
        params = DecoderParams(**json.load(handle))
    match = atd.load(data_dir=args.data_dir)
    if mamba:
        model, std = final_mamba(args.refresh_model, args.model or MAMBA_PATH)
        probs = mamba_probs(model, std, match)
    else:
        model = final_model(args.refresh_model, args.model or MODEL_PATH)
        probs = predict_probs(model, match)
    prior = (
        tags.match_prior(probs, data_dir=args.data_dir, phases=args.tag_phases)
        if args.tags
        else None
    )
    codes = decode(probs, match.period, params, prior).astype(np.int64)
    path = write_submission(args.out, codes)
    np.save(probs_path(path), probs.astype(np.float16))  # for solution.overlay, not submitted
    print("wrote {}".format(path))
    validator = os.path.join(ROOT, "evaluation", "validate_submission.py")
    return subprocess.call([sys.executable, validator, path])


if __name__ == "__main__":
    sys.exit(main())
