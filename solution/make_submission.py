"""The one command that produces our submission from the data bundle, identically every time.

    python -m solution.make_submission                       # -> submissions/zueri_giele.csv

Steps: fetch and convert the seven open IDSSE matches (cached), train the classifier on degraded
copies of all of them (cached under cache/), read the competition ATD, predict per-frame H/A/D
probabilities, decode them per half with the duration-aware Viterbi using the parameters chosen
by solution.validate (solution/decoder_params.json), write X outside the graded halves, and run
the official validator on the written file.

Everything is seeded; rerunning it on the same data gives the same file byte for byte.
"""

import argparse
import json
import os
import pickle
import subprocess
import sys

import numpy as np

from solution import atd, idsse
from solution.common import CACHE_DIR, HERE, LETTERS, ROOT
from solution.decode import DecoderParams, decode
from solution.model import build_training_set, fit, predict_probs

PARAMS_PATH = os.path.join(HERE, "decoder_params.json")
MODEL_PATH = os.path.join(CACHE_DIR, "final_model.pkl")
TEAM = "zueri_giele"


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
    parser.add_argument("--params", default=PARAMS_PATH)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--refresh-model", action="store_true")
    args = parser.parse_args(argv)

    with open(args.params) as handle:
        params = DecoderParams(**json.load(handle))
    model = final_model(args.refresh_model, args.model)
    match = atd.load(data_dir=args.data_dir)
    probs = predict_probs(model, match)
    codes = decode(probs, match.period, params).astype(np.int64)
    path = write_submission(args.out, codes)
    np.save(probs_path(path), probs.astype(np.float16))  # for solution.overlay, not submitted
    print("wrote {}".format(path))
    validator = os.path.join(ROOT, "evaluation", "validate_submission.py")
    return subprocess.call([sys.executable, validator, path])


if __name__ == "__main__":
    sys.exit(main())
