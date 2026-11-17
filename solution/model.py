"""The per-frame classifier: gradient-boosted trees over the features, trained on degraded open data.

Gradient boosting because the features are heterogeneous, partly missing (no ball on ~41% of
frames) and nonlinear in obvious ways (a 1 m proximity gap means everything next to the ball and
nothing 30 m away), and histogram GBMs handle NaN natively. Training rows are every `stride`-th
in-period frame of several independently degraded copies of each training match, each copy also
mirrored (teams swapped) so nothing is learnt about "home" as such. At prediction time the model
is applied to the match and to its mirror and the two answers are averaged, which makes the
output exactly symmetric in the two teams.

Fully deterministic: fixed seeds for the corruption and for the GBM.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from solution.common import A, D, H
from solution.degrade import degrade
from solution.features import build_features

GBM_PARAMS = dict(
    max_iter=300,
    learning_rate=0.08,
    max_leaf_nodes=63,
    min_samples_leaf=200,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=0,
)
STRIDE = 3
COPIES = 2


def copy_seed(match_index, copy):
    """Seed of one degraded training copy. Disjoint from the validation seeds (>= 10_000)."""
    return 1000 * match_index + copy


def training_rows(match, stride=STRIDE):
    """(features, labels) for every stride-th in-period frame of a match and of its mirror."""
    xs, ys = [], []
    for view in (match, match.mirrored()):
        features, names = build_features(view)
        keep = np.flatnonzero(view.period > 0)[::stride]
        xs.append(features[keep])
        ys.append(view.label[keep])
    return np.concatenate(xs), np.concatenate(ys), names


def build_training_set(matches, copies=COPIES, stride=STRIDE):
    """Stack degraded copies of (match_index, clean Match) pairs into one training table."""
    xs, ys = [], []
    for match_index, match in matches:
        for copy in range(copies):
            x, y, names = training_rows(degrade(match, copy_seed(match_index, copy)), stride)
            xs.append(x)
            ys.append(y)
    return np.concatenate(xs), np.concatenate(ys), names


def fit(x, y):
    model = HistGradientBoostingClassifier(**GBM_PARAMS)
    model.fit(x, y)
    return model


def predict_probs(model, match):
    """[T, 3] probabilities over (H, A, D), averaged with the team-swapped view."""
    direct = model.predict_proba(build_features(match)[0])
    swapped = model.predict_proba(build_features(match.mirrored())[0])
    order = list(model.classes_)
    probs = np.zeros((match.n_frames, 3))
    for code, other in ((H, A), (A, H), (D, D)):
        probs[:, code] = 0.5 * (direct[:, order.index(code)] + swapped[:, order.index(other)])
    return probs
