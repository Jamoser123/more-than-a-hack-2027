"""Training and prediction for the Mamba-3 possession classifier (solution/mamba3.py).

A drop-in alternative to the GBM in solution/model.py: same per-frame features, same degraded,
mirrored training copies, same [T, 3] probability output averaged with the team-swapped view, so
the decoder and the validation treat the two identically.

Sequences are taken at 5 fps (every 5th frame, a random one of the five phases per window), in
windows of WINDOW steps (about 51 s) inside one half. At prediction time every phase is run, so
every frame gets its own output, and overlapping windows are blended with triangular weights so no
frame depends on where a window happens to start.

Hyperparameters were fixed before any run (no search): a model about the GBM's size, trained for a
CPU budget of roughly a quarter of an hour per fit. Everything is seeded and torch runs in
deterministic mode with a fixed thread count, so a fit is reproducible on the same machine.
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from solution.common import A, D, H, period_slices
from solution.mamba3 import PossessionMamba

STEP = 5  # frames per sequence step (25 fps -> 5 fps)
WINDOW = 256  # steps per training / prediction window
HOP = 64  # prediction window stride
THREADS = 10
CLIP = 5.0


@dataclass(frozen=True)
class TrainConfig:
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    head_dim: int = 32
    d_state: int = 16
    dropout: float = 0.1
    steps: int = 500
    batch: int = 32
    lr: float = 2e-3
    weight_decay: float = 0.05
    warmup: int = 50
    seed: int = 0


class Standardizer:
    """Per-feature centring and scaling fitted on training frames, NaN -> 0 plus missing flags."""

    def fit(self, arrays, max_rows=400_000, seed=0):
        rng = np.random.default_rng(seed)
        per_array = max(1, max_rows // len(arrays))
        rows = np.concatenate(
            [
                a if len(a) <= per_array else a[rng.choice(len(a), per_array, replace=False)]
                for a in arrays
            ]
        )
        self.mean = np.nan_to_num(np.nanmean(rows, axis=0)).astype(np.float32)
        std = np.nan_to_num(np.nanstd(rows, axis=0))
        self.std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
        self.flags = np.flatnonzero(np.isnan(rows).any(axis=0))
        return self

    @property
    def n_out(self):
        return len(self.mean) + len(self.flags)

    def transform(self, x):
        z = np.clip((x - self.mean) / self.std, -CLIP, CLIP)
        missing = np.isnan(x[:, self.flags]).astype(np.float32)
        return np.concatenate([np.nan_to_num(z), missing], axis=1).astype(np.float32)


def _setup(seed):
    torch.manual_seed(seed)
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)


def _windows(tracks, rng, batch):
    """A batch of (features, labels) windows at STEP spacing from random halves of random tracks."""
    halves = [(i, a, b) for i, (_, _, p) in enumerate(tracks) for _, a, b in period_slices(p)]
    sizes = np.array([b - a for _, a, b in halves], dtype=np.float64)
    picks = rng.choice(len(halves), batch, p=sizes / sizes.sum())
    xs, ys = [], []
    span = WINDOW * STEP
    for k in picks:
        i, a, b = halves[k]
        x, y, _ = tracks[i]
        start = a + int(rng.integers(0, max(1, b - a - span)))
        idx = start + STEP * np.arange(WINDOW)
        idx = idx[idx < b]
        xs.append(x[idx])
        ys.append(y[idx])
    length = min(len(v) for v in ys)
    return (
        torch.from_numpy(np.stack([v[:length] for v in xs]).astype(np.float32)),
        torch.from_numpy(np.stack([v[:length] for v in ys]).astype(np.int64)),
    )


def fit(tracks, config=TrainConfig(), log=print):
    """Train on [(standardised features [T, F], labels [T], period [T])]. Returns the model."""
    _setup(config.seed)
    rng = np.random.default_rng(config.seed)
    n_features = tracks[0][0].shape[1]
    model = PossessionMamba(
        n_features,
        config.d_model,
        config.n_layers,
        config.n_heads,
        config.head_dim,
        config.d_state,
        config.dropout,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    def lr_at(step):
        if step < config.warmup:
            return (step + 1) / config.warmup
        progress = (step - config.warmup) / max(1, config.steps - config.warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    model.train()
    running = None
    for step in range(config.steps):
        x, y = _windows(tracks, rng, config.batch)
        loss = F.cross_entropy(model(x).reshape(-1, 3), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running = loss.item() if running is None else 0.98 * running + 0.02 * loss.item()
        if log and (step + 1) % 100 == 0:
            log("  step {:4d}  loss {:.4f}".format(step + 1, running))
    model.eval()
    model.config = asdict(config)
    return model


@torch.no_grad()
def _probs_one_view(model, x, period):
    """[T, 3] softmax output for one view; frames outside the halves stay at D."""
    out = np.zeros((len(x), 3), dtype=np.float64)
    out[:, D] = 1.0
    ramp = np.minimum(np.arange(1, WINDOW + 1), np.arange(WINDOW, 0, -1)).astype(np.float64)
    for _, a, b in period_slices(period):
        for phase in range(STEP):
            idx = np.arange(a + phase, b, STEP)
            n = len(idx)
            if n == 0:
                continue
            starts = list(range(0, max(1, n - WINDOW + 1), HOP))
            if starts[-1] + WINDOW < n:
                starts.append(n - WINDOW)
            acc = np.zeros((n, 3))
            weight = np.zeros(n)
            for chunk in range(0, len(starts), 32):
                group = starts[chunk : chunk + 32]
                batch = np.stack([x[idx[s : s + WINDOW]] for s in group]) if n >= WINDOW else None
                if batch is None:  # a half shorter than one window
                    batch = x[idx][None]
                batch = torch.from_numpy(batch.astype(np.float32))
                p = torch.softmax(model(batch), dim=-1).double().numpy()
                for s, ps in zip(group, p):
                    w = ramp[: len(ps)]
                    acc[s : s + len(ps)] += ps * w[:, None]
                    weight[s : s + len(ps)] += w
            out[idx] = acc / weight[:, None]
    return out


def predict_from_features(model, x_direct, x_swapped, period):
    """[T, 3] over (H, A, D), averaged with the team-swapped view (exactly symmetric)."""
    torch.set_num_threads(THREADS)
    direct = _probs_one_view(model, x_direct, period)
    swapped = _probs_one_view(model, x_swapped, period)
    probs = np.empty_like(direct)
    for code, other in ((H, A), (A, H), (D, D)):
        probs[:, code] = 0.5 * (direct[:, code] + swapped[:, other])
    return probs
