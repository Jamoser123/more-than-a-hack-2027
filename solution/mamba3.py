"""A small Mamba-3 sequence model (Lahoti et al., 2026, arXiv:2603.15569) in plain PyTorch.

Why: the GBM sees each frame through hand-made window averages; a sequence model can learn its own
temporal context (how a possession builds up, how a dead ball looks over seconds) from the same
per-frame features. Mamba-3 is a linear-time state space model, so it handles long windows cheaply,
and its complex-valued state can track rotations / alternations that a real decay cannot.

The layer follows the paper's SISO Mamba-3 (Section 3, Figure 2):

- exponential-trapezoidal discretisation (Proposition 1):
  h_t = a_t h_{t-1} + b_t B_{t-1} x_{t-1} + g_t B_t x_t, a_t = exp(dt_t A_t),
  b_t = (1 - lam_t) dt_t a_t, g_t = lam_t dt_t, with data-dependent dt, A (Remark 1) and lam;
- complex state via the "RoPE trick" (Propositions 3, 4): B and C are rotated by the cumulative
  data-dependent angle sum_s dt_s theta_s, after which the recurrence is a plain scalar-decay SSM;
- BC normalisation (RMSNorm on B and C) followed by learnable head-specific B / C biases
  initialised to one (Appendix F), and no short convolution;
- multi-value heads: B and C are shared across heads, x, dt, A, lam are per head.

The recurrence is computed in its parallel (SSD) form Y = (L o C B^T) X, where L is the decay mask
times the two-band trapezoidal mask of equation (7). That is quadratic in the window length, which
is fine for the windows used here (a few hundred steps) and needs no custom kernels, so it runs on
a CPU. `Mamba3.recurrent` is the step-by-step form, kept as the reference the tests compare to.

Not taken from the paper: MIMO (a decoding-speed device for GPUs that buys nothing here), and the
blocks are bidirectional (a forward and a backward Mamba-3 per block, summed), because possession
is labelled offline and the future is as informative as the past.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rotate_pairs(v, angle):
    """Rotate consecutive pairs (v[..., 2i], v[..., 2i+1]) by angle[..., i]."""
    even, odd = v[..., 0::2], v[..., 1::2]
    cos, sin = torch.cos(angle), torch.sin(angle)
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class Mamba3(nn.Module):
    """One SISO Mamba-3 mixer, causal over the time axis. Input and output [batch, T, d_model]."""

    def __init__(self, d_model, n_heads=4, head_dim=32, d_state=16):
        super().__init__()
        if d_state % 2:
            raise ValueError("d_state must be even (pairs of real coordinates = one complex)")
        self.h, self.p, self.n = n_heads, head_dim, d_state
        d_inner = n_heads * head_dim
        self.split = [d_inner, d_inner, d_state, d_state, n_heads, n_heads, n_heads, d_state // 2]
        self.in_proj = nn.Linear(d_model, sum(self.split), bias=False)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.b_norm, self.c_norm = RMSNorm(d_state), RMSNorm(d_state)
        self.b_bias = nn.Parameter(torch.ones(n_heads, d_state))
        self.c_bias = nn.Parameter(torch.ones(n_heads, d_state))
        self.skip = nn.Parameter(torch.ones(n_heads))
        # Initial step sizes log-uniform in [1e-3, 1e-1] and decay rates in [1, 16], as in Mamba:
        # memories from about one step to a few hundred at initialisation.
        dt = torch.exp(torch.linspace(math.log(1e-3), math.log(1e-1), n_heads))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        a = torch.linspace(1.0, 16.0, n_heads)
        self.a_bias = nn.Parameter(torch.log(torch.expm1(a)))

    def params(self, u):
        """Per-step quantities shared by the parallel and the recurrent form."""
        x, z, b, c, dt, a, lam, theta = torch.split(self.in_proj(u), self.split, dim=-1)
        batch, steps, _ = u.shape
        x = x.view(batch, steps, self.h, self.p)
        dt = F.softplus(dt + self.dt_bias)  # [b, T, H]
        log_alpha = -dt * F.softplus(a + self.a_bias)  # dt * A with A < 0
        lam = torch.sigmoid(lam)
        gamma = lam * dt
        beta = (1 - lam) * dt * torch.exp(log_alpha)
        b = self.b_norm(b)[:, :, None, :] + self.b_bias  # [b, T, H, N]
        c = self.c_norm(c)[:, :, None, :] + self.c_bias
        angle = dt[..., None] * theta[:, :, None, :]  # [b, T, H, N/2], per-step rotation
        return x, z, b, c, log_alpha, gamma, beta, angle

    def forward(self, u):
        x, z, b, c, log_alpha, gamma, beta, angle = self.params(u)
        # RoPE trick: rotate B and C by minus the cumulative angle (prod of R_s^T up to t).
        phase = torch.cumsum(angle, dim=1)
        b_bar, c_bar = rotate_pairs(b, -phase), rotate_pairs(c, -phase)
        # Decay mask L[t, s] = exp(sum_{k=s+1..t} dt_k A_k) for s <= t, else 0.
        cs = torch.cumsum(log_alpha, dim=1).transpose(1, 2)  # [b, H, T]
        seg = cs[..., :, None] - cs[..., None, :]
        steps = u.shape[1]
        causal = torch.ones(steps, steps, dtype=torch.bool, device=u.device).tril()
        decay = torch.exp(seg.masked_fill(~causal, float("-inf")))
        # Trapezoid: input s enters with g_s, and again one step later with b_{s+1}.
        weight = decay * gamma.transpose(1, 2)[..., None, :]
        late = decay * beta.transpose(1, 2)[..., None, :]
        weight = weight + F.pad(late[..., :, 1:], (0, 1))
        scores = torch.einsum("bthn,bshn->bhts", c_bar, b_bar) * weight
        y = torch.einsum("bhts,bshp->bthp", scores, x) + self.skip[:, None] * x
        y = y.reshape(*u.shape[:2], -1) * F.silu(z)
        return self.out_proj(y)

    def recurrent(self, u):
        """Reference: the step-by-step complex recurrence of Proposition 4, without the RoPE trick."""
        x, z, b, c, log_alpha, gamma, beta, angle = self.params(u)
        batch, steps = u.shape[:2]
        state = u.new_zeros(batch, self.h, self.n, self.p)
        prev_input = u.new_zeros(batch, self.h, self.n, self.p)
        out = []
        for t in range(steps):
            alpha = torch.exp(log_alpha[:, t])[..., None, None]
            rot = angle[:, t]  # rotate the state and the delayed input by R_t (eq. 25)

            def r(m):
                return rotate_pairs(m.transpose(-1, -2), rot[:, :, None, :]).transpose(-1, -2)

            current = b[:, t, :, :, None] * x[:, t, :, None, :]  # B_t x_t^T  [b, H, N, P]
            state = (
                alpha * r(state)
                + beta[:, t, :, None, None] * r(prev_input)
                + gamma[:, t, :, None, None] * current
            )
            prev_input = current
            y = torch.einsum("bhn,bhnp->bhp", c[:, t], state) + self.skip[:, None] * x[:, t]
            out.append(y.reshape(batch, -1) * F.silu(z[:, t]))
        return self.out_proj(torch.stack(out, dim=1))


class Block(nn.Module):
    """Pre-norm residual block: a bidirectional Mamba-3 mixer, then a SwiGLU MLP (Llama layout)."""

    def __init__(self, d_model, n_heads, head_dim, d_state, mlp_mult=2, dropout=0.1):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(d_model), RMSNorm(d_model)
        self.forward_mixer = Mamba3(d_model, n_heads, head_dim, d_state)
        self.backward_mixer = Mamba3(d_model, n_heads, head_dim, d_state)
        hidden = mlp_mult * d_model
        self.gate_up = nn.Linear(d_model, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        mixed = self.forward_mixer(h) + self.backward_mixer(h.flip(1)).flip(1)
        x = x + self.drop(mixed)
        gate, up = self.gate_up(self.norm2(x)).chunk(2, dim=-1)
        return x + self.drop(self.down(F.silu(gate) * up))


class PossessionMamba(nn.Module):
    """Per-step H / A / D logits from per-step features. [batch, T, F] -> [batch, T, 3]."""

    def __init__(
        self, n_features, d_model=64, n_layers=2, n_heads=4, head_dim=32, d_state=16, dropout=0.1
    ):
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, head_dim, d_state, dropout=dropout) for _ in range(n_layers)
        )
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, 3)

    def forward(self, features):
        x = self.embed(features)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))
