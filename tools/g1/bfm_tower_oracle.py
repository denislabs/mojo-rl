"""The ORACLE for `mojo_rl/deep_agents/fb/bfm_towers.mojo` — the reference's
residual towers in torch, fed OUR weights, run on OUR inputs (G3.2).

Imported through Python interop by `tests/nn/test_bfm_towers_vs_torch.mojo`
(run under `pixi run -e act-ref`, the environment with torch):

    build(kind, dims)                      the reference's class, float64
    load_ours(model, kind, dims, names,    OUR flat float32 params by dotted
              values, sizes)               path -> the torch state dict
    run(model, x, grad_out)                out, grad_in, and the param grads
                                           back in OUR names, layout, order

The classes are transcribed from `humanoidverse/agents/nn_models.py`
(`Block`, `ResidualBlock`, `residual_embedding`, `ResidualForwardMap`,
`ResidualActor`, `BackwardMap`, `Norm`) at `num_parallel = 1`, without the
pydantic configs and the gymnasium spaces the module imports — the tensor
arithmetic is theirs line for line, and the transcription is small enough
to read against the source.

THE NAME MAP. Our composition names a param by the combinator path
(`Sequential` child index, `Parallel` branch index, `Residual` child `0`,
`Repeat` copy index), e.g. `0.0.1.1.0.0.gamma`; the torch model names it
by attribute (`embed_sa.1.mlp.0.weight`). `_our_to_torch` walks the same
structure on both sides so every param is set by structure, not by
enumeration order — the reference registers `embed_z` before `embed_sa`,
our `Parallel` lists sa first, and an order-based load would silently
swap two same-shaped embeddings. Linear weights are transposed: ours are
`[IN, OUT]` (`y = x @ W`), torch's `[OUT, IN]`.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# ── the reference's classes (nn_models.py), num_parallel = 1 ─────────────────
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.Mish())

    def forward(self, x):
        return x + self.mlp(x)


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, activation):
        super().__init__()
        seq = [nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim)] + ([nn.Mish()] if activation else [])
        self.mlp = nn.Sequential(*seq)

    def forward(self, x):
        return self.mlp(x)


def residual_embedding(input_dim, hidden_dim, hidden_layers):
    assert hidden_layers >= 2
    seq = [Block(input_dim, hidden_dim, True)]
    for _ in range(hidden_layers - 2):
        seq += [ResidualBlock(hidden_dim)]
    seq += [Block(hidden_dim, hidden_dim // 2, True)]
    return nn.Sequential(*seq)


class Norm(nn.Module):
    def forward(self, x):
        return math.sqrt(x.shape[-1]) * F.normalize(x, dim=-1)


class ResidualForwardMap(nn.Module):
    """`ResidualForwardMap`: embeddings at `hidden_layers`, trunk of `hidden_layers` residual blocks."""

    def __init__(self, obs_dim, z_dim, action_dim, hidden_dim, hidden_layers, output_dim):
        super().__init__()
        self.embed_z = residual_embedding(obs_dim + z_dim, hidden_dim, hidden_layers)
        self.embed_sa = residual_embedding(obs_dim + action_dim, hidden_dim, hidden_layers)
        seq = [ResidualBlock(hidden_dim) for _ in range(hidden_layers)]
        seq += [Block(hidden_dim, output_dim, False)]
        self.Fs = nn.Sequential(*seq)
        self.obs_dim, self.z_dim, self.action_dim = obs_dim, z_dim, action_dim

    def forward(self, row):
        obs = row[:, : self.obs_dim]
        action = row[:, self.obs_dim : self.obs_dim + self.action_dim]
        z = row[:, self.obs_dim + self.action_dim :]
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))
        sa_embedding = self.embed_sa(torch.cat([obs, action], dim=-1))
        return self.Fs(torch.cat([sa_embedding, z_embedding], dim=-1))


class ResidualActor(nn.Module):
    """`ResidualActor`: embeddings at `embedding_layers` (2), trunk of `hidden_layers`, tanh mean."""

    def __init__(self, obs_dim, z_dim, action_dim, hidden_dim, hidden_layers, embedding_layers=2):
        super().__init__()
        self.embed_z = residual_embedding(obs_dim + z_dim, hidden_dim, embedding_layers)
        self.embed_s = residual_embedding(obs_dim, hidden_dim, embedding_layers)
        seq = [ResidualBlock(hidden_dim) for _ in range(hidden_layers)] + [Block(hidden_dim, action_dim, False)]
        self.policy = nn.Sequential(*seq)
        self.obs_dim = obs_dim

    def forward(self, row):
        obs = row[:, : self.obs_dim]
        z = row[:, self.obs_dim :]
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))
        s_embedding = self.embed_s(obs)
        return torch.tanh(self.policy(torch.cat([s_embedding, z_embedding], dim=-1)))


class BackwardMap(nn.Module):
    """`BackwardMap` with `hidden_layers 1`, `norm True`."""

    def __init__(self, obs_dim, z_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, z_dim), Norm())

    def forward(self, x):
        return self.net(x)


class _Slices(nn.Module):
    """lin(cat(x[:, :a], x[:, b:])) — the Parallel[Slice, Slice] -> Linear composition."""

    def __init__(self, n_in, a, b, out):
        super().__init__()
        self.a, self.b = a, b
        self.lin = nn.Linear(a + n_in - b, out)

    def forward(self, x):
        return self.lin(torch.cat([x[:, : self.a], x[:, self.b :]], dim=-1))


class _ParSeq(nn.Module):
    """cat(lin1(x[:, :a]), lin2(x)) — Parallel[Sequential[Slice, Linear], Linear]."""

    def __init__(self, n_in, a, out):
        super().__init__()
        self.a = a
        self.lin1 = nn.Linear(a, out)
        self.lin2 = nn.Linear(n_in, out)

    def forward(self, x):
        return torch.cat([self.lin1(x[:, : self.a]), self.lin2(x)], dim=-1)


def build(kind, dims):
    d = {k: int(v) for k, v in dims.items()}
    if kind == "linmish":
        m = nn.Sequential(nn.Linear(d["in"], d["out"]), nn.Mish())
    elif kind == "linrelu":
        m = nn.Sequential(nn.Linear(d["in"], d["out"]), nn.ReLU())
    elif kind == "lintanh":
        m = nn.Sequential(nn.Linear(d["in"], d["out"]), nn.Tanh())
    elif kind == "block":
        m = Block(d["in"], d["out"], True)
    elif kind == "resblock":
        m = ResidualBlock(d["h"])
    elif kind == "repeat2":
        m = nn.Sequential(ResidualBlock(d["h"]), ResidualBlock(d["h"]))
    elif kind == "slices":
        m = _Slices(d["in"], d["a"], d["b"], d["out"])
    elif kind == "parseq":
        m = _ParSeq(d["in"], d["a"], d["out"])
    elif kind == "f":
        m = ResidualForwardMap(d["obs"], d["d"], d["act"], d["h"], d["l"], d["out"])
    elif kind == "actor":
        m = ResidualActor(d["obs"], d["d"], d["act"], d["h"], d["l"])
    elif kind == "b":
        m = BackwardMap(d["obs"], d["d"], d["hb"])
    else:
        raise ValueError(kind)
    return m.double()


# ── our dotted path -> the torch attribute path ──────────────────────────────
def _ln_or_lin(seg):
    """Inside a Block / ResidualBlock: our Sequential[LayerNorm, Linear*] children 0 / 1
    -> torch `mlp.0` (LayerNorm: weight = gamma, bias = beta) / `mlp.1` (Linear)."""
    idx, pname = seg.split(".", 1)
    if idx == "0":
        return "mlp.0." + {"gamma": "weight", "beta": "bias"}[pname], False
    if idx == "1":
        return "mlp.1." + pname, pname == "weight"
    raise KeyError(seg)


def _embed(rest, L):
    """Our `Embed[.., L]` = Sequential[Block, Repeat[L-2, ResBlock], Block] (L >= 3) or
    `Embed2` = Sequential[Block, Block] (L == 2) -> torch `residual_embedding` index."""
    i, tail = rest.split(".", 1)
    if L == 2:
        return f"{int(i)}." + _ln_or_lin(tail)[0], _ln_or_lin(tail)[1]
    if i == "0":
        p, t = _ln_or_lin(tail)
        return "0." + p, t
    if i == "1":  # Repeat copy k -> Residual child 0 -> Sequential
        k, tail2 = tail.split(".", 1)
        assert tail2.startswith("0."), tail2
        p, t = _ln_or_lin(tail2[2:])
        return f"{1 + int(k)}." + p, t
    if i == "2":
        p, t = _ln_or_lin(tail)
        return f"{L - 1}." + p, t
    raise KeyError(rest)


def _trunk(rest, L, torch_prefix):
    """Our Sequential index 1 = Repeat[L, ResBlock], 2 = BlockLinear -> torch `<prefix>.<k>.mlp.*`."""
    i, tail = rest.split(".", 1)
    if i == "1":
        k, tail2 = tail.split(".", 1)
        assert tail2.startswith("0."), tail2
        p, t = _ln_or_lin(tail2[2:])
        return f"{torch_prefix}.{int(k)}." + p, t
    if i == "2":
        p, t = _ln_or_lin(tail)
        return f"{torch_prefix}.{L}." + p, t
    raise KeyError(rest)


def _our_to_torch(kind, name, L):
    if kind in ("linmish", "linrelu", "lintanh"):
        pname = name.split(".")[-1]
        return "0." + pname, pname == "weight"
    if kind == "block":
        return _ln_or_lin(name)
    if kind == "resblock":
        assert name.startswith("0."), name
        return _ln_or_lin(name[2:])
    if kind == "repeat2":
        k, tail = name.split(".", 1)
        assert tail.startswith("0."), tail
        p, t = _ln_or_lin(tail[2:])
        return f"{int(k)}." + p, t
    if kind == "slices":
        i, pname = name.split(".", 1)
        assert i == "1", name
        return "lin." + pname, pname == "weight"
    if kind == "parseq":
        if name.startswith("0.1."):
            pname = name[len("0.1."):]
            return "lin1." + pname, pname == "weight"
        i, pname = name.split(".", 1)
        assert i == "1", name
        return "lin2." + pname, pname == "weight"
    if kind == "f":
        if name.startswith("0.0.1."):
            p, t = _embed(name[len("0.0.1."):], L)
            return "embed_sa." + p, t
        if name.startswith("0.1.1."):
            p, t = _embed(name[len("0.1.1."):], L)
            return "embed_z." + p, t
        return _trunk(name, L, "Fs")
    if kind == "actor":
        if name.startswith("0.0.1."):
            p, t = _embed(name[len("0.0.1."):], 2)
            return "embed_s." + p, t
        if name.startswith("0.1."):
            p, t = _embed(name[len("0.1."):], 2)
            return "embed_z." + p, t
        return _trunk(name, L, "policy")
    if kind == "b":
        # Sequential[Linear, LayerNorm, Tanh, Linear, StopGradParams[RMSNorm]] vs
        # torch Sequential[Linear, LayerNorm, Tanh, Linear, Norm]; RMSNorm's frozen
        # gamma has no torch counterpart.
        i, pname = name.split(".", 1)
        if i == "0":
            return "net.0." + pname, pname == "weight"
        if i == "1":
            return "net.1." + {"gamma": "weight", "beta": "bias"}[pname], False
        if i == "3":
            return "net.3." + pname, pname == "weight"
        if i == "4":
            return None, False  # frozen RMSNorm gamma (all ones)
        raise KeyError(name)
    raise ValueError(kind)


def load_ours(model, kind, dims, names, values, sizes):
    """Set the torch model's params from OUR flat params. Returns the number set."""
    L = int(dims["l"]) if "l" in dims else 0
    names = [str(n) for n in names]
    values = np.asarray([float(v) for v in values], dtype=np.float64)
    sizes = [int(s) for s in sizes]
    params = dict(model.named_parameters())
    seen = set()
    off = 0
    n_set = 0
    for name, size in zip(names, sizes):
        flat = values[off : off + size]
        off += size
        tname, transpose = _our_to_torch(kind, name, L)
        if tname is None:
            assert np.allclose(flat, 1.0), f"{name}: frozen RMSNorm gamma is not all ones"
            continue
        p = params[tname]
        if transpose:
            out_dim, in_dim = p.shape
            w = flat.reshape(in_dim, out_dim).T
        else:
            w = flat.reshape(p.shape)
        with torch.no_grad():
            p.copy_(torch.from_numpy(np.ascontiguousarray(w)))
        seen.add(tname)
        n_set += 1
    missing = sorted(set(params) - seen)
    assert not missing, f"torch params never set from ours: {missing}"
    assert off == values.shape[0], (off, values.shape)
    return n_set


def run(model, kind, dims, names, sizes, x, batch, grad_out):
    """Forward on `x` [batch, in], backward with `grad_out` [batch, out].
    Returns (out flat, grad_in flat, param grads flat in OUR names/layout/order)."""
    L = int(dims["l"]) if "l" in dims else 0
    names = [str(n) for n in names]
    sizes = [int(s) for s in sizes]
    x = torch.tensor(np.asarray([float(v) for v in x], dtype=np.float64).reshape(int(batch), -1), requires_grad=True)
    go = torch.tensor(np.asarray([float(v) for v in grad_out], dtype=np.float64).reshape(int(batch), -1))
    model.zero_grad(set_to_none=True)
    out = model(x)
    out.backward(go)
    params = dict(model.named_parameters())
    grads = []
    for name, size in zip(names, sizes):
        tname, transpose = _our_to_torch(kind, name, L)
        if tname is None:
            grads.extend([0.0] * size)
            continue
        g = params[tname].grad.detach().numpy()
        if transpose:
            g = g.T
        grads.extend(np.ascontiguousarray(g).reshape(-1).tolist())
    return out.detach().numpy().reshape(-1).tolist(), x.grad.detach().numpy().reshape(-1).tolist(), grads
