"""PR-5b fixture: AC + value-head loss BACKWARD ground truth (jax.vjp).

The actor-critic gradient is the most important backward and is a pure
function of the network outputs, so we validate it with `jax.vjp` of the
ACTUAL reference `imag_loss`/`repl_loss` + the twohot CE — no forward-cache
restructuring needed (that's the WM-module vjp, PR5b part 2).

Cotangent = ones on (policy_loss, value_loss) / (repval) → gradients w.r.t.
the trainable logits: value `vlogits`, policy `pmean`/`pstd_raw`.

Also dumps `rscale` (retnorm scale after the EMA update) so the Mojo
backward uses the identical scalar (retnorm is sg'd → constant w.r.t. grad).

Run: /tmp/dreamer_fixtures_venv/bin/python3 tests/nn2/dreamerv3/fixtures/extract_pr5b.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "references", "dreamerv3-main"))

import numpy as np
import jax
import jax.numpy as jnp
import ninjax as nj
from embodied.jax import outs
from embodied.jax.utils import Normalize
from dreamerv3.agent import imag_loss, repl_loss

rng = np.random.default_rng(11)  # SAME seed as extract_pr5 → identical inputs


def rnd(*shape):
    return jnp.asarray(rng.standard_normal(shape), jnp.float32)


BK, H, T, ACT, BINS = 2, 3, 4, 1, 41
MINSTD, MAXSTD, HORIZON, LAM, ACTENT, SLOWREG = 0.1, 1.0, 333, 0.95, 3e-4, 1.0
bins = jnp.linspace(-10.0, 10.0, BINS).astype(jnp.float32)

out_lines = []


def emit_scalar(k, v):
    out_lines.append(f"{k}={float(v):.9g}")


def emit_flat(name, arr):
    a = np.asarray(arr, np.float64).reshape(-1)
    out_lines.append(f"{name}#size={a.size}")
    for v in a:
        out_lines.append(f"{v:.9g}")


# ── reproduce extract_pr5.py's draw order EXACTLY so inputs match ───────
# (th, nm sections consume rng first in extract_pr5; replicate to align.)
_ = rng.standard_normal((BK, T))          # th_logits (first rnd, unused name)
_ = rng.standard_normal((BK * T, BINS))   # th_logits
_ = rng.uniform(-8, 8, (BK * T,))         # th_target
_ = rng.standard_normal((BK * T, ACT))    # nm_mean_raw
_ = rng.standard_normal((BK * T, ACT))    # nm_std_raw
_ = rng.uniform(-0.9, 0.9, (BK * T, ACT)) # nm_act

il_act = {"action": jnp.asarray(rng.uniform(-0.9, 0.9, (BK, T, ACT)), jnp.float32)}
il_rew = rnd(BK, T)
il_con = jnp.asarray(rng.uniform(0.5, 1.0, (BK, T)), jnp.float32)
il_vlogits = rnd(BK, T, BINS)
il_svlogits = rnd(BK, T, BINS)
il_pmean = rnd(BK, T, ACT)
il_pstd_raw = rnd(BK, T, ACT)


def make_norms():
    retnorm = Normalize("perc", rate=0.01, perclo=5.0, perchi=95.0,
                        limit=1.0, debias=False, name="retnorm")
    valnorm = Normalize("none", rate=0.01, limit=1e-8, name="valnorm")
    advnorm = Normalize("none", rate=0.01, limit=1e-8, name="advnorm")
    return retnorm, valnorm, advnorm


def il_fn(vlogits, pmean, pstd_raw):
    retnorm, valnorm, advnorm = make_norms()
    value = outs.TwoHot(vlogits, bins)
    slowval = outs.TwoHot(il_svlogits, bins)
    std = (MAXSTD - MINSTD) * jax.nn.sigmoid(pstd_raw + 2.0) + MINSTD
    policy = {"action": outs.Agg(outs.Normal(jnp.tanh(pmean), std), 1, jnp.sum)}
    los, _, _ = imag_loss(
        il_act, il_rew, il_con, policy, value, slowval,
        retnorm, valnorm, advnorm, update=True,
        contdisc=True, slowtar=False, horizon=HORIZON,
        lam=LAM, actent=ACTENT, slowreg=SLOWREG,
    )
    return los["policy"], los["value"]


# init nj state, then take rscale + vjp grads (state captured as constant).
args = (il_vlogits, il_pmean, il_pstd_raw)
state, _ = nj.pure(il_fn)({}, *args, seed=0, create=True, modify=True,
                          ignore=True)


def pure_g(vlogits, pmean, pstd_raw):
    _, out = nj.pure(il_fn)(state, vlogits, pmean, pstd_raw, seed=0)
    return out


(pol, val), vjp_fn = jax.vjp(pure_g, *args)
g_vlogits, g_pmean, g_pstd = vjp_fn((jnp.ones_like(pol), jnp.ones_like(val)))

# rscale: retnorm scale after the update (debias False → max(limit, hi-lo))
state2, _ = nj.pure(il_fn)(state, *args, seed=0)
lo = float(np.asarray(state2["retnorm/lo/value"]))
hi = float(np.asarray(state2["retnorm/hi/value"]))
rscale = max(1.0, hi - lo)

emit_scalar("dims.BK", BK)
emit_scalar("dims.T", T)
emit_scalar("dims.ACT", ACT)
emit_scalar("dims.BINS", BINS)
emit_scalar("cfg.minstd", MINSTD)
emit_scalar("cfg.maxstd", MAXSTD)
emit_scalar("cfg.lam", LAM)
emit_scalar("cfg.actent", ACTENT)
emit_scalar("cfg.slowreg", SLOWREG)
emit_scalar("cfg.horizon", HORIZON)
emit_scalar("il.rscale", rscale)
emit_flat("bins", bins)
emit_flat("il.act", il_act["action"])
emit_flat("il.rew", il_rew)
emit_flat("il.con", il_con)
emit_flat("il.vlogits", il_vlogits)
emit_flat("il.svlogits", il_svlogits)
emit_flat("il.pmean", il_pmean)
emit_flat("il.pstd_raw", il_pstd_raw)
emit_flat("il.g_vlogits", g_vlogits)
emit_flat("il.g_pmean", g_pmean)
emit_flat("il.g_pstd_raw", g_pstd)

# ── repl_loss backward (grad w.r.t. vlogits) ───────────────────────────
rl_last_bool = (rng.uniform(0, 1, (BK, T)) < 0.2)
rl_last = jnp.asarray(rl_last_bool, bool)
rl_term = jnp.zeros((BK, T), jnp.float32)
rl_rew = rnd(BK, T)
rl_boot = rnd(BK, T)
rl_vlogits = rnd(BK, T, BINS)
rl_svlogits = rnd(BK, T, BINS)


def rl_fn(vlogits):
    valnorm = Normalize("none", rate=0.01, limit=1e-8, name="valnorm")
    value = outs.TwoHot(vlogits, bins)
    slowval = outs.TwoHot(rl_svlogits, bins)
    los, _, _ = repl_loss(
        rl_last, rl_term, rl_rew, rl_boot, value, slowval, valnorm,
        update=True, slowreg=SLOWREG, slowtar=False, horizon=HORIZON, lam=LAM,
    )
    return los["repval"]


rstate, _ = nj.pure(rl_fn)({}, rl_vlogits, seed=0, create=True, modify=True,
                           ignore=True)


def rl_pure(vlogits):
    _, out = nj.pure(rl_fn)(rstate, vlogits, seed=0)
    return out


rv, rl_vjp = jax.vjp(rl_pure, rl_vlogits)
(g_rl_vlogits,) = rl_vjp(jnp.ones_like(rv))
emit_flat("rl.last", rl_last)
emit_flat("rl.term", rl_term)
emit_flat("rl.rew", rl_rew)
emit_flat("rl.boot", rl_boot)
emit_flat("rl.vlogits", rl_vlogits)
emit_flat("rl.svlogits", rl_svlogits)
emit_flat("rl.g_vlogits", g_rl_vlogits)

# ── standalone twohot CE backward (grad w.r.t. logits) ─────────────────
th_logits = rnd(BK * T, BINS)
th_target = jnp.asarray(rng.uniform(-8, 8, (BK * T,)), jnp.float32)


def th_fn(logits):
    return outs.TwoHot(logits, bins).loss(th_target)


thl, th_vjp = jax.vjp(th_fn, th_logits)
(g_th,) = th_vjp(jnp.ones_like(thl))
emit_flat("th.logits", th_logits)
emit_flat("th.target", th_target)
emit_flat("th.g_logits", g_th)

path = os.path.join(_HERE, "pr5b_fixture.txt")
with open(path, "w") as f:
    f.write("\n".join(out_lines) + "\n")
print("wrote", path)
print("rscale", rscale)
print("g_vlogits[0,0,:4]", np.asarray(g_vlogits).reshape(BK, T, BINS)[0, 0, :4])
print("g_pmean", np.asarray(g_pmean).reshape(-1))
print("g_pstd_raw", np.asarray(g_pstd).reshape(-1))
print("rl.g_vlogits sum", float(np.asarray(g_rl_vlogits).sum()))
