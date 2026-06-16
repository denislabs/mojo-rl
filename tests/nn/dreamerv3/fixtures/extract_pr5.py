"""PR-5a fixture: distributional heads + AC losses ground truth.

Runs the **actual** reference `embodied/jax/outs.py` (TwoHot, Normal) +
`dreamerv3/agent.py:imag_loss/repl_loss` + `embodied/jax/utils.py:Normalize`
(via ninjax), COMPUTE_DTYPE irrelevant here (outs work in f32).

Dumps, for the Mojo side to match ≤1e-4:
  * TwoHot:  bins, logits, pred, loss-vs-target
  * Normal (bounded_normal policy): logp(act), entropy
  * imag_loss: policy_loss, value_loss, ret  (update=True, real norm EMA)
  * repl_loss: repval_loss, ret

Run: /tmp/dreamer_fixtures_venv/bin/python3 tests/nn2/dreamerv3/fixtures/extract_pr5.py
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

rng = np.random.default_rng(11)


def rnd(*shape):
    return jnp.asarray(rng.standard_normal(shape), jnp.float32)


def symexp(x):
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def twohot_bins(bins):
    if bins % 2 == 1:
        half = jnp.linspace(-20, 0, (bins - 1) // 2 + 1, dtype=jnp.float32)
        half = symexp(half)
        return jnp.concatenate([half, -half[:-1][::-1]], 0)
    half = jnp.linspace(-20, 0, bins // 2, dtype=jnp.float32)
    half = symexp(half)
    return jnp.concatenate([half, -half[::-1]], 0)


# ── dims ────────────────────────────────────────────────────────────────
BK = 2          # batch (B·K)
H = 3           # imag_length → time = H+1
T = H + 1
ACT = 1         # Pendulum
BINS = 41          # tame bins for the distributional-math fixture
MINSTD = 0.1
MAXSTD = 1.0
HORIZON = 333
LAM = 0.95
ACTENT = 3e-4
SLOWREG = 1.0

# Tame moderate bins for the distributional-math parity (TwoHot accepts any
# bins). The production symexp bins (range ±4.8e8) make absolute parity
# meaningless and the symmetric-sum order won't match JAX's tree reduction —
# we validate the bin GENERATION separately via `sbins255`.
bins = jnp.linspace(-10.0, 10.0, BINS).astype(jnp.float32)

out_lines = []


def emit_scalar(k, v):
    out_lines.append(f"{k}={float(v):.9g}")


def emit_flat(name, arr):
    a = np.asarray(arr, np.float64).reshape(-1)
    out_lines.append(f"{name}#size={a.size}")
    for v in a:
        out_lines.append(f"{v:.9g}")


emit_scalar("dims.BK", BK)
emit_scalar("dims.H", H)
emit_scalar("dims.T", T)
emit_scalar("dims.ACT", ACT)
emit_scalar("dims.BINS", BINS)
emit_scalar("cfg.minstd", MINSTD)
emit_scalar("cfg.maxstd", MAXSTD)
emit_scalar("cfg.horizon", HORIZON)
emit_scalar("cfg.lam", LAM)
emit_scalar("cfg.actent", ACTENT)
emit_scalar("cfg.slowreg", SLOWREG)
emit_flat("bins", bins)
# Production symexp bins (255) for a Mojo bin-generation parity check.
emit_flat("sbins255", twohot_bins(255))

# ── TwoHot standalone: pred + loss ─────────────────────────────────────
th_logits = rnd(BK, T)  # treat (BK,T) as batch, BINS lanes appended
th_logits = rnd(BK * T, BINS)
th = outs.TwoHot(th_logits, bins)
th_pred = th.pred()                       # [BK*T]
th_target = jnp.asarray(rng.uniform(-8, 8, (BK * T,)), jnp.float32)
th_loss = th.loss(th_target)              # [BK*T]
emit_flat("th.logits", th_logits)
emit_flat("th.pred", th_pred)
emit_flat("th.target", th_target)
emit_flat("th.loss", th_loss)

# ── Normal (bounded_normal) standalone: logp + entropy ─────────────────
nm_mean_raw = rnd(BK * T, ACT)
nm_std_raw = rnd(BK * T, ACT)
nm_std = (MAXSTD - MINSTD) * jax.nn.sigmoid(nm_std_raw + 2.0) + MINSTD
nm = outs.Normal(jnp.tanh(nm_mean_raw), nm_std)
nm_act = jnp.asarray(rng.uniform(-0.9, 0.9, (BK * T, ACT)), jnp.float32)
nm_logp = nm.logp(nm_act)                 # [BK*T, ACT]
nm_ent = nm.entropy()                     # [BK*T, ACT]
emit_flat("nm.mean_raw", nm_mean_raw)
emit_flat("nm.std_raw", nm_std_raw)
emit_flat("nm.act", nm_act)
emit_flat("nm.logp", nm_logp)
emit_flat("nm.entropy", nm_ent)

# ── imag_loss (update=True, real perc EMA from fresh state) ─────────────
il_act = {"action": jnp.asarray(rng.uniform(-0.9, 0.9, (BK, T, ACT)), jnp.float32)}
il_rew = rnd(BK, T)
il_con = jnp.asarray(rng.uniform(0.5, 1.0, (BK, T)), jnp.float32)  # cont prob
il_vlogits = rnd(BK, T, BINS)
il_svlogits = rnd(BK, T, BINS)
il_pmean = rnd(BK, T, ACT)
il_pstd_raw = rnd(BK, T, ACT)


def il_fn(act, rew, con, vlogits, svlogits, pmean, pstd_raw):
    retnorm = Normalize("perc", rate=0.01, perclo=5.0, perchi=95.0,
                        limit=1.0, debias=False, name="retnorm")
    valnorm = Normalize("none", rate=0.01, limit=1e-8, name="valnorm")
    advnorm = Normalize("none", rate=0.01, limit=1e-8, name="advnorm")
    value = outs.TwoHot(vlogits, bins)
    slowval = outs.TwoHot(svlogits, bins)
    std = (MAXSTD - MINSTD) * jax.nn.sigmoid(pstd_raw + 2.0) + MINSTD
    # Head wraps a shaped (ACT,) space in Agg(., 1, sum) → logp/entropy
    # are summed over the action dim.
    policy = {"action": outs.Agg(outs.Normal(jnp.tanh(pmean), std), 1, jnp.sum)}
    return imag_loss(
        act, rew, con, policy, value, slowval,
        retnorm, valnorm, advnorm, update=True,
        contdisc=True, slowtar=False, horizon=HORIZON,
        lam=LAM, actent=ACTENT, slowreg=SLOWREG,
    )


il_args = (il_act, il_rew, il_con, il_vlogits, il_svlogits, il_pmean, il_pstd_raw)
state, _ = nj.pure(il_fn)({}, *il_args, seed=0, create=True, modify=True,
                          ignore=True)
_, (il_losses, il_outs, _) = nj.pure(il_fn)(state, *il_args, seed=0)
emit_flat("il.act", il_act["action"])
emit_flat("il.rew", il_rew)
emit_flat("il.con", il_con)
emit_flat("il.vlogits", il_vlogits)
emit_flat("il.svlogits", il_svlogits)
emit_flat("il.pmean", il_pmean)
emit_flat("il.pstd_raw", il_pstd_raw)
emit_flat("il.policy_loss", il_losses["policy"])   # [BK, T-1]
emit_flat("il.value_loss", il_losses["value"])     # [BK, T-1]
emit_flat("il.ret", il_outs["ret"])                # [BK, T-1]

# ── repl_loss ──────────────────────────────────────────────────────────
rl_last_bool = (rng.uniform(0, 1, (BK, T)) < 0.2)
rl_last = jnp.asarray(rl_last_bool, bool)
rl_term = jnp.zeros((BK, T), jnp.float32)
rl_rew = rnd(BK, T)
rl_boot = rnd(BK, T)
rl_vlogits = rnd(BK, T, BINS)
rl_svlogits = rnd(BK, T, BINS)


def rl_fn(last, term, rew, boot, vlogits, svlogits):
    valnorm = Normalize("none", rate=0.01, limit=1e-8, name="valnorm")
    value = outs.TwoHot(vlogits, bins)
    slowval = outs.TwoHot(svlogits, bins)
    return repl_loss(
        last, term, rew, boot, value, slowval, valnorm,
        update=True, slowreg=SLOWREG, slowtar=False, horizon=HORIZON, lam=LAM,
    )


rl_args = (rl_last, rl_term, rl_rew, rl_boot, rl_vlogits, rl_svlogits)
state, _ = nj.pure(rl_fn)({}, *rl_args, seed=0, create=True, modify=True,
                          ignore=True)
_, (rl_losses, rl_outs, _) = nj.pure(rl_fn)(state, *rl_args, seed=0)
emit_flat("rl.last", rl_last)
emit_flat("rl.term", rl_term)
emit_flat("rl.rew", rl_rew)
emit_flat("rl.boot", rl_boot)
emit_flat("rl.vlogits", rl_vlogits)
emit_flat("rl.svlogits", rl_svlogits)
emit_flat("rl.repval", rl_losses["repval"])        # [BK, T-1]
emit_flat("rl.ret", rl_outs["ret"])                # [BK, T-1]

# ── SlowModel polyak (rate=0.02, every=1) over a flat param ────────────
sm_src = rnd(6)
sm_init = rnd(6)
sm_rate = 0.02
sm = np.asarray(sm_init, np.float64).copy()
src = np.asarray(sm_src, np.float64)
for _ in range(50):
    sm = sm_rate * src + (1 - sm_rate) * sm
emit_scalar("sm.rate", sm_rate)
emit_flat("sm.src", sm_src)
emit_flat("sm.init", sm_init)
emit_flat("sm.after50", sm)

path = os.path.join(_HERE, "pr5_fixture.txt")
with open(path, "w") as f:
    f.write("\n".join(out_lines) + "\n")
print("wrote", path)
print("th.pred[:3]", np.asarray(th_pred)[:3])
print("il.policy_loss", np.asarray(il_losses["policy"]).reshape(-1))
print("il.value_loss", np.asarray(il_losses["value"]).reshape(-1))
print("il.ret", np.asarray(il_outs["ret"]).reshape(-1))
print("rl.repval", np.asarray(rl_losses["repval"]).reshape(-1))
