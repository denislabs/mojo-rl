"""PR-4 fixture: RSSM + Encoder + Decoder forward/loss ground truth.

Runs the **actual** Hafner reference (`references/dreamerv3-main/dreamerv3/
rssm.py` + `embodied/jax/{nets,outs,heads}.py`) through ninjax, with
`COMPUTE_DTYPE` forced to float32 for a clean (non-bf16) parity target.

We dump every parameter by its ninjax path, the random inputs, and the
reference forward outputs for the deterministic building blocks:
  * `_core(deter, stoch, action) -> new_deter`            (BlockLinear GRU)
  * `_prior(deter) -> logit`                              (prior MLP + head)
  * `_observe` obslogit + new_deter (reset=False, single step)
  * dyn/rep KL (reference outs.OneHot/Agg) given post/prior logits
  * Encoder MLP tokens
  * Decoder symlog_mse head pred + per-row recon loss

The Mojo side (rssm.mojo / encoder.mojo / decoder.mojo) loads the dumped
params, runs its own forward, and asserts <= 1e-4. This is the PR-4 gate.

Run:
  /tmp/dreamer_fixtures_venv/bin/python3 tests/nn2/dreamerv3/fixtures/extract_pr4.py
(must be cwd=references/dreamerv3-main OR pass that on sys.path)
"""

import os
import sys

# Locate the reference repo regardless of cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
_REF = os.path.join(_ROOT, "references", "dreamerv3-main")
sys.path.insert(0, _REF)

import numpy as np
import jax
import jax.numpy as jnp
import embodied.jax.nets as nn

nn.COMPUTE_DTYPE = jnp.float32  # clean f32 ground truth (not bf16)

import ninjax as nj
import elements
from dreamerv3 import rssm as R
from embodied.jax import outs

sg = jax.lax.stop_gradient

# ── test dims (small, but exercise every path: blocks>1, multi-layer) ──
B = 2
DETER = 16
HIDDEN = 12
STOCH = 3
CLASSES = 5
BLOCKS = 4
ACT = 2
TOKEN = 8          # = encoder units
OBS = 4
ENC_UNITS = 8
ENC_LAYERS = 2
DEC_UNITS = 8
DEC_LAYERS = 2
UNIMIX = 0.01
FREE_NATS = 1.0
OBSLAYERS = 1
IMGLAYERS = 2
DYNLAYERS = 1

rng = np.random.default_rng(7)


def rnd(*shape):
    return jnp.asarray(rng.standard_normal(shape), jnp.float32)


# ── modules ────────────────────────────────────────────────────────────
act_space = {"action": elements.Space(np.float32, (ACT,))}
obs_space = {"vec": elements.Space(np.float32, (OBS,))}

rssm = R.RSSM(
    act_space, deter=DETER, hidden=HIDDEN, stoch=STOCH, classes=CLASSES,
    blocks=BLOCKS, obslayers=OBSLAYERS, imglayers=IMGLAYERS,
    dynlayers=DYNLAYERS, unimix=UNIMIX, free_nats=FREE_NATS, name="rssm",
)
enc = R.Encoder(
    obs_space, units=ENC_UNITS, layers=ENC_LAYERS, symlog=True, name="enc",
)
dec = R.Decoder(
    obs_space, units=DEC_UNITS, layers=DEC_LAYERS, symlog=True, name="dec",
)

# ── inputs ─────────────────────────────────────────────────────────────
deter0 = rnd(B, DETER)
stoch0 = rnd(B, STOCH, CLASSES)
action = rnd(B, ACT)
tokens = rnd(B, TOKEN)
obs_vec = rnd(B, OBS)
dec_feat = {"deter": rnd(B, DETER), "stoch": rnd(B, STOCH, CLASSES)}
recon_target = rnd(B, OBS)
reset = jnp.zeros((B,), bool)


# ── pure fns ───────────────────────────────────────────────────────────
def core_fn(deter, stoch, action):
    return rssm._core(deter, stoch, action)


def prior_fn(deter):
    return rssm._prior(deter)


def observe_fn(deter, stoch, action, tokens):
    carry = dict(deter=deter, stoch=stoch)
    act_d = {"action": action}
    carry2, (entry, feat) = rssm._observe(carry, tokens, act_d, reset, True)
    return feat["deter"], feat["logit"]


def enc_fn(obs_vec):
    carry, entries, tok = enc({}, {"vec": obs_vec}, reset, True, single=True)
    return tok


def dec_fn(feat):
    carry, entries, recons = dec({}, feat, reset, True, single=True)
    return recons["vec"].pred()


# ── init + apply, collecting params from each ──────────────────────────
def run(fn, *args):
    state, _ = nj.pure(fn)({}, *args, seed=0, create=True, modify=True,
                           ignore=True)
    state2, out = nj.pure(fn)(state, *args, seed=0)
    return state, out


core_params, core_out = run(core_fn, deter0, stoch0, action)
prior_params, prior_out = run(prior_fn, deter0)
obs_params, (obs_deter, obs_logit) = run(observe_fn, deter0, stoch0, action,
                                         tokens)
enc_params, enc_tok = run(enc_fn, obs_vec)
dec_params, dec_pred = run(dec_fn, dec_feat)

# RSSM params: _core gives dynin*/dynhid*/dyngru; _prior gives prior*/
# priorlogit; _observe gives obs*/obslogit (+ the core ones again). Merge.
rssm_params = {}
rssm_params.update(core_params)
rssm_params.update(prior_params)
rssm_params.update(obs_params)

# ── loss: dyn/rep on (post=obs_logit, prior=_prior(obs_deter)) ─────────
prior_on_obsdeter = nj.pure(prior_fn)(prior_params, obs_deter, seed=0)[1]


def dist(logit):
    return outs.Agg(outs.OneHot(logit, UNIMIX), 1, jnp.sum)


post = obs_logit
prior_l = prior_on_obsdeter
dyn = dist(sg(post)).kl(dist(prior_l))     # [B], grad->prior
rep = dist(post).kl(dist(sg(prior_l)))     # [B], grad->post
dyn = np.maximum(np.asarray(dyn), FREE_NATS)
rep = np.maximum(np.asarray(rep), FREE_NATS)

# ── decoder recon loss: sum_o (pred - symlog(target))^2 ────────────────
recon_loss = np.asarray(
    jnp.square(dec_pred - nn.symlog(recon_target)).sum(-1)
)


# ── dump ───────────────────────────────────────────────────────────────
out_lines = []


def emit_scalar(key, v):
    out_lines.append(f"{key}={float(v):.9g}")


def emit_flat(name, arr):
    a = np.asarray(arr, np.float64).reshape(-1)
    out_lines.append(f"{name}#size={a.size}")
    for v in a:
        out_lines.append(f"{v:.9g}")


emit_scalar("dims.B", B)
emit_scalar("dims.DETER", DETER)
emit_scalar("dims.HIDDEN", HIDDEN)
emit_scalar("dims.STOCH", STOCH)
emit_scalar("dims.CLASSES", CLASSES)
emit_scalar("dims.BLOCKS", BLOCKS)
emit_scalar("dims.ACT", ACT)
emit_scalar("dims.TOKEN", TOKEN)
emit_scalar("dims.OBS", OBS)
emit_scalar("dims.ENC_UNITS", ENC_UNITS)
emit_scalar("dims.ENC_LAYERS", ENC_LAYERS)
emit_scalar("dims.DEC_UNITS", DEC_UNITS)
emit_scalar("dims.DEC_LAYERS", DEC_LAYERS)
emit_scalar("cfg.unimix", UNIMIX)
emit_scalar("cfg.free_nats", FREE_NATS)

# params (by ninjax path) — flatten row-major (matches JAX .reshape(-1))
for k in sorted(rssm_params):
    emit_flat("p." + k, rssm_params[k])
for k in sorted(enc_params):
    emit_flat("p." + k, enc_params[k])
for k in sorted(dec_params):
    emit_flat("p." + k, dec_params[k])

# inputs
emit_flat("in.deter0", deter0)
emit_flat("in.stoch0", stoch0)
emit_flat("in.action", action)
emit_flat("in.tokens", tokens)
emit_flat("in.obs_vec", obs_vec)
emit_flat("in.dec_deter", dec_feat["deter"])
emit_flat("in.dec_stoch", dec_feat["stoch"])
emit_flat("in.recon_target", recon_target)

# outputs
emit_flat("out.core", core_out)
emit_flat("out.prior", prior_out)
emit_flat("out.obs_deter", obs_deter)
emit_flat("out.obs_logit", obs_logit)
emit_flat("out.enc_tok", enc_tok)
emit_flat("out.dec_pred", dec_pred)
emit_flat("out.dyn", dyn)
emit_flat("out.rep", rep)
emit_flat("out.recon_loss", recon_loss)

fixture_path = os.path.join(_HERE, "pr4_fixture.txt")
with open(fixture_path, "w") as f:
    f.write("\n".join(out_lines) + "\n")

print("wrote", fixture_path)
print("param paths:")
for k in sorted(rssm_params):
    print("  rssm", k, tuple(np.asarray(rssm_params[k]).shape))
for k in sorted(enc_params):
    print("  enc ", k, tuple(np.asarray(enc_params[k]).shape))
for k in sorted(dec_params):
    print("  dec ", k, tuple(np.asarray(dec_params[k]).shape))
print("core_out[0,:4]", np.asarray(core_out)[0, :4])
print("dyn", dyn, "rep", rep, "recon_loss", recon_loss)
