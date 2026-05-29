"""PR-5b part 2 fixture: WM-module BACKWARD ground truth (jax.vjp).

Reconstructs the SAME Encoder/Decoder/RSSM as `extract_pr4.py` (seed 7,
nj-init seed 0, COMPUTE_DTYPE=f32) so params/inputs are identical to
`pr4_fixture.txt` — the Mojo test loads params/inputs from pr4 and only
reads the new gradients here.

Dumps `jax.vjp` gradients w.r.t. params (by ninjax path) + inputs:
  * Encoder: cotangent `g_tok` (random) → param grads + grad_obs
  * Decoder: recon-loss cotangent ones → param grads + grad feat (deter/stoch)
  * RSSM `_core`: cotangent on new_deter → all dyn* param grads + grad inputs
  * RSSM `_prior`: cotangent on logit → prior* param grads + grad_deter

Run: /tmp/dreamer_fixtures_venv/bin/python3 tests/nn2/dreamerv3/fixtures/extract_pr5b2.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "references", "dreamerv3-main"))

import numpy as np
import jax
import jax.numpy as jnp
import embodied.jax.nets as nn

nn.COMPUTE_DTYPE = jnp.float32

import ninjax as nj
import elements
from dreamerv3 import rssm as R

# dims — identical to extract_pr4.py
B, DETER, HIDDEN, STOCH, CLASSES, BLOCKS, ACT = 2, 16, 12, 3, 5, 4, 2
TOKEN, OBS, ENC_UNITS, ENC_LAYERS, DEC_UNITS, DEC_LAYERS = 8, 4, 8, 2, 8, 2
SC = STOCH * CLASSES

rng = np.random.default_rng(7)  # same as pr4 → same inputs


def rnd(*shape):
    return jnp.asarray(rng.standard_normal(shape), jnp.float32)


act_space = {"action": elements.Space(np.float32, (ACT,))}
obs_space = {"vec": elements.Space(np.float32, (OBS,))}
rssm = R.RSSM(act_space, deter=DETER, hidden=HIDDEN, stoch=STOCH,
              classes=CLASSES, blocks=BLOCKS, obslayers=1, imglayers=2,
              dynlayers=1, unimix=0.01, free_nats=1.0, name="rssm")
enc = R.Encoder(obs_space, units=ENC_UNITS, layers=ENC_LAYERS, symlog=True,
                name="enc")
dec = R.Decoder(obs_space, units=DEC_UNITS, layers=DEC_LAYERS, symlog=True,
                name="dec")

# inputs (same draw order as pr4)
deter0 = rnd(B, DETER)
stoch0 = rnd(B, STOCH, CLASSES)
action = rnd(B, ACT)
tokens = rnd(B, TOKEN)
obs_vec = rnd(B, OBS)
dec_feat = {"deter": rnd(B, DETER), "stoch": rnd(B, STOCH, CLASSES)}
recon_target = rnd(B, OBS)
reset = jnp.zeros((B,), bool)

out_lines = []


def emit_flat(name, arr):
    a = np.asarray(arr, np.float64).reshape(-1)
    out_lines.append(f"{name}#size={a.size}")
    for v in a:
        out_lines.append(f"{v:.9g}")


def init(fn, *args):
    state, _ = nj.pure(fn)({}, *args, seed=0, create=True, modify=True,
                           ignore=True)
    return state


# ── Encoder backward ───────────────────────────────────────────────────
def enc_fwd(obs_vec, state_params):
    carry, _, tok = enc({}, {"vec": obs_vec}, reset, True, single=True)
    return tok


enc_state = init(lambda obs: enc({}, {"vec": obs}, reset, True, single=True)[2],
                 obs_vec)


def enc_g(state, obs):
    _, tok = nj.pure(
        lambda o: enc({}, {"vec": o}, reset, True, single=True)[2]
    )(state, obs, seed=0)
    return tok


g_tok = rnd(B, ENC_UNITS)
(_tok, enc_vjp) = jax.vjp(enc_g, enc_state, obs_vec)
d_state, d_obs = enc_vjp(g_tok)
emit_flat("enc.g_tok", g_tok)
emit_flat("enc.g_obs", d_obs)
for k in sorted(d_state):
    emit_flat("genc." + k.split("/", 1)[1], d_state[k])  # strip 'enc/'

# ── Decoder backward (recon loss) ──────────────────────────────────────
def dec_recon(state, deter, stoch):
    feat = {"deter": deter, "stoch": stoch}
    _, recons = nj.pure(
        lambda f: dec({}, f, reset, True, single=True)[2]
    )(state, feat, seed=0)
    out = recons["vec"]
    return out.loss(recon_target)   # [B]  (Agg sum over OBS)


dec_state = init(lambda f: dec({}, f, reset, True, single=True)[2], dec_feat)
loss, dec_vjp = jax.vjp(dec_recon, dec_state, dec_feat["deter"],
                        dec_feat["stoch"])
d_dstate, d_deter, d_stoch = dec_vjp(jnp.ones_like(loss))
emit_flat("dec.recon_loss", loss)
emit_flat("dec.g_deter", d_deter)
emit_flat("dec.g_stoch", d_stoch)
for k in sorted(d_dstate):
    emit_flat("gdec." + k.split("/", 1)[1], d_dstate[k])  # strip 'dec/'

# ── RSSM _core backward ────────────────────────────────────────────────
core_state = init(lambda d, s, a: rssm._core(d, s, a), deter0, stoch0, action)


def core_g(state, deter, stoch, act):
    _, out = nj.pure(lambda d, s, a: rssm._core(d, s, a))(
        state, deter, stoch, act, seed=0)
    return out


g_core = rnd(B, DETER)
(_c, core_vjp) = jax.vjp(core_g, core_state, deter0, stoch0, action)
d_cstate, d_cdeter, d_cstoch, d_caction = core_vjp(g_core)
emit_flat("core.g_out", g_core)
emit_flat("core.g_deter", d_cdeter)
emit_flat("core.g_stoch", d_cstoch)
emit_flat("core.g_action", d_caction)
for k in sorted(d_cstate):
    emit_flat("gcore." + k.split("/", 1)[1], d_cstate[k])  # strip 'rssm/'

# ── RSSM _prior backward ───────────────────────────────────────────────
prior_state = init(lambda d: rssm._prior(d), deter0)


def prior_g(state, deter):
    _, out = nj.pure(lambda d: rssm._prior(d))(state, deter, seed=0)
    return out


g_prior = rnd(B, STOCH, CLASSES)
(_p, prior_vjp) = jax.vjp(prior_g, prior_state, deter0)
d_pstate, d_pdeter = prior_vjp(g_prior)
emit_flat("prior.g_out", g_prior)
emit_flat("prior.g_deter", d_pdeter)
for k in sorted(d_pstate):
    emit_flat("gprior." + k.split("/", 1)[1], d_pstate[k])  # strip 'rssm/'

# ── full WM dyn/rep loss backward (deterministic; no sampling needed) ──
from embodied.jax import outs

UNIMIX, FREE_NATS = 0.01, 1.0


def _dist(logit):
    return outs.Agg(outs.OneHot(logit, UNIMIX), 1, jnp.sum)


def _wm_body(deter, stoch, action, tokens):
    carry = dict(deter=deter, stoch=stoch)
    _, (entry, feat) = rssm._observe(carry, tokens, {"action": action}, reset,
                                     True)
    prior = rssm._prior(feat["deter"])
    dyn = _dist(jax.lax.stop_gradient(feat["logit"])).kl(_dist(prior))
    rep = _dist(feat["logit"]).kl(_dist(jax.lax.stop_gradient(prior)))
    return jnp.maximum(dyn, FREE_NATS), jnp.maximum(rep, FREE_NATS)


wm_state = init(_wm_body, deter0, stoch0, action, tokens)


def wm_g(state, deter, stoch, action, tokens):
    _, out = nj.pure(_wm_body)(state, deter, stoch, action, tokens, seed=0)
    return out


# isolate OneHotKL: dump post/prior (forward) + grad w.r.t. them
def _post_prior(deter, stoch, action, tokens):
    carry = dict(deter=deter, stoch=stoch)
    _, (entry, feat) = rssm._observe(carry, tokens, {"action": action}, reset,
                                     True)
    return feat["logit"], rssm._prior(feat["deter"])


post_v, prior_v = nj.pure(_post_prior)(
    wm_state, deter0, stoch0, action, tokens, seed=0)[1]


def _kl_fn(post, prior):
    dyn = _dist(jax.lax.stop_gradient(post)).kl(_dist(prior))
    rep = _dist(post).kl(_dist(jax.lax.stop_gradient(prior)))
    return jnp.maximum(dyn, FREE_NATS), jnp.maximum(rep, FREE_NATS)


(_d, _r), kl_vjp = jax.vjp(_kl_fn, post_v, prior_v)
g_post, g_prior = kl_vjp((jnp.ones_like(_d), jnp.ones_like(_r)))
emit_flat("wm.post", post_v)
emit_flat("wm.prior", prior_v)
emit_flat("wm.g_post", g_post)
emit_flat("wm.g_prior", g_prior)

(dyn_o, rep_o), wm_vjp = jax.vjp(wm_g, wm_state, deter0, stoch0, action, tokens)
d_wstate, d_wdeter, d_wstoch, d_waction, d_wtokens = wm_vjp(
    (jnp.ones_like(dyn_o), jnp.ones_like(rep_o)))
# dump the wm_state's OWN params (one nj context creates dynin+obs+prior →
# their RNG-streamed values differ from the per-module inits, so the Mojo
# wm test must load THESE, not pr4's).
for k in sorted(wm_state):
    emit_flat("pwm." + k.split("/", 1)[1], wm_state[k])  # strip 'rssm/'
emit_flat("wm.g_deter", d_wdeter)
emit_flat("wm.g_stoch", d_wstoch)
emit_flat("wm.g_action", d_waction)
emit_flat("wm.g_tokens", d_wtokens)
for k in sorted(d_wstate):
    emit_flat("gwm." + k.split("/", 1)[1], d_wstate[k])  # strip 'rssm/'

# ── reward (twohot) + cont (binary) MLP heads backward ─────────────────
import embodied.jax as ej


def symexp(x):
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def twohot_bins(bins):
    half = jnp.linspace(-20, 0, (bins - 1) // 2 + 1, dtype=jnp.float32)
    half = symexp(half)
    return jnp.concatenate([half, -half[:-1][::-1]], 0)


HFEAT, HU, HBINS = DETER + SC, 8, 255
hbins = twohot_bins(HBINS)
h_feat = rnd(B, HFEAT)
rew_target = jnp.asarray(rng.uniform(-3, 3, (B,)), jnp.float32)
con_target = jnp.asarray((rng.uniform(0, 1, (B,)) < 0.7).astype(np.float32))
scalar_sp = elements.Space(np.float32, ())
binary_sp = elements.Space(bool, (), 0, 2)

rew_head = ej.MLPHead(scalar_sp, "symexp_twohot", layers=1, units=HU,
                      act="gelu", bins=HBINS, name="rew")
con_head = ej.MLPHead(binary_sp, "binary", layers=1, units=HU, act="gelu",
                      name="con")


def rew_loss_fn(state, feat):
    return nj.pure(lambda f: rew_head(f, 1).loss(rew_target))(
        state, feat, seed=0)[1]


def con_loss_fn(state, feat):
    return nj.pure(lambda f: con_head(f, 1).loss(con_target))(
        state, feat, seed=0)[1]


rew_state = init(lambda f: rew_head(f, 1).loss(rew_target), h_feat)
con_state = init(lambda f: con_head(f, 1).loss(con_target), h_feat)


def rew_g(state, feat):
    return nj.pure(lambda f: rew_head(f, 1).loss(rew_target))(
        state, feat, seed=0)[1]


def con_g(state, feat):
    return nj.pure(lambda f: con_head(f, 1).loss(con_target))(
        state, feat, seed=0)[1]


rl_o, rew_vjp = jax.vjp(rew_g, rew_state, h_feat)
d_rstate, d_rfeat = rew_vjp(jnp.ones_like(rl_o))
cl_o, con_vjp = jax.vjp(con_g, con_state, h_feat)
d_cstate2, d_cfeat = con_vjp(jnp.ones_like(cl_o))

emit_flat("hd.bins", hbins)
emit_flat("hd.feat", h_feat)
emit_flat("hd.rew_target", rew_target)
emit_flat("hd.con_target", con_target)
emit_flat("hd.rew_g_feat", d_rfeat)
emit_flat("hd.con_g_feat", d_cfeat)
for k in sorted(d_rstate):
    emit_flat("grew." + k.split("/", 1)[1], d_rstate[k])     # strip 'rew/'
for k in sorted(d_cstate2):
    emit_flat("gcon." + k.split("/", 1)[1], d_cstate2[k])    # strip 'con/'
# head params (single nj context per head → must load these)
for k in sorted(rew_state):
    emit_flat("prew." + k.split("/", 1)[1], rew_state[k])
for k in sorted(con_state):
    emit_flat("pcon." + k.split("/", 1)[1], con_state[k])

path = os.path.join(_HERE, "pr5b2_fixture.txt")
with open(path, "w") as f:
    f.write("\n".join(out_lines) + "\n")
print("wrote", path)
print("enc param-grad keys:", sorted(k.split("/", 1)[1] for k in d_state))
print("core param-grad keys:", sorted(k.split("/", 1)[1] for k in d_cstate))
print("core g_deter[0,:4]", np.asarray(d_cdeter)[0, :4])
