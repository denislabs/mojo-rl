"""DreamerV3 optimizer-chain fixture extractor.

Generates the ground-truth fixture for `tests/nn2/test_dreamer_opt_step_parity.mojo`
and `tests/nn2/test_dreamer_opt_warmup.mojo` using REAL jax + optax.

The three gradient transforms (`clip_by_agc`, `scale_by_rms`,
`scale_by_momentum`) are copied **verbatim** from
`references/dreamerv3-main/embodied/jax/opt.py` (lines 109-164) so the
fixture is an independent source of truth — NOT a numpy re-derivation of
our own understanding. The chain order + hyperparameters mirror
`references/dreamerv3-main/dreamerv3/agent.py:_make_opt` with the config
defaults from `dreamerv3/configs.yaml` (`agc=0.3, eps=1e-20, beta1=0.9,
beta2=0.999, wd=0.0`).

This script does NOT run inside the project's pixi env (which has no
jax). Run it from a throwaway venv:

    python3 -m venv /tmp/dreamer_fixtures_venv
    /tmp/dreamer_fixtures_venv/bin/pip install "jax[cpu]" optax numpy
    /tmp/dreamer_fixtures_venv/bin/python \\
        tests/nn2/dreamerv3/fixtures/extract.py

Output: `dreamer_opt_fixture.txt` (line-based, parsed by the Mojo test).

Pinned versions at generation time are written into the fixture header.
Regenerate whenever the reference opt.py changes.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import optax

jax.config.update("jax_enable_x64", False)  # match nn2 fp32 (DT)

f32 = jnp.float32

# ─────────────────────────────────────────────────────────────────────────
# VERBATIM from references/dreamerv3-main/embodied/jax/opt.py:109-164.
# Do not "improve" — these ARE the reference. Only the `i32`/`f32` and
# optax helper references are project-local equivalents.
# ─────────────────────────────────────────────────────────────────────────


def clip_by_agc(clip=0.3, pmin=1e-3):

  def init_fn(params):
    return ()

  def update_fn(updates, state, params=None):
    def fn(param, update):
      unorm = jnp.linalg.norm(update.flatten(), 2)
      pnorm = jnp.linalg.norm(param.flatten(), 2)
      upper = clip * jnp.maximum(pmin, pnorm)
      return update * (1 / jnp.maximum(1.0, unorm / upper))
    updates = jax.tree.map(fn, params, updates) if clip else updates
    return updates, ()

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_rms(beta=0.999, eps=1e-8):

  def init_fn(params):
    nu = jax.tree.map(lambda t: jnp.zeros_like(t, f32), params)
    step = jnp.zeros((), jnp.int32)
    return (step, nu)

  def update_fn(updates, state, params=None):
    step, nu = state
    step = optax.safe_int32_increment(step)
    nu = jax.tree.map(
        lambda v, u: beta * v + (1 - beta) * (u * u), nu, updates)
    nu_hat = optax.bias_correction(nu, beta, step)
    updates = jax.tree.map(
        lambda u, v: u / (jnp.sqrt(v) + eps), updates, nu_hat)
    return updates, (step, nu)

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_momentum(beta=0.9, nesterov=False):

  def init_fn(params):
    mu = jax.tree.map(lambda t: jnp.zeros_like(t, f32), params)
    step = jnp.zeros((), jnp.int32)
    return (step, mu)

  def update_fn(updates, state, params=None):
    step, mu = state
    step = optax.safe_int32_increment(step)
    mu = optax.update_moment(updates, mu, beta, 1)
    if nesterov:
      mu_nesterov = optax.update_moment(updates, mu, beta, 1)
      mu_hat = optax.bias_correction(mu_nesterov, beta, step)
    else:
      mu_hat = optax.bias_correction(mu, beta, step)
    return mu_hat, (step, mu)

  return optax.GradientTransformation(init_fn, update_fn)


# ─────────────────────────────────────────────────────────────────────────
# Fixture configuration.
# ─────────────────────────────────────────────────────────────────────────

AGC_CLIP = 0.3
AGC_PMIN = 1e-3
BETA1 = 0.9       # momentum
BETA2 = 0.999     # rms
EPS = 1e-20
# Step-parity fixture uses lr=1.0 so the per-step delta is O(1) and the
# 1e-6 absolute-diff comparison is a meaningful test (the LR scale step is
# linear, so AGC+RMS+momentum are fully validated at any lr). The warmup
# *schedule* — which produces the real 4e-5-scale lr — is validated
# separately in the schedule section below.
LR = 1.0
N_STEPS = 6

# Two leaves mirroring a Linear[3,4] walk: weight (12 elems) then bias (4).
# AGC norm is over the full flattened leaf and RMS/momentum are elementwise,
# so 1-D leaves are sufficient — shapes don't matter for this chain.
LEAF_SIZES = [12, 4]


def make_chain(lr):
  # Mirror agent.py:_make_opt with wd=0 (config default) → no
  # add_decayed_weights. scale_by_learning_rate(lr) multiplies by -lr;
  # apply_updates then does param + update = param - lr*g.
  return optax.chain(
      clip_by_agc(AGC_CLIP, AGC_PMIN),
      scale_by_rms(BETA2, EPS),
      scale_by_momentum(BETA1, nesterov=False),
      optax.scale_by_learning_rate(lr),
  )


def make_schedule(lr, warmup):
  # Mirror agent.py:_make_opt schedule='const' branch with warmup.
  sched = optax.constant_schedule(lr)
  if warmup:
    ramp = optax.linear_schedule(0.0, lr, warmup)
    sched = optax.join_schedules([ramp, sched], [warmup])
  return sched


def main():
  rng = np.random.default_rng(20260528)

  # Initial params (flat 1-D leaves). leaf0 ~ moderate; leaf1 small so its
  # gradient stays under the AGC clip threshold (exercises the no-clip
  # branch alongside leaf0's clip branch).
  params = {
      "leaf0": jnp.asarray(rng.standard_normal(LEAF_SIZES[0]) * 0.5, f32),
      "leaf1": jnp.asarray(rng.standard_normal(LEAF_SIZES[1]) * 0.5, f32),
  }

  opt = make_chain(LR)
  state = opt.init(params)

  # Pre-generate per-step grads. leaf0 grads are large (will clip);
  # leaf1 grads are tiny ×0.01 (will NOT clip → scale==1).
  step_grads = []
  for _ in range(N_STEPS):
    g = {
        "leaf0": jnp.asarray(rng.standard_normal(LEAF_SIZES[0]), f32),
        "leaf1": jnp.asarray(rng.standard_normal(LEAF_SIZES[1]) * 0.01, f32),
    }
    step_grads.append(g)

  lines = []

  def emit(s):
    lines.append(s)

  def emit_flat(name, leaves):
    # Flatten in walk order: leaf0 then leaf1.
    flat = np.concatenate(
        [np.asarray(leaves["leaf0"]).ravel(),
         np.asarray(leaves["leaf1"]).ravel()])
    emit(f"{name}#size={flat.size}")
    for v in flat:
      emit(repr(float(v)))

  emit(f"# jax={jax.__version__} optax={optax.__version__} np={np.__version__}")
  emit(f"agc_clip={AGC_CLIP!r}")
  emit(f"agc_pmin={AGC_PMIN!r}")
  emit(f"beta1={BETA1!r}")
  emit(f"beta2={BETA2!r}")
  emit(f"eps={EPS!r}")
  emit(f"lr={LR!r}")
  emit(f"n_leaves={len(LEAF_SIZES)}")
  emit("leaf_sizes=" + ",".join(str(s) for s in LEAF_SIZES))
  emit(f"n_steps={N_STEPS}")

  emit_flat("init", params)

  for t in range(N_STEPS):
    grads = step_grads[t]
    emit_flat(f"step{t}.grad", grads)
    updates, state = opt.update(grads, state, params)
    params = optax.apply_updates(params, updates)
    emit_flat(f"step{t}.param", params)

  # ── Warmup schedule section ──────────────────────────────────────────
  SCHED_LR = 4e-5
  WARMUP = 1000
  sched = make_schedule(SCHED_LR, WARMUP)
  probe_steps = [0, 1, 2, 250, 500, 999, 1000, 1001, 1500, 2000]
  emit(f"sched_lr={SCHED_LR!r}")
  emit(f"sched_warmup={WARMUP}")
  emit(f"sched_n_probe={len(probe_steps)}")
  for s in probe_steps:
    val = float(sched(s))
    emit(f"sched_probe {s} {val!r}")

  out_path = os.path.join(os.path.dirname(__file__), "dreamer_opt_fixture.txt")
  with open(out_path, "w") as f:
    f.write("\n".join(lines) + "\n")
  print(f"wrote {out_path} ({len(lines)} lines)")


if __name__ == "__main__":
  main()
