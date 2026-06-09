"""PR-2 fixtures: lambda_return, OneHotKL (dyn/rep), PercentileNormalize.

Ground truth from real jax, replicating the reference verbatim:
- lambda_return  : `references/.../dreamerv3/agent.py:482` (forward only —
  operates on detached values, no backward in the agent).
- OneHotKL       : `rssm.py:RSSM.loss` dyn/rep with `outs.OneHot(unimix)` +
  `Agg(sum over STOCH)` + free-nats clamp. Gradients via jax.vjp.
- Normalize      : `embodied/jax/utils.py:Normalize`, impl='perc'
  (retnorm config: rate=0.01, perclo=5, perchi=95, limit=1.0, debias=False)
  and impl='none' (valnorm/advnorm → (0,1)).

Run from the throwaway venv (project pixi has no jax).
"""

import os

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", False)
f32 = jnp.float32
sg = jax.lax.stop_gradient


# ── reference lambda_return (verbatim from agent.py:482) ──
def lambda_return(last, term, rew, val, boot, disc, lam):
    rets = [boot[:, -1]]
    live = (1 - f32(term))[:, 1:] * disc
    cont = (1 - f32(last))[:, 1:] * lam
    interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
    for t in reversed(range(live.shape[1])):
        rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
    return jnp.stack(list(reversed(rets))[:-1], 1)


# ── reference OneHot categorical KL with unimix (outs.py Categorical) ──
def unimix_logits(logits, unimix):
    probs = jax.nn.softmax(logits, -1)
    uniform = jnp.ones_like(probs) / probs.shape[-1]
    probs = (1 - unimix) * probs + unimix * uniform
    return jnp.log(probs)


def cat_kl(logits_self, logits_other):
    lp = jax.nn.log_softmax(logits_self, -1)
    lo = jax.nn.log_softmax(logits_other, -1)
    p = jax.nn.softmax(logits_self, -1)
    return (p * (lp - lo)).sum(-1)  # over CLASSES → [..., STOCH]


def dyn_rep(post_logits, prior_logits, unimix, free_nats):
    pl = unimix_logits(post_logits, unimix)
    ql = unimix_logits(prior_logits, unimix)
    # dyn = KL(sg(post) || prior) summed over STOCH, clamped.
    dyn = cat_kl(sg(pl), ql).sum(-1)
    rep = cat_kl(pl, sg(ql)).sum(-1)
    dyn = jnp.maximum(dyn, free_nats)
    rep = jnp.maximum(rep, free_nats)
    return dyn, rep


def main():
    rng = np.random.default_rng(20260530)
    lines = []

    def emit(s):
        lines.append(s)

    def emit_flat(name, arr):
        flat = np.asarray(arr).ravel()
        emit(f"{name}#size={flat.size}")
        for v in flat:
            emit(repr(float(v)))

    emit(f"# jax={jax.__version__} np={np.__version__}")

    # ── lambda_return ──
    B, T = 2, 6
    DISC, LAM = 0.997, 0.95
    last = jnp.asarray((rng.random((B, T)) < 0.2).astype(np.float32))
    term = jnp.asarray((rng.random((B, T)) < 0.1).astype(np.float32))
    rew = jnp.asarray(rng.standard_normal((B, T)).astype(np.float32))
    val = jnp.asarray(rng.standard_normal((B, T)).astype(np.float32))
    boot = jnp.asarray(rng.standard_normal((B, T)).astype(np.float32))
    ret = lambda_return(last, term, rew, val, boot, DISC, LAM)
    emit(f"lr.batch={B}")
    emit(f"lr.t={T}")
    emit(f"lr.disc={DISC!r}")
    emit(f"lr.lam={LAM!r}")
    emit_flat("lr.last", last)
    emit_flat("lr.term", term)
    emit_flat("lr.rew", rew)
    emit_flat("lr.val", val)
    emit_flat("lr.boot", boot)
    emit_flat("lr.ret", ret)  # [B, T-1]

    # ── OneHotKL ──
    KB, STOCH, CLASSES = 2, 3, 4
    UNIMIX, FREE = 0.01, 1.0
    # Row 0: post and prior very different (KL > 1, not clamped).
    # Row 1: post ≈ prior (KL < 1, clamped → zero grad).
    post = rng.standard_normal((KB, STOCH, CLASSES)).astype(np.float32)
    prior = rng.standard_normal((KB, STOCH, CLASSES)).astype(np.float32)
    post[0] *= 3.0          # sharpen row 0 post
    prior[1] = post[1] + 0.01 * rng.standard_normal((STOCH, CLASSES))  # row 1 close
    post_j = jnp.asarray(post)
    prior_j = jnp.asarray(prior)

    dyn, rep = dyn_rep(post_j, prior_j, UNIMIX, FREE)

    # Combined loss with per-row upstream weights d_dyn = d_rep = 1.
    def loss_fn(po, pr):
        d, r = dyn_rep(po, pr, UNIMIX, FREE)
        return d.sum() + r.sum()

    (g_post, g_prior) = jax.grad(loss_fn, argnums=(0, 1))(post_j, prior_j)
    emit(f"kl.batch={KB}")
    emit(f"kl.stoch={STOCH}")
    emit(f"kl.classes={CLASSES}")
    emit(f"kl.unimix={UNIMIX!r}")
    emit(f"kl.free_nats={FREE!r}")
    emit_flat("kl.post", post_j)
    emit_flat("kl.prior", prior_j)
    emit_flat("kl.dyn", dyn)            # [KB]
    emit_flat("kl.rep", rep)            # [KB]
    emit_flat("kl.gpost", g_post)       # [KB, STOCH, CLASSES]
    emit_flat("kl.gprior", g_prior)

    # ── PercentileNormalize (perc, retnorm config) ──
    RATE, PERCLO, PERCHI, LIMIT = 0.01, 5.0, 95.0, 1.0
    n_updates = 50
    lo = 0.0
    hi = 0.0
    # Deterministic update stream (each a [64] sample).
    update_inputs = []
    for _ in range(n_updates):
        update_inputs.append(rng.standard_normal(64).astype(np.float32) * 5.0)
    for u in update_inputs:
        plo = float(jnp.percentile(jnp.asarray(u), PERCLO))
        phi = float(jnp.percentile(jnp.asarray(u), PERCHI))
        lo = (1 - RATE) * lo + RATE * plo
        hi = (1 - RATE) * hi + RATE * phi
    offset = lo
    scale = max(LIMIT, hi - lo)
    emit(f"pn.rate={RATE!r}")
    emit(f"pn.perclo={PERCLO!r}")
    emit(f"pn.perchi={PERCHI!r}")
    emit(f"pn.limit={LIMIT!r}")
    emit(f"pn.n_updates={n_updates}")
    emit(f"pn.sample_size=64")
    # Concatenate all update inputs so Mojo replays the same stream.
    allu = np.concatenate(update_inputs)
    emit_flat("pn.inputs", allu)
    emit(f"pn.offset={offset!r}")
    emit(f"pn.scale={scale!r}")

    out_path = os.path.join(os.path.dirname(__file__), "pr2_fixture.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_path} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
