"""Discrete-policy distribution helpers — DreamerV3 unimix `onehot` actor.

Ports the categorical action head (`embodied/jax/outs.py:OneHotDist` with
`unimix`), the discrete analog of `dists.mojo`'s `bounded_normal`:

  sm   = softmax(logits)                     # [C] pre-mix
  p    = (1-u)·sm + u/C                       # [C] unimix-mixed probs
  logp(k) = log(p_k)                          # k = sampled class index
  entropy = -Σ_m p_m·log(p_m)

REINFORCE actor (action is stop-grad'd), so the only policy gradient flows
through `logp` and `entropy` — identical structure to the continuous head,
just a different per-step distribution. Gradients to the logits:

  d logp(k)/d logits_j = (1/p_k)·(1-u)·sm_k·(δ_kj − sm_j)
  d ent  /d logits_j   = -(1-u)·sm_j·[ (log p_j + 1) − Σ_m sm_m·(log p_m + 1) ]

The unimix matches the latent `OneHotKL` default (u=0.01). These mirror the
softmax+mix already in `onehot_kl.mojo`; kept here as standalone scalar
helpers for the imag-loss per-(b,t) policy term and for unit FD-gradcheck.
"""

from std.memory import alloc
from std.math import log, exp

from mojo_rl.nn2.constants import DT

comptime UNIMIX = Scalar[DT](0.01)


@always_inline
def cat_softmax_mix[
    C: Int
](
    logits: UnsafePointer[Scalar[DT], MutAnyOrigin],
    base: Int,
    u: Scalar[DT],
    out_sm: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [C] pre-mix softmax
    out_p: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [C] unimix-mixed probs
):
    """Softmax(logits[base:base+C]) → out_sm, then unimix → out_p."""
    var mx = logits[base]
    for c in range(1, C):
        if logits[base + c] > mx:
            mx = logits[base + c]
    var s = Scalar[DT](0.0)
    for c in range(C):
        var e = exp(logits[base + c] - mx)
        out_sm[c] = e
        s += e
    var one_m_u = Scalar[DT](1.0) - u
    var uc = u / Scalar[DT](C)
    for c in range(C):
        out_sm[c] = out_sm[c] / s
        out_p[c] = one_m_u * out_sm[c] + uc


@always_inline
def cat_sample[
    C: Int
](
    logits: UnsafePointer[Scalar[DT], MutAnyOrigin],
    base: Int,
    u: Scalar[DT],
    u01: Scalar[DT],  # uniform in [0,1)
) -> Int:
    """Inverse-CDF categorical sample from unimix(softmax(logits))."""
    var sm = alloc[Scalar[DT]](C)
    var pp = alloc[Scalar[DT]](C)
    cat_softmax_mix[C](logits, base, u, sm, pp)
    var acc = Scalar[DT](0.0)
    var k = C - 1
    for c in range(C):
        acc += pp[c]
        if u01 < acc:
            k = c
            break
    sm.free()
    pp.free()
    return k


@always_inline
def cat_argmax[
    C: Int
](logits: UnsafePointer[Scalar[DT], MutAnyOrigin], base: Int) -> Int:
    """Greedy class = argmax logits (= argmax unimix probs)."""
    var k = 0
    var best = logits[base]
    for c in range(1, C):
        if logits[base + c] > best:
            best = logits[base + c]
            k = c
    return k


@always_inline
def cat_fwd[
    C: Int
](
    logits: UnsafePointer[Scalar[DT], MutAnyOrigin],
    base: Int,
    u: Scalar[DT],
    k: Int,  # sampled class index
    out_sm: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [C] scratch (pre-mix)
    out_p: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [C] scratch (mixed)
) -> Tuple[Scalar[DT], Scalar[DT]]:
    """Returns (logp(k), entropy). Fills out_sm/out_p for the backward."""
    cat_softmax_mix[C](logits, base, u, out_sm, out_p)
    var logp = log(out_p[k])
    var ent = Scalar[DT](0.0)
    for m in range(C):
        ent += -out_p[m] * log(out_p[m])
    return Tuple(logp, ent)


@always_inline
def cat_bwd[
    C: Int
](
    sm: UnsafePointer[
        Scalar[DT], MutAnyOrigin
    ],  # [C] pre-mix softmax (from fwd)
    p: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [C] mixed probs (from fwd)
    u: Scalar[DT],
    k: Int,  # sampled class index
    d_logp: Scalar[DT],  # upstream ∂L/∂logp
    d_ent: Scalar[DT],  # upstream ∂L/∂entropy
    grad_logits: UnsafePointer[Scalar[DT], MutAnyOrigin],  # accumulate [.,C]
    base: Int,
):
    """Accumulate ∂L/∂logits at [base:base+C] from logp(k) + entropy paths."""
    var one_m_u = Scalar[DT](1.0) - u
    # entropy: Σ_m sm_m·(log p_m + 1) is shared across j
    var ent_dot = Scalar[DT](0.0)
    for m in range(C):
        ent_dot += sm[m] * (log(p[m]) + Scalar[DT](1.0))
    var inv_pk = Scalar[DT](1.0) / p[k]
    for j in range(C):
        var delta_kj = Scalar[DT](1.0) if j == k else Scalar[DT](0.0)
        # d logp(k)/d logits_j = (1/p_k)·(1-u)·sm_k·(δ_kj − sm_j)
        var dlogp = inv_pk * one_m_u * sm[k] * (delta_kj - sm[j])
        # d ent/d logits_j = -(1-u)·sm_j·[ (log p_j+1) − ent_dot ]
        var dent = -one_m_u * sm[j] * ((log(p[j]) + Scalar[DT](1.0)) - ent_dot)
        grad_logits[base + j] += d_logp * dlogp + d_ent * dent
