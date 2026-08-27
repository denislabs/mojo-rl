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

from std.memory import alloc, dealloc
from std.math import log, exp

from mojo_rl.nn.constants import DT

comptime UNIMIX = Scalar[DT](0.01)


@always_inline
def cat_softmax_mix[
    C: Int,
    logits_o: Origin[mut=True],
    out_sm_o: Origin[mut=True],
    out_p_o: Origin[mut=True],
](
    logits: Pointer[Scalar[DT], logits_o],
    base: Int,
    u: Scalar[DT],
    out_sm: Pointer[Scalar[DT], out_sm_o],  # [C] pre-mix softmax
    out_p: Pointer[Scalar[DT], out_p_o],  # [C] unimix-mixed probs
):
    """Softmax(logits[base:base+C]) → out_sm, then unimix → out_p."""
    var mx = logits[unsafe_offset=base]
    for c in range(1, C):
        if logits[unsafe_offset=base + c] > mx:
            mx = logits[unsafe_offset=base + c]
    var s = Scalar[DT](0.0)
    for c in range(C):
        var e = exp(logits[unsafe_offset=base + c] - mx)
        out_sm[unsafe_offset=c] = e
        s += e
    var one_m_u = Scalar[DT](1.0) - u
    var uc = u / Scalar[DT](C)
    for c in range(C):
        out_sm[unsafe_offset=c] = out_sm[unsafe_offset=c] / s
        out_p[unsafe_offset=c] = one_m_u * out_sm[unsafe_offset=c] + uc


@always_inline
def cat_sample[
    C: Int,
    logits_o: Origin[mut=True],
](
    logits: Pointer[Scalar[DT], logits_o],
    base: Int,
    u: Scalar[DT],
    u01: Scalar[DT],  # uniform in [0,1)
) -> Int:
    """Inverse-CDF categorical sample from unimix(softmax(logits))."""
    var sm_a = alloc[Scalar[DT]]({count = C})
    var sm = sm_a.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var pp_a = alloc[Scalar[DT]]({count = C})
    var pp = pp_a.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    cat_softmax_mix[C](logits, base, u, sm, pp)
    var acc = Scalar[DT](0.0)
    var k = C - 1
    for c in range(C):
        acc += pp[unsafe_offset=c]
        if u01 < acc:
            k = c
            break
    dealloc(sm_a^)
    dealloc(pp_a^)
    return k


@always_inline
def cat_argmax[
    C: Int,
    logits_o: Origin[mut=True],
](logits: Pointer[Scalar[DT], logits_o], base: Int) -> Int:
    """Greedy class = argmax logits (= argmax unimix probs)."""
    var k = 0
    var best = logits[unsafe_offset=base]
    for c in range(1, C):
        if logits[unsafe_offset=base + c] > best:
            best = logits[unsafe_offset=base + c]
            k = c
    return k


@always_inline
def cat_fwd[
    C: Int,
    logits_o: Origin[mut=True],
    out_sm_o: Origin[mut=True],
    out_p_o: Origin[mut=True],
](
    logits: Pointer[Scalar[DT], logits_o],
    base: Int,
    u: Scalar[DT],
    k: Int,  # sampled class index
    out_sm: Pointer[Scalar[DT], out_sm_o],  # [C] scratch (pre-mix)
    out_p: Pointer[Scalar[DT], out_p_o],  # [C] scratch (mixed)
) -> Tuple[Scalar[DT], Scalar[DT]]:
    """Returns (logp(k), entropy). Fills out_sm/out_p for the backward."""
    cat_softmax_mix[C](logits, base, u, out_sm, out_p)
    var logp = log(out_p[unsafe_offset=k])
    var ent = Scalar[DT](0.0)
    for m in range(C):
        ent += -out_p[unsafe_offset=m] * log(out_p[unsafe_offset=m])
    return Tuple(logp, ent)


@always_inline
def cat_bwd[
    C: Int,
    sm_o: Origin[mut=True],
    p_o: Origin[mut=True],
    grad_logits_o: Origin[mut=True],
](
    sm: Pointer[
        Scalar[DT], sm_o
    ],  # [C] pre-mix softmax (from fwd)
    p: Pointer[Scalar[DT], p_o],  # [C] mixed probs (from fwd)
    u: Scalar[DT],
    k: Int,  # sampled class index
    d_logp: Scalar[DT],  # upstream ∂L/∂logp
    d_ent: Scalar[DT],  # upstream ∂L/∂entropy
    grad_logits: Pointer[Scalar[DT], grad_logits_o],  # accumulate [.,C]
    base: Int,
):
    """Accumulate ∂L/∂logits at [base:base+C] from logp(k) + entropy paths."""
    var one_m_u = Scalar[DT](1.0) - u
    # entropy: Σ_m sm_m·(log p_m + 1) is shared across j
    var ent_dot = Scalar[DT](0.0)
    for m in range(C):
        ent_dot += sm[unsafe_offset=m] * (log(p[unsafe_offset=m]) + Scalar[DT](1.0))
    var inv_pk = Scalar[DT](1.0) / p[unsafe_offset=k]
    for j in range(C):
        var delta_kj = Scalar[DT](1.0) if j == k else Scalar[DT](0.0)
        # d logp(k)/d logits_j = (1/p_k)·(1-u)·sm_k·(δ_kj − sm_j)
        var dlogp = inv_pk * one_m_u * sm[unsafe_offset=k] * (delta_kj - sm[unsafe_offset=j])
        # d ent/d logits_j = -(1-u)·sm_j·[ (log p_j+1) − ent_dot ]
        var dent = -one_m_u * sm[unsafe_offset=j] * ((log(p[unsafe_offset=j]) + Scalar[DT](1.0)) - ent_dot)
        grad_logits[unsafe_offset=base + j] += d_logp * dlogp + d_ent * dent
