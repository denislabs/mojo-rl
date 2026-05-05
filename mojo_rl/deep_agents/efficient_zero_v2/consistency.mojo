"""SimSiam-style cosine consistency loss for the EfficientZero V2 unroll.

The dynamics branch's projected+predicted hidden states are pulled toward a
*detached* projection of the representation network's hidden states from
the actual observations. This is the same self-supervised consistency
objective that EfficientZero V1 used (paper Eq. 4):

    L_G = − cos(p_dyn,  sg(proj_obs))

where:
    p_dyn      = predictor(projector(z_dyn))   — gradient flows
    proj_obs   = projector(z_obs)              — stop-gradient via `sg(·)`

Two forms of asymmetry make this safe against the obvious "all features
collapse to zero" degeneracy:

  1. Stop-gradient on the target branch — the projector does not get a
     "match yourself" gradient.
  2. Predictor (1024 → 512 → 1024) only on the online branch — the
     bottleneck breaks symmetry between the two branches.

Both pieces sit in `networks.mojo`; this module only handles the loss +
its gradient.

Mean-over-batch convention: callers receive `(1/B) Σ_b -cos(p_b, t_b)` so
that `grad_seed` is consistent with the rest of the K-step loss
accumulator. The grad written into `grad_online` is also mean-batch'd; if
the caller wants sum, they can rescale `grad_seed` by BATCH.

Phase-2 step 3: CPU-only; the GPU kernel lands alongside the K-step
training loop in step 5+.

Reference:
    Chen & He, *Exploring Simple Siamese Representation Learning*, CVPR
    2021. EZ-V2 paper App. G.
"""

from std.math import sqrt
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype


# ═════════════════════════════════════════════════════════════════════════
# Forward-only (used by gradcheck loops; cheap)
# ═════════════════════════════════════════════════════════════════════════


def cosine_consistency_loss_forward[
    BATCH: Int,
    DIM: Int,
    EPS: Float64 = 1e-12,
](
    online: LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ],
    target: LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ],
) -> Float64:
    """Mean-over-batch negative cosine similarity, no gradients computed.

    L = (1/B) Σ_b − cos(online_b, target_b)

    Args:
        online: Predictor output tensor `[BATCH × DIM]` (gradient
            normally flows through this branch — this fn just doesn't
            *compute* it).
        target: Stop-gradient projection tensor `[BATCH × DIM]`.

    Returns:
        Scalar mean-over-batch loss.
    """
    var loss = Float64(0.0)
    for b in range(BATCH):
        var dot = Float64(0.0)
        var na2 = Float64(0.0)
        var nb2 = Float64(0.0)
        for i in range(DIM):
            var oi = Float64(rebind[Scalar[dtype]](online[b, i]))
            var ti = Float64(rebind[Scalar[dtype]](target[b, i]))
            dot += oi * ti
            na2 += oi * oi
            nb2 += ti * ti
        var na = sqrt(na2 + EPS)
        var nb = sqrt(nb2 + EPS)
        loss += -(dot / (na * nb))
    return loss / Float64(BATCH)


# ═════════════════════════════════════════════════════════════════════════
# Forward + backward (fused, used by the training loop)
# ═════════════════════════════════════════════════════════════════════════


def cosine_consistency_loss[
    BATCH: Int,
    DIM: Int,
    EPS: Float64 = 1e-12,
](
    online: LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ],
    target: LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ],
    mut grad_online: LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ],
    grad_seed: Float64 = 1.0,
) -> Float64:
    """Fused forward + backward of the SimSiam cosine consistency loss.

    Writes the per-element gradient `dL/d(online_{b,i})` into `grad_online`
    and returns the scalar loss. **No gradient is produced for `target`**
    — it's the stop-gradient branch by construction; the function does not
    even take a `grad_target` parameter.

    The mathematical derivation, for one sample (with `na = ||online||`,
    `nb = ||target||`, `s = online · target`, `c = s/(na·nb)`):

        ∂(-c)/∂online_i  =  c · online_i / na² − target_i / (na · nb)

    Mean-over-batch then divides by BATCH (folded into `grad_seed`).

    Args:
        online: Predictor output `[BATCH × DIM]` (the "p" branch).
        target: Detached projector output `[BATCH × DIM]` (the "z" branch).
        grad_online: Output buffer `[BATCH × DIM]` for `dL/d(online)`;
            **overwritten**, not accumulated.
        grad_seed: Upstream gradient w.r.t. this loss term (typically 1.0
            for a top-level loss, or a `λ_G` weight if the consistency
            term is being mixed into a composite loss).

    Returns:
        Scalar mean-over-batch loss.
    """
    var loss = Float64(0.0)
    var inv_batch = grad_seed / Float64(BATCH)
    for b in range(BATCH):
        var dot = Float64(0.0)
        var na2 = Float64(0.0)
        var nb2 = Float64(0.0)
        for i in range(DIM):
            var oi = Float64(rebind[Scalar[dtype]](online[b, i]))
            var ti = Float64(rebind[Scalar[dtype]](target[b, i]))
            dot += oi * ti
            na2 += oi * oi
            nb2 += ti * ti
        var na = sqrt(na2 + EPS)
        var nb = sqrt(nb2 + EPS)
        var na_nb = na * nb
        var inv_na_nb = 1.0 / na_nb
        var inv_na_sq = 1.0 / (na * na)
        var c = dot * inv_na_nb
        loss += -c

        for i in range(DIM):
            var oi = Float64(rebind[Scalar[dtype]](online[b, i]))
            var ti = Float64(rebind[Scalar[dtype]](target[b, i]))
            var g = inv_batch * (c * oi * inv_na_sq - ti * inv_na_nb)
            grad_online[b, i] = Scalar[dtype](g)

    return loss / Float64(BATCH)
