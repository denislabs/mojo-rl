"""The FB losses, as pure functions of activations.

Both losses take `[BATCH, D]` activation buffers and write gradients back onto
them. Nothing here knows about networks, optimizers or replay — the trainer
forwards its nets, calls these, and backpropagates the returned gradients. That
split exists so the mathematically delicate part is testable on its own, which
`docs/BFM_ZERO_SHOT_RL.md` §11 asks for by name.

## The measure loss

    L_FB = E_ij[ ( F(s_i,a_i,z_i)·B(s+_j) - gamma·Fbar(s'_i,a'_i,z_i)·Bbar(s+_j) )^2 ]
           - 2·E_i[ F(s_i,a_i,z_i)·B(s'_i) ]

⚠⚠ **The second term is not a regulariser.** It is what is left when the square
of the successor measure is expanded, and it is the only term that ties `F` to
the immediate transition. Drop it and `L_FB` still descends — the first term
alone is minimised by driving `F·B` towards its own bootstrapped target, which
zero satisfies perfectly. The model then "runs" and encodes nothing.
`test_fb_loss.mojo` pins this by ablation rather than by comment: it checks that
removing the anchor admits a collapsed solution that the full loss rejects.

Note the shapes. The first term pairs EVERY i with EVERY j — that is the
`[BATCH, BATCH]` successor-measure matrix, and why `PairwiseDot` exists. The
second pairs i with i only, so it uses `RowDot`: taking the diagonal of a
`PairwiseDot` would compute BATCH² dot products to keep BATCH.

`s+` comes from a SECOND, INDEPENDENT draw over the dataset. It is not `s'`. The
successor measure asks "starting from s, how often is s+ visited", and s+ has to
range over the whole state distribution for that question to mean anything; if
`s+` were the batch's own next-states, the matrix would only ever be evaluated
on pairs one step apart.

## The orthonormality regulariser

    L_ortho = E_ij[ (B(s_i)·B(s+_j))^2 ] - 2·E_i[ ||B(s_i)||^2 ]

Pushes `E[B B^T]` towards the identity. Without it `B` collapses: a constant `B`
makes the measure loss trivially satisfiable, and — this is the trap
§11 flags — the loss curve looks the same either way. On `point_mass` the
collapse is detectable by hand; on walker it is not, which is why the milestone
validates there first.

## Gradients

All three are exact, not approximations:

    dL_FB/dM     = 2(M - Mtarget)/BATCH^2          M = F·B(s+)^T
    dL_FB/dF    += -2/BATCH · B(s')                 (anchor)
    dL_FB/dB(s') = -2/BATCH · F
    dL_ortho/dO  = 2·O/BATCH^2                     O = B(s)·B(s+)^T
    dL_ortho/dB(s) += -4/BATCH · B(s)

`Mtarget` carries no gradient — it is built from the target networks and is
passed in already scaled by `gamma`.
"""

from std.math import abs

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.pairwise_dot import PairwiseDot, RowDot


def pairwise_matrix[
    D: Int, BATCH: Int
](ref a: Tensor, ref c: Tensor, mut m: Tensor) raises:
    """`m[i,j] = sum_k a[i,k]·c[j,k]`, on the CPU. No gradient bookkeeping.

    Used for the TARGET matrix, which must not carry gradient. Going through
    the same primitive as the differentiable path is deliberate: a
    hand-inlined target would be one edit away from disagreeing with the online
    one, and that disagreement is invisible in the loss value.
    """
    var op = PairwiseDot[D, BATCH].make["cpu", Deterministic](None)
    var ins = TensorPack[2]()
    ins[0].ensure(BATCH * D)
    ins[1].ensure(BATCH * D)
    for i in range(BATCH * D):
        ins[0].data[i] = a.data[i]
        ins[1].data[i] = c.data[i]
    op.forward["cpu", BATCH](TensorRefs[2](ins[0], ins[1]), m, None)


def fb_measure_loss[
    D: Int, BATCH: Int
](
    ref f: Tensor,
    ref b_sp: Tensor,
    ref b_next: Tensor,
    ref m_target: Tensor,
    mut g_f: Tensor,
    mut g_b_sp: Tensor,
    mut g_b_next: Tensor,
    with_anchor: Bool = True,
) raises -> Float64:
    """`L_FB` and its gradients. See the module docstring for the formula.

    Inputs are `[BATCH, D]` except `m_target`, which is `[BATCH, BATCH]` and
    already multiplied by `gamma`.

    `with_anchor=False` drops the `-2·E[F·B(s')]` term. That switch exists ONLY
    so the gate can demonstrate what its absence permits; it is never what a
    trainer wants, and `g_b_next` is then left at zero.
    """
    var op = PairwiseDot[D, BATCH].make["cpu", Deterministic](None)
    var ins = TensorPack[2]()
    ins[0].ensure(BATCH * D)
    ins[1].ensure(BATCH * D)
    for i in range(BATCH * D):
        ins[0].data[i] = f.data[i]
        ins[1].data[i] = b_sp.data[i]
    var m = Tensor.alloc(BATCH * BATCH)
    op.forward["cpu", BATCH](TensorRefs[2](ins[0], ins[1]), m, None)

    # Residual term.
    var n2 = Float64(BATCH) * Float64(BATCH)
    var loss = Float64(0)
    var go = Tensor.alloc(BATCH * BATCH)
    for i in range(BATCH * BATCH):
        var r = Float64(m.data[i]) - Float64(m_target.data[i])
        loss += r * r
        go.data[i] = Scalar[DT](2.0 * r / n2)
    loss /= n2

    var grads = TensorPack[2]()
    op.vjp["cpu", BATCH](
        TensorRefs[2](ins[0], ins[1]), go, TensorRefs[2](grads[0], grads[1]),
        None,
    )
    g_f.ensure(BATCH * D)
    g_b_sp.ensure(BATCH * D)
    g_b_next.ensure(BATCH * D)
    for i in range(BATCH * D):
        g_f.data[i] = grads[0].data[i]
        g_b_sp.data[i] = grads[1].data[i]
        g_b_next.data[i] = Scalar[DT](0)

    if not with_anchor:
        return loss

    # Anchor term: -2·mean_i( F_i · B(s'_i) ). RowDot, not diag(PairwiseDot).
    var rd = RowDot[D].make["cpu", Deterministic](None)
    var rins = TensorPack[2]()
    rins[0].ensure(BATCH * D)
    rins[1].ensure(BATCH * D)
    for i in range(BATCH * D):
        rins[0].data[i] = f.data[i]
        rins[1].data[i] = b_next.data[i]
    var r_out = Tensor.alloc(BATCH)
    rd.forward["cpu", BATCH](TensorRefs[2](rins[0], rins[1]), r_out, None)

    var anchor = Float64(0)
    for i in range(BATCH):
        anchor += Float64(r_out.data[i])
    loss += -2.0 * anchor / Float64(BATCH)

    var r_go = Tensor.alloc(BATCH)
    for i in range(BATCH):
        r_go.data[i] = Scalar[DT](-2.0 / Float64(BATCH))
    var rgrads = TensorPack[2]()
    rd.vjp["cpu", BATCH](
        TensorRefs[2](rins[0], rins[1]), r_go,
        TensorRefs[2](rgrads[0], rgrads[1]), None,
    )
    for i in range(BATCH * D):
        g_f.data[i] = Scalar[DT](
            Float64(g_f.data[i]) + Float64(rgrads[0].data[i])
        )
        g_b_next.data[i] = rgrads[1].data[i]
    return loss


def fb_ortho_loss[
    D: Int, BATCH: Int
](
    ref b_s: Tensor,
    ref b_sp: Tensor,
    mut g_b_s: Tensor,
    mut g_b_sp: Tensor,
) raises -> Float64:
    """`L_ortho` and its gradients.

    OVERWRITES `g_b_s` / `g_b_sp` rather than accumulating. `B(s+)` receives
    gradient from both losses, and making the trainer add them explicitly keeps
    that visible at the call site — an accumulating signature would let a
    forgotten zeroing silently double one contribution.
    """
    var op = PairwiseDot[D, BATCH].make["cpu", Deterministic](None)
    var ins = TensorPack[2]()
    ins[0].ensure(BATCH * D)
    ins[1].ensure(BATCH * D)
    for i in range(BATCH * D):
        ins[0].data[i] = b_s.data[i]
        ins[1].data[i] = b_sp.data[i]
    var o = Tensor.alloc(BATCH * BATCH)
    op.forward["cpu", BATCH](TensorRefs[2](ins[0], ins[1]), o, None)

    var n2 = Float64(BATCH) * Float64(BATCH)
    var loss = Float64(0)
    var go = Tensor.alloc(BATCH * BATCH)
    for i in range(BATCH * BATCH):
        var v = Float64(o.data[i])
        loss += v * v
        go.data[i] = Scalar[DT](2.0 * v / n2)
    loss /= n2

    var grads = TensorPack[2]()
    op.vjp["cpu", BATCH](
        TensorRefs[2](ins[0], ins[1]), go, TensorRefs[2](grads[0], grads[1]),
        None,
    )
    g_b_s.ensure(BATCH * D)
    g_b_sp.ensure(BATCH * D)
    for i in range(BATCH * D):
        g_b_s.data[i] = grads[0].data[i]
        g_b_sp.data[i] = grads[1].data[i]

    # -2·mean_i ||B(s_i)||^2, gradient -4/BATCH · B(s).
    var sq = Float64(0)
    for i in range(BATCH * D):
        var v = Float64(b_s.data[i])
        sq += v * v
    loss += -2.0 * sq / Float64(BATCH)
    var c = -4.0 / Float64(BATCH)
    for i in range(BATCH * D):
        g_b_s.data[i] = Scalar[DT](
            Float64(g_b_s.data[i]) + c * Float64(b_s.data[i])
        )
    return loss
