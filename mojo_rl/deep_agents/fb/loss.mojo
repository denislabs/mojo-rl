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
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT, TPB, TPB_REDUCE
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.pairwise_dot import PairwiseDot, RowDot

from .kernels import (
    residual_grad_kernel,
    sq_diff_reduce_kernel,
    sum_reduce_kernel,
    sumsq_reduce_kernel,
    fill_kernel,
    axpy_kernel,
    scale_kernel,
)


struct FBLossWorkspace[D: Int, BATCH: Int](Movable & ImplicitlyDeletable):
    """Persistent scratch for one FB loss evaluation.

    The `[BATCH, BATCH]` matrices are the reason this exists. At BATCH = 1024
    each is 4 MB and three are live per step; allocating them per call — which
    is what the plain `fb_measure_loss` entry point does — is acceptable for a
    gate and not for 2 M training steps.

    `acc` is a 1-element device scalar the reduction kernels write into. It is
    downloaded ONLY when the caller asks for the loss value: a per-step D2H
    read would serialise the whole pipeline behind a device sync, and the loss
    is diagnostics, not part of the update.
    """

    var pd: PairwiseDot[Self.D, Self.BATCH]
    var rd: RowDot[Self.D]
    var m: Tensor
    var go: Tensor
    var r_out: Tensor
    var r_go: Tensor
    var acc: Tensor
    var ga: Tensor

    def __init__(out self):
        self.pd = PairwiseDot[Self.D, Self.BATCH]()
        self.rd = RowDot[Self.D]()
        self.m = Tensor()
        self.go = Tensor()
        self.r_out = Tensor()
        self.r_go = Tensor()
        self.acc = Tensor()
        self.ga = Tensor()

    def __init__(out self, *, deinit move: Self):
        self.pd = move.pd^
        self.rd = move.rd^
        self.m = move.m^
        self.go = move.go^
        self.r_out = move.r_out^
        self.r_go = move.r_go^
        self.acc = move.acc^
        self.ga = move.ga^

    def prepare[target: StaticString](
        mut self, ctx: Optional[DeviceContext] = None
    ) raises:
        comptime NN = Self.BATCH * Self.BATCH
        comptime ND = Self.BATCH * Self.D
        comptime if target == "cpu":
            self.m.ensure(NN)
            self.go.ensure(NN)
            self.r_out.ensure(Self.BATCH)
            self.r_go.ensure(Self.BATCH)
            self.acc.ensure(1)
            self.ga.ensure(ND)
        else:
            var c = ctx.value()
            self.m.ensure_gpu(c, NN)
            self.go.ensure_gpu(c, NN)
            self.r_out.ensure_gpu(c, Self.BATCH)
            self.r_go.ensure_gpu(c, Self.BATCH)
            self.acc.ensure(1)
            self.acc.ensure_gpu(c, 1)
            self.ga.ensure_gpu(c, ND)


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
    op.forward["cpu", BATCH](TensorRefs[2, MutAnyOrigin](ins[0], ins[1]), m, None)


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
    op.forward["cpu", BATCH](TensorRefs[2, MutAnyOrigin](ins[0], ins[1]), m, None)

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
        TensorRefs[2, MutAnyOrigin](ins[0], ins[1]), go, TensorRefs[2, MutAnyOrigin](grads[0], grads[1]),
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
    rd.forward["cpu", BATCH](TensorRefs[2, MutAnyOrigin](rins[0], rins[1]), r_out, None)

    var anchor = Float64(0)
    for i in range(BATCH):
        anchor += Float64(r_out.data[i])
    loss += -2.0 * anchor / Float64(BATCH)

    var r_go = Tensor.alloc(BATCH)
    for i in range(BATCH):
        r_go.data[i] = Scalar[DT](-2.0 / Float64(BATCH))
    var rgrads = TensorPack[2]()
    rd.vjp["cpu", BATCH](
        TensorRefs[2, MutAnyOrigin](rins[0], rins[1]), r_go,
        TensorRefs[2, MutAnyOrigin](rgrads[0], rgrads[1]), None,
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
    op.forward["cpu", BATCH](TensorRefs[2, MutAnyOrigin](ins[0], ins[1]), o, None)

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
        TensorRefs[2, MutAnyOrigin](ins[0], ins[1]), go, TensorRefs[2, MutAnyOrigin](grads[0], grads[1]),
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


# ══════════════════════════════════════════════════════════════════════
# Target-parameterized implementations (CPU + GPU).
#
# The plain entry points above allocate per call and are CPU-only; they are
# what `test_fb_loss.mojo` gates and what the M1 example uses. These take a
# persistent workspace and run on either device — the M2 path.
#
# The MATH is not duplicated: the CPU branch here is the same sequence of
# operations, expressed once. What differs is only where the elementwise work
# and the reductions run.
# ══════════════════════════════════════════════════════════════════════


def fb_measure_loss_into[
    target: StaticString, D: Int, BATCH: Int
](
    mut ws: FBLossWorkspace[D, BATCH],
    ref [MutAnyOrigin] f: Tensor,
    ref [MutAnyOrigin] b_sp: Tensor,
    ref [MutAnyOrigin] b_next: Tensor,
    ref [MutAnyOrigin] m_target: Tensor,
    ref [MutAnyOrigin] g_f: Tensor,
    ref [MutAnyOrigin] g_b_sp: Tensor,
    ref [MutAnyOrigin] g_b_next: Tensor,
    want_loss: Bool = True,
    ctx: Optional[DeviceContext] = None,
) raises -> Float64:
    """`L_FB` and its gradients, on `target`. See the module docstring.

    `want_loss=False` skips the two reduction kernels AND the device readback,
    leaving the return value at 0. The gradients are unaffected — the loss
    value never enters the update. Use it on the steps you do not log: the D2H
    read is a full device sync, and at 2 M steps paying it every step is the
    difference between a few hours and most of a day.
    """
    ws.prepare[target](ctx)
    comptime NN = BATCH * BATCH
    comptime ND = BATCH * D
    var inv_n = Scalar[DT](1.0 / (Float64(BATCH) * Float64(BATCH)))

    # M = F · B(s+)^T
    ws.pd.forward[target, BATCH](TensorRefs[2, MutAnyOrigin](f, b_sp), ws.m, ctx)

    var loss = Float64(0)
    comptime if target == "cpu":
        for i in range(NN):
            var r = Float64(ws.m.data[i]) - Float64(m_target.data[i])
            if want_loss:
                loss += r * r
            ws.go.data[i] = Scalar[DT](2.0 * r * Float64(inv_n))
        if want_loss:
            loss *= Float64(inv_n)
    else:
        var c = ctx.value()
        c.enqueue_function[residual_grad_kernel[NN]](
            ws.go.dev.value().unsafe_ptr(),
            ws.m.dev.value().unsafe_ptr(),
            m_target.dev.value().unsafe_ptr(),
            inv_n,
            grid_dim=(NN + TPB - 1) // TPB,
            block_dim=TPB,
        )
        if want_loss:
            c.enqueue_function[sq_diff_reduce_kernel[NN]](
                ws.m.dev.value().unsafe_ptr(),
                m_target.dev.value().unsafe_ptr(),
                ws.acc.dev.value().unsafe_ptr(),
                grid_dim=1,
                block_dim=TPB_REDUCE,
            )
            ws.acc.download(c)
            loss = Float64(ws.acc.data[0])

    # dF, dB(s+)
    ws.pd.vjp[target, BATCH](
        TensorRefs[2, MutAnyOrigin](f, b_sp), ws.go, TensorRefs[2, MutAnyOrigin](g_f, g_b_sp), ctx
    )

    # Anchor: -2·mean_i( F_i · B(s'_i) ), via RowDot — NOT diag(PairwiseDot).
    ws.rd.forward[target, BATCH](TensorRefs[2, MutAnyOrigin](f, b_next), ws.r_out, ctx)
    var anchor_scale = Scalar[DT](-2.0 / Float64(BATCH))
    comptime if target == "cpu":
        if want_loss:
            var s = Float64(0)
            for i in range(BATCH):
                s += Float64(ws.r_out.data[i])
            loss += -2.0 * s / Float64(BATCH)
        for i in range(BATCH):
            ws.r_go.data[i] = anchor_scale
    else:
        var c = ctx.value()
        if want_loss:
            c.enqueue_function[sum_reduce_kernel[BATCH]](
                ws.r_out.dev.value().unsafe_ptr(),
                ws.acc.dev.value().unsafe_ptr(),
                grid_dim=1,
                block_dim=TPB_REDUCE,
            )
            ws.acc.download(c)
            loss += -2.0 * Float64(ws.acc.data[0])
        c.enqueue_function[fill_kernel[BATCH]](
            ws.r_go.dev.value().unsafe_ptr(),
            anchor_scale,
            grid_dim=(BATCH + TPB - 1) // TPB,
            block_dim=TPB,
        )

    ws.rd.vjp[target, BATCH](
        TensorRefs[2, MutAnyOrigin](f, b_next), ws.r_go, TensorRefs[2, MutAnyOrigin](ws.ga, g_b_next), ctx
    )
    # g_f += the anchor's contribution.
    comptime if target == "cpu":
        for i in range(ND):
            g_f.data[i] = Scalar[DT](
                Float64(g_f.data[i]) + Float64(ws.ga.data[i])
            )
    else:
        var c = ctx.value()
        c.enqueue_function[axpy_kernel[ND]](
            g_f.dev.value().unsafe_ptr(),
            ws.ga.dev.value().unsafe_ptr(),
            Scalar[DT](1.0),
            grid_dim=(ND + TPB - 1) // TPB,
            block_dim=TPB,
        )
    return loss


def fb_ortho_loss_into[
    target: StaticString, D: Int, BATCH: Int
](
    mut ws: FBLossWorkspace[D, BATCH],
    ref [MutAnyOrigin] b_s: Tensor,
    ref [MutAnyOrigin] b_sp: Tensor,
    ref [MutAnyOrigin] g_b_s: Tensor,
    ref [MutAnyOrigin] g_b_sp: Tensor,
    want_loss: Bool = True,
    ctx: Optional[DeviceContext] = None,
) raises -> Float64:
    """`L_ortho` and its gradients, on `target`. OVERWRITES both gradients."""
    ws.prepare[target](ctx)
    comptime NN = BATCH * BATCH
    comptime ND = BATCH * D
    var inv_n = Scalar[DT](1.0 / (Float64(BATCH) * Float64(BATCH)))

    ws.pd.forward[target, BATCH](TensorRefs[2, MutAnyOrigin](b_s, b_sp), ws.m, ctx)

    var loss = Float64(0)
    comptime if target == "cpu":
        for i in range(NN):
            var v = Float64(ws.m.data[i])
            if want_loss:
                loss += v * v
            ws.go.data[i] = Scalar[DT](2.0 * v * Float64(inv_n))
        if want_loss:
            loss *= Float64(inv_n)
    else:
        var c = ctx.value()
        # The ortho target is ZERO, so the residual IS O and the gradient is
        # just a scaling: `go = 2·O/BATCH^2`. Reusing `residual_grad_kernel`
        # here would mean passing `go` as both its output and its zeroed
        # `mt` input — safe per-thread, but an aliased read/write that the
        # next person to touch this file has to re-derive. One scale instead.
        c.enqueue_function[scale_kernel[NN]](
            ws.go.dev.value().unsafe_ptr(),
            ws.m.dev.value().unsafe_ptr(),
            Scalar[DT](2.0 * Float64(inv_n)),
            grid_dim=(NN + TPB - 1) // TPB,
            block_dim=TPB,
        )
        if want_loss:
            c.enqueue_function[sumsq_reduce_kernel[NN]](
                ws.m.dev.value().unsafe_ptr(),
                ws.acc.dev.value().unsafe_ptr(),
                grid_dim=1,
                block_dim=TPB_REDUCE,
            )
            ws.acc.download(c)
            loss = Float64(ws.acc.data[0])

    ws.pd.vjp[target, BATCH](
        TensorRefs[2, MutAnyOrigin](b_s, b_sp), ws.go, TensorRefs[2, MutAnyOrigin](g_b_s, g_b_sp), ctx
    )

    # -2·mean_i ||B(s_i)||^2 ; gradient -4/BATCH · B(s).
    var c4 = Scalar[DT](-4.0 / Float64(BATCH))
    comptime if target == "cpu":
        if want_loss:
            var sq = Float64(0)
            for i in range(ND):
                var v = Float64(b_s.data[i])
                sq += v * v
            loss += -2.0 * sq / Float64(BATCH)
        for i in range(ND):
            g_b_s.data[i] = Scalar[DT](
                Float64(g_b_s.data[i]) + Float64(c4) * Float64(b_s.data[i])
            )
    else:
        var c = ctx.value()
        if want_loss:
            c.enqueue_function[sumsq_reduce_kernel[ND]](
                b_s.dev.value().unsafe_ptr(),
                ws.acc.dev.value().unsafe_ptr(),
                grid_dim=1,
                block_dim=TPB_REDUCE,
            )
            ws.acc.download(c)
            # the kernel returns the MEAN over ND, so scale back to per-ROW
            loss += -2.0 * Float64(ws.acc.data[0]) * Float64(D)
        c.enqueue_function[axpy_kernel[ND]](
            g_b_s.dev.value().unsafe_ptr(),
            b_s.dev.value().unsafe_ptr(),
            c4,
            grid_dim=(ND + TPB - 1) // TPB,
            block_dim=TPB,
        )
    return loss
