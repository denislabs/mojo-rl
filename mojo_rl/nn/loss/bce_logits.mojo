"""Binary cross-entropy WITH logits — the GAN / discriminator loss.

    loss_i = softplus(l_i) − y_i · l_i            ( = −y log σ(l) − (1−y) log(1−σ(l)) )
    ∂loss_i/∂l_i = σ(l_i) − y_i

Written on the LOGIT, never on a probability: `−log(σ(l))` for a large
negative `l` is `−log(≈0)`, and the two-step form loses the value entirely
in fp32 while this one returns `−l` exactly. BFM-Zero's `fb_cpr/agent.py`
writes the discriminator loss as `−logsigmoid(expert) + softplus(train)`;
that is this function with `y = 1` and `y = 0`.

Two entry points, both target-parameterised (CPU + GPU, one body each):

* `bce_logits_const_t[target, N](logits, label, scale, cot, loss_rows)` —
  one label for the whole batch (the discriminator's "all expert" / "all
  policy" halves). Writes the per-row cotangent `scale · (σ(l) − y)` and
  the per-row loss; reduce the latter with `mean_into_t` when a value is
  wanted. No D2H inside.
* `bce_logits_rows_t[...]` — per-row labels, for a mixed batch.

`scale` is the factor the caller folds in for the mean (`1/N`) and for any
loss weight, so the cotangent is ready to hand to `vjp` as-is.

Stability: `softplus(x) = max(x, 0) + log(1 + exp(−|x|))`, and `σ` is
evaluated on the sign that keeps the exponent negative.

Gate: `tests/nn/test_bce_logits.mojo` — closed form on a grid of logits,
the cotangent against a finite difference of the loss, and the two entry
points agreeing on a constant label.
"""

from std.math import exp, log
from std.gpu import global_idx
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.core.tensor import Tensor


@always_inline
def _sigmoid(x: Scalar[DT]) -> Scalar[DT]:
    if x >= Scalar[DT](0):
        return Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-x))
    var e = exp(x)
    return e / (Scalar[DT](1.0) + e)


@always_inline
def _softplus(x: Scalar[DT]) -> Scalar[DT]:
    var a = x if x >= Scalar[DT](0) else -x
    var m = x if x > Scalar[DT](0) else Scalar[DT](0)
    return m + log(Scalar[DT](1.0) + exp(-a))


@always_inline
def _bce_row(
    l: Scalar[DT], y: Scalar[DT], scale: Scalar[DT],
    mut cot: Scalar[DT], mut loss: Scalar[DT],
):
    cot = scale * (_sigmoid(l) - y)
    loss = _softplus(l) - y * l


def bce_logits_const_kernel[N: Int](
    logits: Pointer[Scalar[DT], MutAnyOrigin],
    cot: Pointer[Scalar[DT], MutAnyOrigin],
    loss_rows: Pointer[Scalar[DT], MutAnyOrigin],
    label: Scalar[DT],
    scale: Scalar[DT],
):
    var i = Int(global_idx.x)
    if i >= N:
        return
    var c: Scalar[DT] = 0
    var lo: Scalar[DT] = 0
    _bce_row(logits[unsafe_offset=i], label, scale, c, lo)
    cot[unsafe_offset=i] = c
    loss_rows[unsafe_offset=i] = lo


def bce_logits_rows_kernel[N: Int](
    logits: Pointer[Scalar[DT], MutAnyOrigin],
    labels: Pointer[Scalar[DT], MutAnyOrigin],
    cot: Pointer[Scalar[DT], MutAnyOrigin],
    loss_rows: Pointer[Scalar[DT], MutAnyOrigin],
    scale: Scalar[DT],
):
    var i = Int(global_idx.x)
    if i >= N:
        return
    var c: Scalar[DT] = 0
    var lo: Scalar[DT] = 0
    _bce_row(logits[unsafe_offset=i], labels[unsafe_offset=i], scale, c, lo)
    cot[unsafe_offset=i] = c
    loss_rows[unsafe_offset=i] = lo


def _ensure[target: StaticString](
    mut t: Tensor, n: Int, ctx: Optional[DeviceContext]
) raises:
    t.ensure(n)
    comptime if target == "gpu":
        t.ensure_gpu(ctx.value(), n)


def bce_logits_const_t[target: StaticString, N: Int](
    mut logits: Tensor, label: Float64, scale: Float64,
    mut cot: Tensor, mut loss_rows: Tensor,
    ctx: Optional[DeviceContext] = None,
) raises:
    """`cot[i] = scale·(σ(l_i) − label)`, `loss_rows[i] = softplus(l_i) − label·l_i`."""
    _ensure[target](cot, N, ctx)
    _ensure[target](loss_rows, N, ctx)
    comptime if target == "cpu":
        var y = Scalar[DT](label)
        var s = Scalar[DT](scale)
        for i in range(N):
            var c: Scalar[DT] = 0
            var lo: Scalar[DT] = 0
            _bce_row(logits.data[i], y, s, c, lo)
            cot.data[i] = c
            loss_rows.data[i] = lo
    else:
        var d = ctx.value()
        d.enqueue_function[bce_logits_const_kernel[N]](
            logits.dev.value().unsafe_ptr(), cot.dev.value().unsafe_ptr(),
            loss_rows.dev.value().unsafe_ptr(),
            Scalar[DT](label), Scalar[DT](scale),
            grid_dim=(N + TPB - 1) // TPB, block_dim=TPB,
        )


def bce_logits_rows_t[target: StaticString, N: Int](
    mut logits: Tensor, mut labels: Tensor, scale: Float64,
    mut cot: Tensor, mut loss_rows: Tensor,
    ctx: Optional[DeviceContext] = None,
) raises:
    """Per-row labels in `[0, 1]`; otherwise `bce_logits_const_t`."""
    _ensure[target](cot, N, ctx)
    _ensure[target](loss_rows, N, ctx)
    comptime if target == "cpu":
        var s = Scalar[DT](scale)
        for i in range(N):
            var c: Scalar[DT] = 0
            var lo: Scalar[DT] = 0
            _bce_row(logits.data[i], labels.data[i], s, c, lo)
            cot.data[i] = c
            loss_rows.data[i] = lo
    else:
        var d = ctx.value()
        d.enqueue_function[bce_logits_rows_kernel[N]](
            logits.dev.value().unsafe_ptr(), labels.dev.value().unsafe_ptr(),
            cot.dev.value().unsafe_ptr(), loss_rows.dev.value().unsafe_ptr(),
            Scalar[DT](scale),
            grid_dim=(N + TPB - 1) // TPB, block_dim=TPB,
        )
