"""Device kernels for the FB training step.

Everything the CPU trainer does with host loops — gathering rows, packing
`[s | a | z]`, forming the twin-min target, the loss residuals, the gradient
sums — has to happen on device for M2, because the alternative is a per-step
host round trip. At `BATCH = 1024`, `D = 128` and walker's `OBS = 24`, packing
`[s|a|z]` on the host and uploading it costs ~650 KB per step; over 2 M steps
that is 1.3 TB across PCIe to avoid arithmetic that takes microseconds on the
GPU. So the dataset stays resident on device, the sampler writes indices on
device (`UniformDeviceSampler.draw_into_device`), and the batch is assembled
by the gather/pack kernels below without ever touching the host.

⚠ **Naive kernels on purpose.** One thread per output element, inner loops over
the contracted axis, no shared-memory tiling except in the reductions. Heavy
blocked kernels hard-crash the Metal compiler on Apple, and this file has to
compile on the laptop the parity gates run on even though the real target is
NVIDIA. The GEMM-shaped work lives in `PairwiseDot`, which carries the same
constraint for the same reason.

⚠ The reductions use the single-block grid-stride form (`block.sum` over a
`TPB_REDUCE` block, each thread striding the whole array) rather than a
multi-block partial-sum pass. Two reasons: it is one launch with no second
buffer, and it avoids a conditional read-modify-write across blocks — the
pattern this project has already been bitten by on NVIDIA, where CUDA drops
conditional RMW stores in reduction kernels.
"""

from std.math import sqrt
from std.gpu import block_dim, block_idx, thread_idx, global_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT, TPB, TPB_REDUCE
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.ptr import mptr
from mojo_rl.nn.random.box_muller import (
    box_muller_normal,
    box_muller_normal_gpu,
)
from mojo_rl.data.resident import IDX_DT


# ══════════════════════════════════════════════════════════════════════
# Gather / pack
# ══════════════════════════════════════════════════════════════════════


def gather_rows_kernel[ROW_DIM: Int, BATCH: Int](
    src: Pointer[Scalar[DT], MutAnyOrigin],
    idx: Pointer[Scalar[IDX_DT], MutAnyOrigin],
    dst: Pointer[Scalar[DT], MutAnyOrigin],
):
    """`dst[i, d] = src[idx[i], d]`.

    Element-parallel, not lane-parallel: a per-lane kernel serialises the row
    copy inside each thread and launches only BATCH threads, which measured at
    ~73% of GPU time on wide rows elsewhere in this repo.
    """
    var t = Int(global_idx.x)
    if t >= BATCH * ROW_DIM:
        return
    var i = t // ROW_DIM
    var d = t % ROW_DIM
    dst[unsafe_offset=t] = src[unsafe_offset=Int(idx[unsafe_offset=i]) * ROW_DIM + d]


def gather_idx_kernel[BATCH: Int](
    table: Pointer[Scalar[IDX_DT], MutAnyOrigin],
    idx: Pointer[Scalar[IDX_DT], MutAnyOrigin],
    dst: Pointer[Scalar[IDX_DT], MutAnyOrigin],
):
    """`dst[i] = table[idx[i]]` — remaps sampled rows through a lookup.

    Used for the `s'` indices: a precomputed `next_row` table encodes the
    episode boundaries, so `s'` is never the first row of the FOLLOWING
    episode. Doing it as a table lookup keeps the boundary logic on the host
    where it is written once, and off the hot path entirely.

    Separate from `gather_rows_kernel` because the index dtype is `IDX_DT`,
    not `DT`: routing indices through the float gather would lose exactness
    above 2^24 rows, which a 10 M-row dataset is not far from.
    """
    var i = Int(global_idx.x)
    if i < BATCH:
        dst[unsafe_offset=i] = table[unsafe_offset=Int(idx[unsafe_offset=i])]


def pack3_kernel[A_DIM: Int, B_DIM: Int, C_DIM: Int, BATCH: Int](
    a: Pointer[Scalar[DT], MutAnyOrigin],
    b: Pointer[Scalar[DT], MutAnyOrigin],
    c: Pointer[Scalar[DT], MutAnyOrigin],
    dst: Pointer[Scalar[DT], MutAnyOrigin],
):
    """`dst[i] = [a[i] | b[i] | c[i]]` — the `[s | a | z]` row of the F net."""
    comptime W = A_DIM + B_DIM + C_DIM
    var t = Int(global_idx.x)
    if t >= BATCH * W:
        return
    var i = t // W
    var k = t % W
    if k < A_DIM:
        dst[unsafe_offset=t] = a[unsafe_offset=i * A_DIM + k]
    elif k < A_DIM + B_DIM:
        dst[unsafe_offset=t] = b[unsafe_offset=i * B_DIM + (k - A_DIM)]
    else:
        dst[unsafe_offset=t] = c[unsafe_offset=i * C_DIM + (k - A_DIM - B_DIM)]


def pack2_kernel[A_DIM: Int, B_DIM: Int, BATCH: Int](
    a: Pointer[Scalar[DT], MutAnyOrigin],
    b: Pointer[Scalar[DT], MutAnyOrigin],
    dst: Pointer[Scalar[DT], MutAnyOrigin],
):
    """`dst[i] = [a[i] | b[i]]` — the `[s | z]` row of the actor."""
    comptime W = A_DIM + B_DIM
    var t = Int(global_idx.x)
    if t >= BATCH * W:
        return
    var i = t // W
    var k = t % W
    dst[unsafe_offset=t] = a[unsafe_offset=i * A_DIM + k] if k < A_DIM else b[unsafe_offset=i * B_DIM + (k - A_DIM)]


def slice_cols_kernel[
    SRC_W: Int, OFFSET: Int, OUT_W: Int, BATCH: Int
](
    src: Pointer[Scalar[DT], MutAnyOrigin],
    dst: Pointer[Scalar[DT], MutAnyOrigin],
):
    """`dst[i, k] = src[i, OFFSET + k]` — pulls the ACTION slice back out of
    the F net's input gradient for the actor update."""
    var t = Int(global_idx.x)
    if t >= BATCH * OUT_W:
        return
    var i = t // OUT_W
    var k = t % OUT_W
    dst[unsafe_offset=t] = src[unsafe_offset=i * SRC_W + OFFSET + k]


# ══════════════════════════════════════════════════════════════════════
# Elementwise
# ══════════════════════════════════════════════════════════════════════


def fill_kernel[N: Int](
    y: Pointer[Scalar[DT], MutAnyOrigin], v: Scalar[DT]
):
    var t = Int(global_idx.x)
    if t < N:
        y[unsafe_offset=t] = v


def axpy_kernel[N: Int](
    y: Pointer[Scalar[DT], MutAnyOrigin],
    x: Pointer[Scalar[DT], MutAnyOrigin],
    alpha: Scalar[DT],
):
    """`y += alpha * x`."""
    var t = Int(global_idx.x)
    if t < N:
        y[unsafe_offset=t] = y[unsafe_offset=t] + alpha * x[unsafe_offset=t]


def scale_kernel[N: Int](
    y: Pointer[Scalar[DT], MutAnyOrigin],
    x: Pointer[Scalar[DT], MutAnyOrigin],
    alpha: Scalar[DT],
):
    """`y = alpha * x`."""
    var t = Int(global_idx.x)
    if t < N:
        y[unsafe_offset=t] = alpha * x[unsafe_offset=t]


def sum3_scaled_kernel[N: Int](
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    a: Pointer[Scalar[DT], MutAnyOrigin],
    b: Pointer[Scalar[DT], MutAnyOrigin],
    c: Pointer[Scalar[DT], MutAnyOrigin],
    w: Scalar[DT],
):
    """`dst = a + b + w*c` — the three gradients arriving at `B(s+)`: one per
    twin from the measure loss, plus the orthonormality term."""
    var t = Int(global_idx.x)
    if t < N:
        dst[unsafe_offset=t] = a[unsafe_offset=t] + b[unsafe_offset=t] + w * c[unsafe_offset=t]


def min_scale_kernel[N: Int](
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    m1: Pointer[Scalar[DT], MutAnyOrigin],
    m2: Pointer[Scalar[DT], MutAnyOrigin],
    gamma: Scalar[DT],
):
    """`dst = gamma * min(m1, m2)` — the twin-min target, ENTRYWISE.

    Every `(i, j)` pair of the successor-measure matrix is its own value
    estimate, so the min is taken per element, not per row.
    """
    var t = Int(global_idx.x)
    if t < N:
        var a = m1[unsafe_offset=t]
        var b = m2[unsafe_offset=t]
        dst[unsafe_offset=t] = gamma * (a if a < b else b)


def residual_grad_kernel[N: Int](
    go: Pointer[Scalar[DT], MutAnyOrigin],
    m: Pointer[Scalar[DT], MutAnyOrigin],
    mt: Pointer[Scalar[DT], MutAnyOrigin],
    inv_n: Scalar[DT],
):
    """`go = 2·(m - mt)·inv_n` — the upstream gradient of `mean((m-mt)^2)`.

    `inv_n` is `1/BATCH^2`, passed in rather than recomputed so the CPU and GPU
    paths divide by exactly the same constant.
    """
    var t = Int(global_idx.x)
    if t < N:
        go[unsafe_offset=t] = Scalar[DT](2.0) * (m[unsafe_offset=t] - mt[unsafe_offset=t]) * inv_n


def sq_diff_reduce_kernel[N: Int](
    m: Pointer[Scalar[DT], MutAnyOrigin],
    mt: Pointer[Scalar[DT], MutAnyOrigin],
    acc: Pointer[Scalar[DT], MutAnyOrigin],
):
    """`acc[0] = mean((m - mt)^2)` over `[N]`. ONE block, grid-stride."""
    var t = Int(thread_idx.x)
    var my_sum: Scalar[DT] = 0.0
    var k = t
    while k < N:
        var r = m[unsafe_offset=k] - mt[unsafe_offset=k]
        my_sum += r * r
        k += TPB_REDUCE
    var total = block.sum[block_size=TPB_REDUCE, broadcast=False](val=my_sum)
    if t == 0:
        acc[unsafe_offset=0] = total[0] / Scalar[DT](N)


def sum_reduce_kernel[N: Int](
    x: Pointer[Scalar[DT], MutAnyOrigin],
    acc: Pointer[Scalar[DT], MutAnyOrigin],
):
    """`acc[0] = mean(x)` over `[N]`. ONE block, grid-stride."""
    var t = Int(thread_idx.x)
    var my_sum: Scalar[DT] = 0.0
    var k = t
    while k < N:
        my_sum += x[unsafe_offset=k]
        k += TPB_REDUCE
    var total = block.sum[block_size=TPB_REDUCE, broadcast=False](val=my_sum)
    if t == 0:
        acc[unsafe_offset=0] = total[0] / Scalar[DT](N)


def sumsq_reduce_kernel[N: Int](
    x: Pointer[Scalar[DT], MutAnyOrigin],
    acc: Pointer[Scalar[DT], MutAnyOrigin],
):
    """`acc[0] = mean(x^2)` over `[N]`. ONE block, grid-stride."""
    var t = Int(thread_idx.x)
    var my_sum: Scalar[DT] = 0.0
    var k = t
    while k < N:
        var v = x[unsafe_offset=k]
        my_sum += v * v
        k += TPB_REDUCE
    var total = block.sum[block_size=TPB_REDUCE, broadcast=False](val=my_sum)
    if t == 0:
        acc[unsafe_offset=0] = total[0] / Scalar[DT](N)


def smooth_action_kernel[N: Int](
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    pi: Pointer[Scalar[DT], MutAnyOrigin],
    noise: Pointer[Scalar[DT], MutAnyOrigin],
    sigma: Scalar[DT],
    clip: Scalar[DT],
):
    """TD3 target-policy smoothing: `clamp(pi + clamp(sigma·n, ±clip), ±1)`."""
    var t = Int(global_idx.x)
    if t >= N:
        return
    var n = noise[unsafe_offset=t] * sigma
    if n > clip:
        n = clip
    elif n < -clip:
        n = -clip
    var v = pi[unsafe_offset=t] + n
    if v > Scalar[DT](1.0):
        v = Scalar[DT](1.0)
    elif v < Scalar[DT](-1.0):
        v = Scalar[DT](-1.0)
    dst[unsafe_offset=t] = v


def project_sphere_kernel[D: Int, BATCH: Int](
    z: Pointer[Scalar[DT], MutAnyOrigin], radius: Scalar[DT]
):
    """Rescale each row of `z` onto the radius-`sqrt(D)` sphere, one thread per
    ROW.

    ⚠ This is the device twin of `z_sampler._project_to_sphere`, and it exists
    for the same reason that one does: a `z` off the sphere crashes nothing and
    trains to a policy that emits plausible garbage. Both must agree, so the
    degenerate-row rule is the same — a row with no direction becomes `radius`
    on the first axis rather than being rescaled by `radius/~0`, which would
    amplify rounding noise into a full-magnitude z pointing nowhere.
    """
    var i = Int(global_idx.x)
    if i >= BATCH:
        return
    var base = i * D
    var acc: Scalar[DT] = 0.0
    for k in range(D):
        var v = z[unsafe_offset=base + k]
        acc += v * v
    var n = sqrt(acc)
    if n < Scalar[DT](1e-12):
        for k in range(D):
            z[unsafe_offset=base + k] = Scalar[DT](0)
        z[unsafe_offset=base] = radius
        return
    var s = radius / n
    for k in range(D):
        z[unsafe_offset=base + k] = z[unsafe_offset=base + k] * s


def z_mixture_kernel[D: Int, BATCH: Int](
    z: Pointer[Scalar[DT], MutAnyOrigin],
    gauss: Pointer[Scalar[DT], MutAnyOrigin],
    b_states: Pointer[Scalar[DT], MutAnyOrigin],
    pick: Pointer[Scalar[DT], MutAnyOrigin],
    uniform_frac: Scalar[DT],
    n_b_rows: Int,
):
    """Half the rows Gaussian, half copied from `B(s+)`. One thread per ROW.

    `pick[i]` carries two independent uniforms per row: `pick[2i]` chooses the
    branch, `pick[2i+1]` chooses which `B` row to copy. Drawing them outside
    keeps this kernel free of RNG state, so the SAME buffer replays identically
    — which is what makes a CPU/GPU parity gate on the mixture possible at all.

    Renormalisation is NOT done here — `project_sphere_kernel` runs immediately
    after, unconditionally, on both branches.
    """
    var i = Int(global_idx.x)
    if i >= BATCH:
        return
    var base = i * D
    var use_uniform = n_b_rows <= 0 or pick[unsafe_offset=2 * i] < uniform_frac
    if use_uniform:
        for k in range(D):
            z[unsafe_offset=base + k] = gauss[unsafe_offset=base + k]
    else:
        var src = Int(pick[unsafe_offset=2 * i + 1] * Scalar[DT](n_b_rows))
        if src >= n_b_rows:
            src = n_b_rows - 1
        if src < 0:
            src = 0
        for k in range(D):
            z[unsafe_offset=base + k] = b_states[unsafe_offset=src * D + k]


# ══════════════════════════════════════════════════════════════════════
# Target-parameterized host-side ops.
#
# These exist so `FBTrainer.train_step` is written ONCE rather than as a CPU
# body and a GPU body that drift apart. Each pairs a host loop with the launch
# of its kernel above; the trainer never writes `comptime if target ==` itself.
#
# Grid dims are computed here, in one place, so a kernel whose element count is
# BATCH*W is never launched with BATCH blocks by mistake.
# ══════════════════════════════════════════════════════════════════════


def _blocks(n: Int) -> Int:
    return (n + TPB - 1) // TPB


def ensure_t[target: StaticString](
    mut t: Tensor, n: Int, ctx: Optional[DeviceContext] = None
) raises:
    """Size a tensor for `target`. On GPU the HOST mirror is sized too — the
    diagnostics and the parity gate read `.data`, and `ensure_gpu` alone leaves
    it empty (a fill would then index out of bounds)."""
    comptime if target == "cpu":
        t.ensure(n)
    else:
        t.ensure(n)
        t.ensure_gpu(ctx.value(), n)


def pack3_t[
    target: StaticString, A_DIM: Int, B_DIM: Int, C_DIM: Int, BATCH: Int
](
    mut dst: Tensor, mut a: Tensor, mut b: Tensor, mut c: Tensor,
    ctx: Optional[DeviceContext] = None,
) raises:
    """`dst[i] = [a[i] | b[i] | c[i]]`."""
    comptime W = A_DIM + B_DIM + C_DIM
    ensure_t[target](dst, BATCH * W, ctx)
    comptime if target == "cpu":
        for i in range(BATCH):
            var o = i * W
            for k in range(A_DIM):
                dst.data[o + k] = a.data[i * A_DIM + k]
            for k in range(B_DIM):
                dst.data[o + A_DIM + k] = b.data[i * B_DIM + k]
            for k in range(C_DIM):
                dst.data[o + A_DIM + B_DIM + k] = c.data[i * C_DIM + k]
    else:
        var d = ctx.value()
        d.enqueue_function[pack3_kernel[A_DIM, B_DIM, C_DIM, BATCH]](
            a.dev.value().unsafe_ptr(), b.dev.value().unsafe_ptr(),
            c.dev.value().unsafe_ptr(), dst.dev.value().unsafe_ptr(),
            grid_dim=_blocks(BATCH * W), block_dim=TPB,
        )


def pack2_t[
    target: StaticString, A_DIM: Int, B_DIM: Int, BATCH: Int
](
    mut dst: Tensor, mut a: Tensor, mut b: Tensor,
    ctx: Optional[DeviceContext] = None,
) raises:
    """`dst[i] = [a[i] | b[i]]`."""
    comptime W = A_DIM + B_DIM
    ensure_t[target](dst, BATCH * W, ctx)
    comptime if target == "cpu":
        for i in range(BATCH):
            var o = i * W
            for k in range(A_DIM):
                dst.data[o + k] = a.data[i * A_DIM + k]
            for k in range(B_DIM):
                dst.data[o + A_DIM + k] = b.data[i * B_DIM + k]
    else:
        var d = ctx.value()
        d.enqueue_function[pack2_kernel[A_DIM, B_DIM, BATCH]](
            a.dev.value().unsafe_ptr(), b.dev.value().unsafe_ptr(),
            dst.dev.value().unsafe_ptr(),
            grid_dim=_blocks(BATCH * W), block_dim=TPB,
        )


def axpy_t[target: StaticString, N: Int](
    mut y: Tensor, mut x: Tensor, alpha: Scalar[DT],
    ctx: Optional[DeviceContext] = None,
) raises:
    """`y += alpha * x`."""
    comptime if target == "cpu":
        for i in range(N):
            y.data[i] = y.data[i] + alpha * x.data[i]
    else:
        var d = ctx.value()
        d.enqueue_function[axpy_kernel[N]](
            y.dev.value().unsafe_ptr(), x.dev.value().unsafe_ptr(), alpha,
            grid_dim=_blocks(N), block_dim=TPB,
        )


def scale_t[target: StaticString, N: Int](
    mut y: Tensor, mut x: Tensor, alpha: Scalar[DT],
    ctx: Optional[DeviceContext] = None,
) raises:
    """`y = alpha * x`."""
    ensure_t[target](y, N, ctx)
    comptime if target == "cpu":
        for i in range(N):
            y.data[i] = alpha * x.data[i]
    else:
        var d = ctx.value()
        d.enqueue_function[scale_kernel[N]](
            y.dev.value().unsafe_ptr(), x.dev.value().unsafe_ptr(), alpha,
            grid_dim=_blocks(N), block_dim=TPB,
        )


def sum3_scaled_t[target: StaticString, N: Int](
    mut dst: Tensor, mut a: Tensor, mut b: Tensor, mut c: Tensor,
    w: Scalar[DT], ctx: Optional[DeviceContext] = None,
) raises:
    """`dst = a + b + w*c`."""
    ensure_t[target](dst, N, ctx)
    comptime if target == "cpu":
        for i in range(N):
            dst.data[i] = a.data[i] + b.data[i] + w * c.data[i]
    else:
        var d = ctx.value()
        d.enqueue_function[sum3_scaled_kernel[N]](
            dst.dev.value().unsafe_ptr(), a.dev.value().unsafe_ptr(),
            b.dev.value().unsafe_ptr(), c.dev.value().unsafe_ptr(), w,
            grid_dim=_blocks(N), block_dim=TPB,
        )


def min_scale_t[target: StaticString, N: Int](
    mut dst: Tensor, mut m1: Tensor, mut m2: Tensor, gamma: Scalar[DT],
    ctx: Optional[DeviceContext] = None,
) raises:
    """`dst = gamma * min(m1, m2)`, entrywise."""
    ensure_t[target](dst, N, ctx)
    comptime if target == "cpu":
        for i in range(N):
            var a = m1.data[i]
            var b = m2.data[i]
            dst.data[i] = gamma * (a if a < b else b)
    else:
        var d = ctx.value()
        d.enqueue_function[min_scale_kernel[N]](
            dst.dev.value().unsafe_ptr(), m1.dev.value().unsafe_ptr(),
            m2.dev.value().unsafe_ptr(), gamma,
            grid_dim=_blocks(N), block_dim=TPB,
        )


def smooth_action_t[target: StaticString, N: Int](
    mut dst: Tensor, mut pi: Tensor, mut noise: Tensor,
    sigma: Scalar[DT], clip: Scalar[DT],
    ctx: Optional[DeviceContext] = None,
) raises:
    """`clamp(pi + clamp(sigma·noise, ±clip), ±1)`."""
    ensure_t[target](dst, N, ctx)
    comptime if target == "cpu":
        for i in range(N):
            var n = noise.data[i] * sigma
            if n > clip:
                n = clip
            elif n < -clip:
                n = -clip
            var v = pi.data[i] + n
            if v > Scalar[DT](1.0):
                v = Scalar[DT](1.0)
            elif v < Scalar[DT](-1.0):
                v = Scalar[DT](-1.0)
            dst.data[i] = v
    else:
        var d = ctx.value()
        d.enqueue_function[smooth_action_kernel[N]](
            dst.dev.value().unsafe_ptr(), pi.dev.value().unsafe_ptr(),
            noise.dev.value().unsafe_ptr(), sigma, clip,
            grid_dim=_blocks(N), block_dim=TPB,
        )


def slice_cols_t[
    target: StaticString, SRC_W: Int, OFFSET: Int, OUT_W: Int, BATCH: Int
](
    mut dst: Tensor, mut src: Tensor, ctx: Optional[DeviceContext] = None
) raises:
    """`dst[i, k] = src[i, OFFSET + k]`."""
    ensure_t[target](dst, BATCH * OUT_W, ctx)
    comptime if target == "cpu":
        for i in range(BATCH):
            for k in range(OUT_W):
                dst.data[i * OUT_W + k] = src.data[i * SRC_W + OFFSET + k]
    else:
        var d = ctx.value()
        d.enqueue_function[slice_cols_kernel[SRC_W, OFFSET, OUT_W, BATCH]](
            src.dev.value().unsafe_ptr(), dst.dev.value().unsafe_ptr(),
            grid_dim=_blocks(BATCH * OUT_W), block_dim=TPB,
        )


def mean_sq_t[target: StaticString, N: Int](
    mut x: Tensor, mut acc: Tensor, ctx: Optional[DeviceContext] = None
) raises -> Float64:
    """`mean(x^2)` over `[N]`. GPU path costs a device sync — diagnostics only.
    """
    comptime if target == "cpu":
        var s = Float64(0)
        for i in range(N):
            var v = Float64(x.data[i])
            s += v * v
        return s / Float64(N)
    else:
        var d = ctx.value()
        d.enqueue_function[sumsq_reduce_kernel[N]](
            x.dev.value().unsafe_ptr(), acc.dev.value().unsafe_ptr(),
            grid_dim=1, block_dim=TPB_REDUCE,
        )
        acc.download(d)
        return Float64(acc.data[0])


def mean_t[target: StaticString, N: Int](
    mut x: Tensor, mut acc: Tensor, ctx: Optional[DeviceContext] = None
) raises -> Float64:
    """`mean(x)` over `[N]`. GPU path costs a device sync — diagnostics only."""
    comptime if target == "cpu":
        var s = Float64(0)
        for i in range(N):
            s += Float64(x.data[i])
        return s / Float64(N)
    else:
        var d = ctx.value()
        d.enqueue_function[sum_reduce_kernel[N]](
            x.dev.value().unsafe_ptr(), acc.dev.value().unsafe_ptr(),
            grid_dim=1, block_dim=TPB_REDUCE,
        )
        acc.download(d)
        return Float64(acc.data[0])


def gaussian_t[target: StaticString, N: Int](
    mut t: Tensor, seed: UInt64, offset: UInt64,
    ctx: Optional[DeviceContext] = None,
) raises:
    """Fill `[N]` with iid N(0,1).

    ⚠ CPU and GPU draw from DIFFERENT generators (host RNG vs Philox), so a
    CPU/GPU parity gate on the trainer must either zero the exploration noise
    or compare distributions rather than values. The alternative — a host draw
    uploaded every step — would put a PCIe round trip in the hot loop to make a
    test easier, which is the wrong trade.
    """
    ensure_t[target](t, N, ctx)
    comptime if target == "cpu":
        box_muller_normal(t.data.unsafe_ptr(), N)
    else:
        var d = ctx.value()
        box_muller_normal_gpu[N](
            d, mptr(t.dev.value().unsafe_ptr()), seed, offset
        )
