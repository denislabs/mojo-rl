"""AMP cast-around-matmul scaffolding.

Two free helpers + one state struct that absorb the bf16 cast
bookkeeping used by `Linear` when `POLICY.compute_dtype == bf16`. The
helpers
collapse those bodies to a single `cast_fp32_to_bf16` /
`cast_bf16_to_fp32` call around the bf16 `linalg.matmul`.

# Surface

  - `cast_fp32_to_bf16[target, N](src, dst, ctx?)` — CPU SIMD or GPU
    kernel, branched on `target`.
  - `cast_bf16_to_fp32[target, N](src, dst, ctx?)` — symmetric upcast.
  - `LinearAMPState[IN, OUT]` — owns the three bf16 scratches
    (w_bf16, in_bf16, ou_bf16) for a Linear-shaped leaf.

# Why no `w_dirty` flag

Earlier revisions cached the bf16 weight cast across steps and gated
on a `w_dirty` flag the optimizer was meant to flip. No optimizer ever
flipped it, so the cache went stale after the first Adam step and the
network silently trained against frozen weights (test_mnist_mlp_cpu_amp
collapsed from 97% to 59%). The cache + flag is gone — Linear re-casts
the fp32 weight on every fwd/bwd call. Cost is IN*OUT scalar ops vs
BATCH*IN*OUT in the matmul, so the dirty optimization wasn't worth the
correctness footgun.

CPU + GPU dual storage; per-instance ~24 bytes for `batch_cap` plus
the scratch lists/buffers.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor

from ..constants import DT, CPU_SIMD_W, TPB


# ──────────────────────────────────────────────────────────────────────
# GPU cast kernels (lifted from `primitives/linear.mojo`).
# ──────────────────────────────────────────────────────────────────────


def _fp32_to_bf16_kernel[
    N: Int,
](
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    dst: LayoutTensor[DType.bfloat16, Layout.row_major(N), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < N:
        var x = rebind[Scalar[DT]](src[i])
        dst[i] = x.cast[DType.bfloat16]()


def _bf16_to_fp32_kernel[
    N: Int,
](
    src: LayoutTensor[DType.bfloat16, Layout.row_major(N), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < N:
        var x = rebind[Scalar[DType.bfloat16]](src[i])
        dst[i] = x.cast[DT]()


# ──────────────────────────────────────────────────────────────────────
# cast_fp32_to_bf16 / cast_bf16_to_fp32 — per-target free helpers.
# Mirror the existing inline SIMD-cast loops in `primitives/linear.mojo`
# but as ONE definition each. Used by Linear/GaussianHead/NormedLinear
# AMP branches.
# ──────────────────────────────────────────────────────────────────────


def cast_fp32_to_bf16[target: StaticString, N: Int](
    src: UnsafePointer[Scalar[DT], MutAnyOrigin],
    dst: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Downcast `N` fp32 elements to bf16. CPU: SIMD strided loop.
    GPU: 1-D launch over N with TPB=128."""
    comptime if target == "cpu":
        var k = 0
        while k + CPU_SIMD_W <= N:
            var v = src.load[width=CPU_SIMD_W](k)
            dst.store(k, v.cast[DType.bfloat16]())
            k += CPU_SIMD_W
        while k < N:
            dst[k] = src[k].cast[DType.bfloat16]()
            k += 1
    else:
        var actx = ctx.value()
        var src_lt = LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](src)
        var dst_lt = LayoutTensor[
            DType.bfloat16, Layout.row_major(N), MutAnyOrigin,
        ](dst)
        comptime n_blocks = (N + TPB - 1) // TPB
        comptime kernel = _fp32_to_bf16_kernel[N]
        actx.enqueue_function[kernel](
            src_lt, dst_lt,
            grid_dim=n_blocks, block_dim=TPB,
        )


def cast_bf16_to_fp32[target: StaticString, N: Int](
    src: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    dst: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Upcast `N` bf16 elements to fp32. CPU: SIMD strided loop.
    GPU: 1-D launch over N with TPB=128."""
    comptime if target == "cpu":
        var k = 0
        while k + CPU_SIMD_W <= N:
            var v = src.load[width=CPU_SIMD_W](k)
            dst.store(k, v.cast[DT]())
            k += CPU_SIMD_W
        while k < N:
            dst[k] = src[k].cast[DT]()
            k += 1
    else:
        var actx = ctx.value()
        var src_lt = LayoutTensor[
            DType.bfloat16, Layout.row_major(N), MutAnyOrigin,
        ](src)
        var dst_lt = LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](dst)
        comptime n_blocks = (N + TPB - 1) // TPB
        comptime kernel = _bf16_to_fp32_kernel[N]
        actx.enqueue_function[kernel](
            src_lt, dst_lt,
            grid_dim=n_blocks, block_dim=TPB,
        )


# ──────────────────────────────────────────────────────────────────────
# LinearAMPState[IN, OUT] — bf16 scratch cluster owned by Linear-shaped
# leaves. One copy per leaf instance, lazy-grown on first bf16 call.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct LinearAMPState[IN: Int, OUT: Int](Movable & ImplicitlyDestructible):
    """Bf16 scratch cluster for the cast-around-matmul path.

    Holds:
      - `w_bf16` — IN*OUT bf16 weight scratch. Re-cast every fwd/bwd.
      - `in_bf16` — BATCH*IN bf16 scratch (input fwd, grad_in bwd).
      - `ou_bf16` — BATCH*OUT bf16 scratch (output fwd, grad_out bwd).
      - `batch_cap` — current BATCH capacity for `in_bf16` / `ou_bf16`.

    CPU + GPU dual storage; only the matching set is populated based on
    the owning leaf's TargetStorage tag."""

    var w_bf16_cpu:  List[Scalar[DType.bfloat16]]
    var in_bf16_cpu: List[Scalar[DType.bfloat16]]
    var ou_bf16_cpu: List[Scalar[DType.bfloat16]]
    var w_bf16_dev:  Optional[DeviceBuffer[DType.bfloat16]]
    var in_bf16_dev: Optional[DeviceBuffer[DType.bfloat16]]
    var ou_bf16_dev: Optional[DeviceBuffer[DType.bfloat16]]
    # GPU-only grad_w bf16 path (Fix 1): `cacheT_bf16` holds cacheᵀ[IN,BATCH]
    # downcast (batch-sized, fused with the transpose); `dW_bf16` holds the
    # bf16 dW GEMM output [IN,OUT] before the fused fp32 accumulate (fixed).
    var cacheT_bf16_dev: Optional[DeviceBuffer[DType.bfloat16]]
    var dW_bf16_dev: Optional[DeviceBuffer[DType.bfloat16]]
    # Once-per-step weight-cast reuse (Fix 2): the forward casts the fp32
    # weight → `w_bf16_dev` every step and flips this True; the backward
    # `grad_input` GEMM then reuses that cast instead of re-casting. Safe
    # because forward always precedes its own backward and the optimizer
    # runs after — so `w_bf16_dev` is fresh whenever a backward reads it.
    # NOT the reverted cross-step `w_dirty` cache (which went stale because
    # the optimizer never invalidated it): here the *forward* re-casts every
    # step, so freshness is guaranteed by the party that owns the invariant.
    var w_step_valid: Bool
    var batch_cap: Int

    @staticmethod
    def make() -> Self:
        """Empty placeholder — buffers grow lazily on first `ensure*`."""
        return Self(
            w_bf16_cpu=List[Scalar[DType.bfloat16]](),
            in_bf16_cpu=List[Scalar[DType.bfloat16]](),
            ou_bf16_cpu=List[Scalar[DType.bfloat16]](),
            w_bf16_dev=None,
            in_bf16_dev=None,
            ou_bf16_dev=None,
            cacheT_bf16_dev=None,
            dW_bf16_dev=None,
            w_step_valid=False,
            batch_cap=0,
        )

    def ensure_cpu(mut self, batch_needed: Int):
        var w_size = Self.IN * Self.OUT
        if len(self.w_bf16_cpu) < w_size:
            self.w_bf16_cpu.resize(w_size, Scalar[DType.bfloat16](0.0))
        if self.batch_cap < batch_needed:
            self.in_bf16_cpu.resize(
                batch_needed * Self.IN, Scalar[DType.bfloat16](0.0),
            )
            self.ou_bf16_cpu.resize(
                batch_needed * Self.OUT, Scalar[DType.bfloat16](0.0),
            )
            self.batch_cap = batch_needed

    def ensure_gpu(mut self, batch_needed: Int, ctx: DeviceContext) raises:
        if not self.w_bf16_dev:
            self.w_bf16_dev = ctx.enqueue_create_buffer[DType.bfloat16](
                Self.IN * Self.OUT,
            )
            # Fixed [IN,OUT] bf16 dW scratch for the grad_w bf16 GEMM (Fix 1).
            self.dW_bf16_dev = ctx.enqueue_create_buffer[DType.bfloat16](
                Self.IN * Self.OUT,
            )
        if self.batch_cap < batch_needed:
            self.in_bf16_dev = ctx.enqueue_create_buffer[DType.bfloat16](
                batch_needed * Self.IN,
            )
            self.ou_bf16_dev = ctx.enqueue_create_buffer[DType.bfloat16](
                batch_needed * Self.OUT,
            )
            # cacheᵀ[IN,BATCH] bf16 scratch — same element count as in_bf16
            # but kept separate so grad_w and grad_input never alias (Fix 1).
            self.cacheT_bf16_dev = ctx.enqueue_create_buffer[DType.bfloat16](
                batch_needed * Self.IN,
            )
            self.batch_cap = batch_needed


# ──────────────────────────────────────────────────────────────────────
# Conv2DAMPState[OC, COL] — bf16 scratch cluster for the Conv2D GPU GEMMs.
# Conv2D reduces to im2col + two GEMMs: forward `out_packed[BS,OC] =
# col[BS,COL] @ Wᵀ[OC,COL]` and backward `dW[OC,COL] = goᵀ[OC,BS] @
# col[BS,COL]`. The dx step is a gather kernel (not a GEMM) so it stays
# fp32, exactly like AMP's cast-around-matmul rule. GPU-only — Conv2D's
# CPU path stays fp32 regardless of POLICY (the benchmark / AMP value is
# the GPU tensor-core GEMM). `BS = BATCH·SPATIAL_OUT`; lazily grown.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct Conv2DAMPState[OC: Int, COL: Int](Movable & ImplicitlyDestructible):
    """Bf16 scratch for the two Conv2D GPU GEMMs.

    Fixed (OC*COL): `w_bf16` (weight downcast, fwd) + `dW_bf16` (dW GEMM
    output, bwd). BS-sized (grown lazily): `col_bf16` (im2col downcast,
    fwd + bwd), `outp_bf16` (fwd GEMM output), `goT_bf16` (transposed
    grad_output downcast, bwd). Weight is re-cast every call (no stale
    cache — see LinearAMPState's `w_dirty` note)."""

    var w_bf16_dev: Optional[DeviceBuffer[DType.bfloat16]]
    var dW_bf16_dev: Optional[DeviceBuffer[DType.bfloat16]]
    var col_bf16_dev: Optional[DeviceBuffer[DType.bfloat16]]
    var outp_bf16_dev: Optional[DeviceBuffer[DType.bfloat16]]
    var goT_bf16_dev: Optional[DeviceBuffer[DType.bfloat16]]
    var bs_cap: Int

    @staticmethod
    def make() -> Self:
        return Self(
            w_bf16_dev=None,
            dW_bf16_dev=None,
            col_bf16_dev=None,
            outp_bf16_dev=None,
            goT_bf16_dev=None,
            bs_cap=0,
        )

    def ensure_gpu(mut self, bs_needed: Int, ctx: DeviceContext) raises:
        if not self.w_bf16_dev:
            self.w_bf16_dev = ctx.enqueue_create_buffer[DType.bfloat16](
                Self.OC * Self.COL,
            )
            self.dW_bf16_dev = ctx.enqueue_create_buffer[DType.bfloat16](
                Self.OC * Self.COL,
            )
        if self.bs_cap < bs_needed:
            self.col_bf16_dev = ctx.enqueue_create_buffer[DType.bfloat16](
                bs_needed * Self.COL,
            )
            self.outp_bf16_dev = ctx.enqueue_create_buffer[DType.bfloat16](
                bs_needed * Self.OC,
            )
            self.goT_bf16_dev = ctx.enqueue_create_buffer[DType.bfloat16](
                Self.OC * bs_needed,
            )
            self.bs_cap = bs_needed
