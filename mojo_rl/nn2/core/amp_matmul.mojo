"""AMP cast-around-matmul scaffolding (audit Follow-up #2).

Two free helpers + one state struct that absorb the bf16 cast bookkeeping
currently duplicated in `Linear`. Today Linear inlines six copies of
the same SIMD cast loop (3 method paths × 2 directions); after the
retrofit those bodies collapse to ~3 lines: bump w_dirty if needed,
call `cast_fp32_to_bf16`, call `linalg.matmul[target=...]`, call
`cast_bf16_to_fp32`.

# Surface

  - `cast_fp32_to_bf16[target, N](src, dst, ctx?)` — CPU SIMD or GPU
    kernel, branched on `target`.
  - `cast_bf16_to_fp32[target, N](src, dst, ctx?)` — symmetric upcast.
  - `LinearAMPState[IN, OUT]` — owns the three bf16 scratches plus a
    `w_dirty` flag the caller flips after every parameter update.

# Why a `w_dirty` flag

Weights only change on `optimizer.step()`. Today Linear re-casts the
full IN×OUT bf16 weight on every forward AND every backward. With
`w_dirty`, we cast once per Adam step. Activations/grad_outputs DO
change every call, so those casts stay.

The optimizer is responsible for flipping the flag — when a parameter
update lands on this Linear's weight, the optimizer (or whatever drove
the update) sets `linear.amp.w_dirty = True`.

CPU + GPU dual storage; per-instance ~24 bytes for the flags/caps plus
the scratch lists/buffers.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor

from ..constants import DT, CPU_SIMD_W


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
        comptime TPB = 128
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
        comptime TPB = 128
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
    """bf16 scratch cluster for the cast-around-matmul path.

    Holds:
      - `w_bf16` — IN*OUT bf16 weight cast. Re-cast only when `w_dirty`.
      - `in_bf16` — BATCH*IN bf16 scratch (input fwd, grad_in bwd).
      - `ou_bf16` — BATCH*OUT bf16 scratch (output fwd, grad_out bwd).
      - `batch_cap` — current BATCH capacity for `in_bf16` / `ou_bf16`.
      - `w_dirty` — True if `w_bf16` is stale vs. the fp32 weight.

    CPU + GPU dual storage; only the matching set is populated based on
    the owning leaf's TargetStorage tag."""

    var w_bf16_cpu:  List[Scalar[DType.bfloat16]]
    var in_bf16_cpu: List[Scalar[DType.bfloat16]]
    var ou_bf16_cpu: List[Scalar[DType.bfloat16]]
    var w_bf16_dev:  Optional[DeviceBuffer[DType.bfloat16]]
    var in_bf16_dev: Optional[DeviceBuffer[DType.bfloat16]]
    var ou_bf16_dev: Optional[DeviceBuffer[DType.bfloat16]]
    var batch_cap: Int
    var w_dirty: Bool

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
            batch_cap=0,
            w_dirty=True,
        )

    def ensure_cpu(mut self, batch_needed: Int):
        var w_size = Self.IN * Self.OUT
        if len(self.w_bf16_cpu) < w_size:
            self.w_bf16_cpu.resize(w_size, Scalar[DType.bfloat16](0.0))
            self.w_dirty = True  # buffer grew — old cast invalid
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
            self.w_dirty = True
        if self.batch_cap < batch_needed:
            self.in_bf16_dev = ctx.enqueue_create_buffer[DType.bfloat16](
                batch_needed * Self.IN,
            )
            self.ou_bf16_dev = ctx.enqueue_create_buffer[DType.bfloat16](
                batch_needed * Self.OUT,
            )
            self.batch_cap = batch_needed
