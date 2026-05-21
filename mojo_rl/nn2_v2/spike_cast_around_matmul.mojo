"""cast_around_matmul helper (Follow-up #2).

Pulls the bf16 cast-around-matmul pattern out of nn2's Linear (six inline
copies, ~500 LOC of duplicated SIMD cast loops) into one reusable helper.

# What this spike validates

  1. A single function handles both `compute_dtype == fp32` (pass-through
     matmul) and `compute_dtype == bf16` (cast-around).
  2. Scratch buffers are passed in from the caller — the helper doesn't
     own state. Linear-shaped callers allocate the scratches once at
     construction time.
  3. The `weights_dirty` flag pattern: weights re-cast only when the
     caller signals the weight matrix changed (i.e. after `opt.step()`),
     not every forward/backward pass.

# Surface

  fn cast_around_matmul[POLICY, M, K, N, BATCH_OUT_ROWS](
      out: [M, N],
      a:   [M, K],   # forward: input (BATCH × IN); backward: grad_out (BATCH × OUT)
      b:   [K, N],   # the WEIGHT matrix (or its T view)
      transpose_b: comptime Bool,
      scratch: BF16Scratch[K_max, batch_max_in, batch_max_out],
      weights_dirty: Bool,  # set True after each Adam step
  ) raises

In the current spike: no `linalg.matmul`, just naive scalar matmul for
clarity. Real retrofit routes through `linalg.matmul[target]`. The point
is the API + the cast bookkeeping; the inner matmul is interchangeable.

# Why this helper changes the cost picture

Today's Linear (`nn2/primitives/linear.mojo`) re-casts fp32→bf16 on
every forward AND every backward AND every backward_input. For a
256×256 weight that's 64K casts × 3 method calls × 2 (forward + backward
of one optimizer step) = ~400K casts/step *per layer*, none of which
change between calls if no Adam step has happened.

With `weights_dirty`, the cast happens once per Adam step. The
inputs/grad_outputs DO change every call, so those casts stay.
"""

from std.math import abs as fabs
from layout import TileTensor


comptime DT = DType.float32


# ──────────────────────────────────────────────────────────────────────
# Tiny AMP policy. Real version is `nn2/core/amp.mojo`; spike keeps it
# minimal so we don't need to thread three dtype params through.
# ──────────────────────────────────────────────────────────────────────


trait AMPPolicy(ImplicitlyDestructible):
    comptime compute_dtype: DType


struct NoAMP(AMPPolicy):
    comptime compute_dtype = DType.float32


struct Bf16Compute(AMPPolicy):
    comptime compute_dtype = DType.bfloat16


# ──────────────────────────────────────────────────────────────────────
# BF16Scratch — owned by the caller (Linear-shaped layer), lazy-grown
# on first non-fp32 call. Pre-sized to comptime upper bounds.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct BF16Scratch(Movable & ImplicitlyDestructible):
    var w_bf16:  List[Scalar[DType.bfloat16]]  # K × N max
    var in_bf16: List[Scalar[DType.bfloat16]]  # batch × K max
    var ou_bf16: List[Scalar[DType.bfloat16]]  # batch × N max
    var batch_cap: Int  # current allocated batch dimension
    var w_dirty: Bool   # True if weight not yet cast or stale

    @staticmethod
    def empty() -> Self:
        return Self(
            w_bf16=List[Scalar[DType.bfloat16]](),
            in_bf16=List[Scalar[DType.bfloat16]](),
            ou_bf16=List[Scalar[DType.bfloat16]](),
            batch_cap=0,
            w_dirty=True,
        )

    def ensure[K: Int, N: Int](mut self, batch_needed: Int):
        var w_size = K * N
        if len(self.w_bf16) < w_size:
            self.w_bf16.resize(w_size, Scalar[DType.bfloat16](0.0))
            self.w_dirty = True  # buffer grew — old cast invalid
        if self.batch_cap < batch_needed:
            self.in_bf16.resize(batch_needed * K, Scalar[DType.bfloat16](0.0))
            self.ou_bf16.resize(batch_needed * N, Scalar[DType.bfloat16](0.0))
            self.batch_cap = batch_needed


# ──────────────────────────────────────────────────────────────────────
# Naive scalar matmul placeholders. Real retrofit routes through
# linalg.matmul[target]. The cast logic is what we're spiking.
# ──────────────────────────────────────────────────────────────────────


def _matmul_fp32[M: Int, K: Int, N: Int](
    a: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    b: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    dst: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    transpose_b_: Bool,
):
    """dst[M,N] = a[M,K] @ b[K,N] (or b^T if transpose_b_)."""
    for i in range(M):
        for j in range(N):
            var s: Scalar[DType.float32] = 0.0
            for k in range(K):
                var b_elem = b[j * K + k] if transpose_b_ else b[k * N + j]
                s += a[i * K + k] * b_elem
            dst[i * N + j] = s


def _matmul_bf16[M: Int, K: Int, N: Int](
    a: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    b: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    dst: UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin],
    transpose_b_: Bool,
):
    for i in range(M):
        for j in range(N):
            var s: Scalar[DType.bfloat16] = 0.0
            for k in range(K):
                var b_elem = b[j * K + k] if transpose_b_ else b[k * N + j]
                s += a[i * K + k] * b_elem
            dst[i * N + j] = s


# ──────────────────────────────────────────────────────────────────────
# The helper.
# ──────────────────────────────────────────────────────────────────────


def cast_around_matmul[
    POLICY: AMPPolicy,
    M: Int, K: Int, N: Int,
](
    a_p: UnsafePointer[Scalar[DT], MutAnyOrigin],     # M × K
    b_p: UnsafePointer[Scalar[DT], MutAnyOrigin],     # K × N  (or N × K if transpose_b)
    out_p: UnsafePointer[Scalar[DT], MutAnyOrigin],   # M × N
    mut scratch: BF16Scratch,
    transpose_b_: Bool = False,
) raises:
    """out = a @ b  (or a @ b^T) under POLICY's compute_dtype.

    fp32 path: direct fp32 matmul, no cast.
    bf16 path: cast a → in_bf16, cast w → w_bf16 (skipped if not dirty),
               bf16 matmul → ou_bf16, cast ou_bf16 → out fp32.

    Scratch's `w_dirty` flag is consumed (set to False) after a cast.
    Caller must set `scratch.w_dirty = True` after any weight update."""

    comptime if POLICY.compute_dtype == DType.float32:
        # ── fp32 fast path ────────────────────────────────────────────
        _matmul_fp32[M, K, N](a_p, b_p, out_p, transpose_b_)
        return

    comptime if POLICY.compute_dtype == DType.bfloat16:
        scratch.ensure[K, N](M)

        # Cast weight fp32 → bf16 ONLY if dirty.
        if scratch.w_dirty:
            var w_bf16_p = scratch.w_bf16.unsafe_ptr()
            # Weight matrix shape depends on transpose_b_: K×N or N×K.
            var w_total = K * N
            for k in range(w_total):
                w_bf16_p[k] = b_p[k].cast[DType.bfloat16]()
            scratch.w_dirty = False

        # Cast activation/grad fp32 → bf16 (always).
        var in_bf16_p = scratch.in_bf16.unsafe_ptr()
        var ou_bf16_p = scratch.ou_bf16.unsafe_ptr()
        var a_total = M * K
        for k in range(a_total):
            in_bf16_p[k] = a_p[k].cast[DType.bfloat16]()

        # bf16 matmul.
        _matmul_bf16[M, K, N](
            in_bf16_p, scratch.w_bf16.unsafe_ptr(), ou_bf16_p, transpose_b_,
        )

        # Cast output bf16 → fp32.
        var ou_total = M * N
        for k in range(ou_total):
            out_p[k] = ou_bf16_p[k].cast[DT]()


# ──────────────────────────────────────────────────────────────────────
# Linear-shaped caller using the helper. The point: ~12 lines per
# method (forward + backward + backward_input) vs. ~80 lines of inlined
# casts in today's nn2 Linear.
# ──────────────────────────────────────────────────────────────────────


struct LinearAMP[IN: Int, OUT: Int](Movable & ImplicitlyDestructible):
    """Linear that uses cast_around_matmul. No 6× inlined cast bodies.

    Owns one BF16Scratch (the fwd matmul). Real retrofit needs two
    (fwd a×W, bwd grad_out×W^T) — easy extension; spike keeps it
    minimal."""

    var weight: List[Scalar[DT]]
    var scratch: BF16Scratch
    # ... bias, grad_w, grad_b, cache pointer — omitted in spike.

    def __init__(out self):
        self.weight = List[Scalar[DT]]()
        self.scratch = BF16Scratch.empty()

    def forward[POLICY: AMPPolicy, BATCH: Int](
        mut self,
        input_p: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mut output_p: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        # That's it. The helper handles fp32 vs bf16 internally.
        cast_around_matmul[POLICY, BATCH, Self.IN, Self.OUT](
            input_p,
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](self.weight.unsafe_ptr()),
            output_p,
            self.scratch,
            transpose_b_=False,
        )

    def mark_weight_dirty(mut self):
        """Called by the optimizer after `step()` — invalidates the
        cached bf16 weight cast."""
        self.scratch.w_dirty = True
