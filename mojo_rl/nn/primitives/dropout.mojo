"""Dropout[DIM] — inverted-dropout regularisation (storage surface).

Transformed from legacy `nn.primitives.Dropout` (surface-only change). The
inverted-dropout math, the per-element PhiloxRandom uniform draw, and the cached
scaled-mask backward are all carried over VERBATIM. The ONLY change is the
surface (TensorRefs input pack + `Tensor` storage + storage RNG idiom).

Math (inverted dropout, identical to PyTorch):
    training:  mask ~ Bernoulli(1 - p), y = x · mask / (1 - p)
    eval:      y = x  (identity)

Backward (mask cached from forward — the scaled 0-or-1/(1-p) mask):
    training:  grad_x = grad_y · mask
    eval:      grad_x = grad_y

RNG idiom (storage): the CPU path keeps the legacy host PhiloxRandom counter
(`call_counter`, host-bumped). The GPU path holds the Philox offset as a 1-elem
device `TensorImpl[uint64]` (`noise_offset`) — mirroring `noisy_linear` /
`rsample` — and the train kernel reads the offset FROM that device buffer (a
LayoutTensor passed via `lt_gpu`, NO raw ptr); after the draw the offset is
advanced with the shared `advance_rng_offset_kernel`. That keeps the device
offset capture-friendly and the surface unsafe_ptr-free.

Math note (carried verbatim): mask = scale if u >= p else 0, with
scale = 1/(1-p), threshold = p, u ~ U(0,1) from PhiloxRandom.step_uniform().

Cache: leaf-owned `[BATCH, DIM]` slab — the SCALED mask (0 or 1/(1-p)) so
backward is one elementwise multiply.
"""

from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from std.random.philox import Random as PhiloxRandom
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.random.box_muller import advance_rng_offset_kernel
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — flat 1-D launch over BATCH·DIM. PhiloxRandom seeded with
# the per-layer SEED and offset (base_offset + idx) so each (forward,
# lane) pair has a fresh seed. Carried VERBATIM from legacy, except the
# train kernel reads `base_offset` from a 1-elem device buffer (storage
# RNG idiom) instead of a baked scalar arg.
# ──────────────────────────────────────────────────────────────────────


def _dropout_train_forward_kernel[
    N: Int, SEED: UInt64
](
    input: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    offset_buf: LayoutTensor[DType.uint64, Layout.row_major(1), MutAnyOrigin],
    threshold: Scalar[DT],
    scale: Scalar[DT],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= N:
        return
    var base_offset = rebind[UInt64](offset_buf[0])
    var rng = PhiloxRandom(
        seed=SEED, offset=base_offset + UInt64(idx),
    )
    var rand = Scalar[DT](rng.step_uniform()[0])
    var mask: Scalar[DT] = scale if rand >= threshold else Scalar[DT](0.0)
    cache[idx] = mask
    output[idx] = rebind[Scalar[DT]](input[idx]) * mask


def _dropout_eval_kernel[N: Int](
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= N:
        return
    dst[idx] = rebind[Scalar[DT]](src[idx])


def _dropout_train_backward_kernel[N: Int](
    grad_output: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    grad_input: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= N:
        return
    grad_input[idx] = (
        rebind[Scalar[DT]](grad_output[idx])
        * rebind[Scalar[DT]](cache[idx])
    )


struct Dropout[DIM_: Int, p: Float64 = 0.5, SEED: UInt64 = 1](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.DIM_)
    comptime OUT_DIM = Self.DIM_

    # Runtime state.
    var training: Bool
    var call_counter: UInt64  # host CPU Philox counter (legacy slot 0)
    var cache_mask: Tensor  # [BATCH, DIM] scaled mask (0 or 1/(1-p))
    var noise_offset: TensorImpl[DType.uint64]  # GPU Philox offset (1 elem)

    def __init__(out self):
        self.training = True
        self.call_counter = 0
        self.cache_mask = Tensor()
        self.noise_offset = TensorImpl[DType.uint64]()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. INIT ignored (no params)."""
        comptime assert target == "cpu" or target == "gpu", (
            "Dropout: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.p >= 0.0 and Self.p < 1.0, (
            "Dropout: p must be in [0, 1)"
        )
        var d = Self()
        comptime if target == "gpu":
            var c = ctx.value()
            d.noise_offset.ensure_gpu(c, 1)
            d.noise_offset.dev.value().enqueue_fill(UInt64(0))
        return d^

    def set_training(mut self, v: Bool):
        self.training = v

    def set_seed(mut self) raises:
        """Reset the per-instance RNG counters to the start of the stream."""
        self.call_counter = 0
        if self.noise_offset.dev:
            self.noise_offset.dev.value().enqueue_fill(UInt64(0))

    def forward[
        target: StaticString,
        B: Int,
        o: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        comptime N = B * Self.DIM_
        comptime if target == "cpu":
            out.ensure(N)
            if not self.training:
                for i in range(N):
                    out.data[i] = in0.data[i]
                # Eval pass doesn't bump the counter — keeps training
                # determinism cleanly separated from eval calls.
                return
            self.cache_mask.ensure(N)
            var scale = Scalar[DT](1.0 / (1.0 - Self.p))
            var threshold = Scalar[DT](Self.p)
            var zero = Scalar[DT](0.0)
            var base_offset = self.call_counter * UInt64(N)
            for b in range(B):
                for i in range(Self.DIM_):
                    var idx = b * Self.DIM_ + i
                    var rng = PhiloxRandom(
                        seed=Self.SEED,
                        offset=base_offset + UInt64(idx),
                    )
                    var rand = Scalar[DT](rng.step_uniform()[0])
                    var mask: Scalar[DT] = scale if rand >= threshold else zero
                    self.cache_mask.data[idx] = mask
                    out.data[idx] = in0.data[idx] * mask
            self.call_counter += 1
        else:
            var c = ctx.value()
            out.ensure_gpu(c, N)
            comptime n_blocks = (N + TPB - 1) // TPB
            comptime lN = Layout.row_major(N)
            comptime loff = Layout.row_major(1)
            if not self.training:
                comptime eval_kernel = _dropout_eval_kernel[N]
                c.enqueue_function[eval_kernel](
                    in0.lt["gpu", lN](),
                    out.lt["gpu", lN](),
                    grid_dim=n_blocks,
                    block_dim=TPB,
                )
                return
            self.cache_mask.ensure_gpu(c, N)
            var scale = Scalar[DT](1.0 / (1.0 - Self.p))
            var threshold = Scalar[DT](Self.p)
            comptime train_kernel = _dropout_train_forward_kernel[N, Self.SEED]
            c.enqueue_function[train_kernel](
                in0.lt["gpu", lN](),
                out.lt["gpu", lN](),
                self.cache_mask.lt["gpu", lN](),
                self.noise_offset.lt["gpu", loff](),
                threshold,
                scale,
                grid_dim=n_blocks,
                block_dim=TPB,
            )
            c.enqueue_function[advance_rng_offset_kernel[N]](
                self.noise_offset.lt["gpu", loff](),
                grid_dim=1,
                block_dim=1,
            )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref gin = grad_inputs[0]
        comptime N = B * Self.DIM_
        comptime if target == "cpu":
            gin.ensure(N)
            if not self.training:
                for i in range(N):
                    gin.data[i] = grad_output.data[i]
                return
            for i in range(N):
                gin.data[i] = grad_output.data[i] * self.cache_mask.data[i]
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, N)
            comptime n_blocks = (N + TPB - 1) // TPB
            comptime lN = Layout.row_major(N)
            if not self.training:
                comptime eval_kernel = _dropout_eval_kernel[N]
                c.enqueue_function[eval_kernel](
                    grad_output.lt["gpu", lN](),
                    gin.lt["gpu", lN](),
                    grid_dim=n_blocks,
                    block_dim=TPB,
                )
                return
            comptime back_kernel = _dropout_train_backward_kernel[N]
            c.enqueue_function[back_kernel](
                grad_output.lt["gpu", lN](),
                self.cache_mask.lt["gpu", lN](),
                gin.lt["gpu", lN](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    # for_each_param / zero_grad / polyak_from inherit the Module reflection
    # defaults (param-less: reflection finds no IsParam fields).
