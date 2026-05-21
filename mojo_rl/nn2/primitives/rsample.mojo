"""RSample — retrofit (Phase E) + Block D (GPU).

Thin Module wrapper around the canonical `squashed_gaussian_forward` /
`squashed_gaussian_backward` free functions from
`nn2/loss/squashed_gaussian.mojo`.

Topology unchanged from v1:
    input  [BATCH, 2*ACT]   packed [mu | log_std]
    output [BATCH, ACT+1]   packed [action | log_prob]

`action_scale` stays a public mut field. RNG advance: each forward call
bumps `_rng_offset` by `2·BATCH·ACT` (the philox offsets consumed by
the box-muller pair kernel). Caller can override `rng_seed` directly.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT
from ..core import Initializer, AMPPolicy, NoAMP, ParamVisitor
from ..core.module import Module
from ..core.target_storage import (
    TargetStorage, assert_tag_for, ensure_cpu_buffer, ensure_gpu_buffer,
)
from ..random.box_muller import box_muller_normal, box_muller_normal_gpu
from ..loss.squashed_gaussian import (
    squashed_gaussian_forward,
    squashed_gaussian_backward,
    squashed_gaussian_forward_gpu,
    squashed_gaussian_backward_gpu,
)


# ──────────────────────────────────────────────────────────────────────
# Small pack/unpack kernels — RSample's packed output convention
# (`[action | log_prob]`) maps onto separate buffers the squashed-Gaussian
# kernels write into; one kernel splice each direction.
# ──────────────────────────────────────────────────────────────────────


def _rsample_pack_kernel[ACT: Int, BATCH: Int](
    action: LayoutTensor[DT, Layout.row_major(BATCH, ACT), MutAnyOrigin],
    log_prob: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    output: LayoutTensor[
        DT, Layout.row_major(BATCH, ACT + 1), MutAnyOrigin,
    ],
):
    var idx = Int(global_idx.x)
    comptime OUT = ACT + 1
    var total = BATCH * OUT
    if idx < total:
        var b = idx // OUT
        var d = idx % OUT
        if d < ACT:
            output[b, d] = rebind[Scalar[DT]](action[b, d])
        else:
            output[b, d] = rebind[Scalar[DT]](log_prob[b])


def _rsample_unpack_kernel[ACT: Int, BATCH: Int](
    grad_output: LayoutTensor[
        DT, Layout.row_major(BATCH, ACT + 1), MutAnyOrigin,
    ],
    grad_action: LayoutTensor[
        DT, Layout.row_major(BATCH, ACT), MutAnyOrigin,
    ],
    grad_log_prob: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    comptime OUT = ACT + 1
    var total = BATCH * OUT
    if idx < total:
        var b = idx // OUT
        var d = idx % OUT
        var v = rebind[Scalar[DT]](grad_output[b, d])
        if d < ACT:
            grad_action[b, d] = v
        else:
            grad_log_prob[b] = v


struct RSample[ACT: Int](Module):
    comptime IN_DIM = 2 * Self.ACT
    comptime OUT_DIM = Self.ACT + 1

    var action_scale: Scalar[DT]

    # Backward caches (CPU). z_cache: fresh noise drawn each forward.
    # in_cache: raw input (mu/log_std) — caller may overwrite between
    # forward and backward, so we copy it.
    var z_cache: List[Scalar[DT]]
    var in_cache: List[Scalar[DT]]
    var cache_n_batch: Int

    # GPU caches (block D). Plus scratch action / log_prob / grad_*
    # buffers because the squashed_gaussian kernels work on separate
    # buffers, and the RSample output convention packs them.
    var z_cache_dev: Optional[DeviceBuffer[DT]]
    var in_cache_dev: Optional[DeviceBuffer[DT]]
    var act_dev: Optional[DeviceBuffer[DT]]
    var lp_dev: Optional[DeviceBuffer[DT]]
    var grad_act_dev: Optional[DeviceBuffer[DT]]
    var grad_lp_dev: Optional[DeviceBuffer[DT]]
    var z_cache_dev_n: Int
    var in_cache_dev_n: Int
    var act_dev_n: Int
    var lp_dev_n: Int
    var grad_act_dev_n: Int
    var grad_lp_dev_n: Int

    # Philox state. Caller can override `rng_seed` directly.
    var rng_seed: UInt64
    var _rng_offset: UInt64

    var ts: TargetStorage

    def __init__(out self):
        self.action_scale = Scalar[DT](1.0)
        self.z_cache = List[Scalar[DT]]()
        self.in_cache = List[Scalar[DT]]()
        self.cache_n_batch = 0
        self.z_cache_dev = None
        self.in_cache_dev = None
        self.act_dev = None
        self.lp_dev = None
        self.grad_act_dev = None
        self.grad_lp_dev = None
        self.z_cache_dev_n = 0
        self.in_cache_dev_n = 0
        self.act_dev_n = 0
        self.lp_dev_n = 0
        self.grad_act_dev_n = 0
        self.grad_lp_dev_n = 0
        self.rng_seed = UInt64(42)
        self._rng_offset = UInt64(0)
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "RSample.make[target='gpu', INIT] requires a DeviceContext"
        )
        var r = Self()
        r.ts = TargetStorage.make_cpu()
        return r^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "RSample.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        var r = Self()
        r.ts = TargetStorage.make_gpu(ctx)
        return r^

    def _ensure_cache_cpu(mut self, batch: Int):
        if self.cache_n_batch < batch:
            self.z_cache.resize(batch * Self.ACT, Scalar[DT](0.0))
            self.in_cache.resize(batch * (2 * Self.ACT), Scalar[DT](0.0))
            self.cache_n_batch = batch

    def _ensure_cache_gpu(mut self, batch: Int) raises:
        var ctx = self.ts.ctx.value()
        ensure_gpu_buffer(
            self.z_cache_dev, self.z_cache_dev_n, batch * Self.ACT, ctx,
        )
        ensure_gpu_buffer(
            self.in_cache_dev, self.in_cache_dev_n, batch * (2 * Self.ACT), ctx,
        )
        ensure_gpu_buffer(
            self.act_dev, self.act_dev_n, batch * Self.ACT, ctx,
        )
        ensure_gpu_buffer(
            self.lp_dev, self.lp_dev_n, batch, ctx,
        )
        ensure_gpu_buffer(
            self.grad_act_dev, self.grad_act_dev_n, batch * Self.ACT, ctx,
        )
        ensure_gpu_buffer(
            self.grad_lp_dev, self.grad_lp_dev_n, batch, ctx,
        )

    # ----- Forward ---------------------------------------------------------

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        input: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        comptime assert input.flat_rank == 2, "input must be rank-2 [BATCH, 2*ACT]"
        comptime assert output.flat_rank == 2, "output must be rank-2 [BATCH, ACT+1]"
        comptime assert Self.ACT >= 1, "RSample[ACT]: ACT >= 1"
        assert_tag_for["RSample", target](self.ts.target_tag)

        comptime if target == "cpu":
            self._ensure_cache_cpu(BATCH)
            # Cache input + draw fresh z.
            for b in range(BATCH):
                for j in range(2 * Self.ACT):
                    self.in_cache[b * (2 * Self.ACT) + j] = input[b, j]
            box_muller_normal(self.z_cache.unsafe_ptr(), BATCH * Self.ACT)

            # Build local TileTensors for action and log_prob slices of
            # the packed output, plus a TileTensor view over the z cache.
            # We can't view output directly because action is [BATCH, ACT]
            # but output is [BATCH, ACT+1] — log_prob lives at column ACT.
            var z_t = TileTensor(self.z_cache, row_major[BATCH, Self.ACT]())

            # Use scratch buffers for action + log_prob (mojo tiles can't
            # alias non-contiguous slices). Then copy into the packed output.
            # For BATCH × ACT this is negligible.
            var act_buf = List[Scalar[DT]](length=BATCH * Self.ACT, fill=0.0)
            var lp_buf = List[Scalar[DT]](length=BATCH, fill=0.0)
            var act_t = TileTensor(act_buf, row_major[BATCH, Self.ACT]())
            var lp_t  = TileTensor(lp_buf,  row_major[BATCH]())

            squashed_gaussian_forward[Self.ACT, BATCH](
                input, z_t, self.action_scale, act_t, lp_t,
            )

            # Pack into output [BATCH, ACT+1]:
            #   output[b, j] = action[b, j]   for j in [0, ACT)
            #   output[b, ACT] = log_prob[b]
            for b in range(BATCH):
                for j in range(Self.ACT):
                    output[b, j] = act_buf[b * Self.ACT + j]
                output[b, Self.ACT] = lp_buf[b]
        else:
            var ctx = self.ts.ctx.value()
            self._ensure_cache_gpu(BATCH)
            var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
            var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)
            var in_cache_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.in_cache_dev.value().unsafe_ptr()
            )
            var z_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.z_cache_dev.value().unsafe_ptr()
            )
            var act_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.act_dev.value().unsafe_ptr()
            )
            var lp_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.lp_dev.value().unsafe_ptr()
            )
            # Cache input via a device-to-device copy.
            ctx.enqueue_copy(self.in_cache_dev.value(), in_p)
            # Draw fresh z via philox+box-muller.
            box_muller_normal_gpu[BATCH * Self.ACT](
                ctx, z_p, self.rng_seed, self._rng_offset,
            )
            # Bump RNG offset for next call (2 philox draws per element).
            self._rng_offset += UInt64(2 * BATCH * Self.ACT)
            # Squashed-gaussian forward into separate action + log_prob bufs.
            squashed_gaussian_forward_gpu[Self.ACT, BATCH](
                ctx, in_p, z_p, self.action_scale, act_p, lp_p,
            )
            # Pack into [BATCH, ACT+1] output.
            var act_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.ACT), MutAnyOrigin,
            ](act_p)
            var lp_lt = LayoutTensor[
                DT, Layout.row_major(BATCH), MutAnyOrigin,
            ](lp_p)
            var out_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.ACT + 1), MutAnyOrigin,
            ](out_p)
            comptime TPB = 128
            comptime total = BATCH * (Self.ACT + 1)
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _rsample_pack_kernel[Self.ACT, BATCH]
            ctx.enqueue_function[kernel](
                act_lt, lp_lt, out_lt,
                grid_dim=n_blocks, block_dim=TPB,
            )

    # ----- Backward --------------------------------------------------------

    def backward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        mut grad_input: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        comptime assert grad_output.flat_rank == 2, "grad_output rank-2"
        comptime assert grad_input.flat_rank == 2, "grad_input rank-2"
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["RSample", target](self.ts.target_tag)

        comptime if target == "cpu":
            # Unpack grad_output [BATCH, ACT+1] → grad_action [BATCH, ACT]
            # + grad_log_prob [BATCH]. Scratch buffers (small, neg cost).
            var ga_buf = List[Scalar[DT]](length=BATCH * Self.ACT, fill=0.0)
            var glp_buf = List[Scalar[DT]](length=BATCH, fill=0.0)
            for b in range(BATCH):
                for j in range(Self.ACT):
                    ga_buf[b * Self.ACT + j] = grad_output[b, j]
                glp_buf[b] = grad_output[b, Self.ACT]
            var ga_t  = TileTensor(ga_buf,  row_major[BATCH, Self.ACT]())
            var glp_t = TileTensor(glp_buf, row_major[BATCH]())
            var z_t   = TileTensor(self.z_cache, row_major[BATCH, Self.ACT]())
            var ic_t  = TileTensor(
                self.in_cache, row_major[BATCH, 2 * Self.ACT](),
            )

            squashed_gaussian_backward[Self.ACT, BATCH](
                ic_t, z_t, ga_t, glp_t, self.action_scale, grad_input,
            )
        else:
            var ctx = self.ts.ctx.value()
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output.ptr)
            var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input.ptr)
            var in_cache_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.in_cache_dev.value().unsafe_ptr()
            )
            var z_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.z_cache_dev.value().unsafe_ptr()
            )
            var ga_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.grad_act_dev.value().unsafe_ptr()
            )
            var glp_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.grad_lp_dev.value().unsafe_ptr()
            )
            # Unpack grad_output → grad_action + grad_log_prob.
            var go_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.ACT + 1), MutAnyOrigin,
            ](go_p)
            var ga_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.ACT), MutAnyOrigin,
            ](ga_p)
            var glp_lt = LayoutTensor[
                DT, Layout.row_major(BATCH), MutAnyOrigin,
            ](glp_p)
            comptime TPB = 128
            comptime total = BATCH * (Self.ACT + 1)
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _rsample_unpack_kernel[Self.ACT, BATCH]
            ctx.enqueue_function[kernel](
                go_lt, ga_lt, glp_lt,
                grid_dim=n_blocks, block_dim=TPB,
            )
            # Squashed-gaussian backward.
            squashed_gaussian_backward_gpu[Self.ACT, BATCH](
                ctx, in_cache_p, z_p, ga_p, glp_p, self.action_scale, gi_p,
            )
