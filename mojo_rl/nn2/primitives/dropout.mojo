"""Dropout[DIM, p, SEED] — inverted-dropout regularisation.

Phase 2 of `nn2/PORTING_PLAN.md`. First nn2 leaf to carry runtime
state beyond `TargetStorage`. Design choice (see PORTING_PLAN.md
Phase 2 train/eval section):

  - **Per-instance runtime `training: Bool` field**, default `True`.
    Set directly (`dropout.training = False`) or through
    `set_attr["training"](v)` where `v > 0.5` ⇒ True. We deliberately
    do NOT thread a comptime mode through the `Module` trait —
    nn2 explicitly removed an old `inference: Bool` flag from
    `TargetStorage` because no consumer ever used it. Dropout is the
    only leaf that needs train/eval, so we keep the surface local.
  - **Per-instance counter `call_counter: UInt64`**, bumped on every
    forward to give each call a unique PhiloxRandom offset. Mirrors
    the legacy `STATE_SIZE=1` GPU counter slot, just on the host —
    the GPU kernel reads it as a scalar arg instead of a device buffer
    (same pattern `box_muller.mojo` uses for SAC noise).

    ⚠️ CUDA-graph footgun: this host-bumped counter is baked into the
    captured kernel args, so under a captured train step (deep_agents2
    SAC *does* capture — see docs/CUDA_GRAPH_TRAIN_STEP.md) the offset
    would FREEZE and every replay would reuse the same dropout mask.
    There is no current overlap (SAC has no Dropout layer), but any
    future captured path that includes Dropout must move this counter
    into a device buffer (the legacy `STATE_SIZE=1` slot) first. SAC's
    own RNG counters (replay/rsample/noisy) were device-promoted for
    exactly this reason.

Math (inverted dropout, identical to PyTorch):
    training:  mask ~ Bernoulli(1 - p), y = x · mask / (1 - p)
    eval:      y = x  (identity)

Backward (mask cached from forward):
    training:  grad_x = grad_y · mask
    eval:      grad_x = grad_y

Cache: leaf-owned `[BATCH, DIM]` slab — we cache the scaled mask
(0 or 1/(1-p)) so backward is a single elementwise multiply. On GPU
the cache lives in a `DeviceBuffer[DT]` sized lazily on first forward.
"""

from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from std.random.philox import Random as PhiloxRandom
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, TPB
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import (
    TargetStorage,
    assert_tag_for,
    ensure_cpu_buffer,
)


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — flat 1-D launch over BATCH·DIM. PhiloxRandom seeded
# with the per-layer SEED (caller-supplied comptime) and offset
# (base_offset + idx) so each (forward, lane) pair has a fresh seed.
# ──────────────────────────────────────────────────────────────────────


def _dropout_train_forward_kernel[
    N: Int, SEED: UInt64,
](
    input: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    base_offset: UInt64,
    threshold: Scalar[DT],
    scale: Scalar[DT],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= N:
        return
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


struct Dropout[DIM: Int, p: Float64, SEED: UInt64](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.DIM)
    comptime OUT_DIM = Self.DIM

    # Runtime state.
    var training: Bool
    var call_counter: UInt64
    # Mask cache [BATCH, DIM] — scaled (0 or 1/(1-p)).
    var cache_mask: List[Scalar[DT]]
    var cache_mask_dev: Optional[DeviceBuffer[DT]]
    var cache_n_batch: Int
    var ts: TargetStorage

    def __init__(out self):
        self.training = True
        self.call_counter = 0
        self.cache_mask = List[Scalar[DT]]()
        self.cache_mask_dev = None
        self.cache_n_batch = 0
        self.ts = TargetStorage.make_uninit()

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
        comptime if target == "cpu":
            d.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("Dropout.make[target='gpu']: ctx required")
            var ctx_v = ctx.value()
            d.cache_mask_dev = ctx_v.enqueue_create_buffer[DT](1)
            d.cache_n_batch = 0
            d.ts = TargetStorage.make_gpu(ctx_v)
        return d^

    def _ensure_cache_gpu(mut self, batch: Int) raises:
        if self.cache_n_batch < batch:
            var ctx = self.ts.ctx.value()
            self.cache_mask_dev = ctx.enqueue_create_buffer[DT](
                batch * Self.DIM
            )
            self.cache_n_batch = batch

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        var *inputs: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["Dropout", target](self.ts.target_tag)
        var input = typed_view[BATCH, Self.IN_DIMS[0]](inputs[0])
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            if not self.training:
                for b in range(BATCH):
                    for i in range(Self.DIM):
                        output_v[b, i] = input[b, i]
                # Eval pass doesn't bump the counter — keeps training
                # determinism cleanly separated from eval calls.
                return
            ensure_cpu_buffer(self.cache_mask, BATCH * Self.DIM)
            var cache_v = TileTensor(
                self.cache_mask, row_major[BATCH, Self.DIM](),
            )
            var scale = Scalar[DT](1.0 / (1.0 - Self.p))
            var threshold = Scalar[DT](Self.p)
            var zero = Scalar[DT](0.0)
            var base_offset = self.call_counter * UInt64(BATCH * Self.DIM)
            for b in range(BATCH):
                for i in range(Self.DIM):
                    var rng = PhiloxRandom(
                        seed=Self.SEED,
                        offset=base_offset
                        + UInt64(b * Self.DIM + i),
                    )
                    var rand = Scalar[DT](rng.step_uniform()[0])
                    var mask: Scalar[DT] = scale if rand >= threshold else zero
                    cache_v[b, i] = mask
                    output_v[b, i] = input[b, i] * mask
            self.call_counter += 1
        else:
            comptime N = BATCH * Self.DIM
            comptime n_blocks = (N + TPB - 1) // TPB
            var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                input.ptr
            )
            var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                output_v.ptr
            )
            var in_lt = LayoutTensor[
                DT, Layout.row_major(N), MutAnyOrigin,
            ](in_p)
            var out_lt = LayoutTensor[
                DT, Layout.row_major(N), MutAnyOrigin,
            ](out_p)
            if not self.training:
                comptime eval_kernel = _dropout_eval_kernel[N]
                self.ts.ctx.value().enqueue_function[eval_kernel](
                    in_lt, out_lt,
                    grid_dim=n_blocks, block_dim=TPB,
                )
                return
            self._ensure_cache_gpu(BATCH)
            var cache_lt = LayoutTensor[
                DT, Layout.row_major(N), MutAnyOrigin,
            ](self.cache_mask_dev.value())
            var scale = Scalar[DT](1.0 / (1.0 - Self.p))
            var threshold = Scalar[DT](Self.p)
            var base_offset = self.call_counter * UInt64(N)
            comptime train_kernel = _dropout_train_forward_kernel[
                N, Self.SEED,
            ]
            self.ts.ctx.value().enqueue_function[train_kernel](
                in_lt, out_lt, cache_lt, base_offset, threshold, scale,
                grid_dim=n_blocks, block_dim=TPB,
            )
            self.call_counter += 1

    def vjp[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut *grad_inputs: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["Dropout", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = typed_view_mut[BATCH, Self.IN_DIMS[0]](
            grad_inputs[0]
        )

        comptime if target == "cpu":
            if not self.training:
                for b in range(BATCH):
                    for i in range(Self.DIM):
                        grad_input_v[b, i] = grad_output_v[b, i]
                return
            var cache_v = TileTensor(
                self.cache_mask, row_major[BATCH, Self.DIM](),
            )
            for b in range(BATCH):
                for i in range(Self.DIM):
                    grad_input_v[b, i] = (
                        grad_output_v[b, i] * cache_v[b, i]
                    )
        else:
            comptime N = BATCH * Self.DIM
            comptime n_blocks = (N + TPB - 1) // TPB
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                grad_output_v.ptr
            )
            var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                grad_input_v.ptr
            )
            var go_lt = LayoutTensor[
                DT, Layout.row_major(N), MutAnyOrigin,
            ](go_p)
            var gi_lt = LayoutTensor[
                DT, Layout.row_major(N), MutAnyOrigin,
            ](gi_p)
            if not self.training:
                comptime eval_kernel = _dropout_eval_kernel[N]
                self.ts.ctx.value().enqueue_function[eval_kernel](
                    go_lt, gi_lt,
                    grid_dim=n_blocks, block_dim=TPB,
                )
                return
            var cache_lt = LayoutTensor[
                DT, Layout.row_major(N), MutAnyOrigin,
            ](self.cache_mask_dev.value())
            comptime back_kernel = _dropout_train_backward_kernel[N]
            self.ts.ctx.value().enqueue_function[back_kernel](
                go_lt, cache_lt, gi_lt,
                grid_dim=n_blocks, block_dim=TPB,
            )

    # `set_attr["training"]` lets ComputeGraph / training-loop callers
    # flip the train/eval flag without naming the field directly. Value
    # convention: > 0.5 ⇒ True, else False (matches Clamp's set_attr
    # `Scalar[DT]` interface — there is no Bool-valued set_attr on the
    # trait).
    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        comptime if ATTR == "training":
            self.training = value > Scalar[DT](0.5)
