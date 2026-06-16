"""SimNorm[DIM, GROUPS] — simplicial normalisation (per-group softmax).

Phase 2 of `nn/PORTING_PLAN.md`. Mirrors the legacy `Mish[DIM]`-paired
TDMPC2 head: reshape `[B, DIM]` into `[B, GROUPS, DIM/GROUPS]`, apply
softmax over the last axis, reshape back. Used on dynamics/encoder
heads to stabilise the latent space (replaces a LayerNorm in TDMPC2
variants).

Math, with `G = DIM/GROUPS` per group:
    sub_g(x) = x[g·G : (g+1)·G]
    y[g·G + k] = exp(sub_g[k] - max(sub_g)) / Σ_j exp(sub_g[j] - max(sub_g))

Backward (per group, standard softmax Jacobian):
    dot_g = Σ_k grad_y[g·G+k] · y[g·G+k]
    grad_x[g·G+k] = y[g·G+k] · (grad_y[g·G+k] - dot_g)

Output-cache leaf (`y` lives in a leaf-owned buffer); backward order is
grad_input only (no params). CPU + GPU paths. The GPU kernel runs one
thread per `(batch, group)` — group sizes are small (≤32 in TDMPC2), so
no in-block reduction is needed.
"""

from std.math import exp
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, TPB
from ..core import Initializer, AMPPolicy, NoAMP, Cache
from ..core.module import Module, typed_view, typed_view_mut
from ..core.tensor_pack import TensorPack
from ..core.target_storage import (
    require_ctx,
    TargetStorage,
    assert_tag_for,
)


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — one thread per (batch, group). The serial inner loop
# over GROUP_SIZE is the canonical TDMPC2 layout (GROUP_SIZE ≤ 32) so
# no in-block reduction is required.
# ──────────────────────────────────────────────────────────────────────


def _sim_norm_forward_kernel[
    BATCH: Int, DIM: Int, GROUPS: Int, GROUP_SIZE: Int,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * GROUPS:
        return
    var b = idx // GROUPS
    var g = idx % GROUPS
    var base = g * GROUP_SIZE

    var max_val = rebind[Scalar[DT]](input[b, base])
    for k in range(1, GROUP_SIZE):
        var v = rebind[Scalar[DT]](input[b, base + k])
        if v > max_val:
            max_val = v

    var sum_exp: Scalar[DT] = 0.0
    for k in range(GROUP_SIZE):
        var v = rebind[Scalar[DT]](input[b, base + k])
        sum_exp += exp(v - max_val)
    var inv_sum = Scalar[DT](1.0) / sum_exp

    for k in range(GROUP_SIZE):
        var v = rebind[Scalar[DT]](input[b, base + k])
        var y = exp(v - max_val) * inv_sum
        output[b, base + k] = y
        cache[b, base + k] = y


def _sim_norm_backward_kernel[
    BATCH: Int, DIM: Int, GROUPS: Int, GROUP_SIZE: Int,
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * GROUPS:
        return
    var b = idx // GROUPS
    var g = idx % GROUPS
    var base = g * GROUP_SIZE

    var dot: Scalar[DT] = 0.0
    for k in range(GROUP_SIZE):
        var dy = rebind[Scalar[DT]](grad_output[b, base + k])
        var y = rebind[Scalar[DT]](cache[b, base + k])
        dot += dy * y

    for k in range(GROUP_SIZE):
        var dy = rebind[Scalar[DT]](grad_output[b, base + k])
        var y = rebind[Scalar[DT]](cache[b, base + k])
        grad_input[b, base + k] = y * (dy - dot)


struct SimNorm[DIM: Int, GROUPS: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.DIM)
    comptime OUT_DIM = Self.DIM
    comptime GROUP_SIZE: Int = Self.DIM // Self.GROUPS

    # Cache holds softmax outputs `[BATCH, DIM]` for backward.
    var cache_y: Cache["cache_y"]
    var ts: TargetStorage

    def __init__(out self):
        self.cache_y = Cache["cache_y"]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. INIT ignored (no params) but accepted
        for `Sequential.make[target, INIT]` uniformity."""
        comptime assert target == "cpu" or target == "gpu", (
            "SimNorm: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.GROUPS > 0, "SimNorm: GROUPS must be > 0"
        comptime assert Self.DIM % Self.GROUPS == 0, (
            "SimNorm: DIM must be divisible by GROUPS"
        )
        var s = Self()
        comptime if target == "cpu":
            s.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["SimNorm.make[target='gpu']"](ctx)
            # Placeholder device buffer — real size set on first forward.
            s.ts = TargetStorage.make_gpu(ctx_v)
        return s^

    def _ensure_cache_gpu(mut self, batch: Int) raises:
        var ctx = self.ts.ctx.value()
        self.cache_y.ensure_gpu(ctx, batch * Self.DIM)
    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorPack[Self.ARITY],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["SimNorm", target](self.ts.target_tag)
        var input = inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            self.cache_y.ensure_cpu(BATCH * Self.DIM)
            var cache_v = TileTensor(
                self.cache_y.cpu, row_major[BATCH, Self.DIM](),
            )
            for b in range(BATCH):
                for g in range(Self.GROUPS):
                    var base = g * Self.GROUP_SIZE
                    var max_val: Scalar[DT] = input[b, base]
                    for k in range(1, Self.GROUP_SIZE):
                        var v = input[b, base + k]
                        if v > max_val:
                            max_val = v
                    var sum_exp: Scalar[DT] = 0.0
                    for k in range(Self.GROUP_SIZE):
                        sum_exp += exp(input[b, base + k] - max_val)
                    var inv_sum = Scalar[DT](1.0) / sum_exp
                    for k in range(Self.GROUP_SIZE):
                        var y = exp(input[b, base + k] - max_val) * inv_sum
                        output_v[b, base + k] = y
                        cache_v[b, base + k] = y
        else:
            self._ensure_cache_gpu(BATCH)
            comptime layout_2d = Layout.row_major(BATCH, Self.DIM)
            var in_p = input.ptr
            var out_p = output_v.ptr
            var in_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](in_p)
            var out_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](out_p)
            var cache_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](
                self.cache_y.dev.value()
            )
            comptime total = BATCH * Self.GROUPS
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _sim_norm_forward_kernel[
                BATCH, Self.DIM, Self.GROUPS, Self.GROUP_SIZE,
            ]
            self.ts.ctx.value().enqueue_function[kernel](
                in_lt, out_lt, cache_lt,
                grid_dim=n_blocks, block_dim=TPB,
            )

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
        grad_inputs: TensorPack[Self.ARITY],
    ) raises:
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["SimNorm", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = grad_inputs.tile[0, BATCH, Self.IN_DIMS[0]]()

        comptime if target == "cpu":
            var cache_v = TileTensor(
                self.cache_y.cpu, row_major[BATCH, Self.DIM](),
            )
            for b in range(BATCH):
                for g in range(Self.GROUPS):
                    var base = g * Self.GROUP_SIZE
                    var dot: Scalar[DT] = 0.0
                    for k in range(Self.GROUP_SIZE):
                        dot += grad_output_v[b, base + k] * cache_v[
                            b, base + k
                        ]
                    for k in range(Self.GROUP_SIZE):
                        var y = cache_v[b, base + k]
                        grad_input_v[b, base + k] = (
                            y * (grad_output_v[b, base + k] - dot)
                        )
        else:
            comptime layout_2d = Layout.row_major(BATCH, Self.DIM)
            var go_p = grad_output_v.ptr
            var gi_p = grad_input_v.ptr
            var go_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](go_p)
            var gi_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](gi_p)
            var cache_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](
                self.cache_y.dev.value()
            )
            comptime total = BATCH * Self.GROUPS
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _sim_norm_backward_kernel[
                BATCH, Self.DIM, Self.GROUPS, Self.GROUP_SIZE,
            ]
            self.ts.ctx.value().enqueue_function[kernel](
                go_lt, cache_lt, gi_lt,
                grid_dim=n_blocks, block_dim=TPB,
            )
