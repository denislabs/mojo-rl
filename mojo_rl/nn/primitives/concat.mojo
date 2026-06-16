"""Concat[*DIMS] — variadic horizontal stack primitive.

Replaces the legacy `BinaryConcat[D0, D1]` + `TernaryConcat[D0, D1, D2]`
with one variadic primitive. `ARITY = DIMS.size` (≥ 2),
`OUT_DIM = Σ DIMS[i]`.

  output[b, off_i + d]   = inputs[i][b, d]        d ∈ [0, DIMS[i])
  grad_inputs[i][b, d]   = grad_output[b, off_i + d]

where `off_i = Σ_{j<i} DIMS[j]` is the cumulative offset (comptime).

CPU + GPU. GPU forward launches N small slab-copy kernels (one per
input); backward symmetric. Variadic inputs follow the same-Layout
hetero-shape workaround (Phase 4.6c): callers construct every variadic
TileTensor with the same `row_major[BATCH, DIMS[0]]()` Layout, and we
rebuild typed views per-input via `typed_view[BATCH, DIMS[i]]`.

Module conformance — note that `IN1_DIM` / `IN2_DIM` collapse to 0 when
arity < that index (helper `_dim_at` gates the access at comptime).
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor

from ..constants import DT, TPB
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module, typed_view, typed_view_mut
from ..core.tensor_pack import TensorPack
from ..core.target_storage import TargetStorage, assert_tag_for


# ──────────────────────────────────────────────────────────────────────
# Comptime helpers — DIMS pack lookup with default-0, total, cumulative
# offset.
# ──────────────────────────────────────────────────────────────────────


def _dim_at[index: Int, *DIMS: Int]() -> Int:
    """Return DIMS[index], or 0 if index >= DIMS.size. Used so IN1_DIM /
    IN2_DIM gracefully collapse to 0 for low-arity instantiations."""
    var s: Int = 0
    comptime for i in range(DIMS.size):
        comptime if i == index:
            s = DIMS[i]
    return s


def _total_dim[*DIMS: Int]() -> Int:
    var s: Int = 0
    comptime for i in range(DIMS.size):
        s += DIMS[i]
    return s


def _cum_offset[index: Int, *DIMS: Int]() -> Int:
    var s: Int = 0
    comptime for j in range(index):
        s += DIMS[j]
    return s


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — one slab-copy per direction. Each launch handles a
# single input/output slab; the comptime-for loop in forward/vjp emits
# N launches.
# ──────────────────────────────────────────────────────────────────────


def _concat_copy_in_kernel[
    BATCH: Int, SRC_DIM: Int, OUT_DIM: Int, DST_OFFSET: Int,
](
    src: LayoutTensor[DT, Layout.row_major(BATCH, SRC_DIM), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = BATCH * SRC_DIM
    if idx < total:
        var b = idx // SRC_DIM
        var d = idx % SRC_DIM
        output[b, DST_OFFSET + d] = rebind[Scalar[DT]](src[b, d])


def _concat_copy_out_kernel[
    BATCH: Int, DST_DIM: Int, OUT_DIM: Int, SRC_OFFSET: Int,
](
    grad_output: LayoutTensor[
        DT, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin,
    ],
    grad_in: LayoutTensor[DT, Layout.row_major(BATCH, DST_DIM), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = BATCH * DST_DIM
    if idx < total:
        var b = idx // DST_DIM
        var d = idx % DST_DIM
        grad_in[b, d] = rebind[Scalar[DT]](grad_output[b, SRC_OFFSET + d])


# ──────────────────────────────────────────────────────────────────────
# Concat[*DIMS]
# ──────────────────────────────────────────────────────────────────────


struct Concat[*DIMS: Int](Module):
    comptime ARITY: Int = Self.DIMS.size
    comptime IN_DIMS = Self._build_in_dims()
    comptime IN0_DIM: Int = _dim_at[0, *Self.DIMS]()
    comptime OUT_DIM: Int = _total_dim[*Self.DIMS]()

    @staticmethod
    def display_label() -> String:
        return String("Concat")

    @staticmethod
    def _build_in_dims() -> InlineArray[Int, Self.DIMS.size]:
        var d = InlineArray[Int, Self.DIMS.size](fill=0)
        comptime for k in range(Self.DIMS.size):
            d[k] = Self.DIMS[k]
        return d

    var ts: TargetStorage

    def __init__(out self):
        comptime assert (
            Self.DIMS.size >= 2
        ), "Concat: needs at least 2 inputs"
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU."""
        comptime assert target == "cpu" or target == "gpu", (
            "Concat: target must be 'cpu' or 'gpu'"
        )
        var c = Self()
        comptime if target == "cpu":
            c.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("Concat.make[target='gpu']: ctx required")
            c.ts = TargetStorage.make_gpu(ctx.value())
        return c^

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
        assert_tag_for["Concat", target](self.ts.target_tag)
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            comptime for i in range(Self.DIMS.size):
                comptime D = Self.DIMS[i]
                comptime OFF = _cum_offset[i, *Self.DIMS]()
                var in_i = inputs.tile[i, BATCH, D]()
                for b in range(BATCH):
                    for d in range(D):
                        output_v[b, OFF + d] = in_i[b, d]
        else:
            var o_p = output_v.ptr
            var o_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin,
            ](o_p)
            comptime for i in range(Self.DIMS.size):
                comptime D = Self.DIMS[i]
                comptime OFF = _cum_offset[i, *Self.DIMS]()
                var in_i = inputs.tile[i, BATCH, D]()
                var i_p = in_i.ptr
                var i_lt = LayoutTensor[
                    DT, Layout.row_major(BATCH, D), MutAnyOrigin,
                ](i_p)
                comptime total = BATCH * D
                comptime n_blocks = (total + TPB - 1) // TPB
                comptime kernel = _concat_copy_in_kernel[
                    BATCH, D, Self.OUT_DIM, OFF,
                ]
                self.ts.ctx.value().enqueue_function[kernel](
                    i_lt, o_lt, grid_dim=n_blocks, block_dim=TPB,
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
        assert_tag_for["Concat", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)

        comptime if target == "cpu":
            comptime for i in range(Self.DIMS.size):
                comptime D = Self.DIMS[i]
                comptime OFF = _cum_offset[i, *Self.DIMS]()
                var gi = grad_inputs.tile[i, BATCH, D]()
                for b in range(BATCH):
                    for d in range(D):
                        gi[b, d] = grad_output_v[b, OFF + d]
        else:
            var go_p = grad_output_v.ptr
            var go_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin,
            ](go_p)
            comptime for i in range(Self.DIMS.size):
                comptime D = Self.DIMS[i]
                comptime OFF = _cum_offset[i, *Self.DIMS]()
                var gi_v = grad_inputs.tile[i, BATCH, D]()
                var gi_p = gi_v.ptr
                var gi_lt = LayoutTensor[
                    DT, Layout.row_major(BATCH, D), MutAnyOrigin,
                ](gi_p)
                comptime total = BATCH * D
                comptime n_blocks = (total + TPB - 1) // TPB
                comptime kernel = _concat_copy_out_kernel[
                    BATCH, D, Self.OUT_DIM, OFF,
                ]
                self.ts.ctx.value().enqueue_function[kernel](
                    go_lt, gi_lt, grid_dim=n_blocks, block_dim=TPB,
                )
