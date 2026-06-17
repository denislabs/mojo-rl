"""SkipConcat[Inner] — `y = concat(x, inner(x))`.

DenseNet-style skip-concat: the output is the input column-stacked with
the inner module's output. Mirrors `Residual` structurally (one child,
one scratch slab for the inner output, lazy-grown to `BATCH × OUT_INNER`),
but the merge op is concat instead of add.

Dimensions
    IN_DIMS[0] = Inner.IN_DIMS[0]                    (passthrough)
    OUT_DIM    = Inner.IN_DIMS[0] + Inner.OUT_DIM    (input + inner output)

Forward
    inner_out = inner(input)                    # scratch [BATCH, OUT_INNER]
    output[:, 0:IN]                = input
    output[:, IN:IN+OUT_INNER]     = inner_out

Backward
    grad_inner_out = grad_output[:, IN:IN+OUT_INNER]    # scratch
    grad_input     = inner.vjp(grad_inner_out)          # overwrite
    grad_input    += grad_output[:, 0:IN]               # add skip path

Used by OFE-style DenseBlocks, where `Inner = Sequential[Linear,
LayerNorm, SiLU]` grows the feature width by `per_unit` each block.
`mode` flows directly into `inner.vjp[mode]` (input_only skips param
grad accumulation in the inner subtree).
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, CPU_SIMD_W, TPB
from ..core import Initializer, AMPPolicy, NoAMP, ParamVisitor
from ..core.module import Module, typed_view, typed_view_mut, mptr
from ..core.tensor_pack import TensorPack
from ..core.target_storage import require_ctx, TargetStorage, assert_tag_for


# ──────────────────────────────────────────────────────────────────────
# GPU kernels.
# ──────────────────────────────────────────────────────────────────────


def _skip_concat_forward_kernel[
    BATCH: Int,
    IN_DIM: Int,
    OUT_INNER: Int,
](
    input: LayoutTensor[
        DT,
        Layout.row_major(BATCH, IN_DIM),
        MutAnyOrigin,
    ],
    inner_out: LayoutTensor[
        DT,
        Layout.row_major(BATCH, OUT_INNER),
        MutAnyOrigin,
    ],
    output: LayoutTensor[
        DT,
        Layout.row_major(BATCH, IN_DIM + OUT_INNER),
        MutAnyOrigin,
    ],
):
    var idx = Int(global_idx.x)
    comptime OUT = IN_DIM + OUT_INNER
    var total = BATCH * OUT
    if idx >= total:
        return
    var b = idx // OUT
    var c = idx % OUT
    if c < IN_DIM:
        output[b, c] = rebind[Scalar[DT]](input[b, c])
    else:
        output[b, c] = rebind[Scalar[DT]](inner_out[b, c - IN_DIM])


def _skip_concat_extract_inner_grad_kernel[
    BATCH: Int,
    IN_DIM: Int,
    OUT_INNER: Int,
](
    grad_output: LayoutTensor[
        DT,
        Layout.row_major(BATCH, IN_DIM + OUT_INNER),
        MutAnyOrigin,
    ],
    grad_inner_out: LayoutTensor[
        DT,
        Layout.row_major(BATCH, OUT_INNER),
        MutAnyOrigin,
    ],
):
    var idx = Int(global_idx.x)
    var total = BATCH * OUT_INNER
    if idx >= total:
        return
    var b = idx // OUT_INNER
    var d = idx % OUT_INNER
    grad_inner_out[b, d] = rebind[Scalar[DT]](grad_output[b, IN_DIM + d])


def _skip_concat_add_skip_grad_kernel[
    BATCH: Int,
    IN_DIM: Int,
    OUT_INNER: Int,
](
    grad_output: LayoutTensor[
        DT,
        Layout.row_major(BATCH, IN_DIM + OUT_INNER),
        MutAnyOrigin,
    ],
    grad_input: LayoutTensor[
        DT,
        Layout.row_major(BATCH, IN_DIM),
        MutAnyOrigin,
    ],
):
    var idx = Int(global_idx.x)
    var total = BATCH * IN_DIM
    if idx >= total:
        return
    var b = idx // IN_DIM
    var d = idx % IN_DIM
    grad_input[b, d] = rebind[Scalar[DT]](grad_input[b, d]) + rebind[
        Scalar[DT]
    ](grad_output[b, d])


# ──────────────────────────────────────────────────────────────────────
# SkipConcat
# ──────────────────────────────────────────────────────────────────────


struct SkipConcat[Inner: Module](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.Inner.IN_DIMS[0])
    comptime OUT_DIM = Self.Inner.IN_DIMS[0] + Self.Inner.OUT_DIM

    var inner: Self.Inner

    # Scratch [BATCH, Inner.OUT_DIM]: forward stores inner's output here
    # (then we concat-copy into the caller-supplied output); backward
    # extracts the inner-portion of grad_output into here, then feeds it
    # to inner.vjp(...).
    var inner_buf_cpu: List[Scalar[DT]]
    var inner_buf_dev: Optional[DeviceBuffer[DT]]
    var inner_buf_cap: Int

    var ts: TargetStorage

    # ----- Defaultable -----------------------------------------------------

    def __init__(out self):
        self.inner = Self.Inner()
        self.inner_buf_cpu = List[Scalar[DT]]()
        self.inner_buf_dev = None
        self.inner_buf_cap = 0
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None,) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU."""
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "SkipConcat: target must be 'cpu' or 'gpu'"
        var s = Self()
        s.inner = Self.Inner.make[target, INIT](ctx=ctx)
        comptime if target == "cpu":
            s.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["SkipConcat.make[target='gpu']"](ctx)
            s.inner_buf_dev = ctx_v.enqueue_create_buffer[DT](1)
            s.ts = TargetStorage.make_gpu(ctx_v)
        return s^

    def _ensure_inner_buf_cpu(mut self, needed: Int):
        # List owns the storage (RAII): grow in place, no manual alloc/free.
        if self.inner_buf_cap < needed:
            self.inner_buf_cpu.resize(needed, Scalar[DT](0))
            self.inner_buf_cap = needed

    def _ensure_inner_buf_gpu(mut self, needed: Int) raises:
        if self.inner_buf_cap < needed:
            self.inner_buf_dev = self.ts.ctx.value().enqueue_create_buffer[DT](
                needed
            )
            self.inner_buf_cap = needed

    # ----- Forward ---------------------------------------------------------

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorPack[Self.ARITY],
        mut output: TileTensor[
            mut=True,
            dtype=DT,
            address_space=AddressSpace.GENERIC,
            element_size=1,
            origin=MutAnyOrigin,
            ...,
        ],
    ) raises:
        assert_tag_for["SkipConcat", target](self.ts.target_tag)
        var input = inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime IN = Self.IN_DIMS[0]
        comptime OUT_INNER = Self.Inner.OUT_DIM

        comptime if target == "cpu":
            self._ensure_inner_buf_cpu(BATCH * OUT_INNER)
            var inner_tt = TileTensor(
                mptr(self.inner_buf_cpu.unsafe_ptr()),
                row_major[BATCH, OUT_INNER](),
            )
            self.inner.forward[target, BATCH, POLICY=POLICY](
                input,
                output=inner_tt,
            )
            var ip = mptr(input.ptr)
            var op = mptr(output_v.ptr)
            var sp = self.inner_buf_cpu.unsafe_ptr()
            for b in range(BATCH):
                var row_out = op + b * (IN + OUT_INNER)
                var row_in = ip + b * IN
                var row_sc = sp + b * OUT_INNER
                # input → first IN columns
                var k = 0
                while k + CPU_SIMD_W <= IN:
                    row_out.store(k, row_in.load[width=CPU_SIMD_W](k))
                    k += CPU_SIMD_W
                while k < IN:
                    row_out[k] = row_in[k]
                    k += 1
                # inner output → next OUT_INNER columns
                k = 0
                var dst = row_out + IN
                while k + CPU_SIMD_W <= OUT_INNER:
                    dst.store(k, row_sc.load[width=CPU_SIMD_W](k))
                    k += CPU_SIMD_W
                while k < OUT_INNER:
                    dst[k] = row_sc[k]
                    k += 1
        else:
            self._ensure_inner_buf_gpu(BATCH * OUT_INNER)
            var in_p_w = mptr(input.ptr)
            var out_p_w = mptr(output_v.ptr)
            var sp: UnsafePointer[
                Scalar[DT], MutAnyOrigin
            ] = self.inner_buf_dev.value().unsafe_ptr()
            var inner_tt = TileTensor(sp, row_major[BATCH, OUT_INNER]())
            self.inner.forward[target, BATCH, POLICY=POLICY](
                input,
                output=inner_tt,
            )
            comptime in_layout = Layout.row_major(BATCH, IN)
            comptime inner_layout = Layout.row_major(BATCH, OUT_INNER)
            comptime out_layout = Layout.row_major(BATCH, IN + OUT_INNER)
            var in_lt = LayoutTensor[DT, in_layout, MutAnyOrigin](in_p_w)
            var inner_lt = LayoutTensor[DT, inner_layout, MutAnyOrigin](
                self.inner_buf_dev.value()
            )
            var out_lt = LayoutTensor[DT, out_layout, MutAnyOrigin](out_p_w)
            comptime n_blocks = (BATCH * (IN + OUT_INNER) + TPB - 1) // TPB
            comptime kernel = _skip_concat_forward_kernel[
                BATCH,
                IN,
                OUT_INNER,
            ]
            self.ts.ctx.value().enqueue_function[kernel](
                in_lt,
                inner_lt,
                out_lt,
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    # ----- Backward --------------------------------------------------------

    def vjp[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT,
            address_space=AddressSpace.GENERIC,
            element_size=1,
            origin=MutAnyOrigin,
            ...,
        ],
        grad_inputs: TensorPack[Self.ARITY],
    ) raises:
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["SkipConcat", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = grad_inputs.tile[0, BATCH, Self.IN_DIMS[0]]()

        comptime IN = Self.IN_DIMS[0]
        comptime OUT_INNER = Self.Inner.OUT_DIM

        comptime if target == "cpu":
            self._ensure_inner_buf_cpu(BATCH * OUT_INNER)
            var gop = mptr(grad_output_v.ptr)
            var sp = self.inner_buf_cpu.unsafe_ptr()
            # 1. Extract inner-portion grad: grad_output[:, IN:IN+OUT_INNER]
            for b in range(BATCH):
                var row_go = gop + b * (IN + OUT_INNER) + IN
                var row_sc = sp + b * OUT_INNER
                var k = 0
                while k + CPU_SIMD_W <= OUT_INNER:
                    row_sc.store(k, row_go.load[width=CPU_SIMD_W](k))
                    k += CPU_SIMD_W
                while k < OUT_INNER:
                    row_sc[k] = row_go[k]
                    k += 1
            # 2. inner.vjp(scratch, grad_input)  ← overwrites grad_input
            var inner_tt = TileTensor(sp, row_major[BATCH, OUT_INNER]())
            self.inner.vjp[
                target,
                BATCH,
                POLICY=POLICY,
                mode=mode,
            ](inner_tt, grad_input_v)
            # 3. grad_input += grad_output[:, 0:IN]
            var gip = mptr(grad_input_v.ptr)
            for b in range(BATCH):
                var row_gi = gip + b * IN
                var row_go = gop + b * (IN + OUT_INNER)
                var k = 0
                while k + CPU_SIMD_W <= IN:
                    row_gi.store(
                        k,
                        row_gi.load[width=CPU_SIMD_W](k)
                        + row_go.load[width=CPU_SIMD_W](k),
                    )
                    k += CPU_SIMD_W
                while k < IN:
                    row_gi[k] = row_gi[k] + row_go[k]
                    k += 1
        else:
            self._ensure_inner_buf_gpu(BATCH * OUT_INNER)
            var go_p_w = mptr(grad_output_v.ptr)
            var gi_p_w = mptr(grad_input_v.ptr)
            comptime go_layout = Layout.row_major(BATCH, IN + OUT_INNER)
            comptime inner_layout = Layout.row_major(BATCH, OUT_INNER)
            comptime gi_layout = Layout.row_major(BATCH, IN)
            var go_lt = LayoutTensor[DT, go_layout, MutAnyOrigin](go_p_w)
            var inner_lt = LayoutTensor[DT, inner_layout, MutAnyOrigin](
                self.inner_buf_dev.value()
            )
            var gi_lt = LayoutTensor[DT, gi_layout, MutAnyOrigin](gi_p_w)
            # 1. Extract inner-portion grad into device scratch.
            comptime n_extract = (BATCH * OUT_INNER + TPB - 1) // TPB
            comptime extract_kernel = _skip_concat_extract_inner_grad_kernel[
                BATCH,
                IN,
                OUT_INNER,
            ]
            self.ts.ctx.value().enqueue_function[extract_kernel](
                go_lt,
                inner_lt,
                grid_dim=n_extract,
                block_dim=TPB,
            )
            # 2. inner.vjp(scratch, grad_input) — overwrites grad_input.
            var inner_tt = TileTensor(
                self.inner_buf_dev.value().unsafe_ptr(),
                row_major[BATCH, OUT_INNER](),
            )
            self.inner.vjp[
                target,
                BATCH,
                POLICY=POLICY,
                mode=mode,
            ](inner_tt, grad_input_v)
            # 3. grad_input += grad_output[:, 0:IN]
            comptime n_add = (BATCH * IN + TPB - 1) // TPB
            comptime add_kernel = _skip_concat_add_skip_grad_kernel[
                BATCH,
                IN,
                OUT_INNER,
            ]
            self.ts.ctx.value().enqueue_function[add_kernel](
                go_lt,
                gi_lt,
                grid_dim=n_add,
                block_dim=TPB,
            )

    # ----- Walkers ---------------------------------------------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["SkipConcat", target](self.ts.target_tag)
        var sep = "." if prefix.byte_length() > 0 else ""
        self.inner.for_each_param[target, V](
            prefix + sep + "inner",
            visitor,
        )

    def for_each_state[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["SkipConcat", target](self.ts.target_tag)
        var sep = "." if prefix.byte_length() > 0 else ""
        self.inner.for_each_state[target, V](
            prefix + sep + "inner",
            visitor,
        )

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["SkipConcat", target](self.ts.target_tag)
        self.inner.zero_grad[target]()
