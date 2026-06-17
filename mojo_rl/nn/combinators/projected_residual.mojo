"""ProjectedResidual[Inner, Skip] — `y = Inner(x) + Skip(x)`.

Like `Residual[Inner]` but the skip path runs its own (parameterised)
module instead of an identity, so it can project the input to match
`Inner.OUT_DIM` when shapes change. This is the ResNet downsampling
block: `Inner` is the 3×3-s2 → BN → ReLU → 3×3-s1 → BN main path and
`Skip` is the 1×1-s2 → BN projection. The external ReLU on the sum is
applied by composing this inside a `Sequential[ProjectedResidual[...],
ReLU[OUT]]` (see `composites.mojo`).

Constraints (comptime-checked at __init__):
  - `Inner.IN_DIMS[0] == Skip.IN_DIMS[0]`  (same input fed to both)
  - `Inner.OUT_DIM   == Skip.OUT_DIM`      (outputs summed elementwise)

Forward:  `output = Inner(x) + Skip(x)`
Backward: `grad_input = Inner.vjp(go) + Skip.vjp(go)` (both branches
          receive the full grad_output; their grad_inputs are summed).

Scratch (4 slabs, lazy-grown to BATCH), mirroring `Parallel`:
  - `inner_out`/`skip_out` (BATCH×OUT_DIM): forward outputs
  - `gi_inner`/`gi_skip`   (BATCH×IN_DIM):  per-branch grad_inputs
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, CPU_SIMD_W, TPB
from ..core import Initializer, AMPPolicy, NoAMP, ParamVisitor
from ..core.module import Module, typed_view, typed_view_mut, mptr
from ..core.tensor_pack import TensorPack
from ..core.target_storage import require_ctx, TargetStorage, assert_tag_for
from .residual import _elementwise_add_kernel


struct ProjectedResidual[Inner: Module, Skip: Module](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.Inner.IN_DIMS[0])
    comptime OUT_DIM = Self.Inner.OUT_DIM

    var inner: Self.Inner
    var skip: Self.Skip

    var inner_out_cpu: List[Scalar[DT]]
    var skip_out_cpu: List[Scalar[DT]]
    var gi_inner_cpu: List[Scalar[DT]]
    var gi_skip_cpu: List[Scalar[DT]]

    var inner_out_dev: Optional[DeviceBuffer[DT]]
    var skip_out_dev: Optional[DeviceBuffer[DT]]
    var gi_inner_dev: Optional[DeviceBuffer[DT]]
    var gi_skip_dev: Optional[DeviceBuffer[DT]]
    var scratch_n_batch: Int

    var ts: TargetStorage

    def __init__(out self):
        comptime assert (
            Self.Inner.IN_DIMS[0] == Self.Skip.IN_DIMS[0]
        ), "ProjectedResidual requires Inner.IN_DIMS[0] == Skip.IN_DIMS[0]"
        comptime assert (
            Self.Inner.OUT_DIM == Self.Skip.OUT_DIM
        ), "ProjectedResidual requires Inner.OUT_DIM == Skip.OUT_DIM"
        self.inner = Self.Inner()
        self.skip = Self.Skip()
        self.inner_out_cpu = List[Scalar[DT]]()
        self.skip_out_cpu = List[Scalar[DT]]()
        self.gi_inner_cpu = List[Scalar[DT]]()
        self.gi_skip_cpu = List[Scalar[DT]]()
        self.inner_out_dev = None
        self.skip_out_dev = None
        self.gi_inner_dev = None
        self.gi_skip_dev = None
        self.scratch_n_batch = 0
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None,) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU."""
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "ProjectedResidual: target must be 'cpu' or 'gpu'"
        var r = Self()
        r.inner = Self.Inner.make[target, INIT](ctx=ctx)
        r.skip = Self.Skip.make[target, INIT](ctx=ctx)
        comptime if target == "cpu":
            r.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["ProjectedResidual.make[target='gpu']"](ctx)
            r.inner_out_dev = ctx_v.enqueue_create_buffer[DT](1)
            r.skip_out_dev = ctx_v.enqueue_create_buffer[DT](1)
            r.gi_inner_dev = ctx_v.enqueue_create_buffer[DT](1)
            r.gi_skip_dev = ctx_v.enqueue_create_buffer[DT](1)
            r.ts = TargetStorage.make_gpu(ctx_v)
        return r^

    def _ensure_scratch_cpu(mut self, batch: Int):
        # List owns the storage (RAII): grow in place, no manual alloc/free.
        if self.scratch_n_batch < batch:
            self.inner_out_cpu.resize(batch * Self.OUT_DIM, Scalar[DT](0))
            self.skip_out_cpu.resize(batch * Self.OUT_DIM, Scalar[DT](0))
            self.gi_inner_cpu.resize(batch * Self.IN_DIMS[0], Scalar[DT](0))
            self.gi_skip_cpu.resize(batch * Self.IN_DIMS[0], Scalar[DT](0))
            self.scratch_n_batch = batch

    def _ensure_scratch_gpu(mut self, batch: Int) raises:
        if self.scratch_n_batch < batch:
            var c = self.ts.ctx.value()
            self.inner_out_dev = c.enqueue_create_buffer[DT](
                batch * Self.OUT_DIM
            )
            self.skip_out_dev = c.enqueue_create_buffer[DT](
                batch * Self.OUT_DIM
            )
            self.gi_inner_dev = c.enqueue_create_buffer[DT](
                batch * Self.IN_DIMS[0]
            )
            self.gi_skip_dev = c.enqueue_create_buffer[DT](
                batch * Self.IN_DIMS[0]
            )
            self.scratch_n_batch = batch

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
        assert_tag_for["ProjectedResidual", target](self.ts.target_tag)
        var input = inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            self._ensure_scratch_cpu(BATCH)
            var inner_out = TileTensor(
                mptr(self.inner_out_cpu.unsafe_ptr()),
                row_major[BATCH, Self.OUT_DIM](),
            )
            var skip_out = TileTensor(
                mptr(self.skip_out_cpu.unsafe_ptr()),
                row_major[BATCH, Self.OUT_DIM](),
            )
            self.inner.forward[target, BATCH, POLICY=POLICY](
                input, output=inner_out
            )
            self.skip.forward[target, BATCH, POLICY=POLICY](
                input, output=skip_out
            )
            var ap = self.inner_out_cpu.unsafe_ptr()
            var bp = self.skip_out_cpu.unsafe_ptr()
            var op = mptr(output_v.ptr)
            comptime N = BATCH * Self.OUT_DIM
            var k = 0
            while k + CPU_SIMD_W <= N:
                op.store(
                    k,
                    ap.load[width=CPU_SIMD_W](k) + bp.load[width=CPU_SIMD_W](k),
                )
                k += CPU_SIMD_W
            while k < N:
                op[k] = ap[k] + bp[k]
                k += 1
        else:
            self._ensure_scratch_gpu(BATCH)
            var out_p_w = mptr(output_v.ptr)
            var inner_out = TileTensor(
                mptr(self.inner_out_dev.value().unsafe_ptr()),
                row_major[BATCH, Self.OUT_DIM](),
            )
            var skip_out = TileTensor(
                mptr(self.skip_out_dev.value().unsafe_ptr()),
                row_major[BATCH, Self.OUT_DIM](),
            )
            self.inner.forward[target, BATCH, POLICY=POLICY](
                input, output=inner_out
            )
            self.skip.forward[target, BATCH, POLICY=POLICY](
                input, output=skip_out
            )
            comptime layout = Layout.row_major(BATCH, Self.OUT_DIM)
            var a_lt = LayoutTensor[DT, layout, MutAnyOrigin](
                self.inner_out_dev.value()
            )
            var b_lt = LayoutTensor[DT, layout, MutAnyOrigin](
                self.skip_out_dev.value()
            )
            var o_lt = LayoutTensor[DT, layout, MutAnyOrigin](out_p_w)
            comptime n_blocks = (BATCH * Self.OUT_DIM + TPB - 1) // TPB
            comptime kernel = _elementwise_add_kernel[BATCH, Self.OUT_DIM]
            self.ts.ctx.value().enqueue_function[kernel](
                a_lt,
                b_lt,
                o_lt,
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
        assert_tag_for["ProjectedResidual", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = grad_inputs.tile[0, BATCH, Self.IN_DIMS[0]]()

        comptime if target == "cpu":
            self._ensure_scratch_cpu(BATCH)
            var gi_inner = TileTensor(
                mptr(self.gi_inner_cpu.unsafe_ptr()),
                row_major[BATCH, Self.IN_DIMS[0]](),
            )
            var gi_skip = TileTensor(
                mptr(self.gi_skip_cpu.unsafe_ptr()),
                row_major[BATCH, Self.IN_DIMS[0]](),
            )
            self.inner.vjp[
                target,
                BATCH,
                POLICY=POLICY,
                mode=mode,
            ](grad_output_v, gi_inner)
            self.skip.vjp[
                target,
                BATCH,
                POLICY=POLICY,
                mode=mode,
            ](grad_output_v, gi_skip)
            var ap = self.gi_inner_cpu.unsafe_ptr()
            var bp = self.gi_skip_cpu.unsafe_ptr()
            var gp = mptr(grad_input_v.ptr)
            comptime N = BATCH * Self.IN_DIMS[0]
            var k = 0
            while k + CPU_SIMD_W <= N:
                gp.store(
                    k,
                    ap.load[width=CPU_SIMD_W](k) + bp.load[width=CPU_SIMD_W](k),
                )
                k += CPU_SIMD_W
            while k < N:
                gp[k] = ap[k] + bp[k]
                k += 1
        else:
            self._ensure_scratch_gpu(BATCH)
            var gi_inner = TileTensor(
                mptr(self.gi_inner_dev.value().unsafe_ptr()),
                row_major[BATCH, Self.IN_DIMS[0]](),
            )
            var gi_skip = TileTensor(
                mptr(self.gi_skip_dev.value().unsafe_ptr()),
                row_major[BATCH, Self.IN_DIMS[0]](),
            )
            self.inner.vjp[
                target,
                BATCH,
                POLICY=POLICY,
                mode=mode,
            ](grad_output_v, gi_inner)
            self.skip.vjp[
                target,
                BATCH,
                POLICY=POLICY,
                mode=mode,
            ](grad_output_v, gi_skip)
            var gi_p_w = mptr(grad_input_v.ptr)
            comptime layout_in = Layout.row_major(BATCH, Self.IN_DIMS[0])
            var gi_a_lt = LayoutTensor[DT, layout_in, MutAnyOrigin](
                self.gi_inner_dev.value()
            )
            var gi_b_lt = LayoutTensor[DT, layout_in, MutAnyOrigin](
                self.gi_skip_dev.value()
            )
            var gi_out_lt = LayoutTensor[DT, layout_in, MutAnyOrigin](gi_p_w)
            comptime n_blocks = (BATCH * Self.IN_DIMS[0] + TPB - 1) // TPB
            comptime sum_kernel = _elementwise_add_kernel[
                BATCH, Self.IN_DIMS[0]
            ]
            self.ts.ctx.value().enqueue_function[sum_kernel](
                gi_a_lt,
                gi_b_lt,
                gi_out_lt,
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    # ----- Walkers ---------------------------------------------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["ProjectedResidual", target](self.ts.target_tag)
        var sep = "." if prefix.byte_length() > 0 else ""
        self.inner.for_each_param[target, V](prefix + sep + "inner", visitor)
        self.skip.for_each_param[target, V](prefix + sep + "skip", visitor)

    def for_each_state[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["ProjectedResidual", target](self.ts.target_tag)
        var sep = "." if prefix.byte_length() > 0 else ""
        self.inner.for_each_state[target, V](prefix + sep + "inner", visitor)
        self.skip.for_each_state[target, V](prefix + sep + "skip", visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["ProjectedResidual", target](self.ts.target_tag)
        self.inner.zero_grad[target]()
        self.skip.zero_grad[target]()

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        self.inner.set_attr[ATTR](value)
        self.skip.set_attr[ATTR](value)
