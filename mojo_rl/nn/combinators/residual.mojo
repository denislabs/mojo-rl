"""Residual[Inner] — `y = inner(x) + x`.

Scaffold: `ts: TargetStorage`,
`backward[mode]` instead of separate `backward_input`, walkers
recurse into Inner.

Phase 10A buffer surface dropped; the single mid slab IS the
inter-module wiring (Residual is itself an orchestrator for one child).

Backward order: `inner.vjp[mode]` runs first (writes into mid),
then `grad_input = mid + grad_output` SIMD-add. Identical to v1.
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


def _elementwise_add_kernel[
    BATCH: Int,
    DIM: Int,
](
    a: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    b: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = BATCH * DIM
    if idx < total:
        var bi = idx // DIM
        var di = idx % DIM
        dst[bi, di] = rebind[Scalar[DT]](a[bi, di]) + rebind[Scalar[DT]](
            b[bi, di]
        )


struct Residual[Inner: Module](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.Inner.IN_DIMS[0])
    comptime OUT_DIM = Self.Inner.OUT_DIM

    var inner: Self.Inner

    var mid_cpu: List[Scalar[DT]]
    var mid_dev: Optional[DeviceBuffer[DT]]
    var mid_cap: Int

    var ts: TargetStorage

    def __init__(out self):
        comptime assert (
            Self.Inner.IN_DIMS[0] == Self.Inner.OUT_DIM
        ), "Residual requires Inner.IN_DIMS[0] == Inner.OUT_DIM"
        self.inner = Self.Inner()
        self.mid_cpu = List[Scalar[DT]]()
        self.mid_dev = None
        self.mid_cap = 0
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None,) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU."""
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "Residual: target must be 'cpu' or 'gpu'"
        var r = Self()
        r.inner = Self.Inner.make[target, INIT](ctx=ctx)
        comptime if target == "cpu":
            r.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["Residual.make[target='gpu']"](ctx)
            r.mid_dev = ctx_v.enqueue_create_buffer[DT](1)
            r.ts = TargetStorage.make_gpu(ctx_v)
        return r^

    def _ensure_mid_cpu(mut self, needed: Int):
        # List owns the storage (RAII): grow in place, no manual alloc/free.
        if self.mid_cap < needed:
            self.mid_cpu.resize(needed, Scalar[DT](0))
            self.mid_cap = needed

    def _ensure_mid_gpu(mut self, needed: Int) raises:
        if self.mid_cap < needed:
            self.mid_dev = self.ts.ctx.value().enqueue_create_buffer[DT](needed)
            self.mid_cap = needed

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
        assert_tag_for["Residual", target](self.ts.target_tag)
        var input = inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            self._ensure_mid_cpu(BATCH * Self.IN_DIMS[0])
            var mid = TileTensor(
                mptr(self.mid_cpu.unsafe_ptr()),
                row_major[BATCH, Self.IN_DIMS[0]](),
            )
            self.inner.forward[target, BATCH, POLICY=POLICY](input, output=mid)
            var ip = mptr(input.ptr)
            var op = mptr(output_v.ptr)
            var mp = self.mid_cpu.unsafe_ptr()
            comptime N = BATCH * Self.IN_DIMS[0]
            var k = 0
            while k + CPU_SIMD_W <= N:
                op.store(
                    k,
                    mp.load[width=CPU_SIMD_W](k) + ip.load[width=CPU_SIMD_W](k),
                )
                k += CPU_SIMD_W
            while k < N:
                op[k] = mp[k] + ip[k]
                k += 1
        else:
            self._ensure_mid_gpu(BATCH * Self.IN_DIMS[0])
            var in_p_w = mptr(input.ptr)
            var out_p_w = mptr(output_v.ptr)
            var mp: UnsafePointer[
                Scalar[DT], MutAnyOrigin
            ] = self.mid_dev.value().unsafe_ptr()
            var mid = TileTensor(mp, row_major[BATCH, Self.IN_DIMS[0]]())
            self.inner.forward[target, BATCH, POLICY=POLICY](input, output=mid)
            comptime layout = Layout.row_major(BATCH, Self.IN_DIMS[0])
            var mid_lt = LayoutTensor[DT, layout, MutAnyOrigin](
                self.mid_dev.value()
            )
            var in_lt = LayoutTensor[DT, layout, MutAnyOrigin](in_p_w)
            var out_lt = LayoutTensor[DT, layout, MutAnyOrigin](out_p_w)
            comptime n_blocks = (BATCH * Self.IN_DIMS[0] + TPB - 1) // TPB
            comptime kernel = _elementwise_add_kernel[BATCH, Self.IN_DIMS[0]]
            self.ts.ctx.value().enqueue_function[kernel](
                mid_lt,
                in_lt,
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
        assert_tag_for["Residual", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = grad_inputs.tile[0, BATCH, Self.IN_DIMS[0]]()

        comptime if target == "cpu":
            self._ensure_mid_cpu(BATCH * Self.IN_DIMS[0])
            var tmp = TileTensor(
                mptr(self.mid_cpu.unsafe_ptr()),
                row_major[BATCH, Self.IN_DIMS[0]](),
            )
            self.inner.vjp[
                target,
                BATCH,
                POLICY=POLICY,
                mode=mode,
            ](grad_output_v, tmp)
            var gop = mptr(grad_output_v.ptr)
            var gip = mptr(grad_input_v.ptr)
            var tp = self.mid_cpu.unsafe_ptr()
            comptime N = BATCH * Self.IN_DIMS[0]
            var k = 0
            while k + CPU_SIMD_W <= N:
                gip.store(
                    k,
                    tp.load[width=CPU_SIMD_W](k)
                    + gop.load[width=CPU_SIMD_W](k),
                )
                k += CPU_SIMD_W
            while k < N:
                gip[k] = tp[k] + gop[k]
                k += 1
        else:
            self._ensure_mid_gpu(BATCH * Self.IN_DIMS[0])
            var go_p_w = mptr(grad_output_v.ptr)
            var gi_p_w = mptr(grad_input_v.ptr)
            var mp: UnsafePointer[
                Scalar[DT], MutAnyOrigin
            ] = self.mid_dev.value().unsafe_ptr()
            var tmp = TileTensor(mp, row_major[BATCH, Self.IN_DIMS[0]]())
            self.inner.vjp[
                target,
                BATCH,
                POLICY=POLICY,
                mode=mode,
            ](grad_output_v, tmp)
            comptime layout = Layout.row_major(BATCH, Self.IN_DIMS[0])
            var tmp_lt = LayoutTensor[DT, layout, MutAnyOrigin](
                self.mid_dev.value()
            )
            var go_lt = LayoutTensor[DT, layout, MutAnyOrigin](go_p_w)
            var gi_lt = LayoutTensor[DT, layout, MutAnyOrigin](gi_p_w)
            comptime n_blocks = (BATCH * Self.IN_DIMS[0] + TPB - 1) // TPB
            comptime kernel = _elementwise_add_kernel[BATCH, Self.IN_DIMS[0]]
            self.ts.ctx.value().enqueue_function[kernel](
                tmp_lt,
                go_lt,
                gi_lt,
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    # ----- Walkers ---------------------------------------------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["Residual", target](self.ts.target_tag)
        var sep = "." if prefix.byte_length() > 0 else ""
        self.inner.for_each_param[target, V](prefix + sep + "inner", visitor)

    def for_each_state[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["Residual", target](self.ts.target_tag)
        var sep = "." if prefix.byte_length() > 0 else ""
        self.inner.for_each_state[target, V](prefix + sep + "inner", visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["Residual", target](self.ts.target_tag)
        self.inner.zero_grad[target]()

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        self.inner.set_attr[ATTR](value)
