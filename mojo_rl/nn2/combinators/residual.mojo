"""Residual[Inner] — `y = inner(x) + x`.

Scaffold: `ts: TargetStorage`,
`backward[mode]` instead of separate `backward_input`, walkers
recurse into Inner.

Phase 10A buffer surface dropped; the single mid slab IS the
inter-module wiring (Residual is itself an orchestrator for one child).

Backward order: `inner.backward[mode]` runs first (writes into mid),
then `grad_input = mid + grad_output` SIMD-add. Identical to v1.
"""

from std.memory import alloc
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, CPU_SIMD_W
from ..core import Initializer, AMPPolicy, NoAMP, ParamVisitor
from ..core.module import Module
from ..core.target_storage import TargetStorage, assert_tag_for


def _elementwise_add_kernel[
    BATCH: Int, DIM: Int,
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
        dst[bi, di] = rebind[Scalar[DT]](a[bi, di]) + rebind[Scalar[DT]](b[bi, di])


struct Residual[Inner: Module](Module):
    comptime IN_DIM = Self.Inner.IN_DIM
    comptime OUT_DIM = Self.Inner.OUT_DIM

    var inner: Self.Inner

    var mid_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var mid_dev: Optional[DeviceBuffer[DT]]
    var mid_cap: Int

    var ts: TargetStorage

    def __init__(out self):
        comptime assert (
            Self.Inner.IN_DIM == Self.Inner.OUT_DIM
        ), "Residual requires Inner.IN_DIM == Inner.OUT_DIM"
        self.inner = Self.Inner()
        self.mid_cpu = alloc[Scalar[DT]](1)
        self.mid_dev = None
        self.mid_cap = 0
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "Residual.make[target='gpu', INIT] requires a DeviceContext"
        )
        var r = Self()
        r.inner = Self.Inner.make[target, INIT]()
        r.ts = TargetStorage.make_cpu()
        return r^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "Residual.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        var r = Self()
        r.inner = Self.Inner.make[target, INIT](ctx)
        r.mid_dev = ctx.enqueue_create_buffer[DT](1)
        r.ts = TargetStorage.make_gpu(ctx)
        return r^

    def __del__(deinit self):
        self.mid_cpu.free()

    def _ensure_mid_cpu(mut self, needed: Int):
        if self.mid_cap < needed:
            self.mid_cpu.free()
            self.mid_cpu = alloc[Scalar[DT]](needed)
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
        input: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        comptime assert input.flat_rank == 2, "input rank-2"
        comptime assert output.flat_rank == 2, "output rank-2"
        assert_tag_for["Residual", target](self.ts.target_tag)

        comptime if target == "cpu":
            self._ensure_mid_cpu(BATCH * Self.IN_DIM)
            var mid = TileTensor(self.mid_cpu, row_major[BATCH, Self.IN_DIM]())
            self.inner.forward[target, BATCH, POLICY=POLICY](input, mid)
            var ip = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
            var op = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)
            var mp = self.mid_cpu
            comptime N = BATCH * Self.IN_DIM
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
            self._ensure_mid_gpu(BATCH * Self.IN_DIM)
            var in_p_w  = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
            var out_p_w = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)
            var mp: UnsafePointer[Scalar[DT], MutAnyOrigin] = self.mid_dev.value().unsafe_ptr()
            var mid = TileTensor(mp, row_major[BATCH, Self.IN_DIM]())
            self.inner.forward[target, BATCH, POLICY=POLICY](input, mid)
            comptime layout = Layout.row_major(BATCH, Self.IN_DIM)
            var mid_lt = LayoutTensor[DT, layout, MutAnyOrigin](self.mid_dev.value())
            var in_lt  = LayoutTensor[DT, layout, MutAnyOrigin](in_p_w)
            var out_lt = LayoutTensor[DT, layout, MutAnyOrigin](out_p_w)
            comptime TPB = 128
            comptime n_blocks = (BATCH * Self.IN_DIM + TPB - 1) // TPB
            comptime kernel = _elementwise_add_kernel[BATCH, Self.IN_DIM]
            self.ts.ctx.value().enqueue_function[kernel](
                mid_lt, in_lt, out_lt,
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
        assert_tag_for["Residual", target](self.ts.target_tag)

        comptime if target == "cpu":
            self._ensure_mid_cpu(BATCH * Self.IN_DIM)
            var tmp = TileTensor(self.mid_cpu, row_major[BATCH, Self.IN_DIM]())
            self.inner.backward[
                target, BATCH, POLICY=POLICY, mode=mode,
            ](grad_output, tmp)
            var gop = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output.ptr)
            var gip = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input.ptr)
            var tp = self.mid_cpu
            comptime N = BATCH * Self.IN_DIM
            var k = 0
            while k + CPU_SIMD_W <= N:
                gip.store(
                    k,
                    tp.load[width=CPU_SIMD_W](k) + gop.load[width=CPU_SIMD_W](k),
                )
                k += CPU_SIMD_W
            while k < N:
                gip[k] = tp[k] + gop[k]
                k += 1
        else:
            self._ensure_mid_gpu(BATCH * Self.IN_DIM)
            var go_p_w = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output.ptr)
            var gi_p_w = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input.ptr)
            var mp: UnsafePointer[Scalar[DT], MutAnyOrigin] = self.mid_dev.value().unsafe_ptr()
            var tmp = TileTensor(mp, row_major[BATCH, Self.IN_DIM]())
            self.inner.backward[
                target, BATCH, POLICY=POLICY, mode=mode,
            ](grad_output, tmp)
            comptime layout = Layout.row_major(BATCH, Self.IN_DIM)
            var tmp_lt = LayoutTensor[DT, layout, MutAnyOrigin](self.mid_dev.value())
            var go_lt  = LayoutTensor[DT, layout, MutAnyOrigin](go_p_w)
            var gi_lt  = LayoutTensor[DT, layout, MutAnyOrigin](gi_p_w)
            comptime TPB = 128
            comptime n_blocks = (BATCH * Self.IN_DIM + TPB - 1) // TPB
            comptime kernel = _elementwise_add_kernel[BATCH, Self.IN_DIM]
            self.ts.ctx.value().enqueue_function[kernel](
                tmp_lt, go_lt, gi_lt,
                grid_dim=n_blocks, block_dim=TPB,
            )

    # ----- Walkers ---------------------------------------------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["Residual", target](self.ts.target_tag)
        var sep = "." if prefix.byte_length() > 0 else ""
        self.inner.for_each_param[target, V](prefix + sep + "inner", visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["Residual", target](self.ts.target_tag)
        self.inner.zero_grad[target]()
