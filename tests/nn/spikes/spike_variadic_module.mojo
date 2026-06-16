from layout import TileTensor, row_major
from std.gpu.memory import AddressSpace
from std.gpu.host import DeviceContext
from mojo_rl.nn.constants import DT
from mojo_rl.nn.core import Initializer, AMPPolicy, NoAMP
from mojo_rl.nn.core.param_visitor import ParamVisitor

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor

from mojo_rl.nn.constants import DT, CPU_SIMD_W
from mojo_rl.nn.core import Initializer, AMPPolicy, NoAMP
from mojo_rl.nn.core.module import typed_view, typed_view_mut
from mojo_rl.nn.core.target_storage import TargetStorage, assert_tag_for


trait VariadicModule(Defaultable & Movable & ImplicitlyDestructible):
    comptime ARITY: Int
    comptime IN_DIMS: InlineArray[Int, Self.ARITY]
    comptime OUT_DIM: Int

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        ...

    @staticmethod
    def make[
        target: StaticString,
        INIT: Initializer,
    ](ctx: DeviceContext) raises -> Self:
        ...

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        var *inputs: TileTensor[
            dtype=DT,
            address_space=AddressSpace.GENERIC,
            element_size=1,
            origin=MutAnyOrigin,
            ...,
        ],
        mut output: TileTensor[
            mut=True,
            dtype=DT,
            address_space=AddressSpace.GENERIC,
            element_size=1,
            origin=MutAnyOrigin,
            ...,
        ],
    ) raises:
        """N-ary forward. Leaf body must:
          1. Use `comptime if Self.ARITY == K:` to branch on arity.
          2. Inside each branch, rebuild typed views with
             `typed_view[BATCH, IN<i>_DIM](inputs[i])` and
             `typed_view_mut[BATCH, Self.OUT_DIM](output)`.
          3. Existing kernel/SIMD bodies follow once views are typed.

        Callers pass `output` as a keyword arg (`output=...`) — required
        by Mojo to disambiguate from the variadic pack."""
        ...

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
        mut *grad_inputs: TileTensor[
            mut=True,
            dtype=DT,
            address_space=AddressSpace.GENERIC,
            element_size=1,
            origin=MutAnyOrigin,
            ...,
        ],
    ) raises:
        """N-ary vector-Jacobian product.

        `mode = "all"` (default): writes all grad_inputs AND accumulates
        param grads (if any).
        `mode = "input_only"`: writes grad_inputs ONLY; skips param-grad
        work. Used by StopGradParams + SAC actor flow through twin
        critics. Param-less leaves ignore `mode`.

        BACKWARD-ORDER INVARIANT: leaves that alias forward inputs by
        pointer (Linear's cached_input_ptr) must compute param grads
        BEFORE writing grad_inputs[i] — clobbering the cache mid-read
        breaks the gradient."""
        ...

    # ──────────────────────────────────────────────────────────────────
    # Provided defaults — parameterless leaves auto-inherit no-ops.
    # ──────────────────────────────────────────────────────────────────

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        """Default: no params. Parameterised leaves override to call
        `for_each_param_auto[Self, V, target]` from `walkers.mojo`."""
        pass

    def zero_grad[target: StaticString](mut self) raises:
        """Default: no params. Override on param-bearing leaves."""
        pass

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        """Per-call runtime attribute mutation. Default no-op. Modules
        with mutable runtime state (e.g. Scale.multiplier, Clamp.min_val)
        override and comptime-branch on `ATTR`."""
        pass


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — init (output = src) and accum (output += src). Forward
# emits 1 init + (N-1) accum launches; vjp emits N init (copy) launches
# (one per grad-input).
# ──────────────────────────────────────────────────────────────────────


def _add_init_kernel[
    N: Int
](
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < N:
        output[idx] = rebind[Scalar[DT]](src[idx])


def _add_accum_kernel[
    N: Int
](
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < N:
        output[idx] = output[idx] + rebind[Scalar[DT]](src[idx])


# ──────────────────────────────────────────────────────────────────────
# Add[DIM, N]
# ──────────────────────────────────────────────────────────────────────


struct Add[DIM_: Int, N_: Int](VariadicModule):
    comptime ARITY: Int = Self.N_
    comptime IN_DIMS: InlineArray[Int, Self.ARITY] = Self._in_dims()
    comptime OUT_DIM: Int = Self.DIM_

    @staticmethod
    def _in_dims() -> InlineArray[Int, Self.ARITY]:
        """Return the input dimensions for the module."""
        var in_dims = InlineArray[Int, Self.ARITY](fill=Self.DIM_)
        return in_dims

    var ts: TargetStorage

    def __init__(out self):
        comptime assert Self.N_ >= 2, "Add: needs at least 2 inputs"
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert (
            target == "cpu"
        ), "Add.make[target='gpu', INIT] requires a DeviceContext"
        var a = Self()
        a.ts = TargetStorage.make_cpu()
        return a^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert (
            target == "gpu"
        ), "Add.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        var a = Self()
        a.ts = TargetStorage.make_gpu(ctx)
        return a^

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        var *inputs: TileTensor[
            dtype=DT,
            address_space=AddressSpace.GENERIC,
            element_size=1,
            origin=MutAnyOrigin,
            ...,
        ],
        mut output: TileTensor[
            mut=True,
            dtype=DT,
            address_space=AddressSpace.GENERIC,
            element_size=1,
            origin=MutAnyOrigin,
            ...,
        ],
    ) raises:
        assert_tag_for["Add", target](self.ts.target_tag)
        comptime TOTAL = BATCH * Self.DIM_

        comptime if target == "cpu":
            var o_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                output.ptr
            )
            var i0_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                inputs[0].ptr
            )
            # Init: output = inputs[0]
            var k = 0
            while k + CPU_SIMD_W <= TOTAL:
                o_p.store(k, i0_p.load[width=CPU_SIMD_W](k))
                k += CPU_SIMD_W
            while k < TOTAL:
                o_p[k] = i0_p[k]
                k += 1
            # Accumulate inputs[1..N)
            comptime for i in range(1, Self.N_):
                var ii_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    inputs[i].ptr
                )
                var kk = 0
                while kk + CPU_SIMD_W <= TOTAL:
                    o_p.store(
                        kk,
                        o_p.load[width=CPU_SIMD_W](kk)
                        + ii_p.load[width=CPU_SIMD_W](kk),
                    )
                    kk += CPU_SIMD_W
                while kk < TOTAL:
                    o_p[kk] = o_p[kk] + ii_p[kk]
                    kk += 1
        else:
            comptime layout = Layout.row_major(TOTAL)
            comptime TPB = 128
            comptime n_blocks = (TOTAL + TPB - 1) // TPB
            var o_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                output.ptr
            )
            var o_lt = LayoutTensor[DT, layout, MutAnyOrigin](o_p)

            # Init from inputs[0].
            var i0_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                inputs[0].ptr
            )
            var i0_lt = LayoutTensor[DT, layout, MutAnyOrigin](i0_p)
            comptime init_kernel = _add_init_kernel[TOTAL]
            self.ts.ctx.value().enqueue_function[init_kernel](
                i0_lt,
                o_lt,
                grid_dim=n_blocks,
                block_dim=TPB,
            )

            # Accumulate inputs[1..N).
            comptime for i in range(1, Self.N_):
                var ii_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    inputs[i].ptr
                )
                var ii_lt = LayoutTensor[DT, layout, MutAnyOrigin](ii_p)
                comptime accum_kernel = _add_accum_kernel[TOTAL]
                self.ts.ctx.value().enqueue_function[accum_kernel](
                    ii_lt,
                    o_lt,
                    grid_dim=n_blocks,
                    block_dim=TPB,
                )

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
        mut *grad_inputs: TileTensor[
            mut=True,
            dtype=DT,
            address_space=AddressSpace.GENERIC,
            element_size=1,
            origin=MutAnyOrigin,
            ...,
        ],
    ) raises:
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["Add", target](self.ts.target_tag)
        comptime TOTAL = BATCH * Self.DIM_

        comptime if target == "cpu":
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                grad_output.ptr
            )
            comptime for i in range(Self.N_):
                var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    grad_inputs[i].ptr
                )
                var k = 0
                while k + CPU_SIMD_W <= TOTAL:
                    gi_p.store(k, go_p.load[width=CPU_SIMD_W](k))
                    k += CPU_SIMD_W
                while k < TOTAL:
                    gi_p[k] = go_p[k]
                    k += 1
        else:
            comptime layout = Layout.row_major(TOTAL)
            comptime TPB = 128
            comptime n_blocks = (TOTAL + TPB - 1) // TPB
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                grad_output.ptr
            )
            var go_lt = LayoutTensor[DT, layout, MutAnyOrigin](go_p)
            comptime for i in range(Self.N_):
                var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    grad_inputs[i].ptr
                )
                var gi_lt = LayoutTensor[DT, layout, MutAnyOrigin](gi_p)
                comptime copy_kernel = _add_init_kernel[TOTAL]
                self.ts.ctx.value().enqueue_function[copy_kernel](
                    go_lt,
                    gi_lt,
                    grid_dim=n_blocks,
                    block_dim=TPB,
                )


def main() raises:
    comptime in_dims = Add[3, 3]._in_dims()
    for i in range(in_dims.size):
        print("in_dims[", i, "] =", in_dims[i])
    print("Hello, World!")
