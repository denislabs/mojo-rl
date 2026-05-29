"""Sequential[*MODULES] — variadic chain of N `Module` children.

Mid-slabs persist on the combinator; lazy-grown to
`BATCH × MODULES[i].OUT_DIM` on first call. Forward and backward share
the same slabs — Sequential IS the orchestrator, the slabs ARE the
inter-module wiring.

Composition rule (checked at compile time): for each adjacent pair
`(child_i, child_{i+1})`, `child_i.OUT_DIM == child_{i+1}.IN_DIM`.

`backward[mode]` flows the comptime `mode` to every child: `"all"`
accumulates param grads, `"input_only"` skips them.

**Backward-order safety**: leaves that obey the param-grads-before-
grad_input invariant (e.g. Linear) are safe when their
`_cached_input_ptr` aliases the predecessor's mid slab — the slab
isn't reused as a grad target until that child's backward returns.
"""

from std.memory import alloc
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from ..constants import DT
from ..core import (
    Initializer, AMPPolicy, NoAMP, ParamVisitor, DisplayStep,
)
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import TargetStorage, assert_tag_for


struct Sequential[*MODULES: Module](Module):
    comptime ARITY: Int = 1
    comptime N = Self.MODULES.size
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.MODULES[0].IN_DIMS[0])
    comptime OUT_DIM = Self.MODULES[Self.N - 1].OUT_DIM

    @staticmethod
    def display_label() -> String:
        return String("Sequential")

    @staticmethod
    def display_steps() -> List[DisplayStep]:
        """Expand the chain — one step per child, carrying its display
        label + output width. Lets `ComputeGraph.describe` exporters open
        a Sequential node instead of showing one opaque box."""
        var steps = List[DisplayStep]()
        comptime for i in range(Self.N):
            steps.append(
                DisplayStep(
                    Self.MODULES[i].display_label(),
                    Self.MODULES[i].OUT_DIM,
                )
            )
        return steps^

    var children: Tuple[*Self.MODULES]

    # Persistent intermediate buffers (N-1 entries each, lazy-grown).
    var mid_cpu: List[UnsafePointer[Scalar[DT], MutAnyOrigin]]
    var mid_dev: List[DeviceBuffer[DT]]
    var mid_caps: List[Int]

    var ts: TargetStorage

    # ----- Defaultable -----------------------------------------------------

    def __init__(out self):
        comptime assert Self.N >= 1, "Sequential requires at least one child"
        comptime if Self.N >= 2:
            comptime for i in range(Self.N - 1):
                comptime assert (
                    Self.MODULES[i].OUT_DIM == Self.MODULES[i + 1].IN_DIMS[0]
                ), "Sequential: adjacent child dims must match"
        self.children = Tuple[*Self.MODULES]()
        self.mid_cpu = List[UnsafePointer[Scalar[DT], MutAnyOrigin]]()
        self.mid_dev = List[DeviceBuffer[DT]]()
        self.mid_caps = List[Int]()
        self.ts = TargetStorage.make_uninit()

    def __init__(out self, var *children: *Self.MODULES):
        """CPU variadic constructor — accepts pre-built CPU children."""
        comptime assert Self.N >= 1, "Sequential requires at least one child"
        comptime if Self.N >= 2:
            comptime for i in range(Self.N - 1):
                comptime assert (
                    Self.MODULES[i].OUT_DIM == Self.MODULES[i + 1].IN_DIMS[0]
                ), "Sequential: adjacent child dims must match"
        self.children = Tuple(*children^)
        self.mid_cpu = List[UnsafePointer[Scalar[DT], MutAnyOrigin]]()
        self.mid_dev = List[DeviceBuffer[DT]]()
        self.mid_caps = List[Int]()
        comptime if Self.N >= 2:
            for _ in range(Self.N - 1):
                var p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](1)
                self.mid_cpu.append(p)
                self.mid_caps.append(0)
        self.ts = TargetStorage.make_cpu()

    def __init__(out self, var *children: *Self.MODULES, ctx: DeviceContext) raises:
        """GPU variadic constructor — accepts pre-built GPU children."""
        comptime assert Self.N >= 1, "Sequential requires at least one child"
        comptime if Self.N >= 2:
            comptime for i in range(Self.N - 1):
                comptime assert (
                    Self.MODULES[i].OUT_DIM == Self.MODULES[i + 1].IN_DIMS[0]
                ), "Sequential: adjacent child dims must match"
        self.children = Tuple(*children^)
        self.mid_cpu = List[UnsafePointer[Scalar[DT], MutAnyOrigin]]()
        self.mid_dev = List[DeviceBuffer[DT]]()
        self.mid_caps = List[Int]()
        comptime if Self.N >= 2:
            for _ in range(Self.N - 1):
                self.mid_dev.append(ctx.enqueue_create_buffer[DT](1))
                self.mid_caps.append(0)
        self.ts = TargetStorage.make_gpu(ctx)

    # ----- Factories -------------------------------------------------------

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory — recurses via
        `MODULES[i].make[target, INIT](ctx=ctx)`."""
        comptime assert target == "cpu" or target == "gpu", (
            "Sequential: target must be 'cpu' or 'gpu'"
        )
        var s = Self()
        comptime for i in range(Self.N):
            s.children[i] = Self.MODULES[i].make[target, INIT](ctx=ctx)
        comptime if target == "cpu":
            comptime if Self.N >= 2:
                for _ in range(Self.N - 1):
                    var p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](1)
                    s.mid_cpu.append(p)
                    s.mid_caps.append(0)
            s.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("Sequential.make[target='gpu']: ctx required")
            var ctx_v = ctx.value()
            comptime if Self.N >= 2:
                for _ in range(Self.N - 1):
                    s.mid_dev.append(ctx_v.enqueue_create_buffer[DT](1))
                    s.mid_caps.append(0)
            s.ts = TargetStorage.make_gpu(ctx_v)
        return s^

    def __del__(deinit self):
        for p in self.mid_cpu:
            p.free()

    # ----- Lazy-grow helpers ----------------------------------------------

    def _ensure_mid_cpu[i: Int](mut self, needed: Int):
        if self.mid_caps[i] < needed:
            self.mid_cpu[i].free()
            self.mid_cpu[i] = alloc[Scalar[DT]](needed)
            self.mid_caps[i] = needed

    def _ensure_mid_gpu[i: Int](mut self, needed: Int) raises:
        if self.mid_caps[i] < needed:
            self.mid_dev[i] = self.ts.ctx.value().enqueue_create_buffer[DT](needed)
            self.mid_caps[i] = needed

    # ----- Forward ---------------------------------------------------------

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
        assert_tag_for["Sequential", target](self.ts.target_tag)
        var input = typed_view[BATCH, Self.IN_DIMS[0]](inputs[0])
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if Self.N == 1:
            self.children[0].forward[target, BATCH, POLICY=POLICY](input, output=output_v)
        else:
            comptime if target == "cpu":
                _forward_cpu[target, BATCH, POLICY=POLICY](self, input, output_v)
            else:
                _forward_gpu[target, BATCH, POLICY=POLICY](self, input, output_v)

    # ----- Backward --------------------------------------------------------

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
        assert_tag_for["Sequential", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = typed_view_mut[BATCH, Self.IN_DIMS[0]](grad_inputs[0])

        comptime if Self.N == 1:
            self.children[0].vjp[
                target, BATCH, POLICY=POLICY, mode=mode,
            ](grad_output_v, grad_input_v)
        else:
            comptime if target == "cpu":
                _backward_cpu[target, BATCH, POLICY=POLICY, mode=mode](
                    self, grad_output_v, grad_input_v,
                )
            else:
                _backward_gpu[target, BATCH, POLICY=POLICY, mode=mode](
                    self, grad_output_v, grad_input_v,
                )

    # ----- Walkers ---------------------------------------------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["Sequential", target](self.ts.target_tag)
        var sep = "." if prefix.byte_length() > 0 else ""
        comptime for i in range(Self.N):
            self.children[i].for_each_param[target, V](
                prefix + sep + String(i), visitor,
            )

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["Sequential", target](self.ts.target_tag)
        comptime for i in range(Self.N):
            self.children[i].zero_grad[target]()


# ──────────────────────────────────────────────────────────────────────
# Free functions for forward / backward bodies.
# ──────────────────────────────────────────────────────────────────────


def _forward_cpu[
    target: StaticString,
    BATCH: Int,
    POLICY: AMPPolicy,
    *MODULES: Module,
](
    mut seq: Sequential[*MODULES],
    input: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC,
        element_size=1, origin=MutAnyOrigin, ...,
    ],
    mut output: TileTensor[
        mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
        element_size=1, origin=MutAnyOrigin, ...,
    ],
) raises:
    comptime N = MODULES.size

    comptime for i in range(N - 1):
        seq._ensure_mid_cpu[i](BATCH * MODULES[i].OUT_DIM)

    comptime for i in range(N):
        comptime if i == 0:
            var out_mid = TileTensor(
                seq.mid_cpu[0], row_major[BATCH, MODULES[0].OUT_DIM](),
            )
            seq.children[0].forward[target, BATCH, POLICY=POLICY](input, output=out_mid)
        elif i == N - 1:
            var in_mid = TileTensor(
                seq.mid_cpu[N - 2], row_major[BATCH, MODULES[N - 1].IN_DIMS[0]](),
            )
            seq.children[i].forward[target, BATCH, POLICY=POLICY](in_mid, output=output)
        else:
            var in_mid  = TileTensor(
                seq.mid_cpu[i - 1], row_major[BATCH, MODULES[i].IN_DIMS[0]](),
            )
            var out_mid = TileTensor(
                seq.mid_cpu[i], row_major[BATCH, MODULES[i].OUT_DIM](),
            )
            seq.children[i].forward[target, BATCH, POLICY=POLICY](in_mid, output=out_mid)


def _forward_gpu[
    target: StaticString,
    BATCH: Int,
    POLICY: AMPPolicy,
    *MODULES: Module,
](
    mut seq: Sequential[*MODULES],
    input: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC,
        element_size=1, origin=MutAnyOrigin, ...,
    ],
    mut output: TileTensor[
        mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
        element_size=1, origin=MutAnyOrigin, ...,
    ],
) raises:
    comptime N = MODULES.size

    comptime for i in range(N - 1):
        seq._ensure_mid_gpu[i](BATCH * MODULES[i].OUT_DIM)

    # Launder slab ptrs through MutAnyOrigin so Mojo's aliasing analysis
    # doesn't flag adjacent slab refs (distinct DeviceBuffers) as aliases.
    comptime for i in range(N):
        comptime if i == 0:
            var p: UnsafePointer[Scalar[DT], MutAnyOrigin] = seq.mid_dev[0].unsafe_ptr()
            var out_mid = TileTensor(p, row_major[BATCH, MODULES[0].OUT_DIM]())
            seq.children[0].forward[target, BATCH, POLICY=POLICY](input, output=out_mid)
        elif i == N - 1:
            var p: UnsafePointer[Scalar[DT], MutAnyOrigin] = seq.mid_dev[N - 2].unsafe_ptr()
            var in_mid = TileTensor(p, row_major[BATCH, MODULES[N - 1].IN_DIMS[0]]())
            seq.children[i].forward[target, BATCH, POLICY=POLICY](in_mid, output=output)
        else:
            var pi: UnsafePointer[Scalar[DT], MutAnyOrigin] = seq.mid_dev[i - 1].unsafe_ptr()
            var po: UnsafePointer[Scalar[DT], MutAnyOrigin] = seq.mid_dev[i].unsafe_ptr()
            var in_mid  = TileTensor(pi, row_major[BATCH, MODULES[i].IN_DIMS[0]]())
            var out_mid = TileTensor(po, row_major[BATCH, MODULES[i].OUT_DIM]())
            seq.children[i].forward[target, BATCH, POLICY=POLICY](in_mid, output=out_mid)


def _backward_cpu[
    target: StaticString,
    BATCH: Int,
    POLICY: AMPPolicy,
    mode: StaticString,
    *MODULES: Module,
](
    mut seq: Sequential[*MODULES],
    grad_output: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC,
        element_size=1, origin=MutAnyOrigin, ...,
    ],
    mut grad_input: TileTensor[
        mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
        element_size=1, origin=MutAnyOrigin, ...,
    ],
) raises:
    comptime N = MODULES.size

    comptime for i in range(N - 1):
        seq._ensure_mid_cpu[i](BATCH * MODULES[i].OUT_DIM)

    comptime for j in range(N):
        comptime i = N - 1 - j
        comptime if i == N - 1:
            var out_grad = TileTensor(
                seq.mid_cpu[N - 2], row_major[BATCH, MODULES[N - 1].IN_DIMS[0]](),
            )
            seq.children[N - 1].vjp[
                target, BATCH, POLICY=POLICY, mode=mode,
            ](grad_output, out_grad)
        elif i == 0:
            var in_grad = TileTensor(
                seq.mid_cpu[0], row_major[BATCH, MODULES[0].OUT_DIM](),
            )
            seq.children[0].vjp[
                target, BATCH, POLICY=POLICY, mode=mode,
            ](in_grad, grad_input)
        else:
            var in_grad  = TileTensor(
                seq.mid_cpu[i], row_major[BATCH, MODULES[i].OUT_DIM](),
            )
            var out_grad = TileTensor(
                seq.mid_cpu[i - 1], row_major[BATCH, MODULES[i].IN_DIMS[0]](),
            )
            seq.children[i].vjp[
                target, BATCH, POLICY=POLICY, mode=mode,
            ](in_grad, out_grad)


def _backward_gpu[
    target: StaticString,
    BATCH: Int,
    POLICY: AMPPolicy,
    mode: StaticString,
    *MODULES: Module,
](
    mut seq: Sequential[*MODULES],
    grad_output: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC,
        element_size=1, origin=MutAnyOrigin, ...,
    ],
    mut grad_input: TileTensor[
        mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
        element_size=1, origin=MutAnyOrigin, ...,
    ],
) raises:
    comptime N = MODULES.size

    comptime for i in range(N - 1):
        seq._ensure_mid_gpu[i](BATCH * MODULES[i].OUT_DIM)

    comptime for j in range(N):
        comptime i = N - 1 - j
        comptime if i == N - 1:
            var p: UnsafePointer[Scalar[DT], MutAnyOrigin] = seq.mid_dev[N - 2].unsafe_ptr()
            var out_grad = TileTensor(p, row_major[BATCH, MODULES[N - 1].IN_DIMS[0]]())
            seq.children[N - 1].vjp[
                target, BATCH, POLICY=POLICY, mode=mode,
            ](grad_output, out_grad)
        elif i == 0:
            var p: UnsafePointer[Scalar[DT], MutAnyOrigin] = seq.mid_dev[0].unsafe_ptr()
            var in_grad = TileTensor(p, row_major[BATCH, MODULES[0].OUT_DIM]())
            seq.children[0].vjp[
                target, BATCH, POLICY=POLICY, mode=mode,
            ](in_grad, grad_input)
        else:
            var pi: UnsafePointer[Scalar[DT], MutAnyOrigin] = seq.mid_dev[i].unsafe_ptr()
            var po: UnsafePointer[Scalar[DT], MutAnyOrigin] = seq.mid_dev[i - 1].unsafe_ptr()
            var in_grad  = TileTensor(pi, row_major[BATCH, MODULES[i].OUT_DIM]())
            var out_grad = TileTensor(po, row_major[BATCH, MODULES[i].IN_DIMS[0]]())
            seq.children[i].vjp[
                target, BATCH, POLICY=POLICY, mode=mode,
            ](in_grad, out_grad)
