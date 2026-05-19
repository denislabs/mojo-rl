"""Sequential[*MODULES] — variadic chain of N Modules, target chosen per call.

Phase 2.4: target is a comptime method param. Because every child is
`Defaultable` (Phase 2.4 invariant), `Tuple[*MODULES]()` works, and
`Sequential.make[target, INIT](ctx?)` can recursively build each child:

```mojo
@staticmethod
def make[target, INIT](ctx) raises -> Self:
    var s = Self()                                  # children = default Tuple
    comptime for i in range(N):
        s.children[i] = MODULES[i].make[target, INIT](ctx)
    return s^
```

Composition rule (checked at compile time):
  - For each adjacent pair `(child_i, child_{i+1})`,
    `child_i.OUT_DIM == child_{i+1}.IN_DIM`.

**Persistent intermediate buffers.** The N-1 transient slabs needed
between adjacent children are stored on `Sequential` and lazily grown
to `BATCH × MODULES[i].OUT_DIM` on the first forward/backward at that
BATCH. Forward and backward share the slabs (no overlap; each child's
internal cache holds whatever backward needs from forward).

Per-call slab allocation would race the GPU queue and exhaust device
memory at scale — same lesson as nn v1's workspace pattern.
"""

from std.memory import alloc
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major

from ..constants import DT
from ..core import (
    Module, ParamVisitor, Initializer,
    AMPPolicy, NoAMP,
    TARGET_UNINIT, TARGET_CPU, TARGET_GPU, target_tag_for,
)


struct Sequential[*MODULES: Module](Module):
    comptime N = Self.MODULES.size
    comptime IN_DIM = Self.MODULES[0].IN_DIM
    comptime OUT_DIM = Self.MODULES[Self.N - 1].OUT_DIM

    var children: Tuple[*Self.MODULES]
    var ctx: Optional[DeviceContext]

    # Persistent intermediate buffers (N-1 entries each, lazy-grown).
    var mid_cpu: List[UnsafePointer[Scalar[DT], MutAnyOrigin]]
    var mid_dev: List[DeviceBuffer[DT]]
    var mid_caps: List[Int]

    var _target_tag: Int8
    var _inference: Bool

    # ------------------------------------------------------------------
    # Defaultable: each child is Defaultable, so `Tuple[*MODULES]()`
    # default-constructs every slot.
    # ------------------------------------------------------------------

    def __init__(out self):
        """Defaultable form — empty placeholders, tag=UNINIT. Required so
        `Tuple[*MODULES]()` default-construction works inside nested
        Sequentials."""
        comptime assert Self.N >= 1, "Sequential requires at least one child"
        comptime if Self.N >= 2:
            comptime for i in range(Self.N - 1):
                comptime assert (
                    Self.MODULES[i].OUT_DIM == Self.MODULES[i + 1].IN_DIM
                ), "Sequential: adjacent child dims must match"
        self.children = Tuple[*Self.MODULES]()
        self.ctx = None
        self.mid_cpu = List[UnsafePointer[Scalar[DT], MutAnyOrigin]]()
        self.mid_dev = List[DeviceBuffer[DT]]()
        self.mid_caps = List[Int]()
        self._target_tag = TARGET_UNINIT
        self._inference = False

    def __init__(out self, var *children: *Self.MODULES):
        """CPU variadic constructor — accepts pre-built CPU children
        (e.g. via `Linear[...].make["cpu", INIT=Kaiming]()`). Sets
        tag=CPU. Each child is responsible for its own tag check."""
        comptime assert Self.N >= 1, "Sequential requires at least one child"
        comptime if Self.N >= 2:
            comptime for i in range(Self.N - 1):
                comptime assert (
                    Self.MODULES[i].OUT_DIM == Self.MODULES[i + 1].IN_DIM
                ), "Sequential: adjacent child dims must match"
        self.children = Tuple(*children^)
        self.ctx = None
        self.mid_cpu = List[UnsafePointer[Scalar[DT], MutAnyOrigin]]()
        self.mid_dev = List[DeviceBuffer[DT]]()
        self.mid_caps = List[Int]()
        comptime if Self.N >= 2:
            for _ in range(Self.N - 1):
                var p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](1)
                self.mid_cpu.append(p)
                self.mid_caps.append(0)
        self._target_tag = TARGET_CPU
        self._inference = False

    def __init__(out self, var *children: *Self.MODULES, ctx: DeviceContext) raises:
        """GPU variadic constructor — accepts pre-built GPU children."""
        comptime assert Self.N >= 1, "Sequential requires at least one child"
        comptime if Self.N >= 2:
            comptime for i in range(Self.N - 1):
                comptime assert (
                    Self.MODULES[i].OUT_DIM == Self.MODULES[i + 1].IN_DIM
                ), "Sequential: adjacent child dims must match"
        self.children = Tuple(*children^)
        self.ctx = ctx
        self.mid_cpu = List[UnsafePointer[Scalar[DT], MutAnyOrigin]]()
        self.mid_dev = List[DeviceBuffer[DT]]()
        self.mid_caps = List[Int]()
        comptime if Self.N >= 2:
            for _ in range(Self.N - 1):
                self.mid_dev.append(ctx.enqueue_create_buffer[DT](1))
                self.mid_caps.append(0)
        self._target_tag = TARGET_GPU
        self._inference = False

    # ------------------------------------------------------------------
    # make[target, INIT] — recursive over children.
    # ------------------------------------------------------------------

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        """CPU factory — recursively builds each child via
        `MODULES[i].make[target='cpu', INIT]()`. Allocates CPU mid slabs
        as length-1 stubs (lazy-grown on first forward)."""
        comptime assert target == "cpu", (
            "Sequential.make[target='gpu', INIT] requires a DeviceContext"
        )
        var s = Self()
        comptime for i in range(Self.N):
            s.children[i] = Self.MODULES[i].make[target, INIT]()
        comptime if Self.N >= 2:
            for _ in range(Self.N - 1):
                var p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](1)
                s.mid_cpu.append(p)
                s.mid_caps.append(0)
        s._target_tag = TARGET_CPU
        return s^

    @staticmethod
    def make[target: StaticString, INIT: Initializer](ctx: DeviceContext) raises -> Self:
        """GPU factory — recurses via `MODULES[i].make[target='gpu', INIT](ctx)`."""
        comptime assert target == "gpu", (
            "Sequential.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        var s = Self()
        comptime for i in range(Self.N):
            s.children[i] = Self.MODULES[i].make[target, INIT](ctx)
        s.ctx = ctx
        comptime if Self.N >= 2:
            for _ in range(Self.N - 1):
                s.mid_dev.append(ctx.enqueue_create_buffer[DT](1))
                s.mid_caps.append(0)
        s._target_tag = TARGET_GPU
        return s^

    def __del__(deinit self):
        for p in self.mid_cpu:
            p.free()

    # ------------------------------------------------------------------
    # Tag check + lazy-grow helpers.
    # ------------------------------------------------------------------

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "Sequential: method called with [target='" + String(target)
                + "'] but module was make'd for a different target "
                + "(tag=" + String(Int(self._target_tag)) + ")"
            )

    def _ensure_mid_cpu[i: Int](mut self, needed: Int):
        if self.mid_caps[i] < needed:
            self.mid_cpu[i].free()
            self.mid_cpu[i] = alloc[Scalar[DT]](needed)
            self.mid_caps[i] = needed

    def _ensure_mid_gpu[i: Int](mut self, needed: Int) raises:
        if self.mid_caps[i] < needed:
            self.mid_dev[i] = self.ctx.value().enqueue_create_buffer[DT](needed)
            self.mid_caps[i] = needed

    # ------------------------------------------------------------------
    # Forward — chain N children through N-1 persistent slabs.
    # ------------------------------------------------------------------

    def forward[
        target: StaticString,
        BATCH: Int,
        LIN: TensorLayout,
        LOUT: TensorLayout,
        OIN: MutOrigin,
        OOUT: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        input: TileTensor[DT, LIN, OIN],
        mut output: TileTensor[DT, LOUT, OOUT],
    ) raises:
        comptime assert input.flat_rank  == 2, "input must be rank-2"
        comptime assert output.flat_rank == 2, "output must be rank-2"
        self._assert_tag[target]()

        comptime if Self.N == 1:
            self.children[0].forward[target, BATCH, POLICY=POLICY](input, output)
        else:
            comptime if target == "cpu":
                _forward_cpu[target, BATCH, POLICY=POLICY](self, input, output)
            else:
                _forward_gpu[target, BATCH, POLICY=POLICY](self, input, output)

    # ------------------------------------------------------------------
    # Backward — reverse traversal, same N-1 slabs.
    # ------------------------------------------------------------------

    def backward[
        target: StaticString,
        BATCH: Int,
        LGO: TensorLayout,
        LGI: TensorLayout,
        OGO: MutOrigin,
        OGI: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        grad_output: TileTensor[DT, LGO, OGO],
        mut grad_input: TileTensor[DT, LGI, OGI],
    ) raises:
        comptime assert grad_output.flat_rank == 2, "grad_output must be rank-2"
        comptime assert grad_input.flat_rank  == 2, "grad_input must be rank-2"
        self._assert_tag[target]()

        comptime if Self.N == 1:
            self.children[0].backward[target, BATCH, POLICY=POLICY](grad_output, grad_input)
        else:
            comptime if target == "cpu":
                _backward_cpu[target, BATCH, POLICY=POLICY](self, grad_output, grad_input)
            else:
                _backward_gpu[target, BATCH, POLICY=POLICY](self, grad_output, grad_input)

    # ------------------------------------------------------------------
    # for_each_param — recurse with indexed prefix.
    # ------------------------------------------------------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](
        mut self,
        prefix: String,
        mut visitor: V,
    ) raises:
        self._assert_tag[target]()
        var sep = "." if prefix.byte_length() > 0 else ""
        comptime for i in range(Self.N):
            self.children[i].for_each_param[target](
                prefix + sep + String(i), visitor
            )

    def set_inference(mut self, value: Bool):
        # Set the flag on this combinator and recurse into every child.
        self._inference = value
        comptime for i in range(Self.N):
            self.children[i].set_inference(value)


# ──────────────────────────────────────────────────────────────────────────
# Free functions for forward/backward bodies.
# ──────────────────────────────────────────────────────────────────────────


def _forward_cpu[
    target: StaticString,
    BATCH: Int,
    LIN: TensorLayout,
    LOUT: TensorLayout,
    OIN: MutOrigin,
    OOUT: MutOrigin,
    POLICY: AMPPolicy,
    *MODULES: Module,
](
    mut seq: Sequential[*MODULES],
    input: TileTensor[DT, LIN, OIN],
    mut output: TileTensor[DT, LOUT, OOUT],
) raises:
    comptime N = MODULES.size

    comptime for i in range(N - 1):
        seq._ensure_mid_cpu[i](BATCH * MODULES[i].OUT_DIM)

    comptime for i in range(N):
        comptime if i == 0:
            var out_mid = TileTensor(seq.mid_cpu[0], row_major[BATCH, MODULES[0].OUT_DIM]())
            seq.children[0].forward[target, BATCH, POLICY=POLICY](input, out_mid)
        elif i == N - 1:
            var in_mid = TileTensor(seq.mid_cpu[N - 2], row_major[BATCH, MODULES[N - 1].IN_DIM]())
            seq.children[i].forward[target, BATCH, POLICY=POLICY](in_mid, output)
        else:
            var in_mid  = TileTensor(seq.mid_cpu[i - 1], row_major[BATCH, MODULES[i].IN_DIM]())
            var out_mid = TileTensor(seq.mid_cpu[i],     row_major[BATCH, MODULES[i].OUT_DIM]())
            seq.children[i].forward[target, BATCH, POLICY=POLICY](in_mid, out_mid)


def _forward_gpu[
    target: StaticString,
    BATCH: Int,
    LIN: TensorLayout,
    LOUT: TensorLayout,
    OIN: MutOrigin,
    OOUT: MutOrigin,
    POLICY: AMPPolicy,
    *MODULES: Module,
](
    mut seq: Sequential[*MODULES],
    input: TileTensor[DT, LIN, OIN],
    mut output: TileTensor[DT, LOUT, OOUT],
) raises:
    comptime N = MODULES.size

    comptime for i in range(N - 1):
        seq._ensure_mid_gpu[i](BATCH * MODULES[i].OUT_DIM)

    # Launder the slab pointers through MutAnyOrigin so Mojo's aliasing
    # analysis doesn't see adjacent slabs (different DeviceBuffers) as
    # potential aliases via `seq.mid_dev`'s origin.
    comptime for i in range(N):
        comptime if i == 0:
            var p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = seq.mid_dev[0].unsafe_ptr()
            var out_mid = TileTensor(p, row_major[BATCH, MODULES[0].OUT_DIM]())
            seq.children[0].forward[target, BATCH, POLICY=POLICY](input, out_mid)
        elif i == N - 1:
            var p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = seq.mid_dev[N - 2].unsafe_ptr()
            var in_mid = TileTensor(p, row_major[BATCH, MODULES[N - 1].IN_DIM]())
            seq.children[i].forward[target, BATCH, POLICY=POLICY](in_mid, output)
        else:
            var pi: UnsafePointer[Scalar[DT], MutAnyOrigin] = seq.mid_dev[i - 1].unsafe_ptr()
            var po: UnsafePointer[Scalar[DT], MutAnyOrigin] = seq.mid_dev[i].unsafe_ptr()
            var in_mid  = TileTensor(pi, row_major[BATCH, MODULES[i].IN_DIM]())
            var out_mid = TileTensor(po, row_major[BATCH, MODULES[i].OUT_DIM]())
            seq.children[i].forward[target, BATCH, POLICY=POLICY](in_mid, out_mid)


def _backward_cpu[
    target: StaticString,
    BATCH: Int,
    LGO: TensorLayout,
    LGI: TensorLayout,
    OGO: MutOrigin,
    OGI: MutOrigin,
    POLICY: AMPPolicy,
    *MODULES: Module,
](
    mut seq: Sequential[*MODULES],
    grad_output: TileTensor[DT, LGO, OGO],
    mut grad_input: TileTensor[DT, LGI, OGI],
) raises:
    comptime N = MODULES.size

    comptime for i in range(N - 1):
        seq._ensure_mid_cpu[i](BATCH * MODULES[i].OUT_DIM)

    comptime for j in range(N):
        comptime i = N - 1 - j
        comptime if i == N - 1:
            var out_grad = TileTensor(seq.mid_cpu[N - 2], row_major[BATCH, MODULES[N - 1].IN_DIM]())
            seq.children[N - 1].backward[target, BATCH, POLICY=POLICY](grad_output, out_grad)
        elif i == 0:
            var in_grad = TileTensor(seq.mid_cpu[0], row_major[BATCH, MODULES[0].OUT_DIM]())
            seq.children[0].backward[target, BATCH, POLICY=POLICY](in_grad, grad_input)
        else:
            var in_grad  = TileTensor(seq.mid_cpu[i],     row_major[BATCH, MODULES[i].OUT_DIM]())
            var out_grad = TileTensor(seq.mid_cpu[i - 1], row_major[BATCH, MODULES[i].IN_DIM]())
            seq.children[i].backward[target, BATCH, POLICY=POLICY](in_grad, out_grad)


def _backward_gpu[
    target: StaticString,
    BATCH: Int,
    LGO: TensorLayout,
    LGI: TensorLayout,
    OGO: MutOrigin,
    OGI: MutOrigin,
    POLICY: AMPPolicy,
    *MODULES: Module,
](
    mut seq: Sequential[*MODULES],
    grad_output: TileTensor[DT, LGO, OGO],
    mut grad_input: TileTensor[DT, LGI, OGI],
) raises:
    comptime N = MODULES.size

    comptime for i in range(N - 1):
        seq._ensure_mid_gpu[i](BATCH * MODULES[i].OUT_DIM)

    # Same MutAnyOrigin laundering as _forward_gpu (see comment there).
    comptime for j in range(N):
        comptime i = N - 1 - j
        comptime if i == N - 1:
            var p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = seq.mid_dev[N - 2].unsafe_ptr()
            var out_grad = TileTensor(p, row_major[BATCH, MODULES[N - 1].IN_DIM]())
            seq.children[N - 1].backward[target, BATCH, POLICY=POLICY](grad_output, out_grad)
        elif i == 0:
            var p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = seq.mid_dev[0].unsafe_ptr()
            var in_grad = TileTensor(p, row_major[BATCH, MODULES[0].OUT_DIM]())
            seq.children[0].backward[target, BATCH, POLICY=POLICY](in_grad, grad_input)
        else:
            var pi: UnsafePointer[Scalar[DT], MutAnyOrigin] = seq.mid_dev[i].unsafe_ptr()
            var po: UnsafePointer[Scalar[DT], MutAnyOrigin] = seq.mid_dev[i - 1].unsafe_ptr()
            var in_grad  = TileTensor(pi, row_major[BATCH, MODULES[i].OUT_DIM]())
            var out_grad = TileTensor(po, row_major[BATCH, MODULES[i].IN_DIM]())
            seq.children[i].backward[target, BATCH, POLICY=POLICY](in_grad, out_grad)
