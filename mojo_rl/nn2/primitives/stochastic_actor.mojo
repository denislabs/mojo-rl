"""StochasticActor[OBS_DIM, ACT_DIM, *TRUNK] — Gaussian policy head.
Phase 5.7.

Topology:
    obs (BATCH × OBS_DIM)
      → Sequential[*TRUNK]                   # variadic trunk
      → (BATCH × HIDDEN)
      → Parallel[Linear[HIDDEN, ACT_DIM],
                 Linear[HIDDEN, ACT_DIM]]    # mu head + log_std head
      → packed output (BATCH × 2*ACT_DIM)

Output packing:
    output[b, :ACT_DIM]            = mu(b)
    output[b, ACT_DIM:2*ACT_DIM]   = log_std(b)

This matches the standard PPO / SAC continuous actor pattern. Callers
slice the output to get (mu, log_std) and compute log_prob /
reparameterized samples downstream — those pieces live outside this
Module (they don't have learnable params).

Constraints (comptime-checked):
    - TRUNK.size >= 1
    - TRUNK[0].IN_DIM == OBS_DIM
    - Consecutive trunk dims match (Sequential checks this)

Why not just a type-alias for `Sequential[*TRUNK, Parallel[Linear, Linear]]`?
HIDDEN (last trunk module's OUT_DIM) is a comptime-derived dim, and
Mojo nightly's type-alias system doesn't easily compute it across a
variadic. A dedicated struct also leaves room for future extensions
(log_std clamping, log_std as a learned constant, etc.) without
touching call sites.

Threads POLICY through both children.
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
from ..combinators import Sequential, Parallel
from .linear import Linear


struct StochasticActor[
    OBS_DIM: Int,
    ACT_DIM: Int,
    *TRUNK: Module,
](Module):
    comptime IN_DIM = Self.OBS_DIM
    comptime OUT_DIM = 2 * Self.ACT_DIM
    comptime N_TRUNK = Self.TRUNK.size
    comptime HIDDEN = Self.TRUNK[Self.N_TRUNK - 1].OUT_DIM

    var trunk: Sequential[*Self.TRUNK]
    var heads: Parallel[
        Linear[Self.HIDDEN, Self.ACT_DIM],
        Linear[Self.HIDDEN, Self.ACT_DIM],
    ]
    var ctx: Optional[DeviceContext]

    # Persistent scratch between trunk and heads (BATCH × HIDDEN).
    var mid_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var mid_dev: Optional[DeviceBuffer[DT]]
    var mid_cap: Int

    var _target_tag: Int8
    var _inference: Bool

    def __init__(out self):
        comptime assert Self.N_TRUNK >= 1, (
            "StochasticActor requires at least one TRUNK module"
        )
        comptime assert (
            Self.TRUNK[0].IN_DIM == Self.OBS_DIM
        ), "StochasticActor: TRUNK[0].IN_DIM must equal OBS_DIM"
        self.trunk = Sequential[*Self.TRUNK]()
        self.heads = Parallel[
            Linear[Self.HIDDEN, Self.ACT_DIM],
            Linear[Self.HIDDEN, Self.ACT_DIM],
        ]()
        self.ctx = None
        self.mid_cpu = alloc[Scalar[DT]](1)
        self.mid_dev = None
        self.mid_cap = 0
        self._target_tag = TARGET_UNINIT
        self._inference = False

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert (
            target == "cpu"
        ), "StochasticActor.make[target='gpu', INIT] requires a DeviceContext"
        var a = Self()
        a.trunk = Sequential[*Self.TRUNK].make[target, INIT]()
        a.heads = Parallel[
            Linear[Self.HIDDEN, Self.ACT_DIM],
            Linear[Self.HIDDEN, Self.ACT_DIM],
        ].make[target, INIT]()
        a._target_tag = TARGET_CPU
        return a^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert (
            target == "gpu"
        ), "StochasticActor.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        var a = Self()
        a.trunk = Sequential[*Self.TRUNK].make[target, INIT](ctx)
        a.heads = Parallel[
            Linear[Self.HIDDEN, Self.ACT_DIM],
            Linear[Self.HIDDEN, Self.ACT_DIM],
        ].make[target, INIT](ctx)
        a.ctx = ctx
        a.mid_dev = ctx.enqueue_create_buffer[DT](1)
        a._target_tag = TARGET_GPU
        return a^

    def __del__(deinit self):
        self.mid_cpu.free()

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "StochasticActor: method called with [target='"
                + String(target)
                + "'] but module was make'd for a different target "
                + "(tag=" + String(Int(self._target_tag)) + ")"
            )

    def _ensure_mid_cpu(mut self, needed: Int):
        if self.mid_cap < needed:
            self.mid_cpu.free()
            self.mid_cpu = alloc[Scalar[DT]](needed)
            self.mid_cap = needed

    def _ensure_mid_gpu(mut self, needed: Int) raises:
        if self.mid_cap < needed:
            self.mid_dev = self.ctx.value().enqueue_create_buffer[DT](needed)
            self.mid_cap = needed

    def forward[
        target: StaticString,
        BATCH: Int,
        LIN: TensorLayout, LOUT: TensorLayout,
        OIN: MutOrigin,    OOUT: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        input: TileTensor[DT, LIN, OIN],
        mut output: TileTensor[DT, LOUT, OOUT],
    ) raises:
        comptime assert input.flat_rank == 2, "input rank-2"
        comptime assert output.flat_rank == 2, "output rank-2"
        self._assert_tag[target]()

        comptime if target == "cpu":
            self._ensure_mid_cpu(BATCH * Self.HIDDEN)
            var mid = TileTensor(self.mid_cpu, row_major[BATCH, Self.HIDDEN]())
            self.trunk.forward[target, BATCH, POLICY=POLICY](input, mid)
            self.heads.forward[target, BATCH, POLICY=POLICY](mid, output)
        else:
            self._ensure_mid_gpu(BATCH * Self.HIDDEN)
            var mp: UnsafePointer[Scalar[DT], MutAnyOrigin] = self.mid_dev.value().unsafe_ptr()
            var mid = TileTensor(mp, row_major[BATCH, Self.HIDDEN]())
            self.trunk.forward[target, BATCH, POLICY=POLICY](input, mid)
            self.heads.forward[target, BATCH, POLICY=POLICY](mid, output)

    def backward[
        target: StaticString,
        BATCH: Int,
        LGO: TensorLayout, LGI: TensorLayout,
        OGO: MutOrigin,    OGI: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        grad_output: TileTensor[DT, LGO, OGO],
        mut grad_input: TileTensor[DT, LGI, OGI],
    ) raises:
        comptime assert grad_output.flat_rank == 2, "grad_output rank-2"
        comptime assert grad_input.flat_rank == 2, "grad_input rank-2"
        self._assert_tag[target]()

        comptime if target == "cpu":
            self._ensure_mid_cpu(BATCH * Self.HIDDEN)
            # heads.backward writes into mid (= ∂L/∂trunk_out).
            var mid = TileTensor(self.mid_cpu, row_major[BATCH, Self.HIDDEN]())
            self.heads.backward[target, BATCH, POLICY=POLICY](grad_output, mid)
            self.trunk.backward[target, BATCH, POLICY=POLICY](mid, grad_input)
        else:
            self._ensure_mid_gpu(BATCH * Self.HIDDEN)
            var mp: UnsafePointer[Scalar[DT], MutAnyOrigin] = self.mid_dev.value().unsafe_ptr()
            var mid = TileTensor(mp, row_major[BATCH, Self.HIDDEN]())
            self.heads.backward[target, BATCH, POLICY=POLICY](grad_output, mid)
            self.trunk.backward[target, BATCH, POLICY=POLICY](mid, grad_input)

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V,) raises:
        self._assert_tag[target]()
        var sep = "." if prefix.byte_length() > 0 else ""
        self.trunk.for_each_param[target](prefix + sep + "trunk", visitor)
        self.heads.for_each_param[target](prefix + sep + "heads", visitor)

    def set_inference(mut self, value: Bool):
        self._inference = value
        self.trunk.set_inference(value)
        self.heads.set_inference(value)
