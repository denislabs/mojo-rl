"""StochasticActor — actor head wrapping a feature trunk + GaussianHead.

  obs → Sequential[*TRUNK] → (BATCH × HIDDEN)
                             → Parallel[Linear, Linear]
                             → packed [mu | log_std]

Composes the Phase C combinators with Phase B Linear. Same scaffold
collapse as everywhere else: `ts: TargetStorage`, `backward[mode]`
collapses v1's `backward` + `backward_input`, walkers recurse into
trunk + heads.

The trunk-to-heads mid slab is owned by this combinator (not by the
inner Sequential or Parallel — they each own their own internal
slabs).
"""

from std.memory import alloc
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from ..constants import DT
from ..core import Initializer, AMPPolicy, NoAMP, ParamVisitor
from ..core.module import Module
from ..core.target_storage import TargetStorage, assert_tag_for
from ..combinators.sequential import Sequential
from ..combinators.parallel import Parallel
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

    var mid_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var mid_dev: Optional[DeviceBuffer[DT]]
    var mid_cap: Int

    var ts: TargetStorage

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
        self.mid_cpu = alloc[Scalar[DT]](1)
        self.mid_dev = None
        self.mid_cap = 0
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "StochasticActor.make[target='gpu', INIT] requires a DeviceContext"
        )
        var a = Self()
        a.trunk = Sequential[*Self.TRUNK].make[target, INIT]()
        a.heads = Parallel[
            Linear[Self.HIDDEN, Self.ACT_DIM],
            Linear[Self.HIDDEN, Self.ACT_DIM],
        ].make[target, INIT]()
        a.ts = TargetStorage.make_cpu()
        return a^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "StochasticActor.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        var a = Self()
        a.trunk = Sequential[*Self.TRUNK].make[target, INIT](ctx)
        a.heads = Parallel[
            Linear[Self.HIDDEN, Self.ACT_DIM],
            Linear[Self.HIDDEN, Self.ACT_DIM],
        ].make[target, INIT](ctx)
        a.mid_dev = ctx.enqueue_create_buffer[DT](1)
        a.ts = TargetStorage.make_gpu(ctx)
        return a^

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
        assert_tag_for["StochasticActor", target](self.ts.target_tag)

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

    # ----- Backward --------------------------------------------------------

    def vjp[
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
        assert_tag_for["StochasticActor", target](self.ts.target_tag)

        comptime if target == "cpu":
            self._ensure_mid_cpu(BATCH * Self.HIDDEN)
            var mid = TileTensor(self.mid_cpu, row_major[BATCH, Self.HIDDEN]())
            self.heads.vjp[
                target, BATCH, POLICY=POLICY, mode=mode,
            ](grad_output, mid)
            self.trunk.vjp[
                target, BATCH, POLICY=POLICY, mode=mode,
            ](mid, grad_input)
        else:
            self._ensure_mid_gpu(BATCH * Self.HIDDEN)
            var mp: UnsafePointer[Scalar[DT], MutAnyOrigin] = self.mid_dev.value().unsafe_ptr()
            var mid = TileTensor(mp, row_major[BATCH, Self.HIDDEN]())
            self.heads.vjp[
                target, BATCH, POLICY=POLICY, mode=mode,
            ](grad_output, mid)
            self.trunk.vjp[
                target, BATCH, POLICY=POLICY, mode=mode,
            ](mid, grad_input)

    # ----- Walkers ---------------------------------------------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["StochasticActor", target](self.ts.target_tag)
        var sep = "." if prefix.byte_length() > 0 else ""
        self.trunk.for_each_param[target, V](prefix + sep + "trunk", visitor)
        self.heads.for_each_param[target, V](prefix + sep + "heads", visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["StochasticActor", target](self.ts.target_tag)
        self.trunk.zero_grad[target]()
        self.heads.zero_grad[target]()
