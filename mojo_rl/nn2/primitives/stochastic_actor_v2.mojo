"""StochasticActor — retrofit (Phase C, deferred from Phase B).

  obs → SequentialV2[*TRUNK] → (BATCH × HIDDEN)
                             → ParallelV2[LinearV2, LinearV2]
                             → packed [mu | log_std]

Composes the Phase C combinators with Phase B LinearV2. Same scaffold
collapse as everywhere else: `ts: TargetStorage`, `backward[mode]`
collapses v1's `backward` + `backward_input`, walkers recurse into
trunk + heads.

The trunk-to-heads mid slab is owned by this combinator (not by the
inner SequentialV2 or ParallelV2 — they each own their own internal
slabs).
"""

from std.memory import alloc
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from ..constants import DT
from ..core import Initializer, AMPPolicy, NoAMP, ParamVisitor
from ..core.module_v2 import ModuleV2
from ..core.target_storage import TargetStorage, assert_tag_for
from ..combinators.sequential_v2 import SequentialV2
from ..combinators.parallel_v2 import ParallelV2
from .linear_v2 import LinearV2


struct StochasticActorV2[
    OBS_DIM: Int,
    ACT_DIM: Int,
    *TRUNK: ModuleV2,
](ModuleV2):
    comptime IN_DIM = Self.OBS_DIM
    comptime OUT_DIM = 2 * Self.ACT_DIM
    comptime N_TRUNK = Self.TRUNK.size
    comptime HIDDEN = Self.TRUNK[Self.N_TRUNK - 1].OUT_DIM

    var trunk: SequentialV2[*Self.TRUNK]
    var heads: ParallelV2[
        LinearV2[Self.HIDDEN, Self.ACT_DIM],
        LinearV2[Self.HIDDEN, Self.ACT_DIM],
    ]

    var mid_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var mid_dev: Optional[DeviceBuffer[DT]]
    var mid_cap: Int

    var ts: TargetStorage

    def __init__(out self):
        comptime assert Self.N_TRUNK >= 1, (
            "StochasticActorV2 requires at least one TRUNK module"
        )
        comptime assert (
            Self.TRUNK[0].IN_DIM == Self.OBS_DIM
        ), "StochasticActorV2: TRUNK[0].IN_DIM must equal OBS_DIM"
        self.trunk = SequentialV2[*Self.TRUNK]()
        self.heads = ParallelV2[
            LinearV2[Self.HIDDEN, Self.ACT_DIM],
            LinearV2[Self.HIDDEN, Self.ACT_DIM],
        ]()
        self.mid_cpu = alloc[Scalar[DT]](1)
        self.mid_dev = None
        self.mid_cap = 0
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "StochasticActorV2.make[target='gpu', INIT] requires a DeviceContext"
        )
        var a = Self()
        a.trunk = SequentialV2[*Self.TRUNK].make[target, INIT]()
        a.heads = ParallelV2[
            LinearV2[Self.HIDDEN, Self.ACT_DIM],
            LinearV2[Self.HIDDEN, Self.ACT_DIM],
        ].make[target, INIT]()
        a.ts = TargetStorage.make_cpu()
        return a^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "StochasticActorV2.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        var a = Self()
        a.trunk = SequentialV2[*Self.TRUNK].make[target, INIT](ctx)
        a.heads = ParallelV2[
            LinearV2[Self.HIDDEN, Self.ACT_DIM],
            LinearV2[Self.HIDDEN, Self.ACT_DIM],
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
        assert_tag_for["StochasticActorV2", target](self.ts.target_tag)

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
        assert_tag_for["StochasticActorV2", target](self.ts.target_tag)

        comptime if target == "cpu":
            self._ensure_mid_cpu(BATCH * Self.HIDDEN)
            var mid = TileTensor(self.mid_cpu, row_major[BATCH, Self.HIDDEN]())
            self.heads.backward[
                target, BATCH, POLICY=POLICY, mode=mode,
            ](grad_output, mid)
            self.trunk.backward[
                target, BATCH, POLICY=POLICY, mode=mode,
            ](mid, grad_input)
        else:
            self._ensure_mid_gpu(BATCH * Self.HIDDEN)
            var mp: UnsafePointer[Scalar[DT], MutAnyOrigin] = self.mid_dev.value().unsafe_ptr()
            var mid = TileTensor(mp, row_major[BATCH, Self.HIDDEN]())
            self.heads.backward[
                target, BATCH, POLICY=POLICY, mode=mode,
            ](grad_output, mid)
            self.trunk.backward[
                target, BATCH, POLICY=POLICY, mode=mode,
            ](mid, grad_input)

    # ----- Walkers ---------------------------------------------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["StochasticActorV2", target](self.ts.target_tag)
        var sep = "." if prefix.byte_length() > 0 else ""
        self.trunk.for_each_param[target, V](prefix + sep + "trunk", visitor)
        self.heads.for_each_param[target, V](prefix + sep + "heads", visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["StochasticActorV2", target](self.ts.target_tag)
        self.trunk.zero_grad[target]()
        self.heads.zero_grad[target]()
