"""TernaryModule trait — 3-input → 1-output ops (Block D-7).

Mirrors `BinaryModule` for ops that naturally take three inputs (e.g.
ternary concatenation, fused 3-way addition). Mojo nightly rejects
variadic `*inputs: TileTensor[...]` packs (see
`feedback_mojo_variadic_tiletensor_blocked`), so fixed-arity-3 is the
ergonomic upper bound for value-level multi-input traits today. Higher
arity uses the packed-tensor convention (`Concat[*BRANCHES]`).

This trait is intentionally minimal — DreamerV3 and MuZero dynamics
modules don't yet plug Ternary nodes into `ComputeGraph` (which still
ranges over `*NODES: GraphNode` with only `UnaryNode` / `BinaryNode`
wrappers). The trait exists so the two concrete impls (`TernaryConcat`,
`TernaryFusedAdd`) compose cleanly when they do — and to mark the slot
in the trait surface.
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor

from ..constants import DT
from .initializer import Initializer
from .amp import AMPPolicy, NoAMP
from .param_visitor import ParamVisitor


trait TernaryModule(Defaultable & Movable & ImplicitlyDestructible):
    comptime IN0_DIM: Int
    comptime IN1_DIM: Int
    comptime IN2_DIM: Int
    comptime OUT_DIM: Int

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        ...

    @staticmethod
    def make[target: StaticString, INIT: Initializer](
        ctx: DeviceContext,
    ) raises -> Self:
        ...

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        in0: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        in1: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        in2: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        ...

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
        mut grad_in0: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
        mut grad_in1: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
        mut grad_in2: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        ...

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        pass

    def zero_grad[target: StaticString](mut self) raises:
        pass
