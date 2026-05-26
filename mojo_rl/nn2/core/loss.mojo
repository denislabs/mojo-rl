"""Loss trait — (logits, targets) → scalar + grad_logits.

Phase 2.4: methods take `target: StaticString` as comptime method param.

Stage B (Phase 10B): tensor args use the partial-spec form
`TileTensor[mut=..., dtype=DT, address_space=AddressSpace.GENERIC,
element_size=1, ...]` so callers can pass tiles built from any source
buffer without intermediate widening. Impls rebind to `MutAnyOrigin`
only at the kernel-launch boundary.
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor

from ..constants import DT
from .amp import AMPPolicy, NoAMP


trait Loss(Defaultable & Movable & ImplicitlyDestructible):
    comptime OUT_DIM: Int

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU
        (impls raise at runtime if missing)."""
        ...

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        logits: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        targets: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
    ) raises -> Scalar[DT]:
        ...

    def vjp[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        targets: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        mut grad_logits: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        """Vector-Jacobian product — gradient of the scalar loss w.r.t.
        `logits` (the input cached by the most recent `forward`). Phase 4
        rename of `Loss.backward`, semantics unchanged."""
        ...
