"""DynamicsSpaceAttention[D, N_HEADS, NSP, NREG, NAGENT, MODE, USE_MAX] — the
space-attention leaf for the Dreamer 4 *dynamics* transformer.

Like `ModalitySpaceAttention` it is a thin wrapper over `MaskedAttention` whose
modality mask is fixed by comptime params, so it drops into a `Sequential`
(qkv-major input, same I/O dims). The difference is the token layout: instead
of the tokenizer's [latent | image] split, it builds the *dynamics* per-frame
layout (model.py:Dynamics)

    [ action | signal | step | spatial×NSP | register×NREG | agent×NAGENT ]

with modality ids 0,1,2,3,4,5. The AGENT modality id is FIXED to 5 (passed
explicitly to `build_modality_mask`), so with `NAGENT = 0` no token carries the
agent id and the `wm_agent_bc` mask collapses to full mixing — bit-identical to
the unconditional dynamics (where every token mixes freely). With `NAGENT > 0`
the `wm_agent_bc` mask enforces the paper §3.3 isolation: agent tokens read the
whole frame; nothing reads back the agent tokens.

Param-free (attention has no params), so `for_each_param` / `zero_grad` inherit
the no-op defaults.
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor

from ..constants import DT
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module
from ..core.target_storage import TargetStorage, assert_tag_for
from .masked_attention import MaskedAttention, build_modality_mask


# Fixed modality ids for the dynamics per-frame token layout.
comptime DYN_MOD_ACTION = 0
comptime DYN_MOD_SIGNAL = 1
comptime DYN_MOD_STEP = 2
comptime DYN_MOD_SPATIAL = 3
comptime DYN_MOD_REGISTER = 4
comptime DYN_MOD_AGENT = 5


struct DynamicsSpaceAttention[
    D: Int,
    N_HEADS: Int,
    NSP: Int,
    NREG: Int,
    NAGENT: Int,
    MODE: StaticString,
    USE_MAX: Bool = True,
](Module):
    comptime S: Int = 3 + Self.NSP + Self.NREG + Self.NAGENT
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.S * Self.D * 3)
    comptime OUT_DIM = Self.S * Self.D

    var inner: MaskedAttention[Self.D, Self.N_HEADS, Self.S, Self.USE_MAX]
    var ts: TargetStorage

    def __init__(out self):
        self.inner = MaskedAttention[
            Self.D, Self.N_HEADS, Self.S, Self.USE_MAX
        ]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def _modality_ids() -> List[Int]:
        var ids = List[Int]()
        ids.append(DYN_MOD_ACTION)
        ids.append(DYN_MOD_SIGNAL)
        ids.append(DYN_MOD_STEP)
        for _ in range(Self.NSP):
            ids.append(DYN_MOD_SPATIAL)
        for _ in range(Self.NREG):
            ids.append(DYN_MOD_REGISTER)
        for _ in range(Self.NAGENT):
            ids.append(DYN_MOD_AGENT)
        return ids^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "DynamicsSpaceAttention: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        m.inner = MaskedAttention[
            Self.D, Self.N_HEADS, Self.S, Self.USE_MAX
        ].make[target=target, INIT=INIT](ctx)
        # Agent modality FIXED to DYN_MOD_AGENT (5): with NAGENT=0 no token has
        # this id → full mixing; with NAGENT>0 the isolation is enforced.
        m.inner.set_mask(
            build_modality_mask[Self.MODE](
                Self._modality_ids(), 0, agent_mod_in=DYN_MOD_AGENT
            )
        )
        comptime if target == "cpu":
            m.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("DynamicsSpaceAttention.make[gpu]: ctx required")
            m.ts = TargetStorage.make_gpu(ctx.value())
        return m^

    @staticmethod
    def display_label() -> String:
        return String("DynamicsSpaceAttention")

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
        assert_tag_for["DynamicsSpaceAttention", target](self.ts.target_tag)
        self.inner.forward[target, BATCH, POLICY=POLICY](
            inputs[0], output=output
        )

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
        assert_tag_for["DynamicsSpaceAttention", target](self.ts.target_tag)
        self.inner.vjp[target, BATCH, POLICY=POLICY, mode=mode](
            grad_output, grad_inputs[0]
        )
