"""DynamicsSpaceAttention[D, N_HEADS, NSP, NREG, NAGENT, MODE, USE_MAX] — the
space-attention leaf for the Dreamer 4 *dynamics* transformer (storage surface).
Transformed from legacy `nn.primitives.dynamics_space_attention` (surface-only
change; the per-frame modality-id layout + mask install carried VERBATIM).

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

Param-free (attention has no params); owns one inner `MaskedAttention` Module
field → forward/vjp delegate; param/state walkers recurse into it.
"""

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from ..core.initializer import Initializer
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.walkers import join_name
from ..core.amp import AMPPolicy, NoAMP
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

    def __init__(out self):
        self.inner = MaskedAttention[
            Self.D, Self.N_HEADS, Self.S, Self.USE_MAX
        ]()

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
        ].make[target, INIT](ctx)
        # Agent modality FIXED to DYN_MOD_AGENT (5): with NAGENT=0 no token has
        # this id → full mixing; with NAGENT>0 the isolation is enforced.
        m.inner.set_mask(
            build_modality_mask[Self.MODE](
                Self._modality_ids(), 0, agent_mod_in=DYN_MOD_AGENT
            ),
            ctx,
        )
        return m^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        self.inner.forward[target, B, POLICY=POLICY](inputs, out, ctx)

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        self.inner.vjp[target, B, POLICY=POLICY](
            forward_input, grad_output, grad_inputs, ctx
        )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.inner.for_each_param[target](
            visitor, ctx, join_name(prefix, String(0))
        )

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.inner.for_each_state[target](
            visitor, ctx, join_name(prefix, String(0))
        )

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        self.inner.zero_grad[target](ctx)
