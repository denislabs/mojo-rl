"""ModalitySpaceAttention[D, N_HEADS, S, N_LATENTS, MODE, USE_MAX] — a
`MaskedAttention` whose modality mask is fixed by comptime params (storage
surface). Transformed from legacy `nn.primitives.modality_space_attention`
(surface-only change; the modality-id layout + mask install carried VERBATIM).

The Module `make[target, INIT](ctx)` signature is fixed by the trait, so a
bare `MaskedAttention` (whose mask is installed at runtime via `set_mask`)
cannot be dropped into a `Sequential` — there is no place to pass the mask.
This thin wrapper closes that gap: its comptime params determine the token
layout, so `make` builds the modality mask itself and installs it into an
inner `MaskedAttention`, then delegates forward/vjp. It is a drop-in
`Sequential` leaf (qkv-major input, same I/O dims as the attention op).

Layout: the first `N_LATENTS` of `S` tokens are latents (modality LATENT);
the remaining `S - N_LATENTS` are a single IMAGE segment (the tokenizer
layout). `MODE` selects the encoder/decoder/wm_agent mask (see
`build_modality_mask`). Param-free (attention has no params), so
`for_each_param` / `zero_grad` inherit the no-op defaults. Owns one inner
`MaskedAttention` Module field → forward/vjp delegate; param/state walkers
recurse into it (no-op, but kept for uniformity).
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


struct ModalitySpaceAttention[
    D: Int,
    N_HEADS: Int,
    S: Int,
    N_LATENTS: Int,
    MODE: StaticString,
    USE_MAX: Bool = True,
](Module):
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
        # First N_LATENTS = LATENT(-1); rest = a single IMAGE(0) segment.
        var ids = List[Int]()
        for _ in range(Self.N_LATENTS):
            ids.append(-1)
        for _ in range(Self.S - Self.N_LATENTS):
            ids.append(0)
        return ids^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "ModalitySpaceAttention: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        m.inner = MaskedAttention[
            Self.D, Self.N_HEADS, Self.S, Self.USE_MAX
        ].make[target, INIT](ctx)
        m.inner.set_mask(
            build_modality_mask[Self.MODE](
                Self._modality_ids(), Self.N_LATENTS
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
