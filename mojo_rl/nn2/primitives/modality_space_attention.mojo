"""ModalitySpaceAttention[D, N_HEADS, S, N_LATENTS, MODE, USE_MAX] — a
`MaskedAttention` whose modality mask is fixed by comptime params.

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
`for_each_param` / `zero_grad` inherit the no-op defaults.
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor

from ..constants import DT
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module
from ..core.tensor_pack import TensorPack
from ..core.target_storage import TargetStorage, assert_tag_for
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
    var ts: TargetStorage

    def __init__(out self):
        self.inner = MaskedAttention[
            Self.D, Self.N_HEADS, Self.S, Self.USE_MAX
        ]()
        self.ts = TargetStorage.make_uninit()

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
        ].make[target=target, INIT=INIT](ctx)
        m.inner.set_mask(
            build_modality_mask[Self.MODE](Self._modality_ids(), Self.N_LATENTS)
        )
        comptime if target == "cpu":
            m.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("ModalitySpaceAttention.make[gpu]: ctx required")
            m.ts = TargetStorage.make_gpu(ctx.value())
        return m^

    @staticmethod
    def display_label() -> String:
        return String("ModalitySpaceAttention")

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorPack[Self.ARITY],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["ModalitySpaceAttention", target](self.ts.target_tag)
        self.inner.forward[target, BATCH, POLICY=POLICY](
            inputs, output=output
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
        grad_inputs: TensorPack[Self.ARITY],
    ) raises:
        assert_tag_for["ModalitySpaceAttention", target](self.ts.target_tag)
        self.inner.vjp[target, BATCH, POLICY=POLICY, mode=mode](
            grad_output, grad_inputs
        )
