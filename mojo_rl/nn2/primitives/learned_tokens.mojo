"""LearnedTokens[N_IN, N_NEW, D, PREPEND] — concatenate learned tokens.

Dreamer 4's encoder prepends learned latent tokens to the projected patches;
the decoder appends learned patch-query tokens to the up-projected latents.
This leaf does that concat with the learned tokens as its (only) parameter,
shared across the whole B·T batch (every frame gets the same learned tokens):

    PREPEND : out = [ learned(N_NEW) ‖ input(N_IN) ]   (encoder latents)
    else    : out = [ input(N_IN) ‖ learned(N_NEW) ]   (decoder queries)

IN_DIM = N_IN·D, OUT_DIM = (N_IN+N_NEW)·D, param = N_NEW·D. The param grad is
batch-reduced (the tokens are shared): grad_tokens[k] = Σ_bt grad_out[bt, new+k].

PHASE 1: CPU forward + vjp (param leaf, `Param` + auto visitors). GPU follows.
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from ..constants import DT
from ..core import (
    Initializer,
    AMPPolicy,
    NoAMP,
    Param,
    ParamVisitor,
    for_each_param_auto,
    zero_grad_auto,
)
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import TargetStorage, assert_tag_for


struct LearnedTokens[N_IN: Int, N_NEW: Int, D: Int, PREPEND: Bool](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.N_IN * Self.D)
    comptime OUT_DIM = (Self.N_IN + Self.N_NEW) * Self.D
    comptime NEW_N: Int = Self.N_NEW * Self.D
    comptime IN_N: Int = Self.N_IN * Self.D
    comptime NEW_OFF: Int = 0 if Self.PREPEND else Self.IN_N
    comptime IN_OFF: Int = Self.NEW_N if Self.PREPEND else 0

    var tokens: Param["tokens", False, Self.NEW_N]
    var ts: TargetStorage

    def __init__(out self):
        self.tokens = Param["tokens", False, Self.NEW_N]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu", "LearnedTokens: PHASE 1 is CPU-only"
        var m = Self()
        m.tokens = Param["tokens", False, Self.NEW_N].make_cpu()
        INIT.init_weight(
            m.tokens.value_unsafe_ptr_cpu(), Self.NEW_N, Self.N_NEW, Self.D
        )
        m.ts = TargetStorage.make_cpu()
        return m^

    @staticmethod
    def display_label() -> String:
        return String("LearnedTokens")

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
        assert_tag_for["LearnedTokens", target](self.ts.target_tag)
        var inp = typed_view[BATCH, Self.IN_N](inputs[0])
        var out = typed_view_mut[BATCH, Self.OUT_DIM](output)
        var tok = TileTensor(self.tokens.value, row_major[Self.NEW_N]())
        for bt in range(BATCH):
            for k in range(Self.NEW_N):
                out[bt, Self.NEW_OFF + k] = tok[k]
            for k in range(Self.IN_N):
                out[bt, Self.IN_OFF + k] = inp[bt, k]

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
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["LearnedTokens", target](self.ts.target_tag)
        var go = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var gi = typed_view_mut[BATCH, Self.IN_N](grad_inputs[0])
        for bt in range(BATCH):
            for k in range(Self.IN_N):
                gi[bt, k] = go[bt, Self.IN_OFF + k]
        comptime if mode == "all":
            var gtok = TileTensor(self.tokens.grad, row_major[Self.NEW_N]())
            for bt in range(BATCH):
                for k in range(Self.NEW_N):
                    gtok[k] += go[bt, Self.NEW_OFF + k]

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["LearnedTokens", target](self.ts.target_tag)
        for_each_param_auto[Self, V, target](self, prefix, visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["LearnedTokens", target](self.ts.target_tag)
        zero_grad_auto[Self, target](self)
