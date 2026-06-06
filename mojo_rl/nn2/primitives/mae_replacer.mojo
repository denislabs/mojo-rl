"""MAEReplacer[NP, D, P_MIN, P_MAX, SEED] — masked-autoencoding patch dropout.

Dreamer 4 trains the tokenizer with masked autoencoding: a random fraction of
the projected patch tokens is replaced by a learned `mask_token`, and the
reconstruction loss is applied only on the replaced patches
(`model.py:MAEReplacer` + `recon_loss_from_mae`). This leaf does the
replacement and remembers which patches were dropped (for the loss + backward).

Operates per frame at nn2-BATCH = B·T: IN_DIM == OUT_DIM == NP·D.

    p_bt   ~ U(P_MIN, P_MAX)                 (one drop-rate per frame)
    keep   = U(0,1) < (1 - p_bt)             (per patch)
    out    = where(keep, input, mask_token)

`mask_token` (D) is the only parameter; its grad accumulates the grad-output
of every dropped patch. Kept patches pass the gradient straight through;
dropped patches get grad_input 0.

RNG: PhiloxRandom seeded by the comptime SEED at offset `base + idx`, with
`base = rng_step * STRIDE`. The step counter is bumped explicitly by
`advance_rng()` (once per training iteration), NOT per forward — so a
gradcheck that never advances sees a frozen mask across its FD forwards.

`mae_mask()` returns the per-patch dropped flags (1.0 = masked / reconstruct).
PHASE 1: CPU forward + vjp. GPU follows.
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.random.philox import Random as PhiloxRandom
from layout import TileTensor, row_major

from ..constants import DT
from ..core import (
    Initializer, AMPPolicy, NoAMP, Param, ParamVisitor,
    for_each_param_auto, zero_grad_auto,
)
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import TargetStorage, assert_tag_for, ensure_cpu_buffer


struct MAEReplacer[
    NP: Int, D: Int, P_MIN: Float64, P_MAX: Float64, SEED: UInt64
](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.NP * Self.D)
    comptime OUT_DIM = Self.NP * Self.D

    var mask_token: Param["mask_token", False, Self.D]
    var keep: List[Scalar[DT]]      # [BATCH*NP] 1.0 kept / 0.0 dropped
    var rng_step: UInt64
    var ts: TargetStorage

    def __init__(out self):
        self.mask_token = Param["mask_token", False, Self.D]()
        self.keep = List[Scalar[DT]]()
        self.rng_step = 0
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu", "MAEReplacer: PHASE 1 is CPU-only"
        var m = Self()
        m.mask_token = Param["mask_token", False, Self.D].make_cpu()
        INIT.init_bias(m.mask_token.value_unsafe_ptr_cpu(), Self.D)
        m.ts = TargetStorage.make_cpu()
        return m^

    @staticmethod
    def display_label() -> String:
        return String("MAEReplacer")

    def advance_rng(mut self):
        """Bump the RNG step (call once per training iteration)."""
        self.rng_step += 1

    def mae_mask_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        """`keep` flags buffer ([BATCH*NP], 1.0=kept). Masked = 1 - keep."""
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.keep.unsafe_ptr()
        )

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
        assert_tag_for["MAEReplacer", target](self.ts.target_tag)
        var inp = typed_view[BATCH, Self.IN_DIMS[0]](inputs[0])
        var out = typed_view_mut[BATCH, Self.OUT_DIM](output)
        ensure_cpu_buffer(self.keep, BATCH * Self.NP)
        var kp = self.keep.unsafe_ptr()
        var mt = self.mask_token.value_unsafe_ptr_cpu()
        comptime STRIDE = UInt64(BATCH * (1 + Self.NP))
        var base = self.rng_step * STRIDE
        var span = Float64(Self.P_MAX - Self.P_MIN)

        for bt in range(BATCH):
            var rp = PhiloxRandom(seed=Self.SEED, offset=base + UInt64(bt))
            var p_bt = Self.P_MIN + span * Float64(rp.step_uniform()[0])
            var keep_prob = 1.0 - p_bt
            for i in range(Self.NP):
                var ri = PhiloxRandom(
                    seed=Self.SEED,
                    offset=base + UInt64(BATCH) + UInt64(bt * Self.NP + i),
                )
                var u = Float64(ri.step_uniform()[0])
                var keep = u < keep_prob
                kp[bt * Self.NP + i] = Scalar[DT](1.0) if keep else Scalar[DT](0.0)
                for d in range(Self.D):
                    if keep:
                        out[bt, i * Self.D + d] = inp[bt, i * Self.D + d]
                    else:
                        out[bt, i * Self.D + d] = mt[d]

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
        assert_tag_for["MAEReplacer", target](self.ts.target_tag)
        var go = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var gi = typed_view_mut[BATCH, Self.IN_DIMS[0]](grad_inputs[0])
        var kp = self.keep.unsafe_ptr()
        var gmt = self.mask_token.grad.unsafe_ptr()
        for bt in range(BATCH):
            for i in range(Self.NP):
                var kept = kp[bt * Self.NP + i] != Scalar[DT](0.0)
                for d in range(Self.D):
                    var g = go[bt, i * Self.D + d]
                    if kept:
                        gi[bt, i * Self.D + d] = g
                    else:
                        gi[bt, i * Self.D + d] = Scalar[DT](0.0)
                        comptime if mode == "all":
                            gmt[d] += g

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["MAEReplacer", target](self.ts.target_tag)
        for_each_param_auto[Self, V, target](self, prefix, visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["MAEReplacer", target](self.ts.target_tag)
        zero_grad_auto[Self, target](self)
