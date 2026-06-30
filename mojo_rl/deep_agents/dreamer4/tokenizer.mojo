"""Dreamer4Tokenizer — encoder + decoder as one module (model.py:Tokenizer).

forward(patches) → reconstructed patches; encoder masks internally and the
masked-reconstruction loss compares the output to the *original* patches on
the dropped positions (mask from `mae_mask_ptr()`). Wrapping both halves in
one Module lets a single optimizer cover all params via `for_each_param`.

    patches (NP·DP) → encoder → z (L·D_BOT) → decoder → pred (NP·DP)

Storage migration: the intermediate latent `z` is a storage `Tensor` (one per
the old cpu-List + device-buffer pair). It is populated during `forward` and
re-read as `dec`/`enc`'s `forward_input` in `vjp` (storage children recompute
from their forward_input), so it MUST persist as a field across forward→vjp.
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.param import Param, ParamVisitor
from mojo_rl.nn.core.initializer import Initializer
from mojo_rl.nn.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn.core.walkers import (
    for_each_param_auto, zero_grad_auto, join_name,
)
from .blocks import Dreamer4Decoder
from .encoder import Dreamer4Encoder


struct Dreamer4Tokenizer[
    DP: Int, D: Int, NH: Int, T: Int, L: Int, NP: Int, D_BOT: Int,
    HID: Int, DEPTH: Int, P_MIN: Float64, P_MAX: Float64, SEED: UInt64,
    USE_MAX: Bool = True,
](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.NP * Self.DP)
    comptime OUT_DIM = Self.NP * Self.DP
    comptime ZN: Int = Self.L * Self.D_BOT

    comptime ENC = Dreamer4Encoder[
        Self.DP, Self.D, Self.NH, Self.T, Self.L, Self.NP, Self.D_BOT,
        Self.HID, Self.DEPTH, Self.P_MIN, Self.P_MAX, Self.SEED, Self.USE_MAX,
    ]
    comptime DEC = Dreamer4Decoder[
        Self.D_BOT, Self.D, Self.NH, Self.T, Self.L, Self.NP, Self.DP,
        Self.HID, Self.DEPTH, Self.USE_MAX,
    ]

    var enc: Self.ENC
    var dec: Self.DEC
    var z: Tensor        # latent scratch [BATCH*ZN]; populated in forward, reused in vjp
    var grad_z: Tensor   # latent grad scratch [BATCH*ZN]

    def __init__(out self):
        self.enc = Self.ENC()
        self.dec = Self.DEC()
        self.z = Tensor()
        self.grad_z = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "Dreamer4Tokenizer: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        m.enc = Self.ENC.make[target=target, INIT=INIT](ctx)
        m.dec = Self.DEC.make[target=target, INIT=INIT](ctx)
        return m^

    @staticmethod
    def display_label() -> String:
        return String("Dreamer4Tokenizer")

    def advance_rng(mut self):
        self.enc.advance_rng()

    def set_mae_p(mut self, p_min: Float64, p_max: Float64):
        self.enc.set_mae_p(p_min, p_max)

    def mae_mask_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self.enc.mae_mask_ptr()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime if target == "cpu":
            self.z.ensure(B * Self.ZN)
        else:
            self.z.ensure_gpu(ctx.value(), B * Self.ZN)
        # enc consumes the module input pack directly → z; dec: z → out.
        self.enc.forward[target, B, POLICY=POLICY](inputs, self.z, ctx)
        self.dec.forward[target, B, POLICY=POLICY](
            TensorRefs[Self.ARITY](self.z), out, ctx
        )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime if target == "cpu":
            self.grad_z.ensure(B * Self.ZN)
        else:
            self.grad_z.ensure_gpu(ctx.value(), B * Self.ZN)
        # dec: forward_input = z (from forward); grad_out = grad_output; grad_in = grad_z.
        self.dec.vjp[target, B, POLICY=POLICY](
            TensorRefs[Self.ARITY](self.z),
            grad_output,
            TensorRefs[Self.ARITY](self.grad_z),
            ctx,
        )
        # enc: forward_input = module input; grad_out = grad_z; grad_in = grad_inputs.
        self.enc.vjp[target, B, POLICY=POLICY](
            forward_input, self.grad_z, grad_inputs, ctx
        )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        self.enc.for_each_param[target, V](
            visitor, ctx, join_name(prefix, "enc")
        )
        self.dec.for_each_param[target, V](
            visitor, ctx, join_name(prefix, "dec")
        )

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        self.enc.zero_grad[target](ctx)
        self.dec.zero_grad[target](ctx)
