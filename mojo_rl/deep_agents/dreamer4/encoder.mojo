"""Dreamer4Encoder — tokenizer encoder (model.py:Encoder).

Input is the per-frame patch tokens (NP × DP); output is the bottleneck z
(L latents × D_BOT). Pipeline at nn-BATCH = B·T:

    patch_proj → MAE (replace dropped patches w/ learned mask_token)
    → prepend learned latents → +positions → encoder transformer
    → slice latents → bottleneck Linear → tanh

This is a bespoke Module (not a pure Sequential) for one reason: MAE emits the
per-patch dropped mask that the reconstruction loss needs, which a
single-output Sequential can't surface. The encoder holds three Module
children — `proj`, `mae`, `body` — and delegates param/grad visiting to all
three; `mae_mask_ptr()` / `advance_rng()` forward to the MAE leaf.

Storage migration: intermediate scratch is storage `Tensor`s (one per cpu-List
+ device-buffer pair). `proj_out` / `masked` are populated during `forward` and
re-read as the children's `forward_input` in `vjp` (storage children recompute
from their forward_input), so they MUST persist as fields across forward→vjp.
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
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.combinators.tokenwise import Tokenwise
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import Tanh
from mojo_rl.nn.primitives.slice import Slice
from mojo_rl.nn.primitives.learned_tokens import LearnedTokens
from mojo_rl.nn.primitives.sinusoidal_pos_bt import SinusoidalPosAddBT
from mojo_rl.nn.primitives.mae_replacer import MAEReplacer
from .blocks import Dreamer4Stack


struct Dreamer4Encoder[
    DP: Int, D: Int, NH: Int, T: Int, L: Int, NP: Int, D_BOT: Int,
    HID: Int, DEPTH: Int, P_MIN: Float64, P_MAX: Float64, SEED: UInt64,
    USE_MAX: Bool = True,
](Module):
    comptime ARITY: Int = 1
    comptime S: Int = Self.L + Self.NP
    comptime ND: Int = Self.NP * Self.D                  # masked-token width
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.NP * Self.DP)
    comptime OUT_DIM = Self.L * Self.D_BOT

    comptime PROJ = Tokenwise[Self.NP, Linear[Self.DP, Self.D]]
    comptime MAE = MAEReplacer[Self.NP, Self.D, Self.P_MIN, Self.P_MAX, Self.SEED]
    comptime BODY = Sequential[
        # Input = NP masked patch tokens; PREPEND L latent register tokens so the
        # sequence is [ L latents | NP patches ] (latents at positions [0, L), as
        # the encoder modality mask and the bottleneck Slice below both assume).
        # INIT_STD=0.02 matches the reference (`normal_(std=0.02)`); the default
        # fan-in=1 Kaiming init gives std~1.4 which collapses the readout.
        LearnedTokens[Self.NP, Self.L, Self.D, True, 0.02],
        # SCALE=True (÷√D): both reference reimpls scale the positions by 1/√D.
        # Without it (D=128) the unit-RMS positions swamp the std-0.02 learned
        # latents at init, so the encoder can't route image CONTENT through the
        # latent bottleneck → the tokenizer fails to reconstruct (RECON = noise).
        SinusoidalPosAddBT[Self.T, Self.S, Self.D, True],
        Dreamer4Stack[
            Self.D, Self.NH, Self.T, Self.S, Self.L, Self.HID, Self.DEPTH,
            "encoder", Self.USE_MAX,
        ],
        Slice[Self.S * Self.D, 0, Self.L * Self.D],
        Tokenwise[Self.L, Linear[Self.D, Self.D_BOT]],
        Tanh[Self.L * Self.D_BOT],
    ]

    var proj: Self.PROJ
    var mae: Self.MAE
    var body: Self.BODY
    var proj_out: Tensor      # scratch [BATCH*ND]; populated in forward, reused in vjp
    var masked: Tensor        # scratch [BATCH*ND]
    var grad_masked: Tensor   # scratch [BATCH*ND]
    var grad_proj: Tensor     # scratch [BATCH*ND]

    def __init__(out self):
        self.proj = Self.PROJ()
        self.mae = Self.MAE()
        self.body = Self.BODY()
        self.proj_out = Tensor()
        self.masked = Tensor()
        self.grad_masked = Tensor()
        self.grad_proj = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "Dreamer4Encoder: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        m.proj = Self.PROJ.make[target=target, INIT=INIT](ctx)
        m.mae = Self.MAE.make[target=target, INIT=INIT](ctx)
        m.body = Self.BODY.make[target=target, INIT=INIT](ctx)
        return m^

    @staticmethod
    def display_label() -> String:
        return String("Dreamer4Encoder")

    def advance_rng(mut self):
        self.mae.advance_rng()

    def set_mae_p(mut self, p_min: Float64, p_max: Float64):
        self.mae.set_p(p_min, p_max)

    def mae_mask_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        """Per-patch `keep` flags ([BATCH*NP], 1.0=kept); masked = 1 - keep."""
        return self.mae.mae_mask_ptr()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime if target == "cpu":
            self.proj_out.ensure(B * Self.ND)
            self.masked.ensure(B * Self.ND)
        else:
            var c = ctx.value()
            self.proj_out.ensure_gpu(c, B * Self.ND)
            self.masked.ensure_gpu(c, B * Self.ND)
        # proj consumes the module input pack directly.
        self.proj.forward[target, B, POLICY=POLICY](inputs, self.proj_out, ctx)
        self.mae.forward[target, B, POLICY=POLICY](
            TensorRefs[Self.ARITY](self.proj_out), self.masked, ctx
        )
        self.body.forward[target, B, POLICY=POLICY](
            TensorRefs[Self.ARITY](self.masked), out, ctx
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
            self.grad_masked.ensure(B * Self.ND)
            self.grad_proj.ensure(B * Self.ND)
        else:
            var c = ctx.value()
            self.grad_masked.ensure_gpu(c, B * Self.ND)
            self.grad_proj.ensure_gpu(c, B * Self.ND)
        # body: forward_input = masked (from forward); grad_in = grad_masked.
        self.body.vjp[target, B, POLICY=POLICY](
            TensorRefs[Self.ARITY](self.masked),
            grad_output,
            TensorRefs[Self.ARITY](self.grad_masked),
            ctx,
        )
        # mae: forward_input = proj_out; grad_out = grad_masked; grad_in = grad_proj.
        self.mae.vjp[target, B, POLICY=POLICY](
            TensorRefs[Self.ARITY](self.proj_out),
            self.grad_masked,
            TensorRefs[Self.ARITY](self.grad_proj),
            ctx,
        )
        # proj: forward_input = module input; grad_out = grad_proj; grad_in = grad_inputs.
        self.proj.vjp[target, B, POLICY=POLICY](
            forward_input, self.grad_proj, grad_inputs, ctx
        )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        self.proj.for_each_param[target, V](
            visitor, ctx, join_name(prefix, "proj")
        )
        self.mae.for_each_param[target, V](
            visitor, ctx, join_name(prefix, "mae")
        )
        self.body.for_each_param[target, V](
            visitor, ctx, join_name(prefix, "body")
        )

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        self.proj.zero_grad[target](ctx)
        self.mae.zero_grad[target](ctx)
        self.body.zero_grad[target](ctx)
