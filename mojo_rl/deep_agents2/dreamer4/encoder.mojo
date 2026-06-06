"""Dreamer4Encoder — tokenizer encoder (model.py:Encoder).

Input is the per-frame patch tokens (NP × DP); output is the bottleneck z
(L latents × D_BOT). Pipeline at nn2-BATCH = B·T:

    patch_proj → MAE (replace dropped patches w/ learned mask_token)
    → prepend learned latents → +positions → encoder transformer
    → slice latents → bottleneck Linear → tanh

This is a bespoke Module (not a pure Sequential) for one reason: MAE emits the
per-patch dropped mask that the reconstruction loss needs, which a
single-output Sequential can't surface. The encoder holds three Module
children — `proj`, `mae`, `body` — and delegates param/grad visiting to all
three; `mae_mask_ptr()` / `advance_rng()` forward to the MAE leaf.
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core import Initializer, AMPPolicy, NoAMP, ParamVisitor
from mojo_rl.nn2.core.module import Module, typed_view, typed_view_mut
from mojo_rl.nn2.core.target_storage import (
    TargetStorage, assert_tag_for, ensure_cpu_buffer,
)
from mojo_rl.nn2.combinators import Sequential, Tokenwise
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.tanh import Tanh
from mojo_rl.nn2.primitives.slice import Slice
from mojo_rl.nn2.primitives.learned_tokens import LearnedTokens
from mojo_rl.nn2.primitives.sinusoidal_pos_bt import SinusoidalPosAddBT
from mojo_rl.nn2.primitives.mae_replacer import MAEReplacer
from .blocks import Dreamer4Stack


def _mao_tile[
    BATCH: Int, N: Int
](mut buf: List[Scalar[DT]]) -> TileTensor[
    DT, type_of(row_major[BATCH, N]()), MutAnyOrigin
]:
    return TileTensor(
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](buf.unsafe_ptr()),
        row_major[BATCH, N](),
    )


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
        LearnedTokens[Self.L, Self.NP, Self.D, True],
        SinusoidalPosAddBT[Self.T, Self.S, Self.D],
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
    var proj_out: List[Scalar[DT]]      # [BATCH*NP*D]
    var masked: List[Scalar[DT]]
    var grad_masked: List[Scalar[DT]]
    var grad_proj: List[Scalar[DT]]
    var ts: TargetStorage

    def __init__(out self):
        self.proj = Self.PROJ()
        self.mae = Self.MAE()
        self.body = Self.BODY()
        self.proj_out = List[Scalar[DT]]()
        self.masked = List[Scalar[DT]]()
        self.grad_masked = List[Scalar[DT]]()
        self.grad_proj = List[Scalar[DT]]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu", "Dreamer4Encoder: PHASE 1 is CPU-only"
        var m = Self()
        m.proj = Self.PROJ.make[target=target, INIT=INIT](ctx)
        m.mae = Self.MAE.make[target=target, INIT=INIT](ctx)
        m.body = Self.BODY.make[target=target, INIT=INIT](ctx)
        m.ts = TargetStorage.make_cpu()
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
        assert_tag_for["Dreamer4Encoder", target](self.ts.target_tag)
        var inp = typed_view[BATCH, Self.IN_DIMS[0]](inputs[0])
        var out = typed_view_mut[BATCH, Self.OUT_DIM](output)
        ensure_cpu_buffer(self.proj_out, BATCH * Self.ND)
        ensure_cpu_buffer(self.masked, BATCH * Self.ND)
        var po = _mao_tile[BATCH, Self.ND](self.proj_out)
        var mk = _mao_tile[BATCH, Self.ND](self.masked)
        self.proj.forward[target, BATCH, POLICY=POLICY](inp, output=po)
        self.mae.forward[target, BATCH, POLICY=POLICY](po, output=mk)
        self.body.forward[target, BATCH, POLICY=POLICY](mk, output=out)

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
        assert_tag_for["Dreamer4Encoder", target](self.ts.target_tag)
        var go = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var gin = typed_view_mut[BATCH, Self.IN_DIMS[0]](grad_inputs[0])
        ensure_cpu_buffer(self.grad_masked, BATCH * Self.ND)
        ensure_cpu_buffer(self.grad_proj, BATCH * Self.ND)
        var gmk = _mao_tile[BATCH, Self.ND](self.grad_masked)
        var gpo = _mao_tile[BATCH, Self.ND](self.grad_proj)
        self.body.vjp[target, BATCH, POLICY=POLICY, mode=mode](go, gmk)
        self.mae.vjp[target, BATCH, POLICY=POLICY, mode=mode](gmk, gpo)
        self.proj.vjp[target, BATCH, POLICY=POLICY, mode=mode](gpo, gin)

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["Dreamer4Encoder", target](self.ts.target_tag)
        self.proj.for_each_param[target, V](prefix + ".proj", visitor)
        self.mae.for_each_param[target, V](prefix + ".mae", visitor)
        self.body.for_each_param[target, V](prefix + ".body", visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["Dreamer4Encoder", target](self.ts.target_tag)
        self.proj.zero_grad[target]()
        self.mae.zero_grad[target]()
        self.body.zero_grad[target]()
