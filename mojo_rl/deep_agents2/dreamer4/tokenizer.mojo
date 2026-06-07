"""Dreamer4Tokenizer — encoder + decoder as one module (model.py:Tokenizer).

forward(patches) → reconstructed patches; encoder masks internally and the
masked-reconstruction loss compares the output to the *original* patches on
the dropped positions (mask from `mae_mask_ptr()`). Wrapping both halves in
one Module lets a single optimizer cover all params via `for_each_param`.

    patches (NP·DP) → encoder → z (L·D_BOT) → decoder → pred (NP·DP)
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core import Initializer, AMPPolicy, NoAMP, ParamVisitor
from mojo_rl.nn2.core.module import Module, typed_view, typed_view_mut
from mojo_rl.nn2.core.tensor_pack import TensorPack
from mojo_rl.nn2.core.target_storage import (
    TargetStorage, assert_tag_for, ensure_cpu_buffer,
)
from .blocks import Dreamer4Decoder
from .encoder import Dreamer4Encoder


def _mao_tile[
    BATCH: Int, N: Int
](mut buf: List[Scalar[DT]]) -> TileTensor[
    DT, type_of(row_major[BATCH, N]()), MutAnyOrigin
]:
    return TileTensor(
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](buf.unsafe_ptr()),
        row_major[BATCH, N](),
    )


def _dev_tile[
    BATCH: Int, N: Int
](buf: DeviceBuffer[DT]) -> TileTensor[
    DT, type_of(row_major[BATCH, N]()), MutAnyOrigin
]:
    return TileTensor(
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](buf.unsafe_ptr()),
        row_major[BATCH, N](),
    )


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
    var z: List[Scalar[DT]]
    var grad_z: List[Scalar[DT]]
    var z_dev: Optional[DeviceBuffer[DT]]
    var gz_dev: Optional[DeviceBuffer[DT]]
    var scratch_n: Int
    var ts: TargetStorage

    def __init__(out self):
        self.enc = Self.ENC()
        self.dec = Self.DEC()
        self.z = List[Scalar[DT]]()
        self.grad_z = List[Scalar[DT]]()
        self.z_dev = None
        self.gz_dev = None
        self.scratch_n = 0
        self.ts = TargetStorage.make_uninit()

    def _ensure_scratch_gpu(mut self, n: Int) raises:
        if self.scratch_n < n:
            var ctx = self.ts.ctx.value()
            self.z_dev = ctx.enqueue_create_buffer[DT](n)
            self.gz_dev = ctx.enqueue_create_buffer[DT](n)
            self.scratch_n = n

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
        comptime if target == "cpu":
            m.ts = TargetStorage.make_cpu()
        else:
            m.ts = TargetStorage.make_gpu(ctx.value())
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
        assert_tag_for["Dreamer4Tokenizer", target](self.ts.target_tag)
        var inp = inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        var out = typed_view_mut[BATCH, Self.OUT_DIM](output)
        comptime LAY = type_of(row_major[BATCH, Self.ZN]())
        var zt: TileTensor[DT, LAY, MutAnyOrigin]
        comptime if target == "cpu":
            ensure_cpu_buffer(self.z, BATCH * Self.ZN)
            zt = _mao_tile[BATCH, Self.ZN](self.z)
        else:
            self._ensure_scratch_gpu(BATCH * Self.ZN)
            zt = _dev_tile[BATCH, Self.ZN](self.z_dev.value())
        self.enc.forward[target, BATCH, POLICY=POLICY](inp, output=zt)
        self.dec.forward[target, BATCH, POLICY=POLICY](zt, output=out)

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
        assert_tag_for["Dreamer4Tokenizer", target](self.ts.target_tag)
        var go = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var gin = grad_inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        comptime LAY = type_of(row_major[BATCH, Self.ZN]())
        var gzt: TileTensor[DT, LAY, MutAnyOrigin]
        comptime if target == "cpu":
            ensure_cpu_buffer(self.grad_z, BATCH * Self.ZN)
            gzt = _mao_tile[BATCH, Self.ZN](self.grad_z)
        else:
            self._ensure_scratch_gpu(BATCH * Self.ZN)
            gzt = _dev_tile[BATCH, Self.ZN](self.gz_dev.value())
        self.dec.vjp[target, BATCH, POLICY=POLICY, mode=mode](go, gzt)
        self.enc.vjp[target, BATCH, POLICY=POLICY, mode=mode](gzt, gin)

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["Dreamer4Tokenizer", target](self.ts.target_tag)
        self.enc.for_each_param[target, V](prefix + ".enc", visitor)
        self.dec.for_each_param[target, V](prefix + ".dec", visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["Dreamer4Tokenizer", target](self.ts.target_tag)
        self.enc.zero_grad[target]()
        self.dec.zero_grad[target]()
