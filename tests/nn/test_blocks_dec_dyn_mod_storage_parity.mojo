"""DecoderBlock / DynamicsSpaceAttention / ModalitySpaceAttention storage parity.

For each block:
  * CPU parity vs the LEGACY block — identical params + inputs → max|Δ| < 1e-6 on
    out + grad_inputs (+ param grads for the param-bearing DecoderBlock).
  * storage GPU vs storage CPU — out is exact; grads drift ~1e-4 from fp32
    accumulation-order differences (Linear/FFN/attention matmuls), so the grad
    tols are looser for these composites (decoder 2e-3, attention 1e-3).

Param sync (DecoderBlock): both frameworks walk params in identical topo order
(same node graph), so a deterministic FILL visitor that writes value(global_idx)
into each param in walk order produces bit-identical weights on both sides. A
COLLECT visitor reads grads back in the same order for comparison.

The two attention blocks are param-free → no sync needed.

  pixi run            mojo run -I . tests/nn/test_blocks_dec_dyn_mod_storage_parity.mojo
  pixi run -e apple   mojo run -I . tests/nn/test_blocks_dec_dyn_mod_storage_parity.mojo
"""

from std.memory import alloc
from std.gpu.memory import AddressSpace
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT

# ── Legacy blocks ────────────────────────────────────────────────────────
from mojo_rl.nn.initializer import Zero as LegacyZero
from mojo_rl.nn.core.tensor_pack import TensorPack as LegacyTensorPack
from mojo_rl.nn.core.param_visitor import ParamVisitor as LegacyParamVisitor
from mojo_rl.nn.primitives.decoder_block import DecoderBlock as LegacyDecoderBlock
from mojo_rl.nn.primitives.dynamics_space_attention import (
    DynamicsSpaceAttention as LegacyDynamicsSpaceAttention,
)
from mojo_rl.nn.primitives.modality_space_attention import (
    ModalitySpaceAttention as LegacyModalitySpaceAttention,
)

# ── Storage blocks ───────────────────────────────────────────────────────
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.tensor_pack import TensorPack
from mojo_rl.nn.storage.core.param import ParamVisitor as SParamVisitor
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.decoder_block import DecoderBlock
from mojo_rl.nn.storage.primitives.dynamics_space_attention import (
    DynamicsSpaceAttention,
)
from mojo_rl.nn.storage.primitives.modality_space_attention import (
    ModalitySpaceAttention,
)


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


def _spread(i: Int, seed: Float64) -> Scalar[DT]:
    var x = seed + 0.7 * Float64(i)
    var t = x - 6.2831853 * Float64(Int(x / 6.2831853))
    return Scalar[DT](0.5 * (t - (t * t * t) / 6.0))


# ── Param fill / grad collect visitors (per framework) ───────────────────
struct _SFill(SParamVisitor):
    var counter: Int

    def __init__(out self):
        self.counter = 0

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        for i in range(N):
            param.data[i] = _spread(self.counter, 3.1)
            self.counter += 1


struct _SGpuFill(SParamVisitor):
    var counter: Int
    var ctx: DeviceContext

    def __init__(out self, ctx: DeviceContext):
        self.counter = 0
        self.ctx = ctx

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        param.ensure(N)
        for i in range(N):
            param.data[i] = _spread(self.counter, 3.1)
            self.counter += 1
        param.n = N
        param.upload(self.ctx)


struct _SGradCollect(SParamVisitor):
    var items: List[Scalar[DT]]

    def __init__(out self):
        self.items = List[Scalar[DT]]()

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        for i in range(N):
            self.items.append(grad.data[i])


struct _LFill(LegacyParamVisitor):
    var counter: Int

    def __init__(out self):
        self.counter = 0

    def visit(
        mut self,
        name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        var p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](param.ptr)
        for i in range(n_elems):
            p[i] = _spread(self.counter, 3.1)
            self.counter += 1


struct _LGradCollect(LegacyParamVisitor):
    var items: List[Scalar[DT]]

    def __init__(out self):
        self.items = List[Scalar[DT]]()

    def visit(
        mut self,
        name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        for i in range(n_elems):
            self.items.append(grad.ptr[i])


# ── DecoderBlock parity ──────────────────────────────────────────────────
def test_decoder_cpu_parity() raises:
    print("DecoderBlock CPU parity (legacy vs storage) ...")
    comptime N = 3
    comptime HID = 4
    comptime FF = 8
    comptime SEQ = N * HID
    comptime B = 2
    comptime DN = B * SEQ
    comptime TOL = Scalar[DT](1e-6)

    var leg = LegacyDecoderBlock[N, HID, FF].make[target="cpu", INIT=LegacyZero]()
    var lf = _LFill()
    leg.for_each_param["cpu", _LFill]("", lf)

    var st = DecoderBlock[N, HID, FF].make["cpu", Deterministic]()
    var sf = _SFill()
    st.for_each_param["cpu", _SFill](sf, None)

    # Inputs.
    var lx = _a(DN); var lc = _a(DN); var ly = _a(DN)
    var sin = TensorPack[2]()
    sin[0].ensure(DN); sin[1].ensure(DN)
    var sout = Tensor.alloc(DN)
    for i in range(DN):
        var xv = _spread(i, 0.5)
        var cv = _spread(i, 1.9)
        lx[i] = xv; lc[i] = cv
        sin[0].data[i] = xv; sin[1].data[i] = cv
    var lxt = TileTensor(lx, row_major[B, SEQ]())
    var lct = TileTensor(lc, row_major[B, SEQ]())
    var lyt = TileTensor(ly, row_major[B, SEQ]())
    leg.forward["cpu", B](LegacyTensorPack[2].of(lxt, lct), output=lyt)
    st.forward["cpu", B](TensorRefs[2](sin[0], sin[1]), sout, None)

    var lgo = _a(DN); var lgx = _a(DN); var lgc = _a(DN)
    var sgo = Tensor.alloc(DN)
    for i in range(DN):
        var gv = _spread(i, 2.3)
        lgo[i] = gv; sgo.data[i] = gv
    var lgot = TileTensor(lgo, row_major[B, SEQ]())
    var lgxt = TileTensor(lgx, row_major[B, SEQ]())
    var lgct = TileTensor(lgc, row_major[B, SEQ]())
    leg.zero_grad["cpu"]()
    leg.vjp["cpu", B](lgot, LegacyTensorPack[2].of(lgxt, lgct))

    var sg = TensorPack[2]()
    st.zero_grad["cpu"](None)
    st.vjp["cpu", B](
        TensorRefs[2](sin[0], sin[1]), sgo, TensorRefs[2](sg[0], sg[1]), None
    )

    var mo: Scalar[DT] = 0
    var mgx: Scalar[DT] = 0
    var mgc: Scalar[DT] = 0
    for i in range(DN):
        if abs(sout.data[i] - ly[i]) > mo: mo = abs(sout.data[i] - ly[i])
        if abs(sg[0].data[i] - lgx[i]) > mgx: mgx = abs(sg[0].data[i] - lgx[i])
        if abs(sg[1].data[i] - lgc[i]) > mgc: mgc = abs(sg[1].data[i] - lgc[i])

    # Param grads in walk order.
    var lgcv = _LGradCollect()
    leg.for_each_param["cpu", _LGradCollect]("", lgcv)
    var sgcv = _SGradCollect()
    st.for_each_param["cpu", _SGradCollect](sgcv, None)
    var mpg: Scalar[DT] = 0
    assert_true(len(lgcv.items) == len(sgcv.items), "param-grad count mismatch")
    for i in range(len(lgcv.items)):
        if abs(sgcv.items[i] - lgcv.items[i]) > mpg:
            mpg = abs(sgcv.items[i] - lgcv.items[i])

    print("  max Δ: out", mo, " gx", mgx, " gc", mgc, " pgrad", mpg,
          " (nparams", len(sgcv.items), ")")
    assert_true(
        mo < TOL and mgx < TOL and mgc < TOL and mpg < TOL,
        "DecoderBlock CPU parity",
    )
    print("  ok")


def test_decoder_gpu_parity() raises:
    print("DecoderBlock GPU vs CPU (storage) ...")
    comptime N = 3
    comptime HID = 4
    comptime FF = 8
    comptime SEQ = N * HID
    comptime B = 2
    comptime DN = B * SEQ
    # Composite (Linear+LayerNorm+FFN+Add through a graph): GPU/CPU fp32
    # accumulation order differs, so the grad path needs a looser tol than a
    # single leaf (out is exact; grads drift ~1e-4 in the deepest branch).
    comptime TOL = Scalar[DT](2e-3)
    var c = DeviceContext()

    var cpu = DecoderBlock[N, HID, FF].make["cpu", Deterministic]()
    var f1 = _SFill()
    cpu.for_each_param["cpu", _SFill](f1, None)
    var gpu = DecoderBlock[N, HID, FF].make["gpu", Deterministic](Optional(c))
    var gf = _SGpuFill(c)
    gpu.for_each_param["gpu", _SGpuFill](gf, Optional(c))

    var cin = TensorPack[2]()
    cin[0].ensure(DN); cin[1].ensure(DN)
    var sgo = Tensor.alloc(DN)
    for i in range(DN):
        cin[0].data[i] = _spread(i, 0.5)
        cin[1].data[i] = _spread(i, 1.9)
        sgo.data[i] = _spread(i, 2.3)
    var cout = Tensor.alloc(DN)
    var cg = TensorPack[2]()
    cpu.forward["cpu", B](TensorRefs[2](cin[0], cin[1]), cout, None)
    cpu.zero_grad["cpu"](None)
    cpu.vjp["cpu", B](
        TensorRefs[2](cin[0], cin[1]), sgo, TensorRefs[2](cg[0], cg[1]), None
    )

    var gin = TensorPack[2]()
    gin[0].ensure(DN); gin[1].ensure(DN)
    var ggo = Tensor.alloc(DN)
    for i in range(DN):
        gin[0].data[i] = cin[0].data[i]
        gin[1].data[i] = cin[1].data[i]
        ggo.data[i] = sgo.data[i]
    gin[0].upload(c); gin[1].upload(c); ggo.upload(c)
    var gout = Tensor.alloc(DN)
    var gg = TensorPack[2]()
    gpu.forward["gpu", B](TensorRefs[2](gin[0], gin[1]), gout, Optional(c))
    gpu.zero_grad["gpu"](Optional(c))
    gpu.vjp["gpu", B](
        TensorRefs[2](gin[0], gin[1]), ggo, TensorRefs[2](gg[0], gg[1]),
        Optional(c)
    )
    gout.download(c); gg[0].download(c); gg[1].download(c)

    var mo: Scalar[DT] = 0
    var mgx: Scalar[DT] = 0
    var mgc: Scalar[DT] = 0
    for i in range(DN):
        if abs(gout.data[i] - cout.data[i]) > mo: mo = abs(gout.data[i] - cout.data[i])
        if abs(gg[0].data[i] - cg[0].data[i]) > mgx: mgx = abs(gg[0].data[i] - cg[0].data[i])
        if abs(gg[1].data[i] - cg[1].data[i]) > mgc: mgc = abs(gg[1].data[i] - cg[1].data[i])
    print("  max Δ: out", mo, " gx", mgx, " gc", mgc)
    assert_true(mo < TOL and mgx < TOL and mgc < TOL, "DecoderBlock GPU vs CPU")
    print("  ok")


# ── Modality attention parity (param-free) ───────────────────────────────
def test_modality_cpu[D: Int, NH: Int, S: Int, L: Int, MODE: StaticString](
    name: String
) raises:
    print("ModalitySpaceAttention CPU parity", name, "...")
    comptime B = 2
    comptime IN_N = B * S * D * 3
    comptime OUT_N = B * S * D
    comptime TOL = Scalar[DT](1e-6)

    var leg = LegacyModalitySpaceAttention[D, NH, S, L, MODE].make[
        target="cpu", INIT=LegacyZero
    ]()
    var st = ModalitySpaceAttention[D, NH, S, L, MODE].make["cpu", Deterministic]()

    var lx = _a(IN_N); var ly = _a(OUT_N); var lgi = _a(IN_N)
    var sx = Tensor.alloc(IN_N); var sout = Tensor.alloc(OUT_N)
    for i in range(IN_N):
        var xv = _spread(i, 1.7)
        lx[i] = xv; sx.data[i] = xv
    var lxt = TileTensor(lx, row_major[B, S * D * 3]())
    var lyt = TileTensor(ly, row_major[B, S * D]())
    leg.forward["cpu", B](LegacyTensorPack[1].of(lxt), output=lyt)
    st.forward["cpu", B](TensorRefs[1](sx), sout, None)

    var lgo = _a(OUT_N)
    var sgo = Tensor.alloc(OUT_N)
    for i in range(OUT_N):
        var gv = _spread(i, 0.9)
        lgo[i] = gv; sgo.data[i] = gv
    var lgot = TileTensor(lgo, row_major[B, S * D]())
    var lgit = TileTensor(lgi, row_major[B, S * D * 3]())
    var sgi = Tensor.alloc(IN_N)
    leg.vjp["cpu", B](lgot, LegacyTensorPack[1].of(lgit))
    st.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var mo: Scalar[DT] = 0
    var mg: Scalar[DT] = 0
    for i in range(OUT_N):
        if abs(sout.data[i] - ly[i]) > mo: mo = abs(sout.data[i] - ly[i])
    for i in range(IN_N):
        if abs(sgi.data[i] - lgi[i]) > mg: mg = abs(sgi.data[i] - lgi[i])
    print("  max Δ: out", mo, " gi", mg)
    assert_true(mo < TOL and mg < TOL, "ModalitySpaceAttention CPU parity")
    print("  ok")


def test_modality_gpu[D: Int, NH: Int, S: Int, L: Int, MODE: StaticString](
    name: String
) raises:
    print("ModalitySpaceAttention GPU vs CPU", name, "...")
    comptime B = 2
    comptime IN_N = B * S * D * 3
    comptime OUT_N = B * S * D
    comptime TOL = Scalar[DT](1e-3)
    var c = DeviceContext()
    var cpu = ModalitySpaceAttention[D, NH, S, L, MODE].make["cpu", Deterministic]()
    var gpu = ModalitySpaceAttention[D, NH, S, L, MODE].make[
        "gpu", Deterministic
    ](Optional(c))

    var sx = Tensor.alloc(IN_N); var sgo = Tensor.alloc(OUT_N)
    for i in range(IN_N): sx.data[i] = _spread(i, 1.7)
    for i in range(OUT_N): sgo.data[i] = _spread(i, 0.9)
    var cout = Tensor.alloc(OUT_N); var cgi = Tensor.alloc(IN_N)
    cpu.forward["cpu", B](TensorRefs[1](sx), cout, None)
    cpu.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](cgi), None)

    var gx = Tensor.alloc(IN_N); var ggo = Tensor.alloc(OUT_N)
    for i in range(IN_N): gx.data[i] = sx.data[i]
    for i in range(OUT_N): ggo.data[i] = sgo.data[i]
    gx.upload(c); ggo.upload(c)
    var gout = Tensor.alloc(OUT_N); var ggi = Tensor.alloc(IN_N)
    gpu.forward["gpu", B](TensorRefs[1](gx), gout, Optional(c))
    gpu.vjp["gpu", B](TensorRefs[1](gx), ggo, TensorRefs[1](ggi), Optional(c))
    gout.download(c); ggi.download(c)

    var mo: Scalar[DT] = 0
    var mg: Scalar[DT] = 0
    for i in range(OUT_N):
        if abs(gout.data[i] - cout.data[i]) > mo: mo = abs(gout.data[i] - cout.data[i])
    for i in range(IN_N):
        if abs(ggi.data[i] - cgi.data[i]) > mg: mg = abs(ggi.data[i] - cgi.data[i])
    print("  max Δ: out", mo, " gi", mg)
    assert_true(mo < TOL and mg < TOL, "ModalitySpaceAttention GPU vs CPU")
    print("  ok")


# ── Dynamics attention parity (param-free) ───────────────────────────────
def test_dynamics_cpu[
    D: Int, NH: Int, NSP: Int, NREG: Int, NAGENT: Int, MODE: StaticString
](name: String) raises:
    print("DynamicsSpaceAttention CPU parity", name, "...")
    comptime S = 3 + NSP + NREG + NAGENT
    comptime B = 2
    comptime IN_N = B * S * D * 3
    comptime OUT_N = B * S * D
    comptime TOL = Scalar[DT](1e-6)

    var leg = LegacyDynamicsSpaceAttention[D, NH, NSP, NREG, NAGENT, MODE].make[
        target="cpu", INIT=LegacyZero
    ]()
    var st = DynamicsSpaceAttention[D, NH, NSP, NREG, NAGENT, MODE].make[
        "cpu", Deterministic
    ]()

    var lx = _a(IN_N); var ly = _a(OUT_N); var lgi = _a(IN_N)
    var sx = Tensor.alloc(IN_N); var sout = Tensor.alloc(OUT_N)
    for i in range(IN_N):
        var xv = _spread(i, 2.1)
        lx[i] = xv; sx.data[i] = xv
    var lxt = TileTensor(lx, row_major[B, S * D * 3]())
    var lyt = TileTensor(ly, row_major[B, S * D]())
    leg.forward["cpu", B](LegacyTensorPack[1].of(lxt), output=lyt)
    st.forward["cpu", B](TensorRefs[1](sx), sout, None)

    var lgo = _a(OUT_N)
    var sgo = Tensor.alloc(OUT_N)
    for i in range(OUT_N):
        var gv = _spread(i, 1.3)
        lgo[i] = gv; sgo.data[i] = gv
    var lgot = TileTensor(lgo, row_major[B, S * D]())
    var lgit = TileTensor(lgi, row_major[B, S * D * 3]())
    var sgi = Tensor.alloc(IN_N)
    leg.vjp["cpu", B](lgot, LegacyTensorPack[1].of(lgit))
    st.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)

    var mo: Scalar[DT] = 0
    var mg: Scalar[DT] = 0
    for i in range(OUT_N):
        if abs(sout.data[i] - ly[i]) > mo: mo = abs(sout.data[i] - ly[i])
    for i in range(IN_N):
        if abs(sgi.data[i] - lgi[i]) > mg: mg = abs(sgi.data[i] - lgi[i])
    print("  max Δ: out", mo, " gi", mg)
    assert_true(mo < TOL and mg < TOL, "DynamicsSpaceAttention CPU parity")
    print("  ok")


def test_dynamics_gpu[
    D: Int, NH: Int, NSP: Int, NREG: Int, NAGENT: Int, MODE: StaticString
](name: String) raises:
    print("DynamicsSpaceAttention GPU vs CPU", name, "...")
    comptime S = 3 + NSP + NREG + NAGENT
    comptime B = 2
    comptime IN_N = B * S * D * 3
    comptime OUT_N = B * S * D
    comptime TOL = Scalar[DT](1e-3)
    var c = DeviceContext()
    var cpu = DynamicsSpaceAttention[D, NH, NSP, NREG, NAGENT, MODE].make[
        "cpu", Deterministic
    ]()
    var gpu = DynamicsSpaceAttention[D, NH, NSP, NREG, NAGENT, MODE].make[
        "gpu", Deterministic
    ](Optional(c))

    var sx = Tensor.alloc(IN_N); var sgo = Tensor.alloc(OUT_N)
    for i in range(IN_N): sx.data[i] = _spread(i, 2.1)
    for i in range(OUT_N): sgo.data[i] = _spread(i, 1.3)
    var cout = Tensor.alloc(OUT_N); var cgi = Tensor.alloc(IN_N)
    cpu.forward["cpu", B](TensorRefs[1](sx), cout, None)
    cpu.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](cgi), None)

    var gx = Tensor.alloc(IN_N); var ggo = Tensor.alloc(OUT_N)
    for i in range(IN_N): gx.data[i] = sx.data[i]
    for i in range(OUT_N): ggo.data[i] = sgo.data[i]
    gx.upload(c); ggo.upload(c)
    var gout = Tensor.alloc(OUT_N); var ggi = Tensor.alloc(IN_N)
    gpu.forward["gpu", B](TensorRefs[1](gx), gout, Optional(c))
    gpu.vjp["gpu", B](TensorRefs[1](gx), ggo, TensorRefs[1](ggi), Optional(c))
    gout.download(c); ggi.download(c)

    var mo: Scalar[DT] = 0
    var mg: Scalar[DT] = 0
    for i in range(OUT_N):
        if abs(gout.data[i] - cout.data[i]) > mo: mo = abs(gout.data[i] - cout.data[i])
    for i in range(IN_N):
        if abs(ggi.data[i] - cgi.data[i]) > mg: mg = abs(ggi.data[i] - cgi.data[i])
    print("  max Δ: out", mo, " gi", mg)
    assert_true(mo < TOL and mg < TOL, "DynamicsSpaceAttention GPU vs CPU")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("DecoderBlock / Dynamics / Modality storage parity")
    print("=" * 70)
    test_decoder_cpu_parity()
    test_modality_cpu[4, 2, 6, 2, "encoder"]("encoder")
    test_modality_cpu[4, 2, 6, 2, "decoder"]("decoder")
    test_dynamics_cpu[4, 2, 2, 1, 0, "wm_agent_bc"]("nagent0")
    test_dynamics_cpu[4, 2, 2, 1, 1, "wm_agent_bc"]("nagent1")

    test_decoder_gpu_parity()
    test_modality_gpu[4, 2, 6, 2, "encoder"]("encoder")
    test_dynamics_gpu[4, 2, 2, 1, 1, "wm_agent_bc"]("nagent1")
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
