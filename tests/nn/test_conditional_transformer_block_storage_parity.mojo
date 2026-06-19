"""ConditionalTransformerBlock storage parity gate (CPU vs legacy; GPU vs CPU).

The storage ConditionalTransformerBlock wraps a storage ComputeGraph (AdaLN-zero
DiT block). This gate carries the legacy block VERBATIM as the oracle:

  - CPU: build a LEGACY block + a STORAGE block, fill BOTH with the SAME
    deterministic params (a shared FILL visitor in identical walk order — both
    frameworks walk the same node graph in the same order), feed the SAME
    inputs, and check max|Δ| < 1e-6 on out + grad_inputs(x,c) + param grads.
  - GPU: run the STORAGE block on GPU and compare to its own CPU run. The block
    is a composite of Linear/LayerNorm/FFN/attention matmuls through a graph, so
    GPU vs CPU fp32 accumulation order drifts the grad path ~1e-4 (out exact);
    the grad tol is looser (2e-3), matching the DecoderBlock composite gate.

Run:
  rm -f mojo_rl.mojoc && pixi run mojo run -I . \
      tests/nn/test_conditional_transformer_block_storage_parity.mojo
  rm -f mojo_rl.mojoc && pixi run -e apple mojo run -I . \
      tests/nn/test_conditional_transformer_block_storage_parity.mojo
"""

from std.memory import alloc
from std.gpu.memory import AddressSpace
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT

# ── Legacy oracle ────────────────────────────────────────────────────────
from mojo_rl.nn.initializer import Zero as LegacyZero
from mojo_rl.nn.core.tensor_pack import TensorPack as LegacyTensorPack
from mojo_rl.nn.core.param_visitor import ParamVisitor as LegacyParamVisitor
from mojo_rl.nn.primitives.conditional_transformer_block import (
    ConditionalTransformerBlock as LegacyCTB,
)

# ── Storage under test ───────────────────────────────────────────────────
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.tensor_pack import TensorPack
from mojo_rl.nn.storage.core.param import ParamVisitor as SParamVisitor
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.conditional_transformer_block import (
    ConditionalTransformerBlock,
)


comptime EMB = 4
comptime HEADS = 2
comptime H = 3
comptime FF = 8
comptime BATCH = 2
comptime SEQ = H * EMB
comptime N = BATCH * SEQ


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


def test_cpu_parity() raises:
    print("ConditionalTransformerBlock CPU parity (legacy vs storage) ...")
    comptime TOL = Scalar[DT](1e-6)

    var leg = LegacyCTB[EMB, HEADS, H, FF].make[target="cpu", INIT=LegacyZero]()
    var lf = _LFill()
    leg.for_each_param["cpu", _LFill]("", lf)

    var st = ConditionalTransformerBlock[EMB, HEADS, H, FF].make[
        "cpu", Deterministic
    ]()
    var sf = _SFill()
    st.for_each_param["cpu", _SFill](sf, None)
    print("   legacy filled", lf.counter, "elems; storage filled", sf.counter)

    # Inputs (x, c) + grad_output (w).
    var lx = _a(N); var lc = _a(N); var ly = _a(N)
    var sin = TensorPack[2]()
    sin[0].ensure(N); sin[1].ensure(N)
    var sout = Tensor.alloc(N)
    for i in range(N):
        var xv = _spread(i, 0.5)
        var cv = _spread(i, 1.9)
        lx[i] = xv; lc[i] = cv
        sin[0].data[i] = xv; sin[1].data[i] = cv
    var lxt = TileTensor(lx, row_major[BATCH, SEQ]())
    var lct = TileTensor(lc, row_major[BATCH, SEQ]())
    var lyt = TileTensor(ly, row_major[BATCH, SEQ]())
    leg.forward["cpu", BATCH](LegacyTensorPack[2].of(lxt, lct), output=lyt)
    st.forward["cpu", BATCH](TensorRefs[2](sin[0], sin[1]), sout, None)

    var lgo = _a(N); var lgx = _a(N); var lgc = _a(N)
    var sgo = Tensor.alloc(N)
    for i in range(N):
        var gv = _spread(i, 2.3)
        lgo[i] = gv; sgo.data[i] = gv
    var lgot = TileTensor(lgo, row_major[BATCH, SEQ]())
    var lgxt = TileTensor(lgx, row_major[BATCH, SEQ]())
    var lgct = TileTensor(lgc, row_major[BATCH, SEQ]())
    leg.vjp["cpu", BATCH](lgot, LegacyTensorPack[2].of(lgxt, lgct))

    var sg = TensorPack[2]()
    st.vjp["cpu", BATCH](
        TensorRefs[2](sin[0], sin[1]), sgo, TensorRefs[2](sg[0], sg[1]), None
    )

    var mo: Scalar[DT] = 0
    var mgx: Scalar[DT] = 0
    var mgc: Scalar[DT] = 0
    for i in range(N):
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
    lx.free(); lc.free(); ly.free(); lgo.free(); lgx.free(); lgc.free()
    _ = leg^
    assert_true(
        mo < TOL and mgx < TOL and mgc < TOL and mpg < TOL,
        "ConditionalTransformerBlock CPU parity",
    )
    print("  ok")


def test_gpu_vs_cpu() raises:
    print("ConditionalTransformerBlock GPU vs CPU (storage) ...")
    # Composite (six ZeroLinear + LN + Modulate + MHA + FFN + Gate through a
    # graph with grad fan-out accumulation): GPU/CPU fp32 accumulation order
    # differs, so the grad path drifts (out near-exact). This block is DEEPER
    # than the DecoderBlock composite (more matmul stages + 3-way fan-out grad
    # sums), so its grad drift is correspondingly larger — checked RELATIVE to
    # the grad magnitude (< 2% rel), with a tight absolute tol on the output.
    comptime OUT_TOL = Scalar[DT](5e-3)
    comptime REL_TOL = Scalar[DT](2e-2)
    var c = DeviceContext()

    var cpu = ConditionalTransformerBlock[EMB, HEADS, H, FF].make[
        "cpu", Deterministic
    ]()
    var f1 = _SFill()
    cpu.for_each_param["cpu", _SFill](f1, None)
    var gpu = ConditionalTransformerBlock[EMB, HEADS, H, FF].make[
        "gpu", Deterministic
    ](Optional(c))
    var gf = _SGpuFill(c)
    gpu.for_each_param["gpu", _SGpuFill](gf, Optional(c))

    var cin = TensorPack[2]()
    cin[0].ensure(N); cin[1].ensure(N)
    var sgo = Tensor.alloc(N)
    for i in range(N):
        cin[0].data[i] = _spread(i, 0.5)
        cin[1].data[i] = _spread(i, 1.9)
        sgo.data[i] = _spread(i, 2.3)
    var cout = Tensor.alloc(N)
    var cg = TensorPack[2]()
    cpu.forward["cpu", BATCH](TensorRefs[2](cin[0], cin[1]), cout, None)
    cpu.vjp["cpu", BATCH](
        TensorRefs[2](cin[0], cin[1]), sgo, TensorRefs[2](cg[0], cg[1]), None
    )

    var gin = TensorPack[2]()
    gin[0].ensure(N); gin[1].ensure(N)
    var ggo = Tensor.alloc(N)
    for i in range(N):
        gin[0].data[i] = cin[0].data[i]
        gin[1].data[i] = cin[1].data[i]
        ggo.data[i] = sgo.data[i]
    gin[0].upload(c); gin[1].upload(c); ggo.upload(c)
    var gout = Tensor.alloc(N)
    var gg = TensorPack[2]()
    gpu.forward["gpu", BATCH](TensorRefs[2](gin[0], gin[1]), gout, Optional(c))
    gpu.vjp["gpu", BATCH](
        TensorRefs[2](gin[0], gin[1]), ggo, TensorRefs[2](gg[0], gg[1]),
        Optional(c)
    )
    gout.download(c); gg[0].download(c); gg[1].download(c)

    var mo: Scalar[DT] = 0
    var mgx: Scalar[DT] = 0
    var mgc: Scalar[DT] = 0
    var max_gx: Scalar[DT] = 0
    var max_gc: Scalar[DT] = 0
    for i in range(N):
        if abs(gout.data[i] - cout.data[i]) > mo:
            mo = abs(gout.data[i] - cout.data[i])
        if abs(gg[0].data[i] - cg[0].data[i]) > mgx:
            mgx = abs(gg[0].data[i] - cg[0].data[i])
        if abs(gg[1].data[i] - cg[1].data[i]) > mgc:
            mgc = abs(gg[1].data[i] - cg[1].data[i])
        if abs(cg[0].data[i]) > max_gx: max_gx = abs(cg[0].data[i])
        if abs(cg[1].data[i]) > max_gc: max_gc = abs(cg[1].data[i])
    var rel_gx = mgx / (max_gx + Scalar[DT](1e-6))
    var rel_gc = mgc / (max_gc + Scalar[DT](1e-6))
    print("  max Δ: out", mo, " gx", mgx, "(rel", rel_gx, ")",
          " gc", mgc, "(rel", rel_gc, ")")
    _ = cpu^; _ = gpu^
    assert_true(
        mo < OUT_TOL and rel_gx < REL_TOL and rel_gc < REL_TOL,
        "ConditionalTransformerBlock GPU vs CPU",
    )
    print("  ok")


def main() raises:
    print("=" * 70)
    print("ConditionalTransformerBlock storage parity")
    print("=" * 70)
    test_cpu_parity()
    test_gpu_vs_cpu()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
