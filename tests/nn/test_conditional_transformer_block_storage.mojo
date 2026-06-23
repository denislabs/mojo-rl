"""ConditionalTransformerBlock storage primitive — CPU correctness (golden) +
GPU vs CPU.

Standalone storage test (no legacy oracle — converted from the former
`_storage_parity` test in legacy-removal Phase 0b). The storage
ConditionalTransformerBlock wraps a storage ComputeGraph (AdaLN-zero DiT block).
The CPU check asserts the storage forward/backward against golden fingerprints
(S = Σ vᵢ, W = Σ vᵢ·(i+1) — the weight catches sign/position errors that a plain
sum would cancel), captured from the bit-identical legacy↔storage run the parity
test used to verify. The GPU check is storage-only (GPU vs CPU consistency).

This block has a CONDITIONING input (ARITY=2): forward/vjp take a (x, c) pack.

Run:
  rm -f mojo_rl.mojoc && pixi run -e apple mojo run -I . \
      tests/nn/test_conditional_transformer_block_storage.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT

from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.core.param import ParamVisitor as SParamVisitor
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.conditional_transformer_block import (
    ConditionalTransformerBlock,
)


comptime EMB = 4
comptime HEADS = 2
comptime H = 3
comptime FF = 8
comptime BATCH = 2
comptime SEQ = H * EMB
comptime N = BATCH * SEQ


def _spread(i: Int, seed: Float64) -> Scalar[DT]:
    var x = seed + 0.7 * Float64(i)
    var t = x - 6.2831853 * Float64(Int(x / 6.2831853))
    return Scalar[DT](0.5 * (t - (t * t * t) / 6.0))


def _check(name: String, data: Tensor, n: Int,
           es: Scalar[DT], ew: Scalar[DT], tol: Scalar[DT]) -> Bool:
    """Assert tensor fingerprint (Σ vᵢ, Σ vᵢ·(i+1)) matches golden (es, ew)."""
    var s: Scalar[DT] = 0
    var w: Scalar[DT] = 0
    for i in range(n):
        s += data.data[i]
        w += data.data[i] * Scalar[DT](i + 1)
    var ok = abs(s - es) < tol and abs(w - ew) < tol
    print("  ", name, "S", s, "(exp", es, ") W", w, "(exp", ew, ")", "OK" if ok else "FAIL")
    return ok


# ── Param fill / grad collect visitors ───────────────────────────────────
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


struct _SGradCheck(SParamVisitor):
    """Accumulate the (S, W) fingerprint over ALL param grads in walk order."""
    var s: Scalar[DT]
    var w: Scalar[DT]
    var idx: Int

    def __init__(out self):
        self.s = 0
        self.w = 0
        self.idx = 0

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        for i in range(N):
            self.s += grad.data[i]
            self.w += grad.data[i] * Scalar[DT](self.idx + 1)
            self.idx += 1


def test_ctb_cpu_golden() raises:
    print("test_ctb_cpu_golden (storage CPU vs golden) ...")
    # Fingerprints reach ~1e7 (276 params, composite block); fp32 ULP at that
    # magnitude is ~1, so the absolute tol is loosened accordingly.
    comptime TOL = Scalar[DT](5.0)

    var st = ConditionalTransformerBlock[EMB, HEADS, H, FF].make[
        "cpu", Deterministic
    ]()
    var sf = _SFill()
    st.for_each_param["cpu", _SFill](sf, None)
    print("   storage filled", sf.counter, "elems")

    # Inputs (x, c) + grad_output.
    var sin = TensorPack[2]()
    sin[0].ensure(N); sin[1].ensure(N)
    var sout = Tensor.alloc(N)
    for i in range(N):
        sin[0].data[i] = _spread(i, 0.5)
        sin[1].data[i] = _spread(i, 1.9)
    st.forward["cpu", BATCH](TensorRefs[2](sin[0], sin[1]), sout, None)

    var sgo = Tensor.alloc(N)
    for i in range(N):
        sgo.data[i] = _spread(i, 2.3)

    var sg = TensorPack[2]()
    st.vjp["cpu", BATCH](
        TensorRefs[2](sin[0], sin[1]), sgo, TensorRefs[2](sg[0], sg[1]), None
    )

    var ok = _check("out", sout, N, 32065.723, 459018.72, TOL)
    ok = _check("gx", sg[0], N, -88.53027, 15791.227, TOL) and ok
    ok = _check("gc", sg[1], N, -183787.02, -194723.27, TOL) and ok

    # Param grads fingerprint in walk order.
    comptime PG_S = Scalar[DT](-190910.84)
    comptime PG_W = Scalar[DT](-39046290.0)
    comptime PG_TOL = Scalar[DT](4000.0)
    var pgc = _SGradCheck()
    st.for_each_param["cpu", _SGradCheck](pgc, None)
    var pok = abs(pgc.s - PG_S) < PG_TOL and abs(pgc.w - PG_W) < PG_TOL
    print("   pgrad S", pgc.s, "(exp", PG_S, ") W", pgc.w,
          "(exp", PG_W, ")", "(nparams", pgc.idx, ")",
          "OK" if pok else "FAIL")
    ok = pok and ok

    assert_true(ok, "ConditionalTransformerBlock CPU golden")
    print("  ok")


def test_ctb_gpu_vs_cpu() raises:
    print("test_ctb_gpu_vs_cpu (storage GPU vs storage CPU) ...")
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
    print("ConditionalTransformerBlock storage primitive (CPU golden + GPU vs CPU)")
    print("=" * 70)
    test_ctb_cpu_golden()
    test_ctb_gpu_vs_cpu()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
