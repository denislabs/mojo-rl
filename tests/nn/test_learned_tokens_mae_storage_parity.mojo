"""LearnedTokens + LearnedQueries + MAEReplacer — legacy↔storage parity.

Per leaf: storage CPU vs LEGACY CPU (identical param table + inputs, max|Δ| <
1e-6 on out + grad_input + the table grad); storage GPU vs storage CPU (~2e-5).

    pixi run        mojo run -I . tests/nn/test_learned_tokens_mae_storage_parity.mojo
    pixi run -e apple mojo run -I . tests/nn/test_learned_tokens_mae_storage_parity.mojo
"""

from std.memory import alloc
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT

# legacy leaves
from mojo_rl.nn.primitives.learned_tokens import LearnedTokens as LegacyLearnedTokens
from mojo_rl.nn.primitives.learned_queries import LearnedQueries as LegacyLearnedQueries
from mojo_rl.nn.primitives.mae_replacer import MAEReplacer as LegacyMAEReplacer
from mojo_rl.nn.initializer import Zero as LegacyZero

# storage leaves
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.learned_tokens import LearnedTokens
from mojo_rl.nn.storage.primitives.learned_queries import LearnedQueries
from mojo_rl.nn.storage.primitives.mae_replacer import MAEReplacer


def _mao(p: UnsafePointer[Scalar[DT], _]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](p)


# ───────────────────────── LearnedTokens ──────────────────────────────
def _lt_check[
    target: StaticString, N_IN: Int, N_NEW: Int, D: Int, PREPEND: Bool
](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime B = 6
    comptime IN_N = B * N_IN * D
    comptime OUT_N = B * (N_IN + N_NEW) * D
    comptime PN = N_NEW * D
    comptime TOL = Scalar[DT](1e-6) if target == "cpu" else Scalar[DT](2e-5)

    # legacy CPU reference
    var leg = LegacyLearnedTokens[N_IN, N_NEW, D, PREPEND].make[
        target="cpu", INIT=LegacyZero
    ]()
    var lp = leg.tokens.value_unsafe_ptr_cpu()
    for k in range(PN):
        lp[k] = Scalar[DT]((k % 9) - 4) * 0.06
    var x = alloc[Scalar[DT]](IN_N)
    var y = alloc[Scalar[DT]](OUT_N)
    var go = alloc[Scalar[DT]](OUT_N)
    var gi = alloc[Scalar[DT]](IN_N)
    for i in range(IN_N):
        x[i] = Scalar[DT]((i % 5) - 2) * 0.3
    for i in range(OUT_N):
        go[i] = Scalar[DT]((i % 7) - 3) * 0.2
    var xc = TileTensor(_mao(x), row_major[B, N_IN * D]())
    var yc = TileTensor(_mao(y), row_major[B, (N_IN + N_NEW) * D]())
    var goc = TileTensor(_mao(go), row_major[B, (N_IN + N_NEW) * D]())
    var gic = TileTensor(_mao(gi), row_major[B, N_IN * D]())
    leg.forward["cpu", B](xc, output=yc)
    leg.zero_grad["cpu"]()
    leg.vjp["cpu", B](goc, gic)

    # storage
    var st = LearnedTokens[N_IN, N_NEW, D, PREPEND].make[target, Deterministic](
        ctx
    )
    for k in range(PN):
        st.tokens.val.data[k] = lp[k]
    var sx = Tensor.alloc(IN_N)
    var sgo = Tensor.alloc(OUT_N)
    var sout = Tensor.alloc(OUT_N)
    var sgi = Tensor.alloc(IN_N)
    for i in range(IN_N):
        sx.data[i] = x[i]
    for i in range(OUT_N):
        sgo.data[i] = go[i]
    comptime if target == "cpu":
        st.forward["cpu", B](TensorRefs[1](sx), sout, None)
        st.zero_grad["cpu"](None)
        st.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)
    else:
        var c = ctx.value()
        st.tokens.val.upload(c)
        sx.upload(c); sgo.upload(c)
        st.forward["gpu", B](TensorRefs[1](sx), sout, ctx)
        st.zero_grad["gpu"](ctx)
        st.vjp["gpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), ctx)
        sout.download(c); sgi.download(c); st.tokens.grd.download(c)

    var ok = True
    for i in range(OUT_N):
        if abs(sout.data[i] - y[i]) > TOL: ok = False
    for i in range(IN_N):
        if abs(sgi.data[i] - gi[i]) > TOL: ok = False
    for k in range(PN):
        if abs(st.tokens.grd.data[k] - leg.tokens.grd.cpu[k]) > TOL: ok = False
    return ok


# ───────────────────────── LearnedQueries ─────────────────────────────
def _lq_check[
    target: StaticString, IGNORE_DIM: Int, N: Int, D: Int
](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime B = 5
    comptime GI_N = B * IGNORE_DIM
    comptime OUT_N = B * N * D
    comptime PN = N * D
    comptime TOL = Scalar[DT](1e-6) if target == "cpu" else Scalar[DT](2e-5)

    var leg = LegacyLearnedQueries[IGNORE_DIM, N, D].make[
        target="cpu", INIT=LegacyZero
    ]()
    var lp = leg.queries.value_unsafe_ptr_cpu()
    for k in range(PN):
        lp[k] = Scalar[DT]((k % 11) - 5) * 0.07
    var x = alloc[Scalar[DT]](GI_N)
    var y = alloc[Scalar[DT]](OUT_N)
    var go = alloc[Scalar[DT]](OUT_N)
    var gi = alloc[Scalar[DT]](GI_N)
    for i in range(GI_N):
        x[i] = Scalar[DT]((i % 4) - 1) * 0.9  # ignored, but exercise it
    for i in range(OUT_N):
        go[i] = Scalar[DT]((i % 7) - 3) * 0.25
    var xc = TileTensor(_mao(x), row_major[B, IGNORE_DIM]())
    var yc = TileTensor(_mao(y), row_major[B, N * D]())
    var goc = TileTensor(_mao(go), row_major[B, N * D]())
    var gic = TileTensor(_mao(gi), row_major[B, IGNORE_DIM]())
    leg.forward["cpu", B](xc, output=yc)
    leg.zero_grad["cpu"]()
    leg.vjp["cpu", B](goc, gic)

    var st = LearnedQueries[IGNORE_DIM, N, D].make[target, Deterministic](ctx)
    for k in range(PN):
        st.queries.val.data[k] = lp[k]
    var sx = Tensor.alloc(GI_N)
    var sgo = Tensor.alloc(OUT_N)
    var sout = Tensor.alloc(OUT_N)
    var sgi = Tensor.alloc(GI_N)
    for i in range(GI_N):
        sx.data[i] = x[i]
    for i in range(OUT_N):
        sgo.data[i] = go[i]
    comptime if target == "cpu":
        st.forward["cpu", B](TensorRefs[1](sx), sout, None)
        st.zero_grad["cpu"](None)
        st.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)
    else:
        var c = ctx.value()
        st.queries.val.upload(c)
        sx.upload(c); sgo.upload(c)
        st.forward["gpu", B](TensorRefs[1](sx), sout, ctx)
        st.zero_grad["gpu"](ctx)
        st.vjp["gpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), ctx)
        sout.download(c); sgi.download(c); st.queries.grd.download(c)

    var ok = True
    for i in range(OUT_N):
        if abs(sout.data[i] - y[i]) > TOL: ok = False
    for i in range(GI_N):
        if abs(sgi.data[i] - gi[i]) > TOL: ok = False
    for k in range(PN):
        if abs(st.queries.grd.data[k] - leg.queries.grd.cpu[k]) > TOL: ok = False
    return ok


# ───────────────────────── MAEReplacer ────────────────────────────────
def _mae_check[
    target: StaticString, NP: Int, D: Int, SEED: UInt64
](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime B = 7
    comptime M = B * NP * D
    comptime PN = D
    comptime P_MIN = 0.2
    comptime P_MAX = 0.6
    comptime TOL = Scalar[DT](1e-6) if target == "cpu" else Scalar[DT](2e-5)

    var leg = LegacyMAEReplacer[NP, D, P_MIN, P_MAX, SEED].make[
        target="cpu", INIT=LegacyZero
    ]()
    var lp = leg.mask_token.value_unsafe_ptr_cpu()
    for k in range(PN):
        lp[k] = Scalar[DT]((k % 5) - 2) * 0.11
    var x = alloc[Scalar[DT]](M)
    var y = alloc[Scalar[DT]](M)
    var go = alloc[Scalar[DT]](M)
    var gi = alloc[Scalar[DT]](M)
    for i in range(M):
        x[i] = Scalar[DT]((i % 9) - 4) * 0.13
        go[i] = Scalar[DT]((i % 6) - 3) * 0.21
    var xc = TileTensor(_mao(x), row_major[B, NP * D]())
    var yc = TileTensor(_mao(y), row_major[B, NP * D]())
    var goc = TileTensor(_mao(go), row_major[B, NP * D]())
    var gic = TileTensor(_mao(gi), row_major[B, NP * D]())
    leg.forward["cpu", B](xc, output=yc)
    leg.zero_grad["cpu"]()
    leg.vjp["cpu", B](goc, gic)

    var st = MAEReplacer[NP, D, P_MIN, P_MAX, SEED].make[target, Deterministic](
        ctx
    )
    for k in range(PN):
        st.mask_token.val.data[k] = lp[k]
    var sx = Tensor.alloc(M)
    var sgo = Tensor.alloc(M)
    var sout = Tensor.alloc(M)
    var sgi = Tensor.alloc(M)
    for i in range(M):
        sx.data[i] = x[i]
        sgo.data[i] = go[i]
    comptime if target == "cpu":
        st.forward["cpu", B](TensorRefs[1](sx), sout, None)
        st.zero_grad["cpu"](None)
        st.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), None)
    else:
        var c = ctx.value()
        st.mask_token.val.upload(c)
        sx.upload(c); sgo.upload(c)
        st.forward["gpu", B](TensorRefs[1](sx), sout, ctx)
        st.zero_grad["gpu"](ctx)
        st.vjp["gpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](sgi), ctx)
        sout.download(c); sgi.download(c); st.mask_token.grd.download(c)

    var ok = True
    for i in range(M):
        if abs(sout.data[i] - y[i]) > TOL: ok = False
    for i in range(M):
        if abs(sgi.data[i] - gi[i]) > TOL: ok = False
    for k in range(PN):
        if abs(st.mask_token.grd.data[k] - leg.mask_token.grd.cpu[k]) > TOL:
            ok = False
    return ok


def main() raises:
    print("=" * 70)
    print("LearnedTokens + LearnedQueries + MAEReplacer storage parity")
    print("=" * 70)
    var c = DeviceContext()
    var ok = True

    var a = _lt_check["cpu", 3, 2, 4, True](None)
    print("LearnedTokens prepend CPU (legacy↔storage):", "OK" if a else "FAIL")
    ok = a and ok
    var a2 = _lt_check["cpu", 3, 2, 4, False](None)
    print("LearnedTokens append  CPU (legacy↔storage):", "OK" if a2 else "FAIL")
    ok = a2 and ok
    var ag = _lt_check["gpu", 3, 2, 4, True](Optional(c))
    print("LearnedTokens prepend GPU (vs storage CPU):", "OK" if ag else "FAIL")
    ok = ag and ok

    var b = _lq_check["cpu", 5, 3, 4](None)
    print("LearnedQueries CPU (legacy↔storage):", "OK" if b else "FAIL")
    ok = b and ok
    var bg = _lq_check["gpu", 5, 3, 4](Optional(c))
    print("LearnedQueries GPU (vs storage CPU):", "OK" if bg else "FAIL")
    ok = bg and ok

    var d = _mae_check["cpu", 6, 4, 1234](None)
    print("MAEReplacer CPU (legacy↔storage):", "OK" if d else "FAIL")
    ok = d and ok
    var dg = _mae_check["gpu", 6, 4, 1234](Optional(c))
    print("MAEReplacer GPU (vs storage CPU):", "OK" if dg else "FAIL")
    ok = dg and ok

    assert_true(ok, "learned tokens / queries / mae parity")
    print("ALL PASSED")
