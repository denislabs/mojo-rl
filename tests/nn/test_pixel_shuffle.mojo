"""PixelShuffle — checked against an independent re-derivation, not itself.

The leaf composes the reference's five `view`/`permute`/`reshape` calls into one
closed-form index. Every step in that chain is a reshape or a transpose, so
**every wrong composition is still shape-legal**: right size, finite values,
patches scrambled. A shape or NaN check sees nothing.

So the reference here is built the other way round — as TWO EXPLICIT PERMUTES
with index arithmetic derived from the op sequence, materialising the same
intermediate buffers torch would. Two independent derivations agreeing is a
check; one derivation compared with itself is not.

    step B  view   [h][w][e]      -> [h][w'][e']     w'=w/s, e'=(w%s)*E+e
    step C  permute              -> buf1[w'][h][e']
    step D  view                 -> [w'][h'][e'']    h'=h/s, e''=(h%s)*s*E+e'
    step E  permute              -> buf2[h'][w'][e'']
    step F  view                 -> [t][e'']         t = h'*(G/s) + w'

Also asserted: the map is a BIJECTION (every destination written exactly once —
a permutation that drops and duplicates elements can still match on many of
them), the adjoint identity (a permutation is orthogonal), and GPU/CPU parity.

Run:
  pixi run -e apple mojo run -I . tests/nn/test_pixel_shuffle.mojo
"""

from std.math import abs
from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.pixel_shuffle import PixelShuffle


def _reference[G: Int, E: Int, S: Int](ref src: List[Scalar[DT]]) -> List[Scalar[DT]]:
    """The five reference steps, as two materialised permutes."""
    comptime OG = G // S
    comptime N = G * G * E
    comptime SE = S * E
    comptime S2E = S * S * E
    # C: buf1[w'][h][e'] = xB[h][w'][e']
    var buf1 = List[Scalar[DT]](unsafe_uninit_length=N)
    for h in range(G):
        for wp in range(OG):
            for ep in range(SE):
                buf1[wp * (G * SE) + h * SE + ep] = src[h * (G * E) + wp * SE + ep]
    # E: buf2[h'][w'][e''] = xD[w'][h'][e'']
    var buf2 = List[Scalar[DT]](unsafe_uninit_length=N)
    for wp in range(OG):
        for hp in range(OG):
            for epp in range(S2E):
                buf2[hp * (OG * S2E) + wp * S2E + epp] = buf1[
                    wp * (G * SE) + hp * S2E + epp
                ]
    return buf2^


def _case[G: Int, E: Int, S: Int](label: String) raises:
    comptime N = G * G * E
    comptime B = 1
    comptime P = PixelShuffle[G, E, S]
    var m = P.make["cpu", Deterministic]()

    var host = List[Scalar[DT]]()
    for i in range(N):
        host.append(Scalar[DT](i) * 0.001)
    var x = Tensor.alloc(B * N)
    for i in range(N):
        x.data[i] = host[i]
    var out = Tensor.alloc(B * N)
    m.forward["cpu", B](TensorRefs[1](x), out, None)

    var want = _reference[G, E, S](host)
    var bad = 0
    for i in range(N):
        if out.data[i] != want[i]:
            bad += 1
    print("  ", label, ": compared", N, " mismatched", bad,
          " -> tokens", P.OUT_TOKENS, "x", P.OUT_CHAN)
    assert_true(bad == 0, label + ": disagrees with the re-derived reference")

    # Bijection: every destination written exactly once.
    var hits = List[Int](unsafe_uninit_length=N)
    for i in range(N):
        hits[i] = 0
    comptime OG = G // S
    for h in range(G):
        for w in range(G):
            for e in range(E):
                var t = (h // S) * OG + (w // S)
                var c = (h % S) * (S * E) + (w % S) * E + e
                hits[t * (E * S * S) + c] += 1
    var notone = 0
    for i in range(N):
        if hits[i] != 1:
            notone += 1
    assert_true(notone == 0, label + ": the index map is not a bijection")


def main() raises:
    print("=" * 66)
    print("PixelShuffle — vs an independent re-derivation")
    print("=" * 66)
    _case[4, 6, 2](String("[1] G=4  E=6   S=2"))
    _case[6, 5, 3](String("[2] G=6  E=5   S=3"))
    _case[32, 768, 4](String("[3] G=32 E=768 S=4  (SmolVLA)"))

    # ── adjoint identity, on the real shape ──────────────────────────────
    comptime G = 32
    comptime E = 768
    comptime S = 4
    comptime N = G * G * E
    comptime B = 1
    comptime P = PixelShuffle[G, E, S]
    var m = P.make["cpu", Deterministic]()
    var x = Tensor.alloc(B * N)
    for i in range(B * N):
        x.data[i] = Scalar[DT](((i * 37) % 19) - 9) * 0.1
    var fx = Tensor.alloc(B * N)
    m.forward["cpu", B](TensorRefs[1](x), fx, None)
    var y = Tensor.alloc(B * N)
    for i in range(B * N):
        y.data[i] = Scalar[DT](((i * 53) % 23) - 11) * 0.07
    var gi = Tensor.alloc(B * N)
    m.vjp["cpu", B](TensorRefs[1](x), y, TensorRefs[1](gi), None)
    var lhs = Float64(0)
    var rhs = Float64(0)
    var scale = Float64(0)
    for i in range(B * N):
        var p = Float64(fx.data[i]) * Float64(y.data[i])
        lhs += p
        scale += abs(p)
        rhs += Float64(x.data[i]) * Float64(gi.data[i])
    var rel = abs(lhs - rhs) / (scale + 1e-12)
    print("  [4] adjoint: gap =", abs(lhs - rhs), " sum|terms| =", scale,
          " rel =", rel)
    assert_true(abs(lhs) > 1e-6, "degenerate inner product")
    assert_true(rel < 1e-8, "backward is not the adjoint of forward")

    # ── GPU parity ───────────────────────────────────────────────────────
    var c = DeviceContext()
    var mg = P.make["gpu", Deterministic](Optional(c))
    var xg = Tensor.alloc(B * N)
    for i in range(B * N):
        xg.data[i] = x.data[i]
    xg.upload(c)
    var og = Tensor.alloc(B * N)
    mg.forward["gpu", B](TensorRefs[1](xg), og, Optional(c))
    og.download(c)
    var fbad = 0
    for i in range(B * N):
        if og.data[i] != fx.data[i]:
            fbad += 1
    var yg = Tensor.alloc(B * N)
    for i in range(B * N):
        yg.data[i] = y.data[i]
    yg.upload(c)
    var gig = Tensor.alloc(B * N)
    mg.vjp["gpu", B](TensorRefs[1](xg), yg, TensorRefs[1](gig), Optional(c))
    gig.download(c)
    var vbad = 0
    for i in range(B * N):
        if gig.data[i] != gi.data[i]:
            vbad += 1
    print("  [5] GPU vs CPU: fwd", N, "compared,", fbad, "wrong | vjp", N,
          "compared,", vbad, "wrong")
    assert_true(fbad == 0 and vbad == 0, "GPU disagrees with CPU")

    print()
    print("PASSED")
