"""RepeatKVHeads — the GQA grouping rule, and the adjoint of its backward.

`g // REP` and `g % N_KV` are both plausible groupings of the same head count
and produce identically shaped tensors, so a shape or NaN check cannot tell
them apart. Every query head would simply be paired with the wrong key head:
finite, right-shaped, a different model.

So kv head `k` is filled with the marker value `k + 1` and the output is read
back. With N_KV=5, REP=3 the two conventions disagree from output head 1
onward (`g//REP` says head 1 reads kv 0; `g%N_KV` says it reads kv 1), which
makes the test decisive rather than suggestive.

The backward is a sum, not a copy — each input head collects REP output heads.
Copying one instead would scale the K/V gradients by 1/REP and surface only as
slow training. It is checked by the adjoint identity `<f(x), y> == <x, vjp(y)>`,
exact for a linear map, plus a direct check that a uniform incoming gradient
comes back as exactly REP.

Run:
  pixi run mojo run -I . tests/nn/test_repeat_kv_heads.mojo
  pixi run -e apple mojo run -I . tests/nn/test_repeat_kv_heads.mojo
"""

from std.math import abs
from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.repeat_kv_heads import RepeatKVHeads

comptime SEQ = 4
comptime N_KV = 5
comptime REP = 3          # 15 query heads over 5 kv heads — SmolLM2's ratio
comptime HD = 8
comptime B = 2
comptime RK = RepeatKVHeads[SEQ, N_KV, REP, HD]
comptime IN_N = RK.IN_N
comptime OUT_N = RK.OUT_N


def main() raises:
    print("=" * 66)
    print("RepeatKVHeads — GQA grouping and the backward's sum")
    print("=" * 66)
    var m = RK.make["cpu", Deterministic]()

    # ── [1] the grouping rule ────────────────────────────────────────────
    var x = Tensor.alloc(B * IN_N)
    for b in range(B):
        for t in range(SEQ):
            for kv in range(N_KV):
                for d in range(HD):
                    x.data[b * IN_N + t * (N_KV * HD) + kv * HD + d] = (
                        Scalar[DT](kv + 1)
                    )
    var out = Tensor.alloc(B * OUT_N)
    m.forward["cpu", B](TensorRefs[1](x), out, None)

    var checked = 0
    var wrong = 0
    for b in range(B):
        for t in range(SEQ):
            for g in range(N_KV * REP):
                for d in range(HD):
                    checked += 1
                    var got = out.data[
                        b * OUT_N + t * (N_KV * REP * HD) + g * HD + d
                    ]
                    var want = Scalar[DT]((g // REP) + 1)
                    if got != want:
                        wrong += 1
    print("  [1] grouping: compared", checked, " wrong", wrong)
    assert_true(checked == B * OUT_N, "must compare every output element")
    assert_true(wrong == 0, "output head g does not read kv head g//REP — the"
                            " GQA grouping is the interleaved one")
    # Say out loud that the two conventions actually differ on this input.
    var h1 = out.data[1 * HD]
    var h3 = out.data[3 * HD]
    print("      head1 =", h1, "(g//REP -> 1; g%N_KV would give 2)",
          " head3 =", h3, "(g//REP -> 2; g%N_KV would give 4)")
    assert_true(h1 == Scalar[DT](1) and h3 == Scalar[DT](2),
                "markers do not discriminate the two conventions")

    # ── [2] backward sums REP, it does not copy ──────────────────────────
    var go = Tensor.alloc(B * OUT_N)
    for i in range(B * OUT_N):
        go.data[i] = Scalar[DT](1)
    var gi = Tensor.alloc(B * IN_N)
    m.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), None)
    var bad = 0
    for i in range(B * IN_N):
        if gi.data[i] != Scalar[DT](REP):
            bad += 1
    print("  [2] backward on uniform ones: compared", B * IN_N,
          " not-equal-to-REP", bad, " (REP =", REP, ")")
    assert_true(bad == 0, "backward does not SUM the REP copies")

    # ── [3] adjoint identity ─────────────────────────────────────────────
    var xr = Tensor.alloc(B * IN_N)
    for i in range(B * IN_N):
        xr.data[i] = Scalar[DT](((i * 37) % 19) - 9) * 0.1
    var fx = Tensor.alloc(B * OUT_N)
    m.forward["cpu", B](TensorRefs[1](xr), fx, None)
    var y = Tensor.alloc(B * OUT_N)
    for i in range(B * OUT_N):
        y.data[i] = Scalar[DT](((i * 53) % 23) - 11) * 0.07
    var gi2 = Tensor.alloc(B * IN_N)
    m.vjp["cpu", B](TensorRefs[1](xr), y, TensorRefs[1](gi2), None)
    var lhs = Float64(0)
    var rhs = Float64(0)
    var scale = Float64(0)
    for i in range(B * OUT_N):
        var p = Float64(fx.data[i]) * Float64(y.data[i])
        lhs += p
        scale += abs(p)
    for i in range(B * IN_N):
        rhs += Float64(xr.data[i]) * Float64(gi2.data[i])
    # ⚠ Normalise by the sum of |terms|, NOT by |lhs|. These inner products
    # cancel heavily (here sum|terms| is ~788x the result), so gap/|lhs| reports
    # ~4e-7 for a gap of 1.0e-7 that is simply fp32 epsilon on a 3-term sum —
    # a conditioning artefact, not an error in the vjp. Dividing by the
    # cancelled result measures the cancellation, not the code.
    var gap = abs(lhs - rhs)
    var rel = gap / (scale + 1e-12)
    print("  [3] adjoint: <f(x),y> =", lhs, " <x,vjp(y)> =", rhs)
    print("      gap =", gap, " sum|terms| =", scale, " rel =", rel,
          " (cancellation", scale / (abs(lhs) + 1e-12), "x)")
    assert_true(abs(lhs) > 1e-6, "degenerate inner product — identity would"
                                 " hold vacuously")
    assert_true(rel < 1e-8, "backward is not the adjoint of forward")

    # ── [4] GPU == CPU, both directions ──────────────────────────────────
    var c = DeviceContext()
    var mg = RK.make["gpu", Deterministic](Optional(c))
    var xg = Tensor.alloc(B * IN_N)
    for i in range(B * IN_N):
        xg.data[i] = xr.data[i]
    xg.upload(c)
    var og = Tensor.alloc(B * OUT_N)
    mg.forward["gpu", B](TensorRefs[1](xg), og, Optional(c))
    og.download(c)
    var fbad = 0
    for i in range(B * OUT_N):
        if abs(og.data[i] - fx.data[i]) > 1e-6:
            fbad += 1
    var yg = Tensor.alloc(B * OUT_N)
    for i in range(B * OUT_N):
        yg.data[i] = y.data[i]
    yg.upload(c)
    var gig = Tensor.alloc(B * IN_N)
    mg.vjp["gpu", B](TensorRefs[1](xg), yg, TensorRefs[1](gig), Optional(c))
    gig.download(c)
    var vbad = 0
    for i in range(B * IN_N):
        if abs(gig.data[i] - gi2.data[i]) > 1e-6:
            vbad += 1
    print("  [4] GPU vs CPU: fwd compared", B * OUT_N, "wrong", fbad,
          " | vjp compared", B * IN_N, "wrong", vbad)
    assert_true(fbad == 0 and vbad == 0, "GPU disagrees with CPU")

    print()
    print("PASSED")
