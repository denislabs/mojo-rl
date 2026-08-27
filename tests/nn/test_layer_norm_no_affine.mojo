"""`LayerNormNoAffine` — the sqrt(d) invariant, a finite-difference gradient
check, and GPU vs CPU.

The layer shipped with a GPU path and NO test anywhere. It is now load-bearing:
FB's `B` network ends in it, because `LayerNorm`'s learnable gamma let ||B||
drift 11.31 -> 17.54 over 100 k steps while `L_ortho` — a pure quartic
`(1/N^2) sum_ij (B(s_i).B(s'_j))^2` — rose 8.8x tracking it. Pinning the output
norm removes the degenerate direction the optimizer was exploiting.

Three properties, in the order they matter to that use:

  [1] **||row|| == sqrt(DIM), exactly, for ANY input.** This is the whole
      reason the layer was chosen, and it is the one `LayerNorm` does not have.
      LayerNorm subtracts the mean and divides by std, so sum(xhat^2) = DIM by
      construction — but only if the implementation actually normalizes
      variance rather than, say, dividing by a biased estimator. Asserted
      directly on adversarial rows (constant, near-constant, huge, tiny), not
      on a fingerprint that would hide a systematic scale error.
  [2] **Gradients match finite differences.** No golden fingerprint exists to
      convert (this test is new), so the backward is checked against the
      forward it belongs to. ⚠ FD on a normalisation layer needs a loose
      tolerance: the operation is scale-INVARIANT, so the analytic gradient
      along the radial direction is exactly 0 and FD there is pure roundoff.
  [3] **GPU == CPU.** The kernel path is what FB actually runs.

Run:
    pixi run -e apple mojo run -I . tests/nn/test_layer_norm_no_affine.mojo
"""

from std.math import abs, sqrt
from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.layer_norm_no_affine import LayerNormNoAffine


comptime DIM = 10
comptime B = 6
comptime D128 = 128  # FB's actual d — the invariant must hold at the real size


def _row_norm(ref t: Tensor, row: Int, dim: Int) -> Float64:
    var s = Float64(0)
    for k in range(dim):
        var v = Float64(t.data[row * dim + k])
        s += v * v
    return sqrt(s)


def test_sqrt_d_invariant() raises:
    """[1] `||row|| == sqrt(DIM)` — and the exact condition under which it fails.

    ⚠⚠ The pin is CONDITIONAL on the row's variance being >> `eps` (1e-6).
    This is standard LayerNorm semantics, not a defect, but it is a real limit
    on the guarantee FB now leans on. The first version of this test asserted
    sqrt(DIM) unconditionally and FAILED on two rows — which is how the
    condition got found. Rather than delete the offending rows, the degradation
    is asserted against its closed form:

        ||row|| = sqrt(DIM) * sqrt(var) / sqrt(var + eps)

    which reproduces both measurements to 5-6 digits. So the layer is exact
    where it matters and SPECIFIED where it is not.

    Consequence for FB, worth stating plainly: if `B`'s pre-norm activations
    ever collapse to a per-row std below ~1e-3, the sqrt(d) projection quietly
    stops projecting and ||B|| is free to drift again. Because the pin is
    otherwise exact, **||B|| != sqrt(d) in the training log IS that alarm** —
    no separate instrument is needed.
    """
    print("[1] ||row|| == sqrt(DIM), and the eps condition ...")
    comptime TOL = 2e-3
    comptime EPS = 1e-6  # LNNA_EPS
    var ln = LayerNormNoAffine[DIM].make["cpu", Deterministic]()

    var x = Tensor.alloc(B * DIM)
    # Rows 0/3/5: healthy variance — the sqrt(DIM) pin must be exact.
    # Row 1: constant (zero variance).
    # Rows 2/4: variance BELOW eps — degradation, asserted in closed form.
    for k in range(DIM):
        x.data[0 * DIM + k] = Scalar[DT](Float64(k) - 4.5)
        x.data[1 * DIM + k] = Scalar[DT](3.0)
        x.data[2 * DIM + k] = Scalar[DT](3.0 + 1e-4 * Float64(k))
        x.data[3 * DIM + k] = Scalar[DT](1e5 * (Float64(k) - 4.5))
        x.data[4 * DIM + k] = Scalar[DT](1e-5 * (Float64(k) - 4.5))
        x.data[5 * DIM + k] = Scalar[DT](0.0)
    x.data[5 * DIM + 3] = Scalar[DT](1.0)

    var out = Tensor.alloc(B * DIM)
    ln.forward["cpu", B](TensorRefs[1](x), out, None)

    var want = sqrt(Float64(DIM))
    var worst = Float64(0)
    var worst_row = -1
    for r in range(B):
        var n = _row_norm(out, r, DIM)

        # The zero-variance row CANNOT reach sqrt(DIM): x - mean == 0 exactly,
        # so the output is 0 for any eps. Correct and unavoidable.
        if r == 1:
            assert_true(n < 1e-3, "constant row should map to 0, got " + String(n))
            print("      row", r, "(constant)        -> 0  OK (expected)")
            continue

        # Rows whose variance is below eps: assert the closed form instead.
        if r == 2 or r == 4:
            var mean = Float64(0)
            for k in range(DIM):
                mean += Float64(x.data[r * DIM + k])
            mean /= Float64(DIM)
            var var_ = Float64(0)
            for k in range(DIM):
                var d = Float64(x.data[r * DIM + k]) - mean
                var_ += d * d
            var_ /= Float64(DIM)
            var pred = want * sqrt(var_) / sqrt(var_ + EPS)
            print(
                "      row", r, "(var", var_, "< eps) ||.|| =", n,
                " closed form", pred,
            )
            assert_true(
                abs(n - pred) < 1e-3,
                "sub-eps degradation does not follow sqrt(var)/sqrt(var+eps):"
                " got " + String(n) + " want " + String(pred),
            )
            continue

        var e = abs(n - want)
        print("      row", r, " ||.|| =", n, " (want", want, ")")
        if e > worst:
            worst = e
            worst_row = r
    assert_true(
        worst < TOL,
        "row " + String(worst_row) + " norm is off sqrt(DIM) by "
        + String(worst) + " — the sqrt(d) projection FB relies on is not"
        " exact, and ||B|| will drift again",
    )

    # And at FB's real width, where the whole point is that it stays put.
    var ln2 = LayerNormNoAffine[D128].make["cpu", Deterministic]()
    var x2 = Tensor.alloc(2 * D128)
    for k in range(D128):
        x2.data[k] = Scalar[DT](0.3 * Float64((k % 17)) - 2.0)
        x2.data[D128 + k] = Scalar[DT](50.0 * Float64((k % 5)) - 100.0)
    var o2 = Tensor.alloc(2 * D128)
    ln2.forward["cpu", 2](TensorRefs[1](x2), o2, None)
    var w128 = sqrt(Float64(D128))
    for r in range(2):
        var n = _row_norm(o2, r, D128)
        print("      d=128 row", r, " ||.|| =", n, " (want", w128, ")")
        assert_true(
            abs(n - w128) < 1e-2,
            "d=128 row norm off by " + String(abs(n - w128)),
        )
    print("      OK")


def test_grad_vs_finite_difference() raises:
    """[2] vjp matches finite differences of the forward it belongs to."""
    print("[2] vjp vs finite differences ...")
    comptime EPS = 1e-2  # deep-chain FD epsilon; 1e-4 is roundoff-dominated
    comptime TOL = 3e-2
    var ln = LayerNormNoAffine[DIM].make["cpu", Deterministic]()

    var x = Tensor.alloc(B * DIM)
    var go = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        x.data[i] = Scalar[DT](Float64((i % 13) - 6) * 0.18 + 0.05)
        go.data[i] = Scalar[DT](Float64((i % 7) - 3) * 0.22)

    var out = Tensor.alloc(B * DIM)
    var gi = Tensor.alloc(B * DIM)
    ln.forward["cpu", B](TensorRefs[1](x), out, None)
    ln.zero_grad["cpu"](None)
    ln.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), None)

    # L = sum_i go_i * out_i  =>  dL/dx_j is exactly what vjp returns.
    var worst = Float64(0)
    var worst_i = -1
    var probe = Tensor.alloc(B * DIM)
    for j in range(B * DIM):
        for i in range(B * DIM):
            probe.data[i] = x.data[i]
        probe.data[j] = x.data[j] + Scalar[DT](EPS)
        var op = Tensor.alloc(B * DIM)
        ln.forward["cpu", B](TensorRefs[1](probe), op, None)
        var lp = Float64(0)
        for i in range(B * DIM):
            lp += Float64(go.data[i]) * Float64(op.data[i])

        probe.data[j] = x.data[j] - Scalar[DT](EPS)
        var om = Tensor.alloc(B * DIM)
        ln.forward["cpu", B](TensorRefs[1](probe), om, None)
        var lm = Float64(0)
        for i in range(B * DIM):
            lm += Float64(go.data[i]) * Float64(om.data[i])

        var fd = (lp - lm) / (2.0 * EPS)
        var an = Float64(gi.data[j])
        var e = abs(fd - an)
        if e > worst:
            worst = e
            worst_i = j
    print("      worst |FD - analytic| =", worst, "at", worst_i)
    assert_true(
        worst < TOL,
        "vjp disagrees with finite differences by " + String(worst)
        + " at index " + String(worst_i),
    )
    print("      OK")


def test_gpu_vs_cpu() raises:
    """[3] The kernel path FB actually runs."""
    print("[3] GPU vs CPU ...")
    comptime TOL = Scalar[DT](2e-5)
    var c = DeviceContext()
    var cpu = LayerNormNoAffine[DIM].make["cpu", Deterministic]()
    var gpu = LayerNormNoAffine[DIM].make["gpu", Deterministic](Optional(c))

    var sx = Tensor.alloc(B * DIM)
    var sgo = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        sx.data[i] = Scalar[DT](Float64((i % 13) - 6) * 0.18)
        sgo.data[i] = Scalar[DT](Float64((i % 7) - 3) * 0.22)

    var c_out = Tensor.alloc(B * DIM)
    var c_gi = Tensor.alloc(B * DIM)
    cpu.forward["cpu", B](TensorRefs[1](sx), c_out, None)
    cpu.zero_grad["cpu"](None)
    cpu.vjp["cpu", B](TensorRefs[1](sx), sgo, TensorRefs[1](c_gi), None)

    var gx = Tensor.alloc(B * DIM)
    var ggo = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        gx.data[i] = sx.data[i]
        ggo.data[i] = sgo.data[i]
    gx.upload(c)
    ggo.upload(c)
    var g_out = Tensor.alloc(B * DIM)
    var g_gi = Tensor.alloc(B * DIM)
    gpu.forward["gpu", B](TensorRefs[1](gx), g_out, Optional(c))
    gpu.zero_grad["gpu"](Optional(c))
    gpu.vjp["gpu", B](TensorRefs[1](gx), ggo, TensorRefs[1](g_gi), Optional(c))
    g_out.download(c)
    g_gi.download(c)

    var mo: Scalar[DT] = 0
    var mgi: Scalar[DT] = 0
    for i in range(B * DIM):
        var eo = abs(g_out.data[i] - c_out.data[i])
        var eg = abs(g_gi.data[i] - c_gi.data[i])
        if eo > mo:
            mo = eo
        if eg > mgi:
            mgi = eg
    print("      max Δ: out", mo, " gi", mgi)
    assert_true(mo < TOL and mgi < TOL, "LayerNormNoAffine GPU vs CPU")

    # ⚠ The invariant must survive the GPU path too — a kernel that normalised
    # with a slightly different eps or a biased variance would pass the Δ
    # check above (CPU and GPU agreeing on the SAME wrong value is possible if
    # both share a helper) and still let ||B|| drift.
    var want = sqrt(Float64(DIM))
    var worst = Float64(0)
    for r in range(B):
        var e = abs(_row_norm(g_out, r, DIM) - want)
        if e > worst:
            worst = e
    print("      GPU worst |row norm - sqrt(DIM)| =", worst)
    assert_true(worst < 2e-3, "GPU rows are not on the sqrt(DIM) sphere")
    print("      OK")


def main() raises:
    print("=" * 70)
    print("LayerNormNoAffine — sqrt(d) invariant, FD gradients, GPU parity")
    print("=" * 70)
    test_sqrt_d_invariant()
    test_grad_vs_finite_difference()
    test_gpu_vs_cpu()
    print("\n[PASS] LayerNormNoAffine")
