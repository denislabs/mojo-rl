# +--------------------------------------------------------------------------+ #
# | The camera -> base fit, and the reflection that fits it better than a rotation
# +--------------------------------------------------------------------------+ #
"""Gate for `mojo_rl/vision/extrinsics.mojo`.

    pixi run build-opencv     # ONCE — the fit calls `svd_3x3`
    pixi run mojo run -I . tests/vision/test_extrinsics_kabsch.mojo

⚠ **NO CAMERA, NO ARM, NO FIXTURE.** Every case here is synthesised from a
transform this file chose, so the truth is exact and the gate is a statement
about the SOLVER rather than about a rig. What it therefore does NOT gate is
the thing a rig decides — whether the marker offset was measured right, whether
the FK and the detection were taken at the same instant — and those only show
up as a residual on real correspondences.

⚠⚠ **THE CASE THIS FILE EXISTS FOR IS `test_reflection_is_refused`.** Kabsch's
textbook one-liner `R = V Uᵀ` maximises over ORTHOGONAL matrices, so it can
return a reflection: a mirror world that fits the data *better than any
rotation can* and describes a robot that does not exist. Every other check here
passes with the sign fix deleted. That one is the reason the sign fix is
gateable at all — measured, by deleting it:

| injected defect | result |
|---|---|
| drop `w[2] = -1.0`, the reflection fix | **FAIL** — mirrored input gives `det = -1.0` and **rms 0.000 mm** |
| build the cross-covariance transposed, `H = sum y xᵀ` | **FAIL** — the exact case's translation moves 391.8 mm |
| remove the collinear refusal | **FAIL** — a line fits to 1.6e-13 mm, its rotation free |

⚠ A LOW RESIDUAL IS WHAT A WRONG ANSWER LOOKS LIKE HERE, in all three rows.
That is why none of them is gated on the residual alone.
"""

from std.math import abs, sqrt

from mojo_rl.math3d import Mat3 as Mat3Generic, Vec3 as Vec3Generic
from mojo_rl.vision.extrinsics import RigidFit, fit_rigid
from mojo_rl.vision.opencv import opencv_shim_available

comptime Vec3d = Vec3Generic[DType.float64]
comptime Mat3d = Mat3Generic[DType.float64]


def _rnd(mut s: UInt64) -> Float64:
    """A deterministic [-1, 1). ⚠ NOT `random`: a gate whose input changes per
    run cannot be re-run against a defect, which is the only use a gate has."""
    s = (s * 1103515245 + 12345) & 0x7FFFFFFF
    return Float64(s) / 1073741823.5 - 1.0


def _grid(nx: Int, ny: Int, nz: Int, step: Float64) -> List[Float64]:
    """A point cloud with real extent on ALL THREE axes.

    ⚠ THE Z EXTENT IS LOAD-BEARING FOR THE REFLECTION CASE: a mirror about the
    z plane acts as the IDENTITY on points that are all in that plane, so a
    flat cloud cannot tell a reflection from a rotation and the falsification
    would pass vacuously.
    """
    var out = List[Float64]()
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                out.append(Float64(i) * step - 0.1)
                out.append(Float64(j) * step + 0.05)
                out.append(Float64(k) * step + 0.25)
    return out^


def _transform(
    pts: List[Float64], rot: Mat3d, trans: Vec3d
) -> List[Float64]:
    var out = List[Float64]()
    for k in range(len(pts) // 3):
        var p = rot * Vec3d(pts[k * 3], pts[k * 3 + 1], pts[k * 3 + 2]) + trans
        out.append(p.x)
        out.append(p.y)
        out.append(p.z)
    return out^


def _max_rot_err(a: Mat3d, b: Mat3d) -> Float64:
    var e = 0.0
    var d = [
        a.m00 - b.m00, a.m01 - b.m01, a.m02 - b.m02,
        a.m10 - b.m10, a.m11 - b.m11, a.m12 - b.m12,
        a.m20 - b.m20, a.m21 - b.m21, a.m22 - b.m22,
    ]
    for i in range(9):
        if abs(d[i]) > e:
            e = abs(d[i])
    return e


def _truth() -> Tuple[Mat3d, Vec3d]:
    """One arbitrary but FIXED pose, not near any axis or any right angle."""
    var axis = Vec3d(0.37, -0.82, 0.44).normalized()
    return (Mat3d.rotation_axis(axis, 0.7391), Vec3d(0.31, -0.22, 0.455))


def main() raises:
    print("=" * 70)
    print("camera -> base extrinsics — the Kabsch fit")
    print("=" * 70)

    if not opencv_shim_available():
        raise Error(
            "the OpenCV shim is not built — `pixi run build-opencv`. This gate"
            " needs it for `svd_3x3` and REFUSES to skip: a gate that reports"
            " success when it ran nothing is the failure it is here to catch."
        )

    var checks = 0
    var failures = 0
    var t = _truth()
    var rot_true = t[0]
    var trans_true = t[1]

    # ── 1. noise-free recovery ──────────────────────────────────────────────
    var cam = _grid(3, 2, 2, 0.06)
    var base = _transform(cam, rot_true, trans_true)
    var fit = fit_rigid(cam, base)

    var rot_err = _max_rot_err(fit.rot, rot_true)
    var t_err_mm = (fit.trans - trans_true).length() * 1000.0
    checks += 1
    if rot_err > 1.0e-12:
        print("  FAIL: exact case, rotation off by", rot_err)
        failures += 1
    checks += 1
    if t_err_mm > 1.0e-9:
        print("  FAIL: exact case, translation off by", t_err_mm, "mm")
        failures += 1
    checks += 1
    if fit.rms_mm > 1.0e-9:
        print("  FAIL: exact case, residual", fit.rms_mm, "mm")
        failures += 1
    checks += 1
    if fit.n != len(cam) // 3:
        print("  FAIL: fit used", fit.n, "of", len(cam) // 3, "points")
        failures += 1
    print(
        "  exact:      ",
        fit.n,
        "poses, rot err",
        rot_err,
        " t err",
        t_err_mm,
        "mm, rms",
        fit.rms_mm,
        "mm",
    )

    # ── 2. the transform is what `apply` applies ────────────────────────────
    #
    # ⚠ NOT A TAUTOLOGY. `rms_mm` is computed inside the fit; this checks the
    # PUBLIC accessor against an independently transformed point, so a fit that
    # stored a transposed rotation would still report a perfect residual and
    # fail here.
    var probe = Vec3d(0.123, -0.456, 0.789)
    var want = rot_true * probe + trans_true
    var got = fit.apply(probe)
    checks += 1
    var apply_mm = (got - want).length() * 1000.0
    if apply_mm > 1.0e-9:
        print("  FAIL: apply() disagrees by", apply_mm, "mm")
        failures += 1
    print("  apply:       agrees to", apply_mm, "mm")

    # ── 3. ⚠⚠ THE REFLECTION, WHICH FITS BETTER THAN ANY ROTATION ───────────
    var mirrored = List[Float64]()
    for k in range(len(cam) // 3):
        mirrored.append(cam[k * 3])
        mirrored.append(cam[k * 3 + 1])
        mirrored.append(-cam[k * 3 + 2])
    var mfit = fit_rigid(cam, mirrored)
    var mdet = mfit.rot.determinant()
    checks += 1
    if abs(mdet - 1.0) > 1.0e-9:
        print("  FAIL: mirrored input produced det =", mdet, "— a REFLECTION")
        failures += 1
    checks += 1
    # A proper rotation CANNOT explain a mirror, so it must not pretend to.
    if mfit.rms_mm < 10.0:
        print(
            "  FAIL: mirrored input fitted to",
            mfit.rms_mm,
            "mm — a rotation cannot explain a reflection, so this is one",
        )
        failures += 1
    print(
        "  reflection:  det",
        mdet,
        " rms",
        mfit.rms_mm,
        "mm (must be large — a rotation cannot mirror)",
    )

    # ── 4. noise lands where noise should ───────────────────────────────────
    var seed = UInt64(20260909)
    var noisy = List[Float64]()
    for i in range(len(base)):
        noisy.append(base[i] + _rnd(seed) * 0.001)
    var nfit = fit_rigid(cam, noisy)
    checks += 1
    # 1 mm uniform per axis is 0.577 mm RMS per axis, 1.0 mm in 3D, and the fit
    # absorbs 6 of the 36 degrees of freedom — so a band, not a point.
    if nfit.rms_mm < 0.2 or nfit.rms_mm > 2.0:
        print("  FAIL: 1 mm noise gave rms", nfit.rms_mm, "mm, expected ~1")
        failures += 1
    checks += 1
    if _max_rot_err(nfit.rot, rot_true) > 0.02:
        print("  FAIL: 1 mm noise moved the rotation by", _max_rot_err(nfit.rot, rot_true))
        failures += 1
    checks += 1
    if nfit.max_mm < nfit.rms_mm:
        print("  FAIL: max_mm", nfit.max_mm, "below rms_mm", nfit.rms_mm)
        failures += 1
    print(
        "  1 mm noise:  rms",
        nfit.rms_mm,
        "mm, worst",
        nfit.max_mm,
        "mm at pose",
        nfit.worst,
    )

    # ── 5. spread reports the geometry that was actually sampled ────────────
    checks += 1
    if not (
        fit.spread_mm[0] >= fit.spread_mm[1]
        and fit.spread_mm[1] >= fit.spread_mm[2]
    ):
        print("  FAIL: spread is not descending")
        failures += 1
    # A flat sweep is LEGAL — three non-collinear points determine the
    # transform — but its third extent must read ~0 so nobody mistakes it for
    # a volume.
    var flat = List[Float64]()
    for i in range(4):
        for j in range(3):
            flat.append(Float64(i) * 0.07 - 0.1)
            flat.append(Float64(j) * 0.07 + 0.05)
            flat.append(0.3)
    var flat_base = _transform(flat, rot_true, trans_true)
    var ffit = fit_rigid(flat, flat_base)
    checks += 1
    if ffit.spread_mm[2] > 1.0e-6:
        print("  FAIL: coplanar poses reported a third extent", ffit.spread_mm[2])
        failures += 1
    checks += 1
    if ffit.rms_mm > 1.0e-9:
        print("  FAIL: coplanar poses must still fit exactly, got", ffit.rms_mm)
        failures += 1
    print(
        "  spread:      volume",
        fit.spread_mm[0], "/", fit.spread_mm[1], "/", fit.spread_mm[2],
        " mm;  plane third extent", ffit.spread_mm[2], "mm",
    )

    # ── 6. what must be REFUSED rather than answered ────────────────────────
    var line = List[Float64]()
    for i in range(6):
        line.append(Float64(i) * 0.04)
        line.append(0.1)
        line.append(0.3)
    var line_base = _transform(line, rot_true, trans_true)
    checks += 1
    var refused = False
    try:
        var bad = fit_rigid(line, line_base)
        print("  FAIL: a COLLINEAR set fitted to", bad.rms_mm, "mm instead of raising")
    except:
        refused = True
    if not refused:
        failures += 1

    checks += 1
    var refused_short = False
    try:
        var two = List[Float64](length=6, fill=0.0)
        _ = fit_rigid(two, two)
    except:
        refused_short = True
    if not refused_short:
        print("  FAIL: two correspondences did not raise")
        failures += 1

    checks += 1
    var refused_mismatch = False
    try:
        var short_base = List[Float64](length=len(base) - 3, fill=0.0)
        _ = fit_rigid(cam, short_base)
    except:
        refused_mismatch = True
    if not refused_mismatch:
        print("  FAIL: mismatched counts did not raise")
        failures += 1
    print("  refusals:    collinear, n < 3 and count mismatch all raise")

    print("-" * 70)
    if failures == 0:
        print("PASS —", checks, "checks")
    else:
        print("FAIL —", failures, "of", checks, "checks")
    print("=" * 70)
    # ⚠⚠ THE RAISE IS THE GATE. `scripts/run_tests.sh` reads the EXIT CODE and
    # nothing else, so a test that only prints "FAIL" is reported as a pass.
    if failures != 0:
        raise (
            String("extrinsics: ")
            + String(failures)
            + " of "
            + String(checks)
            + " checks failed"
        )
