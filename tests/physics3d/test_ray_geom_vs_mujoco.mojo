"""`ray/geom.mojo` vs `mju_rayGeom` — randomised sweep over the six pure geoms.

    pixi run mojo run -I . tests/physics3d/test_ray_geom_vs_mujoco.mojo

Both sides get the SAME pose, size and ray, generated here by a fixed LCG and
handed to MuJoCo through its own binding, so a mismatch is our arithmetic and
nothing else. No model, no `Data`, no parser — `ray/geom.mojo` takes scalars.

⚠⚠ VACUITY IS THE FAILURE MODE THIS GATE IS BUILT AGAINST. A ray sampled
uniformly in a big box misses almost everything, and "0 mismatches over 200
rays that all missed" is the same output as a correct implementation. So every
type prints **hits / cases** beside the residual, and asserts a hit FRACTION —
the sampler aims rays at a point jittered around the geom precisely so the hit
rate stays high enough to mean something. A drop in the hit count is a
regression in the TEST, and it is checked as one.

⚠ THE SECOND FAILURE MODE IS A COUNT DISAGREEMENT, NOT A VALUE ONE. When one
side reports a hit and the other a miss, there is no residual to average — the
difference hides completely in a `max |dt|` taken over agreed rows. Those are
counted separately as `split` and asserted at zero.
[[feedback_compare_row_sets_before_reading_buckets]]

WHAT THIS GATE WAS PROVEN ABLE TO FAIL
======================================
Five defects were injected into `ray/geom.mojo` one at a time and the sweep
re-run. It is recorded because two of them are invisible to the obvious
version of this test:

  injected defect                                  caught by      |dt|
  ----------------------------------------------   ------------   --------
  capsule caps take ray_quad's BEST root, not both  10 splits      UNCHANGED
  ray_map rotates forward, not inverse              724 splits     9.8
  plane drops the `size <= 0` infinite branch       220 splits     UNCHANGED
  ellipsoid normal = centre direction, not gradient |dnormal| 1.07 UNCHANGED
  cylinder flat-cap normal loses its sign           |dnormal| 2.00 UNCHANGED

⚠⚠ FOUR OF THE FIVE LEAVE `max |dt|` EXACTLY AS IT WAS. A gate that compared
only the distance over rows both sides agree on would have passed all four.
The `split` counter and the normal column are not thoroughness — they are the
only things that see two thirds of these.

⚠ `geomtype` TRANSLATION IS NOT OPTIONAL. `mjtGeom` and this tree's enum have
never matched — `mjtGeom` is PLANE 0, HFIELD 1, SPHERE 2, CAPSULE 3,
ELLIPSOID 4, CYLINDER 5, BOX 6, MESH 7, while `physics3d/constants.mojo` is
PLANE 0, SPHERE 1, CAPSULE 2, BOX 3, CYLINDER 4, MESH 5, ELLIPSOID 6,
HFIELD 7. Passing ours to MuJoCo silently compares a sphere against a
heightfield. `_MJ_TYPE` is the map and `test_geom_enum_translation` pins it by
NAME so a future insertion into either enum fails loudly.
"""

from std.math import abs, sqrt
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.math3d import Vec3 as Vec3Generic, Quat as QuatGeneric
from mojo_rl.physics3d.constants import (
    GEOM_PLANE,
    GEOM_SPHERE,
    GEOM_CAPSULE,
    GEOM_BOX,
    GEOM_CYLINDER,
    GEOM_ELLIPSOID,
)
from mojo_rl.physics3d.ray import ray_geom

comptime Vec3 = Vec3Generic[DType.float64]
comptime Quat = QuatGeneric[DType.float64]

comptime NCASE = 1200
"""⚠ SIZED BY FALSIFICATION, not by taste. At 300 the sweep caught the
best-root-only capsule defect exactly ONCE out of 236 hits — a real catch, but
one that a different seed would lose. The defect appears on ~1.8% of capsule
hits (measured by brute force in numpy over 400k rays), so the sample count is
set to put ~8 differing rays in the sweep rather than ~1. Lowering it back is
how this gate quietly stops being able to fail."""


struct Lcg(Copyable, Movable):
    """Numerical Recipes LCG. Deterministic on purpose: a sweep whose inputs
    change per run cannot be bisected, and a residual nobody can reproduce is
    not a measurement."""

    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def u01(mut self) -> Float64:
        self.s = self.s * 1664525 + 1013904223
        return Float64((self.s >> 16) & 0xFFFFFFF) / Float64(0x10000000)

    def sym(mut self, a: Float64) -> Float64:
        return (self.u01() * 2.0 - 1.0) * a


def _mj_type(ours: Int) -> Int:
    """This tree's geom enum -> `mjtGeom`."""
    if ours == GEOM_PLANE:
        return 0  # mjGEOM_PLANE
    if ours == GEOM_SPHERE:
        return 2  # mjGEOM_SPHERE
    if ours == GEOM_CAPSULE:
        return 3  # mjGEOM_CAPSULE
    if ours == GEOM_ELLIPSOID:
        return 4  # mjGEOM_ELLIPSOID
    if ours == GEOM_CYLINDER:
        return 5  # mjGEOM_CYLINDER
    if ours == GEOM_BOX:
        return 6  # mjGEOM_BOX
    return -1


def _pad(s: String, n: Int) -> String:
    var out = s
    while out.byte_length() < n:
        out += " "
    return out


def _name(ours: Int) -> String:
    if ours == GEOM_PLANE:
        return "plane"
    if ours == GEOM_SPHERE:
        return "sphere"
    if ours == GEOM_CAPSULE:
        return "capsule"
    if ours == GEOM_ELLIPSOID:
        return "ellipsoid"
    if ours == GEOM_CYLINDER:
        return "cylinder"
    return "box"


def test_geom_enum_translation() raises:
    """`_MJ_TYPE` by NAME, so an insertion into either enum fails loudly."""
    var mujoco = Python.import_module("mujoco")
    var pairs = List[Tuple[Int, String]]()
    pairs.append((GEOM_PLANE, String("mjGEOM_PLANE")))
    pairs.append((GEOM_SPHERE, String("mjGEOM_SPHERE")))
    pairs.append((GEOM_CAPSULE, String("mjGEOM_CAPSULE")))
    pairs.append((GEOM_ELLIPSOID, String("mjGEOM_ELLIPSOID")))
    pairs.append((GEOM_CYLINDER, String("mjGEOM_CYLINDER")))
    pairs.append((GEOM_BOX, String("mjGEOM_BOX")))
    for p in pairs:
        var want = Int(py=mujoco.mjtGeom.__getattr__(p[1]))
        var got = _mj_type(p[0])
        assert_true(
            got == want,
            "ours " + String(p[0]) + " maps to " + String(got) + " but "
            + p[1] + " is " + String(want),
        )
    # And the thing that makes the translation necessary in the first place.
    assert_true(
        _mj_type(GEOM_SPHERE) != GEOM_SPHERE,
        "the two enums now agree on SPHERE — if they were unified on purpose,"
        " delete `_mj_type`; if by accident, this is the warning",
    )
    print("  enum translation OK (and the two enums still differ)")


def _sweep(gtype: Int) raises -> Tuple[Int, Int, Int, Float64, Float64]:
    """Returns (hits, split, cases, worst |dt|, worst |dnormal|)."""
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")

    var a_pos = np.zeros(3)
    var a_mat = np.zeros(9)
    var a_size = np.zeros(3)
    var a_pnt = np.zeros(3)
    var a_vec = np.zeros(3)
    var a_nrm = np.zeros(3)
    var a_quat = np.zeros(4)

    var rng = Lcg(0x9E3779B9 + UInt64(gtype) * 2654435761)
    var hits = 0
    var split = 0
    var worst_t = 0.0
    var worst_n = 0.0

    for _ in range(NCASE):
        var pos = Vec3(rng.sym(0.5), rng.sym(0.5), rng.sym(0.5))

        # Random orientation via a normalised random quaternion. ⚠ Not
        # axis-angle: a uniform axis with a uniform angle over-samples small
        # rotations, and a plane or a box only shows an orientation bug when
        # it is actually turned.
        var q = Quat(rng.sym(1.0), rng.sym(1.0), rng.sym(1.0), rng.sym(1.0))
        var qn = sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z)
        if qn < 1e-6:
            q = Quat(1.0, 0.0, 0.0, 0.0)
        else:
            q = Quat(q.w / qn, q.x / qn, q.y / qn, q.z / qn)

        var size = Vec3(
            0.05 + rng.u01() * 0.4,
            0.05 + rng.u01() * 0.4,
            0.05 + rng.u01() * 0.4,
        )
        # A plane's size[0]/size[1] are RENDERED half-extents and 0 means
        # infinite; sample both regimes so the `size <= 0` branch is covered.
        if gtype == GEOM_PLANE and rng.u01() < 0.5:
            size = Vec3(0.0, 0.0, size.z)

        # ── THREE RAY FAMILIES, and the second one is not optional ──────
        # ⚠⚠ THE FIRST FAMILY ALONE IS VACUOUS ON THE CAPS. A sampler that
        # only aims at a jittered centre was MEASURED to miss a real defect:
        # rewriting the capsule's cap loops to take `ray_quad`'s best root
        # instead of both roots changed NOTHING — same 104/300 hits, same
        # 1.24e-14, same 0 splits. Every hit it generated landed on the round
        # side or on a cap where the two roots agree. The defect only shows on
        # a ray travelling near the geom's OWN AXIS, where the round-side
        # quadratic degenerates (`a = lvec.x^2 + lvec.y^2 -> 0`, rejected by
        # `a < mjMINVAL`) and the caps alone decide the answer — which is
        # exactly what `studio/pick.mojo::_hit_capsule` documents about
        # itself, from the other direction.
        var axis = q.rotate_vec(Vec3(0.0, 0.0, 1.0))
        var pick = rng.u01()
        var eye: Vec3
        var aim: Vec3
        if pick < 0.25:
            # (4) ORIGIN INSIDE THE GEOM, pointing out. ⚠⚠ THIS IS THE FAMILY
            # THAT CATCHES THE CAP DEFECT, and it took two failed attempts to
            # find. Families (1)-(3) all start well outside, where the near
            # root of a cap sphere is the answer and `ray_quad`'s best root
            # equals the right one. From INSIDE, the near root is negative or
            # lands on the wrong half, so only the FAR root is a real hit —
            # take the best root alone and the ray reports MISS where MuJoCo
            # reports a distance. Measured: best-root-only differs on ~1.8% of
            # capsule hits, and every differing case has the origin inside or
            # just below the body.
            #
            # It is not a capsule curiosity either: a ray starting inside a
            # geom is what a camera mounted inside a robot's shell and a
            # rangefinder sitting flush with a body both produce.
            eye = pos + Vec3(
                rng.sym(size.x), rng.sym(size.y), rng.sym(size.z)
            ) * 0.7
            aim = eye + Vec3(rng.sym(1.0), rng.sym(1.0), rng.sym(1.0))
        elif pick < 0.5:
            # (1) general: aimed at a jittered centre from anywhere.
            eye = Vec3(rng.sym(3.0), rng.sym(3.0), rng.sym(3.0))
            aim = pos + Vec3(rng.sym(0.6), rng.sym(0.6), rng.sym(0.6))
        elif pick < 0.8:
            # (2) ALONG THE AXIS, both directions, with lateral jitter small
            # relative to the radius so the ray stays inside the cap.
            var sgn = 1.0 if rng.u01() < 0.5 else -1.0
            var lat = Vec3(rng.sym(0.05), rng.sym(0.05), rng.sym(0.05))
            eye = pos + axis * (sgn * (1.5 + rng.u01() * 2.0)) + lat
            aim = pos + Vec3(rng.sym(0.02), rng.sym(0.02), rng.sym(0.02))
        else:
            # (3) GRAZING: aimed a hair outside the silhouette, so the accept
            # tests at the rim and the face borders are the deciding branch.
            var r = 0.5 * (size.x + size.y)
            eye = Vec3(rng.sym(3.0), rng.sym(3.0), rng.sym(3.0))
            aim = pos + Vec3(rng.sym(1.0), rng.sym(1.0), rng.sym(1.0)).normalized() * (
                r * (0.9 + rng.u01() * 0.25)
            )
        var vec = aim - eye
        # ⚠ Deliberately NOT normalised on half the cases: `x` is in units of
        # `|vec|`, and a routine that normalises internally would agree on
        # every unit ray and be wrong here.
        if rng.u01() < 0.5:
            vec = vec * (0.3 + rng.u01() * 2.0)

        var ours = ray_geom[DType.float64](pos, q, size, eye, vec, gtype)

        a_pos[0] = pos.x
        a_pos[1] = pos.y
        a_pos[2] = pos.z
        a_size[0] = size.x
        a_size[1] = size.y
        a_size[2] = size.z
        a_pnt[0] = eye.x
        a_pnt[1] = eye.y
        a_pnt[2] = eye.z
        a_vec[0] = vec.x
        a_vec[1] = vec.y
        a_vec[2] = vec.z
        a_quat[0] = q.w
        a_quat[1] = q.x
        a_quat[2] = q.y
        a_quat[3] = q.z
        _ = mujoco.mju_quat2Mat(a_mat, a_quat)
        var t_mj = Float64(
            py=mujoco.mju_rayGeom(
                a_pos, a_mat, a_size, a_pnt, a_vec, _mj_type(gtype), a_nrm
            )
        )

        var t_ours = Float64(ours[0])
        var hit_ours = t_ours >= 0.0
        var hit_mj = t_mj >= 0.0
        if hit_ours != hit_mj:
            split += 1
            continue
        if not hit_mj:
            continue

        hits += 1
        worst_t = max(worst_t, abs(t_ours - t_mj))
        var n = ours[1]
        worst_n = max(worst_n, abs(Float64(n.x) - Float64(py=a_nrm[0])))
        worst_n = max(worst_n, abs(Float64(n.y) - Float64(py=a_nrm[1])))
        worst_n = max(worst_n, abs(Float64(n.z) - Float64(py=a_nrm[2])))

    return (hits, split, NCASE, worst_t, worst_n)


def test_ray_geom_vs_mujoco() raises:
    var types = List[Int]()
    types.append(GEOM_PLANE)
    types.append(GEOM_SPHERE)
    types.append(GEOM_CAPSULE)
    types.append(GEOM_ELLIPSOID)
    types.append(GEOM_CYLINDER)
    types.append(GEOM_BOX)

    var total_hits = 0
    var total_split = 0
    var worst_t_all = 0.0
    var worst_n_all = 0.0

    print(
        "    geom          hits/cases    worst |dt|      worst |dnormal|"
        "   split"
    )
    for g in types:
        var r = _sweep(g)
        var hits = r[0]
        var split = r[1]
        var cases = r[2]
        total_hits += hits
        total_split += split
        worst_t_all = max(worst_t_all, r[3])
        worst_n_all = max(worst_n_all, r[4])
        print(
            "    " + _pad(_name(g), 12),
            String(hits) + "/" + String(cases),
            "     ",
            r[3],
            "  ",
            r[4],
            "  ",
            split,
        )
        # Per-type non-vacuity: a type that stopped hitting proves nothing,
        # however green the residual next to it looks.
        assert_true(
            hits > cases // 8,
            _name(g) + " hit only " + String(hits) + " of " + String(cases)
            + " rays — this row is vacuous, fix the SAMPLER",
        )

    print("  total hits", total_hits, " splits", total_split)
    assert_true(
        total_split == 0,
        String(total_split)
        + " rays where one side hit and the other missed — a count"
        " disagreement, which no residual over agreed rows can show",
    )
    # The tolerance is the cost of the one documented deviation: we rotate by
    # QUATERNION where the reference multiplies by a 3x3. Same map, different
    # roundings. Anything materially above this is arithmetic, not spelling.
    assert_true(
        worst_t_all < 1e-9,
        "worst |dt| " + String(worst_t_all),
    )
    assert_true(
        worst_n_all < 1e-9,
        "worst |dnormal| " + String(worst_n_all),
    )


def test_mesh_and_hfield_return_no_hit_not_a_wrong_hit() raises:
    """The two types `mju_rayGeom` refuses, and what we do instead.

    The reference calls `mjERROR` on MESH and HFIELD because a "pure geom"
    signature has nowhere to put vertices or an elevation grid. We return
    NO HIT. That is the safer half of the trade and the more dangerous half is
    that it is SILENT — a caller that forgets to special-case a mesh sees an
    empty scene rather than a crash. Pinned here so the behaviour is a
    decision on record rather than a discovery.
    """
    var q = Quat(1.0, 0.0, 0.0, 0.0)
    var pos = Vec3(0.0, 0.0, 0.0)
    var size = Vec3(0.5, 0.5, 0.5)
    # A ray straight through the origin: every implemented type hits this.
    var eye = Vec3(-5.0, 0.0, 0.0)
    var vec = Vec3(1.0, 0.0, 0.0)

    var sphere = ray_geom[DType.float64](pos, q, size, eye, vec, GEOM_SPHERE)
    assert_true(
        Float64(sphere[0]) > 0.0,
        "the control ray misses a sphere at the origin — this test is broken,"
        " not the dispatch",
    )
    for g in [5, 7]:  # GEOM_MESH, GEOM_HFIELD
        var r = ray_geom[DType.float64](pos, q, size, eye, vec, g)
        assert_true(
            Float64(r[0]) == -1.0,
            "geom type " + String(g) + " returned " + String(Float64(r[0]))
            + " — it must be NO HIT until its own routine lands, never a"
            " plausible-looking distance from a fallback",
        )
    print("  mesh/hfield return NO HIT (control: sphere hits at",
          Float64(sphere[0]), ")")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
