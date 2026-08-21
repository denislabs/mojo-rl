"""Three ways a body's mass and inertia came out wrong, and MuJoCo's answers.

    pixi run mojo run -I . tests/physics3d/test_body_inertia_from_geoms_vs_mujoco.mojo

All three were invisible in the step-1 sweep until the models they hit were the
last ones left, and each of them made a body HEAVIER than the reference, never
lighter — the direction a fallback default always errs in.

(1) `<geom density>` NEVER REACHED A MESH. `geom_volume` returns 0 for a mesh,
    so the parser leaves `mass = -1` — "compute me from density times volume" —
    and the mesh branch of `_geom_mass_and_inertia` then multiplied by
    `MJ_DEFAULT_DENSITY` (1000), throwing the geom's own density away on the
    one geom type that cannot state a mass any other way.
    ⚠⚠ `density="0"` IS THE COMMON CASE. Menagerie's
    `<default class="visual"><geom density="0"/>` is how a render-only shell is
    told to weigh nothing, and every one of them was charged 1000 kg/m^3.
    trs_so_arm100's `Base` carries the same mesh twice, once visual and once
    collision, and weighed 1.168357 kg against MuJoCo's 0.562466 — the visual
    copy again, plus the motor shell.

(2) A FITTED PRIMITIVE WAS WEIGHED BEFORE IT HAD A SIZE. `mjCMesh::FitGeom`
    runs in the compiler, so `<geom mesh="base_link" class="collision"/>` whose
    class says `type="capsule"` reaches this parser with NO size and keeps
    `GeomData`'s placeholder 0.5. `density * volume` on a half-metre capsule is
    `pi*0.25*(4*0.5/3 + 2*0.5) = 1.309 m^3`, and arx_l5's base_link weighed
    **1308.997 kg** against MuJoCo's 0.128420.
    ⚠ rby1's 49 fitted spheres escaped ONLY by luck: every one of their bodies
    declares an explicit `<inertial>`, so the geom-derived pass skipped them
    and the 523 kg apiece never landed.

(3) AN `<inertial>` WITHOUT `diaginertia` KEPT A DEFAULT OF 0.01.
    `<inertial pos="0 0 0" mass="0"/>` is a legal and common way to spell a
    massless frame body; MuJoCo compiles it to `body_inertia [0, 0, 0]`. We
    only wrote the inertia when `diaginertia` was present, so the constructor
    default stood — 0.01 kg m^2 on all three axes, which at rby1's wrist is
    comparable to the whole forearm.
    ⚠ THE SAME DEFECT LOOKED LIKE TWO DIFFERENT ROBOTS. rby1 v1.3 has three
    such bodies (`EE_GR_TF_L/R` on the arms, `NECK_0`) and diverged 2.1e-03;
    v1.2 has only `NECK_0`, on a head nothing drives, and sat at 2.1e-05.

MEASURED CONSEQUENCE, worst |d(qpos)| against MuJoCo 3.10.0, one step:

    rainbow_robotics_rby1/scene_rby1m_1.3   2.142e-03 -> 2.043e-13
    wonik_allegro/scene_right               2.033e-04 -> 7.806e-18
    wonik_allegro/scene_left                2.033e-04 -> 7.373e-18
    umi_gripper/scene                       2.401e-05 -> 1.084e-19
    rainbow_robotics_rby1/scene_rby1a_1.2   2.860e-05 -> 2.047e-16

and the count of Menagerie scenes whose body masses disagree with the runtime
at all fell from 11 to 3.
"""

from std.math import abs
from max.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, read_model_source,
)
from mojo_rl.physics3d.fields import Model, DynDims
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE, BODY_IDX_MASS, BODY_IDX_IXX, BODY_IDX_IYY, BODY_IDX_IZZ,
)

comptime DT = DType.float64

comptime SO100 = String(
    "references/mujoco_menagerie-main/trs_so_arm100/scene.xml"
)
comptime ARX = String("references/mujoco_menagerie-main/arx_l5/scene.xml")
comptime RBY13 = String(
    "references/mujoco_menagerie-main/rainbow_robotics_rby1/"
    "scene_rby1m_1.3.xml"
)
comptime ALLEGRO = String(
    "references/mujoco_menagerie-main/wonik_allegro/scene_right.xml"
)
# ⚠ THE NEGATIVE CONTROL. g1 was already exact; if a "fix" to the density
# arithmetic broke the ordinary path, its total mass moves.
comptime G1 = String("references/mujoco_menagerie-main/unitree_g1/scene.xml")

# MuJoCo 3.10.0, `sum(m.body_mass)`.
comptime MJ_TOTAL_SO100 = 1.1714309911322078
comptime MJ_TOTAL_ARX = 3.595660380050618
comptime MJ_TOTAL_RBY13 = 137.47926800000002
comptime MJ_TOTAL_ALLEGRO = 0.7450265130483191
comptime MJ_TOTAL_G1 = 33.341142000000005

# `m.body_mass[1]` / `m.body_inertia[1]` for the two bodies each defect hit.
comptime MJ_SO100_BASE_M = 0.562465591132208
comptime MJ_SO100_BASE_IXX = 0.00061498630577883
comptime MJ_ARX_BASE_M = 0.1284201800506184
comptime MJ_ARX_BASE_IXX = 6.382318726491087e-05
# rby1 v1.3's three massless frame bodies.
comptime RBY13_NECK0 = 12
comptime RBY13_EE_R = 20
comptime RBY13_EE_L = 30


def _bodies(path: String) raises -> List[Float64]:
    """`[mass, ixx, iyy, izz]` per body, flat, straight off the built model."""
    var src = read_model_source(path)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    var verts = 32768
    var dims = dims_from_flat(fmd, max_contacts=16, nmesh_verts=verts)
    var m = Model[DT, DynDims](dims)
    var tries = 0
    while True:
        try:
            build_model_runtime[DT](fmd, dims, m)
            break
        except e:
            if String(e).find("mesh vertex capacity") == -1 or tries > 24:
                raise e
            tries += 1
            verts = verts * 2
            dims = dims_from_flat(fmd, max_contacts=16, nmesh_verts=verts)
            m = Model[DT, DynDims](dims)
    var out = List[Float64]()
    for b in range(dims.get_nbody()):
        var o = b * MODEL_BODY_SIZE
        out.append(Float64(m.bodies.data[o + BODY_IDX_MASS]))
        out.append(Float64(m.bodies.data[o + BODY_IDX_IXX]))
        out.append(Float64(m.bodies.data[o + BODY_IDX_IYY]))
        out.append(Float64(m.bodies.data[o + BODY_IDX_IZZ]))
    return out^


def _total(v: List[Float64]) -> Float64:
    var t = 0.0
    for i in range(0, len(v), 4):
        t += v[i]
    return t


def test_geom_density_reaches_a_mesh() raises:
    """trs_so_arm100: the same mesh visual + collision, `density="0"` on one."""
    print("=== <geom density> on a mesh — trs_so_arm100 ===")
    var v = _bodies(SO100)
    assert_true(
        len(v) == 8 * 4,
        "so_arm100 has 8 bodies; got " + String(len(v) // 4),
    )
    print("  Base mass", v[4], " MuJoCo", MJ_SO100_BASE_M)
    print("  Base ixx ", v[5], " MuJoCo", MJ_SO100_BASE_IXX)
    assert_true(
        abs(v[4] - MJ_SO100_BASE_M) < 1e-12,
        "Base weighs " + String(v[4]) + " against MuJoCo's "
        + String(MJ_SO100_BASE_M) + ". Its `Base` mesh appears twice — once"
        " in a `density=\"0\"` visual class, once as collision — so a value"
        " near twice MuJoCo's means the visual copy was charged 1000 kg/m^3.",
    )
    assert_true(
        abs(v[5] - MJ_SO100_BASE_IXX) < 1e-15,
        "Base ixx is " + String(v[5]) + " against MuJoCo's "
        + String(MJ_SO100_BASE_IXX) + " — the moments scale with the mass"
        " that was wrong.",
    )
    var t = _total(v)
    print("  total mass", t, " MuJoCo", MJ_TOTAL_SO100)
    assert_true(
        abs(t - MJ_TOTAL_SO100) < 1e-12,
        "total mass " + String(t) + " vs MuJoCo " + String(MJ_TOTAL_SO100),
    )
    print("  PASS")


def test_a_fitted_primitive_is_weighed_after_it_is_sized() raises:
    """arx_l5: a `class="collision"` capsule fitted to a mesh."""
    print("=== a fitted capsule — arx_l5 ===")
    var v = _bodies(ARX)
    assert_true(
        len(v) == 10 * 4, "arx_l5 has 10 bodies; got " + String(len(v) // 4)
    )
    print("  base_link mass", v[4], " MuJoCo", MJ_ARX_BASE_M)
    print("  base_link ixx ", v[5], " MuJoCo", MJ_ARX_BASE_IXX)
    # ⚠ THE FAILING VALUE IS NAMED. 1308.997 kg is `1000 * pi*0.25*(4*0.5/3 +
    # 2*0.5)`, the volume of a capsule at the placeholder half-metre size, and
    # seeing it in the message says "sized after fitting" went missing rather
    # than "the capsule formula is wrong".
    assert_true(
        abs(v[4] - MJ_ARX_BASE_M) < 1e-12,
        "base_link weighs " + String(v[4]) + " against MuJoCo's "
        + String(MJ_ARX_BASE_M) + ". A value near 1308.997 is the placeholder"
        " 0.5 m capsule, i.e. the mass was computed before `FitGeom` ran.",
    )
    assert_true(
        abs(v[5] - MJ_ARX_BASE_IXX) < 1e-15,
        "base_link ixx is " + String(v[5]) + " vs MuJoCo "
        + String(MJ_ARX_BASE_IXX),
    )
    var t = _total(v)
    print("  total mass", t, " MuJoCo", MJ_TOTAL_ARX)
    assert_true(
        abs(t - MJ_TOTAL_ARX) < 1e-12,
        "total mass " + String(t) + " vs MuJoCo " + String(MJ_TOTAL_ARX),
    )
    print("  PASS")


def test_an_inertial_without_diaginertia_is_zero() raises:
    """rby1 v1.3: `<inertial pos="0 0 0" mass="0"/>` on three frame bodies."""
    print("=== <inertial> with no diaginertia — rby1 v1.3 ===")
    var v = _bodies(RBY13)
    assert_true(
        len(v) == 35 * 4, "rby1 v1.3 has 35 bodies; got " + String(len(v) // 4)
    )
    var ids: List[Int] = [RBY13_NECK0, RBY13_EE_R, RBY13_EE_L]
    var names: List[String] = [
        String("NECK_0"), String("EE_GR_TF_R"), String("EE_GR_TF_L"),
    ]
    for k in range(len(ids)):
        var o = ids[k] * 4
        print(
            "  ", names[k], " mass", v[o], " inertia", v[o + 1], v[o + 2],
            v[o + 3],
        )
        assert_true(
            v[o] == 0.0 and v[o + 1] == 0.0 and v[o + 2] == 0.0
            and v[o + 3] == 0.0,
            names[k] + " has mass " + String(v[o]) + " and inertia ("
            + String(v[o + 1]) + ", " + String(v[o + 2]) + ", "
            + String(v[o + 3]) + "); MuJoCo compiles all four to 0. A 0.01"
            " here is `BodyData`'s constructor default surviving an"
            " `<inertial>` that gave a mass and no `diaginertia`.",
        )
    var t = _total(v)
    print("  total mass", t, " MuJoCo", MJ_TOTAL_RBY13)
    assert_true(
        abs(t - MJ_TOTAL_RBY13) < 1e-9,
        "total mass " + String(t) + " vs MuJoCo " + String(MJ_TOTAL_RBY13),
    )
    print("  PASS")


def test_totals_on_two_more_models() raises:
    """One more that moved, and one that must not have.

    ⚠ ALLEGRO IS NOT REDUNDANT: its hand is built from `density="0"` visual
    meshes over collision primitives, a different mix from so_arm100's, and it
    moved 2.033e-04 -> 7.806e-18 on the sweep.

    ⚠ G1 IS THE NEGATIVE CONTROL. Every one of its bodies states an explicit
    `<inertial>` with a `diaginertia`, so none of the three defects can reach
    it. If its total mass moves, the fix broke the ordinary path.
    """
    print("=== totals: wonik_allegro (moved) and unitree_g1 (must not) ===")
    var ta = _total(_bodies(ALLEGRO))
    print("  allegro", ta, " MuJoCo", MJ_TOTAL_ALLEGRO)
    assert_true(
        abs(ta - MJ_TOTAL_ALLEGRO) < 1e-12,
        "wonik_allegro total mass " + String(ta) + " vs MuJoCo "
        + String(MJ_TOTAL_ALLEGRO),
    )
    var tg = _total(_bodies(G1))
    print("  g1     ", tg, " MuJoCo", MJ_TOTAL_G1)
    assert_true(
        abs(tg - MJ_TOTAL_G1) < 1e-9,
        "unitree_g1 total mass " + String(tg) + " vs MuJoCo "
        + String(MJ_TOTAL_G1) + " — g1 declares every inertia explicitly and"
        " none of these three defects can touch it, so a mismatch here means"
        " the ordinary geom-derived path regressed.",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
