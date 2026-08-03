"""Per-contact solver parameters vs MuJoCo — priority, solref mixing, condim.

WHY THIS EXISTS. Until 2026-08-03 every contact row in the engine read ONE
MODEL-LEVEL solref/solimp (`MODEL_META_IDX_SOLREF_CONTACT_*`), while the geom
record already carried per-geom `solref`/`solimp` that `fields_build` populated
on every build and NOTHING read. `<geom priority>` was not parsed at all, and
friction/condim were mixed with an unconditional elementwise max regardless of
it. So three separate things were wrong at once and none of them had a gate.

THE MODEL exercises all three branches of `engine_collision_driver.c:1426-1480`
in a single forward pass, which is the point — a gate that covers one branch
would pass while the other two are broken:

  * `ga` has `priority="1"`. MuJoCo takes its condim, solref, solimp AND
    friction WHOLESALE, with no mixing: dim 6, solref (-10000, -30), its own
    solimp, friction 0.7 — even though the floor it touches declares condim 3,
    solref (0.02, 1) and friction 1.0. This is dm_control's quadruped ball.
  * `gb` has equal priority and a POSITIVE solref, so solref and solimp are
    elementwise MEANS — (0.02, 1) with (0.006, 0.5) gives (0.013, 0.75) — while
    condim takes the max and friction the elementwise max.
  * `gc` has equal priority and a NEGATIVE solref, so solref is the elementwise
    MIN rather than a mean: (0.02, 1) with (-500, -7) gives (-500, -7). A
    direct solref therefore wins WITHOUT being averaged into nonsense. solimp
    still means.

⚠ THE MEAN IS A SPECIAL CASE. MuJoCo weights it `mix = solmix1/(solmix1 +
solmix2)`, which is 0.5 only because every geom defaults to `solmix = 1`. No
suite model sets one and `full_parser` REJECTS a non-default value rather than
silently substituting the mean — so this file gates the rule as implemented,
and the parser guards the case it does not implement. A probe of default-valued
models cannot distinguish the two; the source is what says which is which.

⚠ FRICTION IS 5-WIDE IN MuJoCo AND 3-WIDE HERE. `contact.friction` unpacks as
[slide, slide, spin, roll, roll] (`engine_collision_driver.c:1486-1490`), so
ours[FRICTION] is its [0], ours[FRICTION_SPIN] its [2], ours[FRICTION_ROLL] its
[3]. Comparing index-for-index would look like a spin/roll swap.

Both detection paths are gated, because they are separate implementations of
the mixing block — `broadphase_sap` carries its own copy in TWO pair loops (a
plane pass and a sweep), and an early version of this work rewired one of them
and not the other. The assert that caught that is the reason both are here.

Run: pixi run mojo run -I . tests/physics3d/test_contact_solparams_vs_mujoco.mojo
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from std.gpu.host import DeviceContext

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.fields import Data, Model
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.collision.contact_detection import detect_contacts
from mojo_rl.physics3d.collision.broadphase_sap import detect_contacts_sap
from mojo_rl.physics3d.gpu.constants import (
    CONTACT_SIZE,
    META_IDX_NUM_CONTACTS,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_FRICTION,
    CONTACT_IDX_FRICTION_SPIN,
    CONTACT_IDX_FRICTION_ROLL,
    CONTACT_IDX_CONDIM,
    CONTACT_IDX_SOLREF_0,
    CONTACT_IDX_SOLREF_1,
    CONTACT_IDX_SOLIMP_0,
    CONTACT_IDX_SOLIMP_1,
    CONTACT_IDX_SOLIMP_2,
    CONTACT_IDX_SOLIMP_3,
    CONTACT_IDX_SOLIMP_4,
)


comptime DTYPE = DType.float64

comptime SOLPAR_XML = """
<mujoco model="solparams">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1" solref="0.02 1" solimp="0.9 0.95 0.001 0.5 2" friction="1 0.005 0.0001" condim="3"/>
    <body name="a" pos="0 0 0.145">
      <joint name="ja" type="slide" axis="0 0 1"/>
      <geom name="ga" size=".15" priority="1" condim="6" friction=".7 .005 .005" solref="-10000 -30" solimp="0.8 0.99 0.002 0.4 3"/>
    </body>
    <body name="b" pos="1 0 0.145">
      <joint name="jb" type="slide" axis="0 0 1"/>
      <geom name="gb" size=".15" condim="4" friction="2 .01 .02" solref="0.006 0.5" solimp="0.7 0.99 0.003 0.6 4"/>
    </body>
    <body name="c" pos="2 0 0.145">
      <joint name="jc" type="slide" axis="0 0 1"/>
      <geom name="gc" size=".15" solref="-500 -7"/>
    </body>
  </worldbody>
</mujoco>
"""

comptime pp = parse_xml(SOLPAR_XML)
comptime PM = ModelDefFromXML[
    xml=SOLPAR_XML,
    nbody=pp.NBODY, njoint=pp.NJOINT, nq=pp.NQ, nv=pp.NV,
    ngeom=pp.NGEOM, nact=pp.NACT, ntex=pp.NTEX, nmat=pp.NMAT,
    nlight=pp.NLIGHT, ncam=pp.NCAM, nsite=pp.NSITE,
    max_tendon=pp.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=32,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=pp.TIMESTEP,
]

# Both sides read the same decimal literals and apply the same mean/min, so
# these are rounding budgets.
comptime TOL: Float64 = 1e-12

comptime Dat = Data[DTYPE, PM.NQ, PM.NV, PM.NBODY, PM.MAX_CONTACTS, PM.NSITE, 1]
comptime Mod = Model[
    DTYPE, PM.NV, PM.NBODY, PM.NJOINT, PM.NGEOM, PM.MAX_EQUALITY,
    PM.MAX_TENDON, PM.NSITE, PM.NEXCLUDE, 0,
]


def _build() raises -> Mod:
    var ctx = DeviceContext()
    var mf = Mod()
    PM.init_fields[DTYPE, 0](ctx, mf)
    return mf^


def _check(d: Dat, label: String) raises:
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(materialize[SOLPAR_XML]())
    var md = mujoco.MjData(m)
    mujoco.mj_forward(m, md)

    var n_ours = Int(d.meta.data[META_IDX_NUM_CONTACTS])
    var n_mj = Int(py=md.ncon)
    print("  [" + label + "] contacts: ours", n_ours, " MuJoCo", n_mj)
    assert_true(
        n_ours == 3 and n_mj == 3,
        label + ": expected exactly one contact per test body — the model is"
        " built so each of the three mixing branches produces one",
    )

    var worst = Float64(0)
    for c in range(n_ours):
        var b = c * CONTACT_SIZE
        var ba = Int(d.contacts.data[b + CONTACT_IDX_BODY_A])
        var bb = Int(d.contacts.data[b + CONTACT_IDX_BODY_B])
        var other = ba if (bb == 0 or bb == -1) else bb

        # MuJoCo's contact for the same body
        var k_match = -1
        for k in range(n_mj):
            var cc = md.contact[k]
            var q1 = Int(py=m.geom_bodyid[cc.geom1])
            var q2 = Int(py=m.geom_bodyid[cc.geom2])
            if q1 == other or q2 == other:
                k_match = k
                break
        assert_true(k_match >= 0, label + ": no MuJoCo contact for that body")
        var cc = md.contact[k_match]

        var names = [String("?"), String("ga"), String("gb"), String("gc")]
        var nm = names[other] if other < 4 else String("?")

        # condim
        var our_dim = Int(d.contacts.data[b + CONTACT_IDX_CONDIM])
        var mj_dim = Int(py=cc.dim)
        assert_true(
            our_dim == mj_dim,
            label + " " + nm + ": condim " + String(our_dim) + " != MuJoCo's "
            + String(mj_dim) + ". For `ga` this is the PRIORITY rule — its"
            " condim=6 must override the floor's 3 rather than being maxed.",
        )

        # solref / solimp, elementwise
        var pairs = [
            (CONTACT_IDX_SOLREF_0, Float64(py=cc.solref[0]), String("solref0")),
            (CONTACT_IDX_SOLREF_1, Float64(py=cc.solref[1]), String("solref1")),
            (CONTACT_IDX_SOLIMP_0, Float64(py=cc.solimp[0]), String("solimp0")),
            (CONTACT_IDX_SOLIMP_1, Float64(py=cc.solimp[1]), String("solimp1")),
            (CONTACT_IDX_SOLIMP_2, Float64(py=cc.solimp[2]), String("solimp2")),
            (CONTACT_IDX_SOLIMP_3, Float64(py=cc.solimp[3]), String("solimp3")),
            (CONTACT_IDX_SOLIMP_4, Float64(py=cc.solimp[4]), String("solimp4")),
        ]
        for p in pairs:
            var got = Float64(d.contacts.data[b + p[0]])
            var e = abs(got - p[1])
            if e > worst:
                worst = e
            assert_true(
                e <= TOL,
                label + " " + nm + " " + p[2] + ": ours " + String(got)
                + " != MuJoCo's " + String(p[1]),
            )

        # friction — ⚠ MuJoCo's is 5-wide [slide, slide, spin, roll, roll]
        var fr = [
            (CONTACT_IDX_FRICTION, 0, String("slide")),
            (CONTACT_IDX_FRICTION_SPIN, 2, String("spin")),
            (CONTACT_IDX_FRICTION_ROLL, 3, String("roll")),
        ]
        for f in fr:
            var got = Float64(d.contacts.data[b + f[0]])
            var want = Float64(py=cc.friction[f[1]])
            var e = abs(got - want)
            if e > worst:
                worst = e
            assert_true(
                e <= TOL,
                label + " " + nm + " friction " + f[2] + ": ours "
                + String(got) + " != MuJoCo's " + String(want),
            )

        print("    ", nm, " dim", our_dim, " solref [",
              Float64(d.contacts.data[b + CONTACT_IDX_SOLREF_0]),
              Float64(d.contacts.data[b + CONTACT_IDX_SOLREF_1]), "]")
    print("    worst elementwise error:", worst)


def test_contact_solparams_naive() raises:
    """`detect_contacts` — the all-pairs path."""
    print("--- contact solparams: detect_contacts ---")
    var mf = _build()
    var d = Dat()
    PM.reset_data(d)
    forward_kinematics["cpu"](d, mf)
    detect_contacts["cpu"](d, mf)
    _check(d, String("naive"))


def test_contact_solparams_sap() raises:
    """`detect_contacts_sap` — a SEPARATE implementation, in two pair loops."""
    print("--- contact solparams: detect_contacts_sap ---")
    var mf = _build()
    var d = Dat()
    PM.reset_data(d)
    forward_kinematics["cpu"](d, mf)
    detect_contacts_sap["cpu"](d, mf)
    _check(d, String("sap"))


def test_every_mixing_branch_is_exercised() raises:
    """Non-vacuity: the three contacts must genuinely differ.

    If the model ever degenerated so that all three took the same branch, every
    assertion above would still pass while testing one third of the rule. The
    branches are distinguishable by their solref alone.
    """
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(materialize[SOLPAR_XML]())
    var md = mujoco.MjData(m)
    mujoco.mj_forward(m, md)
    assert_true(Int(py=md.ncon) == 3, "expected three contacts")

    var sr0 = List[Float64]()
    var dims = List[Int]()
    for k in range(3):
        sr0.append(Float64(py=md.contact[k].solref[0]))
        dims.append(Int(py=md.contact[k].dim))
    print("--- non-vacuity: solref[0] per contact =", sr0[0], sr0[1], sr0[2])
    print("                 condim per contact    =", dims[0], dims[1], dims[2])

    # priority-wins (a large negative), positive-mean, negative-min.
    var n_neg = 0
    var n_pos = 0
    for k in range(3):
        if sr0[k] < 0.0:
            n_neg += 1
        else:
            n_pos += 1
    assert_true(
        n_neg == 2 and n_pos == 1,
        "the model no longer exercises both the direct and standard solref"
        " branches — it gates a third of the rule",
    )
    assert_true(
        dims[0] != dims[1] or dims[1] != dims[2],
        "every contact now has the same condim, so the priority rule and the"
        " max rule are indistinguishable here",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
