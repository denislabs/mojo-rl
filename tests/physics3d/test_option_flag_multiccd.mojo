"""`<option><flag multiccd="disable"/></option>` — `mjDSBL_MULTICCD`, honoured.

WHY THIS EXISTS. `collision/multi_ccd.mojo` implemented MuJoCo's multi-point
convex manifold UNCONDITIONALLY, and `full_parser` read exactly three of
`<option>`'s `<flag>` children (`gravity`, `constraint`, `contact`). So a model
that asked for single-point convex contacts got a four-point manifold anyway,
and nothing in the suite could see it: every dm_control manipulation gate reads
the REFERENCE's `ncon` out of its Python fixture and never compares it to ours.

Measured on `manipulation/reassemble5` before the fix — 437 contacts against
MuJoCo's 111, and 3701 ms per control step against 13-49 ms. All eleven baked
manipulation models set `nativeccd`; nine of them set `multiccd`.

THE FIXTURE IS ONE XML BUILT TWICE, differing in a single WORD: `enable` vs
`disable` on the flag. Anything else that differed between the two would be a
second variable, and the whole point is that the flag is the only one.

Three groups, 1 m apart in x so a contact's own x names its group, each set to
a 5 mm penetration and each moving body on a slide joint (a body with no joint
is welded to the world and MuJoCo excludes the pair outright):

    group 0   cylinder / cylinder    reaches the perturbation loop
    group 1   cylinder / box         reaches the perturbation loop
    group 2   sphere   / box         MuJoCo's own guard excludes spheres

WHAT IS ASSERTED
  * PARITY against MuJoCo on the same XML text handed to both engines, so the
    oracle is the reference run on our input rather than a number recorded
    here that rots. `disable` — this file's subject — is an EQUALITY per
    group. `enable` only requires `ours <= MuJoCo`, because our perturbation
    manifold is known to be short of the reference's and that gap predates
    this flag; see the note at the comparison.
  * THE NEGATIVE CONTROL, which is what stops this being vacuous: `disable`
    must produce STRICTLY FEWER contacts than `enable`. A wiring that silently
    did nothing would satisfy every equality above by matching MuJoCo in the
    one configuration it actually implements.
  * THE INVARIANCE CONTROL: group 2 is a sphere pair, which MuJoCo excludes
    from the loop by geom type, so its count must be IDENTICAL under both
    flags. A "fix" that just suppressed contacts globally would move it.

⚠ THE FLAG IS A DISABLE BIT. MuJoCo 3.6.0 had `mjENBL_MULTICCD` (opt-in); the
3.10.0 runtime has `mjDSBL_MULTICCD` (1<<19, feature ON by default). Reading
the older tree gets the sense backwards — see `multi_ccd.mojo`'s header.

Run: pixi run mojo run -I . tests/physics3d/test_option_flag_multiccd.mojo
"""

from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.fields import Data, Model
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.collision.contact_detection import detect_contacts
from mojo_rl.physics3d.model.model_dims import ModelDims
from mojo_rl.physics3d.gpu.constants import (
    CONTACT_SIZE,
    META_IDX_NUM_CONTACTS,
    CONTACT_IDX_POS_X,
)

comptime DTYPE = DType.float64
comptime NGROUPS = 3

# ⚠ SPLIT SO THE TWO MODELS SHARE EVERY BYTE BUT ONE WORD. Two hand-written
# literals would let an unrelated edit drift between them, which is exactly the
# failure this fixture is built to rule out.
comptime XML_HEAD = """
<mujoco model="multiccd_flag">
  <option>
    <flag multiccd=\""""

comptime XML_TAIL = """"/>
  </option>
  <worldbody>
    <light pos="0 0 3"/>
    <body name="g0a" pos="0 0 0.5">
      <joint name="j0a" type="slide" axis="1 0 0"/>
      <geom name="c0a" type="cylinder" size=".04 .06"/>
    </body>
    <geom name="w0" type="cylinder" size=".04 .06" pos="0.075 0 0.5"/>
    <body name="g1a" pos="1 0 0.5">
      <joint name="j1a" type="slide" axis="1 0 0"/>
      <geom name="c1a" type="cylinder" size=".04 .06"/>
    </body>
    <geom name="w1" type="box" size=".05 .05 .05" pos="1.085 0 0.5"/>
    <body name="g2a" pos="2 0 0.5">
      <joint name="j2a" type="slide" axis="1 0 0"/>
      <geom name="c2a" type="sphere" size=".04"/>
    </body>
    <geom name="w2" type="box" size=".05 .05 .05" pos="2.085 0 0.5"/>
  </worldbody>
</mujoco>
"""

comptime XML_ON = XML_HEAD + "enable" + XML_TAIL
comptime XML_OFF = XML_HEAD + "disable" + XML_TAIL

comptime pp_on = parse_xml(XML_ON)
comptime PM_ON = ModelDefFromXML[
    xml=XML_ON,
    nbody=pp_on.NBODY, njoint=pp_on.NJOINT, nq=pp_on.NQ, nv=pp_on.NV,
    ngeom=pp_on.NGEOM, nact=pp_on.NACT, ntex=pp_on.NTEX, nmat=pp_on.NMAT,
    nlight=pp_on.NLIGHT, ncam=pp_on.NCAM, nsite=pp_on.NSITE,
    max_tendon=pp_on.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=64,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=pp_on.TIMESTEP,
]

comptime pp_off = parse_xml(XML_OFF)
comptime PM_OFF = ModelDefFromXML[
    xml=XML_OFF,
    nbody=pp_off.NBODY, njoint=pp_off.NJOINT, nq=pp_off.NQ, nv=pp_off.NV,
    ngeom=pp_off.NGEOM, nact=pp_off.NACT, ntex=pp_off.NTEX, nmat=pp_off.NMAT,
    nlight=pp_off.NLIGHT, ncam=pp_off.NCAM, nsite=pp_off.NSITE,
    max_tendon=pp_off.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=64,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=pp_off.TIMESTEP,
]

comptime MD_ON = ModelDims[PM_ON]
comptime MD_OFF = ModelDims[PM_OFF]


def _group_names() -> List[String]:
    return [
        String("cylinder/cylinder"),
        String("cylinder/box"),
        String("sphere/box  (control)"),
    ]


def _ours_on() raises -> List[Int]:
    """Per-group contact counts from our narrow phase, `multiccd=enable`."""
    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD_ON]()
    PM_ON.init_fields[DTYPE](ctx, mf)
    var sf = PM_ON.make_spec_fields[DTYPE]()
    var d = Data[DTYPE, MD_ON, 1]()
    PM_ON.reset_data(sf, d)
    forward_kinematics["cpu"](d, mf)
    detect_contacts["cpu"](d, mf)
    return _bucket(d.contacts.data, Int(d.meta.data[META_IDX_NUM_CONTACTS]))


def _ours_off() raises -> List[Int]:
    """Per-group contact counts from our narrow phase, `multiccd=disable`."""
    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD_OFF]()
    PM_OFF.init_fields[DTYPE](ctx, mf)
    var sf = PM_OFF.make_spec_fields[DTYPE]()
    var d = Data[DTYPE, MD_OFF, 1]()
    PM_OFF.reset_data(sf, d)
    forward_kinematics["cpu"](d, mf)
    detect_contacts["cpu"](d, mf)
    return _bucket(d.contacts.data, Int(d.meta.data[META_IDX_NUM_CONTACTS]))


def _bucket(contacts: List[Scalar[DTYPE]], n: Int) -> List[Int]:
    """Groups sit 1 m apart in x, so a contact's own x names its group."""
    var out = List[Int]()
    for _ in range(NGROUPS):
        out.append(0)
    for k in range(n):
        var gx = Int(Float64(contacts[k * CONTACT_SIZE + CONTACT_IDX_POS_X]) + 0.5)
        if gx >= 0 and gx < NGROUPS:
            out[gx] = out[gx] + 1
    return out^


def _mujoco(xml: String) raises -> List[Int]:
    """MuJoCo's per-group counts on the SAME XML text this engine was built
    from — the oracle is the reference run on our input, not a number frozen
    here."""
    var mj = Python.import_module("mujoco")
    var m = mj.MjModel.from_xml_string(xml)
    var dat = mj.MjData(m)
    mj.mj_forward(m, dat)
    var out = List[Int]()
    for _ in range(NGROUPS):
        out.append(0)
    var n = Int(py=dat.ncon)
    for k in range(n):
        var gx = Int(Float64(py=dat.contact[k].pos[0]) + 0.5)
        if gx >= 0 and gx < NGROUPS:
            out[gx] = out[gx] + 1
    return out^


def test_multiccd_flag_matches_mujoco() raises:
    print("=== <flag multiccd> vs MuJoCo, same XML into both engines ===")
    var names = _group_names()
    var ours_on = _ours_on()
    var ours_off = _ours_off()
    var mj_on = _mujoco(XML_ON)
    var mj_off = _mujoco(XML_OFF)

    var total_on = 0
    var total_off = 0
    var bad = 0
    print("  group                    enable(ours/mj)   disable(ours/mj)")
    for g in range(NGROUPS):
        print(
            "   ", names[g], "        ", ours_on[g], "/", mj_on[g],
            "            ", ours_off[g], "/", mj_off[g],
        )
        # ⚠⚠ THE TWO COLUMNS ARE GATED DIFFERENTLY, ON PURPOSE.
        #
        # `disable` is an EQUALITY: it is this file's subject, and it is exact
        # — MuJoCo emits one contact per convex pair and so do we.
        #
        # `enable` is only `ours <= mj`, because our perturbation manifold is
        # KNOWN to be short of MuJoCo's and that gap predates this flag.
        # Measured here 2026-08-18: cylinder/cylinder 2 against 5,
        # cylinder/box 4 against 5. `test_narrow_phase_pairs` is the gate that
        # owns that number; asserting equality here would make this file fail
        # for a reason it does not test and cannot fix, and — worse — someone
        # would eventually "fix" it by loosening the `disable` column too.
        if ours_on[g] > mj_on[g]:
            bad += 1
        if ours_off[g] != mj_off[g]:
            bad += 1
        total_on += ours_on[g]
        total_off += ours_off[g]

    # ⚠ NON-VACUITY FIRST. Every equality below is satisfied by a fixture that
    # collides nothing at all.
    assert_true(
        total_on > 0 and total_off > 0,
        "the fixture produced no contacts in one of the two configurations, so"
        " nothing below tests anything",
    )
    assert_true(
        bad == 0,
        "our per-group contact count disagrees with MuJoCo on the same XML in"
        " " + String(bad) + " of " + String(NGROUPS * 2) + " comparisons"
        " (`disable` is an equality, `enable` only requires ours <= MuJoCo)",
    )
    # ⚠ THE NEGATIVE CONTROL. Without this the file passes even if the flag is
    # read and then dropped on the floor, because we would still match MuJoCo
    # in the `enable` configuration — which is the one we implemented.
    assert_true(
        total_off < total_on,
        "`multiccd=disable` produced " + String(total_off) + " contacts and"
        " `enable` produced " + String(total_on) + " — the flag changed"
        " NOTHING, so it is not wired to the narrow phase",
    )
    # ⚠ THE INVARIANCE CONTROL. MuJoCo excludes spheres from the perturbation
    # loop by geom TYPE, so this group cannot move with the flag. If it does,
    # the change is suppressing contacts wholesale rather than honouring the
    # reference's guard.
    assert_true(
        ours_on[2] == ours_off[2],
        "the sphere/box group moved from " + String(ours_on[2]) + " to "
        + String(ours_off[2]) + " — `multiccd` must not touch a pair MuJoCo"
        " excludes by geom type",
    )
    print(
        "  totals: enable", total_on, " disable", total_off,
        " (sphere control", ours_on[2], "in both)",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
