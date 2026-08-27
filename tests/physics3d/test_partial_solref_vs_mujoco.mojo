"""A one-value `solref` must keep its dampratio, not zero it.

    pixi run mojo run -I . tests/physics3d/test_partial_solref_vs_mujoco.mojo

WHAT WENT WRONG. Every `solref` reader took the whole vector through
`_parse_vec3`, which returns 0 for any component the string does not contain.
MJCF allows a partial `solref`, and the second component is the DAMPRATIO, so

    solref="0.01"   ->   (0.01, 0.0)      instead of MuJoCo's (0.01, 1.0)

⚠⚠ A ZERO DAMPRATIO IS A DIVISION BY ZERO WEARING A NUMBER'S CLOTHES.
`contact_solve` builds the constraint stiffness as MuJoCo does,
`K = 1/(dmax^2 * timeconst^2 * dampratio^2)` (`engine_core_constraint.c:1432`),
so the contact became infinitely stiff. Measured on Menagerie's trossen_wxai,
whose gripper pads declare `solref="0.01"`, from its own keyframe:

    contact normal force : 7.3e13 N       ->  821.13   (MuJoCo 821.204067)
    qacc|max             : 5.6e13 rad/s^2 ->  ~640     (MuJoCo 641.684860)
    step 1, max |qpos - MuJoCo| : 0.2     ->  5.4e-06
    step 100                    : 3.57    ->  5.0e-05

⚠ THE 0.2 WAS THE VELOCITY CLAMP, NOT A PHYSICAL NUMBER, and reading it as one
cost real time. `euler.mojo` clamps |qvel| at 100 and the model's dt is 0.002,
so 100*dt = 0.2 rad of travel in a single step, on several joints at once,
to ten digits. An exact, repeated, suspiciously round offset in a POSITION is
worth checking against `clamp * dt` before it is worth explaining.

⚠ AND IT WAS NOT THE SOLVER, WHICH IS WHERE A 7e13 FORCE POINTS. PGS, Newton,
pyramidal and elliptic all produced the identical explosion — four "different"
configurations agreeing to the digit is itself the evidence that none of them
is the variable. The contact SET was already right: all 21 contacts matched
MuJoCo's positions, normals and depths to ~3e-6.

⚠ THE FILL RULE IS MuJoCo'S, MEASURED ON THE 3.10.0 RUNTIME on a bare sphere
rather than assumed:

    (absent)            -> solref (0.02, 1)
    solref="0.01"       -> solref (0.01, 1)      <- component 1 KEPT
    solref="0.01 0.5"   -> solref (0.01, 0.5)
    solimp="0.8"        -> solimp (0.8, 0.95, 0.001, 0.5, 2)
    solimp="0.8 0.99"   -> solimp (0.8, 0.99, 0.001, 0.5, 2)

Supplied components overwrite; omitted ones keep whatever they had — the
default, or the value inherited from the `<default>` class. Every `solimp`
reader in the parser already did this one component at a time; only `solref`
took the vector whole. Six sites, including one that required BOTH components
and so dropped a one-value `solref` entirely.
"""

from std.math import abs
from max.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
    read_model_source,
)
from mojo_rl.physics3d.studio.stepping import StudioIntegEll
from mojo_rl.physics3d.dynamics.actuation import apply_actions_fields
from mojo_rl.physics3d.gpu.constants import (
    CONTACT_SIZE, CONTACT_IDX_SOLREF_0, CONTACT_IDX_SOLREF_1,
    CONTACT_IDX_FORCE_N, META_IDX_NUM_CONTACTS, KEY_IDX_NCTRL,
)

comptime DT = DType.float64
comptime WXAI = String(
    "references/mujoco_menagerie-main/trossen_wxai/scene.xml"
)

# One geom states a partial `solref`, one a full one, one states it in a
# `<default>` class, and one states none — so a reader that zero-fills, one
# that ignores partial values, and one that applies the class to everything
# all fail on a different row.
comptime XML = String(
    """<mujoco>
  <compiler angle="radian"/>
  <default>
    <default class="pad"><geom solref="0.03"/></default>
  </default>
  <worldbody>
    <geom name='a' type='sphere' size='.1' solref='0.01'/>
    <geom name='b' type='sphere' size='.1' solref='0.01 0.5'/>
    <geom name='c' type='sphere' size='.1' class='pad'/>
    <geom name='d' type='sphere' size='.1'/>
  </worldbody>
</mujoco>"""
)


def test_partial_solref_keeps_its_dampratio() raises:
    """The parse, against MuJoCo's compiled `geom_solref`."""
    print("=== a one-value solref keeps dampratio 1 ===")
    var fmd = parse_xml_full(XML, String(""))
    assert_true(
        len(fmd.geoms) == 4,
        "fixture did not parse four geoms — the gate would be vacuous",
    )
    for i in range(4):
        print("  geom", i, " solref", fmd.geoms[i].solref_0,
              fmd.geoms[i].solref_1)
    assert_true(
        abs(fmd.geoms[0].solref_0 - 0.01) < 1e-15
        and abs(fmd.geoms[0].solref_1 - 1.0) < 1e-15,
        "`solref='0.01'` must compile to (0.01, 1.0) — MuJoCo keeps the"
        " default dampratio for the component the attribute omits. We got ("
        + String(fmd.geoms[0].solref_0) + ", "
        + String(fmd.geoms[0].solref_1)
        + "). A 0 there is a division by zero in the stiffness.",
    )
    # ⚠ THE NEGATIVE CONTROL: a fully-specified solref must still be taken
    # verbatim, or "keep the default" has been applied too eagerly.
    assert_true(
        abs(fmd.geoms[1].solref_0 - 0.01) < 1e-15
        and abs(fmd.geoms[1].solref_1 - 0.5) < 1e-15,
        "`solref='0.01 0.5'` must be taken verbatim; got ("
        + String(fmd.geoms[1].solref_0) + ", "
        + String(fmd.geoms[1].solref_1) + ")",
    )
    assert_true(
        abs(fmd.geoms[2].solref_0 - 0.03) < 1e-15
        and abs(fmd.geoms[2].solref_1 - 1.0) < 1e-15,
        "a partial `solref` in a <default> class must inherit the same way;"
        " got (" + String(fmd.geoms[2].solref_0) + ", "
        + String(fmd.geoms[2].solref_1) + ")",
    )
    assert_true(
        abs(fmd.geoms[3].solref_0 - 0.02) < 1e-15
        and abs(fmd.geoms[3].solref_1 - 1.0) < 1e-15,
        "a geom stating no `solref` keeps MuJoCo's default (0.02, 1); got ("
        + String(fmd.geoms[3].solref_0) + ", "
        + String(fmd.geoms[3].solref_1) + ")",
    )
    print("  PASS")


def test_wxai_contact_force_and_trajectory_match_mujoco() raises:
    """The model it was found on: the force, then 100 steps.

    ⚠ MEASURED ON THE 3.10.0 RUNTIME from keyframe 0: `efc_force|max`
    821.204067, `qacc|max` 641.684860, ncon 21, and after 100 steps
    left/joint_2 = 1.102644430353, right/joint_3 = 0.084924813452.
    """
    print("=== trossen_wxai: contact force and 100 steps ===")
    var src = read_model_source(WXAI)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    var verts = 262144
    var dims = dims_from_flat(fmd, max_contacts=128, nmesh_verts=verts)
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
            dims = dims_from_flat(fmd, max_contacts=128, nmesh_verts=verts)
            m = Model[DT, DynDims](dims)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var d = Data[DT, DynDims, 1](dims)
    assert_true(
        dims.get_nkey() > 0,
        "this gate drives trossen_wxai's keyframe; nkey = "
        + String(dims.get_nkey()),
    )
    for i in range(dims.get_nq()):
        d.qpos.data[i] = sf.key_qpos.data[i]
    for i in range(dims.get_nv()):
        d.qvel.data[i] = Scalar[DT](0)
    var nact = dims.get_nact()
    var actions = List[Float64](length=nact, fill=0.0)
    var act = List[Scalar[DT]](length=nact, fill=Scalar[DT](0))
    var nct = Int(Float64(sf.key_meta.data[KEY_IDX_NCTRL]))
    for a in range(min(nct, nact)):
        actions[a] = Float64(sf.key_ctrl.data[a])

    var integ = StudioIntegEll(dims)
    apply_actions_fields[DT](sf, d, actions, act, fmd.timestep)
    integ.step["cpu"](d, m)

    var nc = Int(Float64(d.meta.data[META_IDX_NUM_CONTACTS]))
    var fmax = 0.0
    var sr1_min = 1e30
    for k in range(nc):
        var o = k * CONTACT_SIZE
        var f = abs(Float64(d.contacts.data[o + CONTACT_IDX_FORCE_N]))
        if f > fmax:
            fmax = f
        var s1 = Float64(d.contacts.data[o + CONTACT_IDX_SOLREF_1])
        if s1 < sr1_min:
            sr1_min = s1
    print("  ncon", nc, " (MuJoCo 21)")
    print("  min contact solref dampratio", sr1_min, " (MuJoCo 1.0)")
    print("  max contact normal force", fmax, " (MuJoCo efc_force 821.204)")
    assert_true(
        nc == 21,
        "MuJoCo reports 21 contacts at this keyframe; we report " + String(nc),
    )
    # ⚠ THE ROW THAT NAMES THE BUG. Every mixed dampratio must be positive;
    # a single zero is enough to make one contact infinitely stiff.
    assert_true(
        sr1_min > 0.0,
        "a contact carries solref dampratio " + String(sr1_min)
        + ". The stiffness divides by its square, so zero is infinite"
        " stiffness — this is a partial `solref` that lost its second"
        " component.",
    )
    assert_true(
        fmax < 5.0e3,
        "the largest contact normal force is " + String(fmax)
        + " N against MuJoCo's 821. Unfixed this read 7.3e13.",
    )

    for _ in range(99):
        apply_actions_fields[DT](sf, d, actions, act, fmd.timestep)
        integ.step["cpu"](d, m)
    var j2 = Float64(d.qpos.data[2])
    var rj3 = Float64(d.qpos.data[11])
    print("  after 100: left/joint_2", j2, " (MuJoCo 1.102644430353)")
    print("             right/joint_3", rj3, " (MuJoCo 0.084924813452)")
    # ⚠ 1e-3, NOT TIGHTER, AND THE REASON IS NAMED: 19 of the 21 contacts are
    # condim 6 and the studio's integrator is built with MAX_CONDIM=3, so the
    # torsional and rolling friction rows are clamped away. Measured, that is
    # NOT what the residual is — stepping the same model at MAX_CONDIM=6 gives
    # the identical 4.956e-05 — but it is an honest reason not to claim more
    # precision than the configuration supports.
    assert_true(
        abs(j2 - 1.102644430353) < 1e-3,
        "left/joint_2 after 100 steps is " + String(j2)
        + " against MuJoCo's 1.102644430353",
    )
    assert_true(
        abs(rj3 - 0.084924813452) < 1e-3,
        "right/joint_3 after 100 steps is " + String(rj3)
        + " against MuJoCo's 0.084924813452",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
