"""A `<compiler>` inside an `<include>` must survive the merge.

    pixi run mojo run -I . tests/physics3d/test_compiler_merge_vs_mujoco.mojo

WHAT WENT WRONG. `resolve_includes` splices every included file into ONE
document and then calls `merge_mjcf` on it — and `merge_mjcf` collected its
singleton tags with `_extract_singleton_tag`, which returns THE FIRST MATCH.
So of the several `<compiler>` (and `<option>`, `<statistic>`, `<size>`)
elements in a spliced document, only the host's survived and every included
file's was discarded. The careful last-wins attribute merge underneath
(`_merge_singleton_attrs`, which is correct) never saw them.

⚠⚠ IT COST A FACTOR OF 57.3 ON EVERY ANGLE IN THE MODEL. Menagerie's aloha
opens `scene.xml` with

    <compiler meshdir="assets" texturedir="assets"/>     <- no `angle`
    <include file="aloha.xml"/>                          <- angle="radian"

MJCF's default is DEGREE, so dropping the included `angle="radian"` compiled
every joint range as if it were degrees: the waist came out +-0.0548 rad
instead of +-3.14159. Every arm joint then sat far outside a limit it should
never have reached, the limit constraints answered with everything they had,
and `qacc` was ~2200 rad/s^2 where MuJoCo has 0.15. Measured, from aloha's own
neutral keyframe:

    step   1, max |qpos - MuJoCo| : 1.02e-02  ->  3.4e-10
    step 100, max |qpos - MuJoCo| : 1.14e+00  ->  1.2e-07

i.e. the arms held their pose instead of collapsing.

⚠ THE MERGE RULE IS "LAST TAG THAT STATES THE ATTRIBUTE", and all four
branches of it were measured on the 3.10.0 runtime rather than assumed —
`angle` on a hinge with `range="-1.5 1.5"`:

    <compiler angle="radian"/> then <compiler angle="degree"/>  -> degree
    <include (radian)/>        then <compiler angle="degree"/>  -> degree
    <compiler meshdir=.../>    then <include (radian)/>         -> radian
    <compiler angle="radian"/> then <compiler meshdir=.../>     -> radian

So it is NOT "the included file wins" — putting the `<include>` first makes
the parent win — and a later tag that OMITS the attribute does not reset it.

⚠ THE DIRECTORY ATTRIBUTES ARE DELIBERATELY EXCLUDED. `meshdir`/`assetdir`/
`texturedir` are PATHS, and `expand_mjcf` has already rebased every spliced
`file=` against the directory of the file that wrote it (`50d99683`).
Re-resolving them here would double-apply that.
"""

from std.math import abs
from max.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.xml_parser import merge_mjcf
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
    read_model_source,
)
from mojo_rl.physics3d.studio.stepping import StudioIntegEll, StudioIntegPyr
from mojo_rl.physics3d.dynamics.actuation import apply_actions_fields
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.gpu.constants import KEY_META_SIZE, KEY_IDX_NCTRL

comptime DT = DType.float64
comptime ALOHA = String(
    "references/mujoco_menagerie-main/aloha/scene.xml"
)

# The shape `resolve_includes` produces: one document, two `<compiler>`s, the
# host's first and without `angle`.
comptime SPLICED = String(
    """<mujoco>
  <compiler meshdir="assets" texturedir="assets"/>
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <body><joint name='j' type='hinge' axis='0 0 1' range='-1.5 1.5'/>
    <geom type='sphere' size='0.1' mass='1'/></body>
  </worldbody>
</mujoco>"""
)
# ⚠ THE ORDER REVERSED — the negative control for "the included file wins".
comptime SPLICED_REV = String(
    """<mujoco>
  <compiler angle="radian"/>
  <compiler angle="degree"/>
  <worldbody>
    <body><joint name='j' type='hinge' axis='0 0 1' range='-1.5 1.5'/>
    <geom type='sphere' size='0.1' mass='1'/></body>
  </worldbody>
</mujoco>"""
)


def test_merge_keeps_every_compiler_tag() raises:
    """`merge_mjcf` on one spliced document must see BOTH `<compiler>`s."""
    print("=== merge_mjcf keeps the included <compiler> ===")
    var merged = merge_mjcf(SPLICED)
    print("  merged:", merged[byte=0 : min(220, merged.byte_length())])
    assert_true(
        merged.find("angle=\"radian\"") != -1,
        "the second <compiler>'s angle=\"radian\" did not survive the merge."
        " `_extract_singleton_tag` returns only the FIRST match, and"
        " `resolve_includes` hands `merge_mjcf` a single spliced document.",
    )
    # ⚠ AND THE HOST'S OWN ATTRIBUTES MUST STILL BE THERE. A "fix" that took
    # the LAST tag wholesale would pass the line above and lose `meshdir`,
    # which is how the assets stop loading.
    assert_true(
        merged.find("meshdir=\"assets\"") != -1
        and merged.find("texturedir=\"assets\"") != -1,
        "the host's meshdir/texturedir were lost — the merge is per-ATTRIBUTE,"
        " not whole-tag replacement",
    )
    print("  PASS")


def test_document_order_decides_not_include_depth() raises:
    """`radian` then `degree` must give DEGREE — measured on the runtime.

    ⚠ THIS IS THE ROW THAT PINS THE RULE. "The included file wins" also
    explains aloha, and is wrong: MuJoCo takes the LAST tag in document order,
    so an `<include>` placed before the parent's own `<compiler>` loses.
    """
    print("=== last in document order wins ===")
    var fmd = parse_xml_full(SPLICED_REV, String(""))
    var lo = fmd.joints[0].range_min
    var hi = fmd.joints[0].range_max
    print("  radian-then-degree range:", lo, hi,
          " (MuJoCo: -0.0261799388, 0.0261799388)")
    assert_true(
        abs(lo - (-0.026179938779914945)) < 1e-12
        and abs(hi - 0.026179938779914945) < 1e-12,
        "with `radian` then `degree`, MuJoCo applies DEGREE and compiles"
        " -1.5/1.5 to -0.02618/0.02618; we got " + String(lo) + "/"
        + String(hi),
    )
    # And the forward case still resolves to radian.
    var fmd2 = parse_xml_full(SPLICED, String(""))
    print("  bare-then-radian range:", fmd2.joints[0].range_min,
          fmd2.joints[0].range_max, " (MuJoCo: -1.5, 1.5)")
    assert_true(
        abs(fmd2.joints[0].range_min - (-1.5)) < 1e-12,
        "a host without `angle` followed by an included `angle=\"radian\"`"
        " must compile the range as RADIANS",
    )
    print("  PASS")


def test_aloha_ranges_and_trajectory_match_mujoco() raises:
    """The real model, ranges and motion, against MuJoCo's own numbers.

    ⚠ DRIVEN FROM THE NEUTRAL KEYFRAME, NOT qpos0, AND THAT IS THE POINT OF
    THE FIXTURE. aloha's `qpos0` (all zeros) folds the two arms into each
    other — MuJoCo itself reports 66 contacts there, up to 3.5 cm deep. Two
    engines legitimately diverge in a configuration like that, so measuring
    there says nothing; the model ships `neutral_pose` (ncon 0) precisely to
    avoid it. The keyframe also carries `ctrl`, which MuJoCo's reset applies —
    without driving it the arms go limp and fall, which is a property of the
    HARNESS and not of the engine.
    """
    print("=== aloha: ranges, and 100 steps from neutral_pose ===")
    var src = read_model_source(ALOHA)
    var flat = expand_mjcf(src[0], src[1])
    var fmd = parse_xml_full(flat, src[1])
    # MuJoCo `m.jnt_range`: waist +-3.14158, shoulder [-1.85005, 1.25664].
    print("  waist   ", fmd.joints[0].range_min, fmd.joints[0].range_max)
    print("  shoulder", fmd.joints[1].range_min, fmd.joints[1].range_max)
    assert_true(
        abs(fmd.joints[0].range_min - (-3.14158)) < 1e-5
        and abs(fmd.joints[0].range_max - 3.14158) < 1e-5,
        "aloha's waist range must be MuJoCo's +-3.14158 rad; got "
        + String(fmd.joints[0].range_min) + "/"
        + String(fmd.joints[0].range_max)
        + ". A value near +-0.0548 is this range divided by 57.3, i.e. the"
        " included <compiler angle=\"radian\"> was dropped.",
    )
    assert_true(
        abs(fmd.joints[1].range_min - (-1.85005)) < 1e-5
        and abs(fmd.joints[1].range_max - 1.25664) < 1e-5,
        "aloha's shoulder range must be MuJoCo's [-1.85005, 1.25664]",
    )

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
    var nq = dims.get_nq()
    assert_true(
        dims.get_nkey() > 0,
        "aloha ships a `neutral_pose` keyframe and this gate needs it; the"
        " parse reports nkey = " + String(dims.get_nkey()),
    )
    for i in range(nq):
        d.qpos.data[i] = sf.key_qpos.data[i]
    for i in range(dims.get_nv()):
        d.qvel.data[i] = Scalar[DT](0)

    var nact = dims.get_nact()
    var actions = List[Float64](length=nact, fill=0.0)
    var act = List[Scalar[DT]](length=nact, fill=Scalar[DT](0))
    var nct = Int(Float64(sf.key_meta.data[KEY_IDX_NCTRL]))
    assert_true(
        nct > 0,
        "aloha's keyframe carries `ctrl` and the servos need it; parsed"
        " nctrl = " + String(nct),
    )
    for a in range(min(nct, nact)):
        actions[a] = Float64(sf.key_ctrl.data[a])

    var ell = StudioIntegEll(dims)
    var pyr = StudioIntegPyr(dims)
    var ell_cone = fmd.cone == ConeType.ELLIPTIC
    for _ in range(100):
        apply_actions_fields[DT](sf, d, actions, act, fmd.timestep)
        if ell_cone:
            ell.step["cpu"](d, m)
        else:
            pyr.step["cpu"](d, m)

    # MuJoCo, 100 steps from key 0.
    var elbow = Float64(d.qpos.data[2])
    var shoulder = Float64(d.qpos.data[1])
    print("  after 100: elbow", elbow, " (MuJoCo 1.177073028803)")
    print("             shoulder", shoulder, " (MuJoCo -0.959942754977)")
    # ⚠ THE BOUND IS TIGHT BECAUSE THERE ARE NO CONTACTS. `neutral_pose` has
    # ncon 0 in both engines, so this is pure smooth dynamics plus servos and
    # there is nothing legitimate to disagree about at 1e-3.
    assert_true(
        abs(elbow - 1.177073028803) < 1e-3,
        "aloha's elbow after 100 steps is " + String(elbow)
        + " against MuJoCo's 1.177073028803. Before the fix it read 0.033 —"
        " the arm had collapsed.",
    )
    assert_true(
        abs(shoulder - (-0.959942754977)) < 1e-3,
        "aloha's shoulder after 100 steps is " + String(shoulder)
        + " against MuJoCo's -0.959942754977",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
