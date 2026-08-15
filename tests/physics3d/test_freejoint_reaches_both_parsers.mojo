"""`<freejoint/>` must be understood by BOTH parsers, not just `merge_mjcf`.

`_normalize_freejoint` rewrites the tag as `<joint type="free" ...>` because
~20 scanners downstream look for the literal `"<joint"`. It used to be called
from `merge_mjcf` ALONE. Composer-built models (dm_control's humanoid,
quadruped, dog) go through that function and were fine; a single-file MJCF
handed straight to `parse_xml` / `ModelDefFromXML` was not — and single-file is
exactly the shape Menagerie / SO-ARM / ToddlerBot ports arrive in.

⚠⚠ THE FAILURE WAS SILENT AND TOTAL, WHICH IS WHY IT NEEDS A GATE:

  * `<freejoint` matches no `find("<joint")`, so NJOINT/NQ/NV came out **0** —
    the body had NO DEGREES OF FREEDOM and could not move;
  * with no joint, `body_weldid` stayed 0 (the world's), and
    `pair_body_filtered`'s FIRST clause is `if weld_i == weld_j: return True`,
    so EVERY contact pair that body belonged to was discarded before the
    narrow phase ran.

Neither shows up as an error. It presents as "the object is stuck and passes
through things", and it was found only because a hand-written two-geom fixture
reported ZERO contacts where MuJoCo reported one.

WHAT IS ASSERTED, in the order that localises a regression:

  1. the COMPTIME counter (`parse_xml`) gets 7/6/1 for a `<freejoint/>` body;
  2. the two spellings — `<freejoint/>` and `<joint type="free"/>` — agree on
     dimensions, since the alias is the entire point;
  3. the RUNTIME parser (`parse_xml_full` via `init_fields`) sets
     `body_weldid` to the body's own index, which is the field the contact
     filter reads — checked against MuJoCo's `body_weldid`;
  4. an overlapping pair actually PRODUCES a contact. Without this, 1-3 are
     satisfiable while the pair is still filtered downstream.

⚠ 3 AND 4 ARE THE ONES THAT MATTER. Dimensions alone would have passed on a
model whose weldid was right for another reason, and weldid alone proves
nothing about whether the pair survives the filter.

Run with:
    pixi run mojo run -I . tests/physics3d/test_freejoint_reaches_both_parsers.mojo
"""

from std.python import Python
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.fields import Model, Data
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.collision.contact_detection import detect_contacts
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE, BODY_IDX_WELDID, META_IDX_NUM_CONTACTS,
)
from max.gpu.host import DeviceContext

comptime DTYPE = DType.float64
comptime NMV: Int = 64

# A free body overlapping a STATIC geom. The overlap is unambiguous — the
# sphere's centre is inside the box's x/y span and 8 mm above its top face
# against a 30 mm radius — so a zero-contact result cannot be a near-miss.
comptime XML_FREEJOINT = String(
    """<mujoco><option gravity='0 0 0'/>
  <worldbody>
    <geom name='g_box' type='box' size='0.05 0.04 0.03'/>
    <body name='b1' pos='0.030 0.020 0.038'>
      <freejoint/>
      <geom name='g_sph' type='sphere' size='0.03' mass='1'/>
    </body>
  </worldbody>
</mujoco>"""
)

# The same model with the LONGHAND spelling. MuJoCo treats them as different
# only in default inheritance, which this model has none of, so every dimension
# must agree.
comptime XML_LONGHAND = String(
    """<mujoco><option gravity='0 0 0'/>
  <worldbody>
    <geom name='g_box' type='box' size='0.05 0.04 0.03'/>
    <body name='b1' pos='0.030 0.020 0.038'>
      <joint type="free"/>
      <geom name='g_sph' type='sphere' size='0.03' mass='1'/>
    </body>
  </worldbody>
</mujoco>"""
)

comptime PM_FJ = parse_xml(XML_FREEJOINT)
comptime PM_LH = parse_xml(XML_LONGHAND)

comptime MD_FJ = ModelDefFromXML[
    xml=XML_FREEJOINT, nbody=PM_FJ.NBODY, njoint=PM_FJ.NJOINT, nq=PM_FJ.NQ,
    nv=PM_FJ.NV, ngeom=PM_FJ.NGEOM, nact=PM_FJ.NACT, ntex=PM_FJ.NTEX,
    nmat=PM_FJ.NMAT, nlight=PM_FJ.NLIGHT, ncam=PM_FJ.NCAM, nsite=PM_FJ.NSITE,
    neq=PM_FJ.NEQ, nexclude=PM_FJ.NEXCLUDE, npair=PM_FJ.NPAIR,
    max_tendon=PM_FJ.NTENDON, max_condim=PM_FJ.MAX_CONDIM, max_equality=1,
    max_contacts=16, timestep=PM_FJ.TIMESTEP,
]


def test_comptime_parser_understands_freejoint() raises:
    """`parse_xml` — 0/0/0 was the bug, 7/6/1 is MuJoCo."""
    print("=== parse_xml on <freejoint/> ===")
    print("  NQ/NV/NJOINT =", PM_FJ.NQ, PM_FJ.NV, PM_FJ.NJOINT)
    assert_true(
        PM_FJ.NJOINT == 1 and PM_FJ.NQ == 7 and PM_FJ.NV == 6,
        "`<freejoint/>` produced NQ/NV/NJOINT " + String(PM_FJ.NQ) + "/"
        + String(PM_FJ.NV) + "/" + String(PM_FJ.NJOINT) + " instead of 7/6/1."
        " The tag matches no `find(\"<joint\")`, so the body has NO DOFS: it"
        " cannot move, and its weldid stays 0 which makes the contact filter"
        " discard every pair it is in",
    )
    print("  PASS")


def test_both_spellings_agree() raises:
    """The alias must be an alias."""
    print("=== <freejoint/> vs <joint type=\"free\"/> ===")
    print("  freejoint:", PM_FJ.NQ, PM_FJ.NV, PM_FJ.NJOINT,
          "  longhand:", PM_LH.NQ, PM_LH.NV, PM_LH.NJOINT)
    assert_true(
        PM_FJ.NQ == PM_LH.NQ
        and PM_FJ.NV == PM_LH.NV
        and PM_FJ.NJOINT == PM_LH.NJOINT
        and PM_FJ.NBODY == PM_LH.NBODY,
        "the two spellings of a free root disagree on dimensions, so one of"
        " them is not being recognised",
    )
    print("  PASS")


def test_runtime_parser_sets_weldid_and_the_pair_collides() raises:
    """The RUNTIME parser's `body_weldid`, and the contact it gates.

    ⚠ A DIFFERENT PARSER FROM THE ONE ABOVE. `init_fields` goes through
    `parse_xml_full`, which had NO reference to `<freejoint>` at all — the
    comptime counter could be fixed on its own and this would still fail.
    """
    var sf = MD_FJ.make_spec_fields[DTYPE]()
    print("=== runtime parser: weldid, and the contact it gates ===")
    var sys = Python.import_module("sys")
    _ = sys.path.insert(0, "tests/dm_control")
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    var mujoco = Python.import_module("mujoco")

    var m = mujoco.MjModel.from_xml_string(XML_FREEJOINT)
    var md = mujoco.MjData(m)
    mujoco.mj_forward(m, md)
    var mj_ncon = Int(py=md.ncon)

    var ctx = DeviceContext()
    var mf = Model[
        DTYPE, MD_FJ.NV, MD_FJ.NBODY, MD_FJ.NJOINT, MD_FJ.NGEOM,
        MD_FJ.MAX_EQUALITY, MD_FJ.MAX_TENDON, MD_FJ.NSITE, MD_FJ.NEXCLUDE,
        NMV, MD_FJ.NPAIR,
    ]()
    MD_FJ.init_fields[DTYPE, NMV](ctx, mf)
    var d = Data[
        DTYPE, MD_FJ.NQ, MD_FJ.NV, MD_FJ.NBODY, MD_FJ.MAX_CONTACTS,
        MD_FJ.NSITE, 1,
    ]()
    MD_FJ.reset_data[DTYPE](sf, d)
    forward_kinematics["cpu"](d, mf)
    detect_contacts["cpu"](d, mf)
    var our_ncon = Int(Float64(d.meta.data[META_IDX_NUM_CONTACTS]))

    for b in range(MD_FJ.NBODY):
        var ours = Int(
            Float64(mf.bodies.data[b * MODEL_BODY_SIZE + BODY_IDX_WELDID])
        )
        var theirs = Int(py=m.body_weldid[b])
        print("  body", b, " weldid ours", ours, " MuJoCo", theirs)
        assert_true(
            ours == theirs,
            "body " + String(b) + " weldid " + String(ours) + " against"
            " MuJoCo's " + String(theirs) + ". `pair_body_filtered` reads this"
            " field and discards the pair when the two sides are equal, so a"
            " wrong 0 here silently removes contacts rather than erroring",
        )

    print("  ncon ours", our_ncon, " MuJoCo", mj_ncon)
    # MuJoCo must actually see the overlap, or this gate proves nothing.
    assert_true(
        mj_ncon > 0,
        "MuJoCo reports no contact for this fixture, so the geoms are not"
        " overlapping and the assertion below is vacuous — move the sphere"
        " further into the box",
    )
    assert_true(
        our_ncon > 0,
        "the overlapping pair produced ZERO contacts where MuJoCo reports "
        + String(mj_ncon) + ". The narrow phase is not at fault — `gjk_epa`"
        " on these two geoms is correct to 7.6e-6 — the pair never reaches it",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
