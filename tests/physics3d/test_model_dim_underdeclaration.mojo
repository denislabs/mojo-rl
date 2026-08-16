"""An UNDER-DECLARED model dimension must fail the build, not truncate.

    pixi run mojo run -I . tests/physics3d/test_model_dim_underdeclaration.mojo

`ModelDefFromXML` takes `nexclude`, `npair`, `max_equality` and `max_tendon` as
OPTIONAL parameters defaulting to 0 or written by hand. Every one of them sizes
a fixed allocation that `fields_build` fills with a bounded loop, so a value
smaller than the model needs used to mean "quietly build a different robot".
Three of the four already raised by the time this file was written; equality
was the last hole (2026-08-13).

⚠⚠ WHY THIS FAMILY IS SO GOOD AT HIDING. The truncated model is not broken in
any way a normal gate can see: it has the right bodies, the right joints, the
right geoms, and it steps. It merely does not enforce a constraint, or lets a
pair collide that should not. `fish` shipped with one of its two tendons
dropped for a day; a weld silently vanished behind `max_equality` the same way,
and `MODEL_META_IDX_NEQUALITY` is CLAMPED to the cap, so even a count printed
from the built model agrees with the wrong number.

⚠ `max_equality` DOES DOUBLE DUTY, which is how it stayed open longest.
`fields_build`'s fill loop breaks on `num_eq >= MAX_EQUALITY` — a RECORD cap —
while every model in the tree sizes it as a ROW budget (sawyer passes 6 for one
weld, "1 weld = 6 rows"). A generous number over-satisfies both readings, so
nobody hit the record cap; only a deliberately small one exposes it.

⚠ THE FIXTURE IS DELIBERATELY OVER-CONSTRAINED — two `<weld>`s, two `<fixed>`
tendons and one `<exclude>` in eleven lines — so that ONE model can drive every
member of the family and a future addition to it has somewhere obvious to go.

⚠ THIS GATES THE GUARD, NOT THE PHYSICS. It says nothing about whether a weld
or a tendon is solved correctly; `test_*_vs_mujoco` does that. What it pins is
that a mis-sized declaration can never again produce a running model.
"""

from max.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.fields import Model, Dims
from mojo_rl.physics3d.gpu.constants import (
    MODEL_META_IDX_NEXCLUDE,
    MODEL_META_IDX_NEQUALITY,
    MODEL_META_IDX_NTENDON,
)

# Two equalities, two fixed tendons, one exclude — one fixture, four guards.
comptime X = """<mujoco>
  <worldbody>
    <body name="a"><joint name="ja" type="hinge" axis="0 1 0"/><geom type="box" size=".1 .1 .1"/></body>
    <body name="b" pos="0 0 .5"><joint name="jb" type="hinge" axis="0 1 0"/><geom type="box" size=".1 .1 .1"/></body>
    <body name="c" pos="0 0 1"><joint name="jc" type="hinge" axis="0 1 0"/><geom type="box" size=".1 .1 .1"/></body>
  </worldbody>
  <tendon>
    <fixed name="t1"><joint joint="ja" coef="1"/><joint joint="jb" coef="-1"/></fixed>
    <fixed name="t2"><joint joint="jb" coef="1"/><joint joint="jc" coef="-1"/></fixed>
  </tendon>
  <equality>
    <weld body1="a" body2="b"/>
    <weld body1="b" body2="c"/>
  </equality>
  <contact><exclude body1="a" body2="b"/></contact>
</mujoco>"""

comptime pm = parse_xml(X)

comptime TRUTHFUL = ModelDefFromXML[
    xml=X, nbody=pm.NBODY, njoint=pm.NJOINT, nq=pm.NQ, nv=pm.NV,
    ngeom=pm.NGEOM, nact=pm.NACT, ntex=pm.NTEX, nmat=pm.NMAT,
    nlight=pm.NLIGHT, ncam=pm.NCAM, nsite=pm.NSITE, neq=pm.NEQ,
    nexclude=pm.NEXCLUDE, npair=pm.NPAIR, max_equality=12, max_tendon=2,
    timestep=pm.TIMESTEP,
]

# ⚠ Only ONE dimension is wrong in each of these. With two wrong at once the
# first raise masks the second, and a guard that never fires looks like a guard
# that works — which is exactly what happened while this was being written.
comptime SHORT_EQUALITY = ModelDefFromXML[
    xml=X, nbody=pm.NBODY, njoint=pm.NJOINT, nq=pm.NQ, nv=pm.NV,
    ngeom=pm.NGEOM, nact=pm.NACT, ntex=pm.NTEX, nmat=pm.NMAT,
    nlight=pm.NLIGHT, ncam=pm.NCAM, nsite=pm.NSITE, neq=pm.NEQ,
    nexclude=pm.NEXCLUDE, npair=pm.NPAIR, max_equality=1, max_tendon=2,
    timestep=pm.TIMESTEP,
]

comptime SHORT_TENDON = ModelDefFromXML[
    xml=X, nbody=pm.NBODY, njoint=pm.NJOINT, nq=pm.NQ, nv=pm.NV,
    ngeom=pm.NGEOM, nact=pm.NACT, ntex=pm.NTEX, nmat=pm.NMAT,
    nlight=pm.NLIGHT, ncam=pm.NCAM, nsite=pm.NSITE, neq=pm.NEQ,
    nexclude=pm.NEXCLUDE, npair=pm.NPAIR, max_equality=12, max_tendon=1,
    timestep=pm.TIMESTEP,
]

comptime SHORT_EXCLUDE = ModelDefFromXML[
    xml=X, nbody=pm.NBODY, njoint=pm.NJOINT, nq=pm.NQ, nv=pm.NV,
    ngeom=pm.NGEOM, nact=pm.NACT, ntex=pm.NTEX, nmat=pm.NMAT,
    nlight=pm.NLIGHT, ncam=pm.NCAM, nsite=pm.NSITE, neq=pm.NEQ,
    nexclude=0, npair=pm.NPAIR, max_equality=12, max_tendon=2,
    timestep=pm.TIMESTEP,
]


def test_truthful_dims_build() raises:
    """The control. Without it, a guard that rejects EVERYTHING passes.

    Also pins the three counts the guards protect, read back off the BUILT
    model — `MODEL_META_IDX_NEQUALITY` is clamped to the cap, so reading it is
    what would have shown the truncation had anyone looked.
    """
    var ctx = DeviceContext()
    var mf = Model[DType.float64, Dims[nv=TRUTHFUL.NV, nbody=TRUTHFUL.NBODY, njoint=TRUTHFUL.NJOINT, ngeom=TRUTHFUL.NGEOM, nequality=TRUTHFUL.MAX_EQUALITY, ntendon=TRUTHFUL.MAX_TENDON, nsite=TRUTHFUL.NSITE, nexclude=TRUTHFUL.NEXCLUDE, nmesh_verts=0, npair=TRUTHFUL.NPAIR]]()
    TRUTHFUL.init_fields[DType.float64, 0](ctx, mf)
    var neq = Int(mf.meta.data[MODEL_META_IDX_NEQUALITY])
    var nten = Int(mf.meta.data[MODEL_META_IDX_NTENDON])
    var nexc = Int(mf.meta.data[MODEL_META_IDX_NEXCLUDE])
    print("  truthful dims build — neq", neq, " ntendon", nten,
          " nexclude", nexc)
    assert_true(neq == 2, "nequality is " + String(neq) + ", want 2")
    assert_true(nten == 2, "ntendon is " + String(nten) + ", want 2")
    assert_true(nexc == 1, "nexclude is " + String(nexc) + ", want 1")


def test_short_max_equality_raises() raises:
    """THE ONE THAT WAS SILENT until 2026-08-13.

    Before the guard this BUILT and reported `nequality = 1`: the second weld
    was simply never enforced, and nothing anywhere said so.
    """
    var ctx = DeviceContext()
    var raised = False
    try:
        var mf = Model[DType.float64, Dims[nv=SHORT_EQUALITY.NV, nbody=SHORT_EQUALITY.NBODY, njoint=SHORT_EQUALITY.NJOINT, ngeom=SHORT_EQUALITY.NGEOM, nequality=SHORT_EQUALITY.MAX_EQUALITY, ntendon=SHORT_EQUALITY.MAX_TENDON, nsite=SHORT_EQUALITY.NSITE, nexclude=SHORT_EQUALITY.NEXCLUDE, nmesh_verts=0, npair=SHORT_EQUALITY.NPAIR]]()
        SHORT_EQUALITY.init_fields[DType.float64, 0](ctx, mf)
    except e:
        raised = True
        print("  max_equality=1 for 2 equalities raised:", e)
    assert_true(
        raised,
        "max_equality=1 against 2 <equality> records BUILT — the fill loop"
        " breaks at the cap, so the model runs with one weld unenforced and"
        " `nequality` clamped to agree with it",
    )


def test_short_max_tendon_raises() raises:
    """Sibling guard, pinned so the family stays uniform.

    fish shipped with one of its two tendons dropped for a day before this
    existed.
    """
    var ctx = DeviceContext()
    var raised = False
    try:
        var mf = Model[DType.float64, Dims[nv=SHORT_TENDON.NV, nbody=SHORT_TENDON.NBODY, njoint=SHORT_TENDON.NJOINT, ngeom=SHORT_TENDON.NGEOM, nequality=SHORT_TENDON.MAX_EQUALITY, ntendon=SHORT_TENDON.MAX_TENDON, nsite=SHORT_TENDON.NSITE, nexclude=SHORT_TENDON.NEXCLUDE, nmesh_verts=0, npair=SHORT_TENDON.NPAIR]]()
        SHORT_TENDON.init_fields[DType.float64, 0](ctx, mf)
    except e:
        raised = True
        print("  max_tendon=1 for 2 tendons raised:", e)
    assert_true(raised, "max_tendon=1 against 2 <fixed> tendons BUILT")


def test_short_nexclude_raises() raises:
    """Sibling guard. `nexclude` defaults to 0, so this is the easiest to hit —
    forget the parameter entirely and the excluded pair collides forever."""
    var ctx = DeviceContext()
    var raised = False
    try:
        var mf = Model[DType.float64, Dims[nv=SHORT_EXCLUDE.NV, nbody=SHORT_EXCLUDE.NBODY, njoint=SHORT_EXCLUDE.NJOINT, ngeom=SHORT_EXCLUDE.NGEOM, nequality=SHORT_EXCLUDE.MAX_EQUALITY, ntendon=SHORT_EXCLUDE.MAX_TENDON, nsite=SHORT_EXCLUDE.NSITE, nexclude=SHORT_EXCLUDE.NEXCLUDE, nmesh_verts=0, npair=SHORT_EXCLUDE.NPAIR]]()
        SHORT_EXCLUDE.init_fields[DType.float64, 0](ctx, mf)
    except e:
        raised = True
        print("  nexclude=0 for 1 exclude raised:", e)
    assert_true(raised, "nexclude=0 against 1 <exclude> BUILT")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
