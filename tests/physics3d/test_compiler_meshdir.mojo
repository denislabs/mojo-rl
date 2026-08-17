"""`<compiler meshdir>` / `assetdir` resolution — vs MuJoCo's `geom_rbound`.

    pixi run mojo run -I . tests/physics3d/test_compiler_meshdir.mojo

⚠ RUN FROM THE REPO ROOT. MuJoCo resolves an asset directory relative to the
MODEL FILE; we parse a STRING and have no file path, so both sides here resolve
against the PROCESS CWD. That difference is real and deliberate — see
`full_parser`'s note — and it is why the fixture paths are repo-root-relative.

WHAT THIS GATES. `meshdir` was unparsed until 2026-08-13. `mesh_asset_files`
held the `file=` attribute verbatim, so `load_mesh_hull` got the bare stem, the
open failed, and `fields_build` printed `Warning: failed to load mesh:` and
carried on — a model that builds and steps with every mesh geom non-colliding.
96 XML files across 69 Menagerie models declare `meshdir` or `assetdir`, so
almost none of them could be loaded unmodified.

⚠⚠ THE ASSERTION IS `geom_rbound` AGAINST MuJoCo, NOT "did it warn". Three
reasons, and the third is the one that matters:

  · A warning is not assertable from inside the test.
  · `rbound` is DERIVED FROM THE LOADED HULL, so it can only be right if the
    mesh actually loaded and was correct — it is the cheapest end-to-end
    signal available.
  · ⚠ THE FAILING VALUE IS NOT ZERO. Measured before the fix, the geom kept a
    **0.5** fallback radius and a `GEOM_IDX_MESH_ID` still pointing at its
    asset index, so the broadphase went on accepting pairs for a geom with no
    hull behind it. A test asserting `rbound != 0` would have passed while the
    model was badly broken.

⚠ EVERY CASE MUST NAME THE SAME MESH so one reference number covers them all;
a case that silently loaded a DIFFERENT file would otherwise look correct.
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from max.gpu.host import DeviceContext
from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.fields import Model, Dims
from mojo_rl.physics3d.gpu.constants import GEOM_IDX_RBOUND
from mojo_rl.physics3d.model.model_dims import ModelDims

# A tiny real hull: 8 vertices, 684 bytes. Big enough to be a mesh, small
# enough that the fixture costs nothing.
comptime DIR = "mojo_rl/envs/robots/assets/so_arm100/"
comptime FILE = "Moving_Jaw_Collision_2.stl"

comptime BODY = """
  <worldbody>
    <body name="a"><joint name="ja" type="hinge" axis="0 1 0"/>
      <geom type="mesh" mesh="jaw"/></body>
  </worldbody>
</mujoco>"""

# 1. meshdir + a bare filename — the Menagerie spelling, and the one that broke.
comptime X_MESHDIR = (
    """<mujoco>
  <compiler meshdir=\"""" + DIR + """\"/>
  <asset><mesh name="jaw" file=\"""" + FILE + """\"/></asset>""" + BODY
)

# 2. assetdir alone — MEASURED on 3.10.0 to act as meshdir's fallback.
comptime X_ASSETDIR = (
    """<mujoco>
  <compiler assetdir=\"""" + DIR + """\"/>
  <asset><mesh name="jaw" file=\"""" + FILE + """\"/></asset>""" + BODY
)

# 3. Both, disagreeing. MEASURED: `meshdir` WINS. `assetdir="."` would not
#    find the file, so this case fails loudly if the precedence is inverted.
comptime X_BOTH = (
    """<mujoco>
  <compiler assetdir="." meshdir=\"""" + DIR + """\"/>
  <asset><mesh name="jaw" file=\"""" + FILE + """\"/></asset>""" + BODY
)

# 4. Control: no directory attribute, full path on `file=`. This is what every
#    ported model baked by hand while `meshdir` was unparsed, and it must keep
#    working — the fix must not start prefixing a path that is already whole.
comptime X_FULLPATH = (
    """<mujoco>
  <asset><mesh name="jaw" file=\"""" + DIR + FILE + """\"/></asset>""" + BODY
)

comptime p1 = parse_xml(X_MESHDIR)
comptime p2 = parse_xml(X_ASSETDIR)
comptime p3 = parse_xml(X_BOTH)
comptime p4 = parse_xml(X_FULLPATH)

comptime M1 = ModelDefFromXML[xml=X_MESHDIR, nbody=p1.NBODY, njoint=p1.NJOINT,
    nq=p1.NQ, nv=p1.NV, ngeom=p1.NGEOM, nact=p1.NACT, ntex=p1.NTEX,
    nmat=p1.NMAT, nlight=p1.NLIGHT, ncam=p1.NCAM, nsite=p1.NSITE, neq=p1.NEQ,
    nexclude=p1.NEXCLUDE, npair=p1.NPAIR, timestep=p1.TIMESTEP]
comptime M2 = ModelDefFromXML[xml=X_ASSETDIR, nbody=p2.NBODY, njoint=p2.NJOINT,
    nq=p2.NQ, nv=p2.NV, ngeom=p2.NGEOM, nact=p2.NACT, ntex=p2.NTEX,
    nmat=p2.NMAT, nlight=p2.NLIGHT, ncam=p2.NCAM, nsite=p2.NSITE, neq=p2.NEQ,
    nexclude=p2.NEXCLUDE, npair=p2.NPAIR, timestep=p2.TIMESTEP]
comptime M3 = ModelDefFromXML[xml=X_BOTH, nbody=p3.NBODY, njoint=p3.NJOINT,
    nq=p3.NQ, nv=p3.NV, ngeom=p3.NGEOM, nact=p3.NACT, ntex=p3.NTEX,
    nmat=p3.NMAT, nlight=p3.NLIGHT, ncam=p3.NCAM, nsite=p3.NSITE, neq=p3.NEQ,
    nexclude=p3.NEXCLUDE, npair=p3.NPAIR, timestep=p3.TIMESTEP]
comptime M4 = ModelDefFromXML[xml=X_FULLPATH, nbody=p4.NBODY, njoint=p4.NJOINT,
    nq=p4.NQ, nv=p4.NV, ngeom=p4.NGEOM, nact=p4.NACT, ntex=p4.NTEX,
    nmat=p4.NMAT, nlight=p4.NLIGHT, ncam=p4.NCAM, nsite=p4.NSITE, neq=p4.NEQ,
    nexclude=p4.NEXCLUDE, npair=p4.NPAIR, timestep=p4.TIMESTEP]

comptime NMV = 64
comptime MD_4 = ModelDims[M4, 64]
comptime MD_3 = ModelDims[M3, 64]
comptime MD_2 = ModelDims[M2, 64]
comptime MD = ModelDims[M1, 64]


def _mj_rbound(xml: String) raises -> Float64:
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(xml)
    return Float64(py=m.geom_rbound[0])


def _check(name: String, xml: String, ours: Float64) raises:
    var want = _mj_rbound(xml)
    print("  ", name, " ours rbound", ours, "  MuJoCo", want)
    assert_true(
        abs(ours - want) < 1e-12,
        name + ": rbound " + String(ours) + " vs MuJoCo " + String(want)
        + " — 0.5 means the hull never loaded and the geom kept its fallback"
        " radius; any other value means a DIFFERENT mesh was read",
    )


def test_meshdir_resolves() raises:
    """The case that was broken: `meshdir` + a bare `file=`."""
    var ctx = DeviceContext()
    var mf = Model[DType.float64, MD]()
    M1.init_fields[DType.float64](ctx, mf)
    _check("meshdir      ", X_MESHDIR, Float64(mf.geoms.data[GEOM_IDX_RBOUND]))


def test_assetdir_is_the_fallback() raises:
    """`assetdir` alone stands in for `meshdir` — measured on 3.10.0."""
    var ctx = DeviceContext()
    var mf = Model[DType.float64, MD_2]()
    M2.init_fields[DType.float64](ctx, mf)
    _check("assetdir     ", X_ASSETDIR, Float64(mf.geoms.data[GEOM_IDX_RBOUND]))


def test_meshdir_wins_over_assetdir() raises:
    """Precedence. `assetdir="."` cannot find the file, so an inverted
    precedence fails loudly rather than coincidentally passing."""
    var ctx = DeviceContext()
    var mf = Model[DType.float64, MD_3]()
    M3.init_fields[DType.float64](ctx, mf)
    _check("both (meshdir)", X_BOTH, Float64(mf.geoms.data[GEOM_IDX_RBOUND]))


def test_full_path_still_works() raises:
    """Control: a whole path on `file=` with no directory attribute.

    Every ported model baked this by hand while `meshdir` was unparsed, so the
    fix must not start prefixing something already complete.
    """
    var ctx = DeviceContext()
    var mf = Model[DType.float64, MD_4]()
    M4.init_fields[DType.float64](ctx, mf)
    _check("full path    ", X_FULLPATH, Float64(mf.geoms.data[GEOM_IDX_RBOUND]))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
