"""Mesh POLYGON topology against `mjModel.mesh_poly*`.

The native multi-contact path clips one geom's contact FACE against the other's.
For a mesh that face has to come from stored topology, which
`collision/mesh_polygons.mojo` builds by merging the hull's coplanar triangles.
This gates that build directly against MuJoCo's, before any collision result
depends on it — a wrong polygon set would show up downstream as a wrong
manifold, which is a much worse place to debug it from.

WHAT IS COMPARED, AND WHAT DELIBERATELY IS NOT. MuJoCo emits its polygons in
`std::unordered_map` iteration order, which is a hash artefact of libstdc++ and
is not reproducible by any port. So polygons are matched as a SET, by normal,
and then compared on their vertex CYCLE (positions, up to rotation of the
cycle). Index-by-index equality would be gating the reference's hash function.

Vertex ORDER differs too — our hull compacts vertices in input order, MuJoCo's
comes out of qhull — so the comparison is on POSITIONS throughout, never on
vertex indices. `test_mesh_frames_are_identity` in the manifold gate is what
lets positions be compared at all; if MuJoCo re-framed the fixture, the two
engines would be describing different solids.

Run: pixi run mojo run -I . tests/physics3d/test_mesh_polygons_vs_mujoco.mojo
"""

from std.math import abs, sqrt
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.fields import Data, Model
from mojo_rl.physics3d.gpu.constants import (
    MODEL_MESH_META_SIZE,
    MODEL_MESH_POLY_SIZE,
    MESH_META_IDX_VERTADR,
    MESH_META_IDX_VERTNUM,
    MESH_META_IDX_POLYADR,
    MESH_META_IDX_POLYNUM,
    MESH_POLY_IDX_VERTADR,
    MESH_POLY_IDX_VERTNUM,
    MESH_POLY_IDX_NX,
    MESH_POLY_IDX_NY,
    MESH_POLY_IDX_NZ,
)


comptime DTYPE = DType.float64

# Both fixtures, each on its own geom so both hulls are actually built. They
# never touch, so nothing here depends on collision at all.
comptime MP_XML = """
<mujoco model="mesh polygons">
  <option timestep="0.002"/>
  <asset>
    <mesh name="cube" file="tests/physics3d/assets/mc_cube.stl"/>
    <mesh name="hex" file="tests/physics3d/assets/mc_hex.stl"/>
  </asset>
  <worldbody>
    <body name="a" pos="0 0 0.5">
      <joint name="ja" type="free"/>
      <geom name="ga" type="mesh" mesh="cube"/>
    </body>
    <body name="b" pos="3 0 0.5">
      <joint name="jb" type="free"/>
      <geom name="gb" type="mesh" mesh="hex"/>
    </body>
  </worldbody>
</mujoco>
"""

comptime mp = parse_xml(MP_XML)
comptime MPM = ModelDefFromXML[
    xml=MP_XML,
    nbody=mp.NBODY, njoint=mp.NJOINT, nq=mp.NQ, nv=mp.NV,
    ngeom=mp.NGEOM, nact=mp.NACT, ntex=mp.NTEX, nmat=mp.NMAT,
    nlight=mp.NLIGHT, ncam=mp.NCAM, nsite=mp.NSITE,
    max_tendon=mp.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=16,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=mp.TIMESTEP,
]

comptime NMESHV: Int = 64
comptime Mod = Model[
    DTYPE, MPM.NV, MPM.NBODY, MPM.NJOINT, MPM.NGEOM, MPM.MAX_EQUALITY,
    MPM.MAX_TENDON, MPM.NSITE, MPM.NEXCLUDE, NMESHV,
]

comptime TOL: Float64 = 1e-6


def test_mesh_polygons_vs_mujoco() raises:
    var ctx = DeviceContext()
    var mf = Mod()
    MPM.init_fields[DTYPE, NMESHV](ctx, mf)

    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(String(MP_XML))
    var nmesh = Int(py=m.nmesh)
    print("--- mesh polygons: nmesh =", nmesh)

    var total_matched = 0
    var rotated = 0
    for mi in range(nmesh):
        var mj_va = Int(py=m.mesh_vertadr[mi])
        var mj_vn = Int(py=m.mesh_vertnum[mi])
        var mj_pa = Int(py=m.mesh_polyadr[mi])
        var mj_pn = Int(py=m.mesh_polynum[mi])

        var o = mi * MODEL_MESH_META_SIZE
        var our_va = Int(mf.mesh_meta.data[o + MESH_META_IDX_VERTADR])
        var our_vn = Int(mf.mesh_meta.data[o + MESH_META_IDX_VERTNUM])
        var our_pa = Int(mf.mesh_meta.data[o + MESH_META_IDX_POLYADR])
        var our_pn = Int(mf.mesh_meta.data[o + MESH_META_IDX_POLYNUM])

        print(
            "  mesh", mi, ": verts MuJoCo", mj_vn, "ours", our_vn,
            " polygons MuJoCo", mj_pn, "ours", our_pn,
        )
        assert_true(
            our_vn == mj_vn,
            String("mesh ") + String(mi) + " hull vertex count "
            + String(our_vn) + " != MuJoCo " + String(mj_vn),
        )
        assert_true(
            our_pn == mj_pn,
            String("mesh ") + String(mi) + " POLYGON count " + String(our_pn)
            + " != MuJoCo " + String(mj_pn) + ". A cube coming back as 12"
            " means coplanar triangles did not merge — check the hull's face"
            " WINDING, which is what the edge cancellation depends on.",
        )

        # Every MuJoCo polygon must have exactly one of ours with the same
        # normal AND the same vertex cycle (as positions, up to rotation).
        for pj in range(mj_pn):
            var mnx = Float64(py=m.mesh_polynormal[mj_pa + pj][0])
            var mny = Float64(py=m.mesh_polynormal[mj_pa + pj][1])
            var mnz = Float64(py=m.mesh_polynormal[mj_pa + pj][2])
            var madr = Int(py=m.mesh_polyvertadr[mj_pa + pj])
            var mnum = Int(py=m.mesh_polyvertnum[mj_pa + pj])

            var hit = -1
            for pk in range(our_pn):
                var po = (our_pa + pk) * MODEL_MESH_POLY_SIZE
                var e = max(
                    abs(Float64(mf.mesh_polys.data[po + MESH_POLY_IDX_NX]) - mnx),
                    max(
                        abs(Float64(mf.mesh_polys.data[po + MESH_POLY_IDX_NY]) - mny),
                        abs(Float64(mf.mesh_polys.data[po + MESH_POLY_IDX_NZ]) - mnz),
                    ),
                )
                if e < TOL:
                    hit = pk
                    break
            assert_true(
                hit >= 0,
                String("mesh ") + String(mi) + " polygon " + String(pj)
                + " with normal (" + String(mnx) + ", " + String(mny) + ", "
                + String(mnz) + ") has no counterpart in ours",
            )

            var po = (our_pa + hit) * MODEL_MESH_POLY_SIZE
            var onum = Int(mf.mesh_polys.data[po + MESH_POLY_IDX_VERTNUM])
            var oadr = Int(mf.mesh_polys.data[po + MESH_POLY_IDX_VERTADR])
            assert_true(
                onum == mnum,
                String("mesh ") + String(mi) + " polygon with normal ("
                + String(mnx) + ", " + String(mny) + ", " + String(mnz)
                + ") has " + String(onum) + " vertices, MuJoCo has "
                + String(mnum),
            )

            # Positions of MuJoCo's cycle.
            var mxs = List[Float64]()
            var mys = List[Float64]()
            var mzs = List[Float64]()
            for k in range(mnum):
                var vi = Int(py=m.mesh_polyvert[madr + k])
                mxs.append(Float64(py=m.mesh_vert[mj_va + vi][0]))
                mys.append(Float64(py=m.mesh_vert[mj_va + vi][1]))
                mzs.append(Float64(py=m.mesh_vert[mj_va + vi][2]))
            # ...and ours.
            var oxs = List[Float64]()
            var oys = List[Float64]()
            var ozs = List[Float64]()
            for k in range(onum):
                var vi = Int(mf.mesh_polyvert.data[oadr + k])
                oxs.append(Float64(mf.mesh_verts.data[(our_va + vi) * 3 + 0]))
                oys.append(Float64(mf.mesh_verts.data[(our_va + vi) * 3 + 1]))
                ozs.append(Float64(mf.mesh_verts.data[(our_va + vi) * 3 + 2]))

            # Same cycle, same direction, some rotation of the start point.
            # ⚠ DIRECTION MATTERS AND IS NOT NORMALISED AWAY: the winding is
            # what makes `cross(p1-p0, p2-p0)` agree with the stored normal,
            # and `polygonClip` reads the polygon as an ordered boundary.
            var found_shift = -1
            for s in range(mnum):
                var ok = True
                for k in range(mnum):
                    var t = (s + k) % mnum
                    if (
                        abs(oxs[t] - mxs[k]) > TOL
                        or abs(oys[t] - mys[k]) > TOL
                        or abs(ozs[t] - mzs[k]) > TOL
                    ):
                        ok = False
                        break
                if ok:
                    found_shift = s
                    break
            assert_true(
                found_shift >= 0,
                String("mesh ") + String(mi) + " polygon with normal ("
                + String(mnx) + ", " + String(mny) + ", " + String(mnz)
                + ") has the right vertex COUNT but a different cycle than"
                " MuJoCo's — if the vertices match as a set, the WINDING is"
                " reversed",
            )
            total_matched += 1
            if found_shift != 0:
                rotated += 1
                print(
                    "    mesh", mi, " polygon n = (", mnx, mny, mnz,
                    ") nvert", mnum, " starts at MuJoCo index", found_shift,
                )

    print("  polygons matched (normal + cycle):", total_matched)
    # ⚠ A ROTATION IS NOT COSMETIC. `polygonQuad` is a GREEDY four-pointer walk
    # seeded at ring index 0, not a global maximiser, so the clipped ring's
    # starting vertex changes which quad survives whenever the clip leaves more
    # than four points. Rotation is invisible to every group whose clip returns
    # <= 4 (no pruning runs), which is exactly why the box groups looked exact.
    print("  polygons whose cycle START differs from MuJoCo's:", rotated)
    assert_true(
        total_matched == 14,
        String("expected 14 polygons across the two fixtures (cube 6, hex 8),"
               " matched ") + String(total_matched),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
