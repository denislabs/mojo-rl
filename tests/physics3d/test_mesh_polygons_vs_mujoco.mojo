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

from std.math import abs, sqrt, acos
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.collision.convex_hull import load_mesh_hull
from mojo_rl.physics3d.collision.mesh_polygons import polygon_normal
from mojo_rl.physics3d.model.mesh_inertia import MeshInertia
from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.fields import Data, Model, Dims
from mojo_rl.physics3d.model.model_dims import ModelDims
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
    <mesh name="ankle" file="references/mujoco_menagerie-main/toddlerbot_2xc/assets/left_ankle_roll_link_collision.stl"/>
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
    <body name="c" pos="6 0 0.5">
      <joint name="jc" type="free"/>
      <geom name="gc" type="mesh" mesh="ankle"/>
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
comptime MD = ModelDims[MPM, 64]
comptime Mod = Model[DTYPE, MD]

comptime TOL: Float64 = 1e-6


def test_mesh_polygons_vs_mujoco() raises:
    var ctx = DeviceContext()
    var mf = Mod()
    MPM.init_fields[DTYPE](ctx, mf)

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
        total_matched == 61,
        String("expected 61 polygons across the three fixtures (cube 6,"
               " hex 8, ankle 47), matched ") + String(total_matched),
    )


def test_a_fixture_has_a_non_identity_mesh_frame() raises:
    """⚠⚠ THE FIRST TWO FIXTURES CANNOT SEE THE FRAME DEFECT, BY CONSTRUCTION.

    `mc_cube` and `mc_hex` are authored centred and axis-aligned, so MuJoCo's
    principal-axis transform is the identity and every frame is the same
    frame. That is what made this gate GREEN for the whole life of the defect
    below: the polygon partition is decided by the QUANTISED DIRECTION of each
    hull triangle's normal, and a rotation of zero degrees moves nothing
    across a bucket boundary.

    `mjCMesh::Process` (user_mesh.cc:1350) runs `MakeGraph` (:1387) and
    `MakePolygons` (:1422) on `dvert` while it still holds the RAW FILE
    VALUES, then `ApplyTransformations` (:1444), then the CoM shift and the
    principal-axis `Rotate` (:1517-1524), and only then
    `MakePolygonNormals` (:1538). We partitioned in the principal frame, so on
    any mesh MuJoCo actually re-frames we merged a different set of triangles.

    So this row is the gate on the gate: it asserts that at least one fixture
    IS re-framed, and prints how far. Delete the ankle mesh from `MP_XML` and
    this fails rather than silently going vacuous again.

    Measured on the ankle mesh — polygon count against MuJoCo's 47:

        this engine, principal frame (what shipped)   46
        Python transcription on MuJoCo's OWN hull faces:
          principal frame                             45
          file frame, without `+ 0.0` in the key       48
          file frame, with it                          47

    ⚠ The two disagree by one because the transcription is fed MuJoCo's hull
    and this engine builds its own; that is the point of running both.
    """
    print("=== is any fixture actually re-framed? ===")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(String(MP_XML))
    var worst_ang = Float64(0)
    var worst_off = Float64(0)
    for mi in range(Int(py=m.nmesh)):
        var qw = abs(Float64(py=m.mesh_quat[mi][0]))
        if qw > 1.0:
            qw = 1.0
        var ang = 2.0 * acos(qw) * (180.0 / 3.141592653589793)
        var off = sqrt(
            Float64(py=m.mesh_pos[mi][0]) ** 2
            + Float64(py=m.mesh_pos[mi][1]) ** 2
            + Float64(py=m.mesh_pos[mi][2]) ** 2
        )
        print("  mesh", mi, " principal rotation ~", ang, "deg  CoM offset",
              off)
        if ang > worst_ang:
            worst_ang = ang
        if off > worst_off:
            worst_off = off
    assert_true(
        worst_ang > 5.0,
        String("every fixture mesh has an (almost) identity principal"
               " rotation — worst is ") + String(worst_ang) + " deg, so this"
        " file cannot see a frame defect in the polygon partition at all",
    )


def _normals_are_unit(name: String, path: String) raises -> Int:
    """Load one real mesh and check every stored polygon normal is a UNIT
    vector. Returns how many polygons the REFERENCE's rule could not answer.

    ⚠⚠ THE RETURN VALUE IS THE NON-VACUITY COUNT, not a diagnostic. A mesh
    whose first-three-path-vertices rule never hits a degenerate triple cannot
    fail this assertion however broken the fallback is, so a caller that gets
    0 back has proved nothing and says so.
    """
    var mesh_vert = List[Scalar[DTYPE]]()
    var mesh_vertadr = List[Int]()
    var mesh_vertnum = List[Int]()
    var num_meshes = 0
    var mesh_polyadr = List[Int]()
    var mesh_polynum = List[Int]()
    var poly_vert = List[Int]()
    var poly_vertadr = List[Int]()
    var poly_vertnum = List[Int]()
    var poly_normal = List[Scalar[DTYPE]]()
    var polymap = List[Int]()
    var polymap_adr = List[Int]()
    var polymap_num = List[Int]()
    var edge_adr = List[Int]()
    var edge_list = List[Int]()
    # ⚠ The triangle SOUP, added when `mj_rayMesh` needed the mesh's original
    # faces. This file did not compile from that change until now — and it is
    # the gate that compares our merged polygons against MuJoCo's, so it is the
    # one a hull change most needs.
    var mesh_tri = List[Scalar[DTYPE]]()
    var mesh_triadr = List[Int]()
    var mesh_trinum = List[Int]()
    var mi = MeshInertia[DTYPE]()
    _ = load_mesh_hull[DTYPE](
        path, mesh_vert, mesh_vertadr, mesh_vertnum, num_meshes,
        mesh_polyadr, mesh_polynum, poly_vert, poly_vertadr, poly_vertnum,
        poly_normal, polymap, polymap_adr, polymap_num, edge_adr, edge_list,
        mesh_tri, mesh_triadr, mesh_trinum,
        mi,
    )
    var npoly = len(poly_vertnum)
    var worst = Float64(1e30)
    var degenerate = 0
    for p in range(npoly):
        var nx = Float64(poly_normal[p * 3 + 0])
        var ny = Float64(poly_normal[p * 3 + 1])
        var nz = Float64(poly_normal[p * 3 + 2])
        var l2 = nx * nx + ny * ny + nz * nz
        if l2 < worst:
            worst = l2
        # how many polygons the reference's own rule cannot answer: the cross
        # product of the FIRST THREE path vertices is shorter than `mjEPS`.
        var adr = poly_vertadr[p]
        var a = poly_vert[adr + 0] * 3
        var b = poly_vert[adr + 1] * 3
        var c = poly_vert[adr + 2] * 3
        var ux = Float64(mesh_vert[b + 0] - mesh_vert[a + 0])
        var uy = Float64(mesh_vert[b + 1] - mesh_vert[a + 1])
        var uz = Float64(mesh_vert[b + 2] - mesh_vert[a + 2])
        var vx = Float64(mesh_vert[c + 0] - mesh_vert[a + 0])
        var vy = Float64(mesh_vert[c + 1] - mesh_vert[a + 1])
        var vz = Float64(mesh_vert[c + 2] - mesh_vert[a + 2])
        var cx = uy * vz - uz * vy
        var cy = uz * vx - ux * vz
        var cz = ux * vy - uy * vx
        if sqrt(cx * cx + cy * cy + cz * cz) < 1e-14:
            degenerate += 1
    print("  ", name, " polygons", npoly, " worst |n|^2", worst,
          " first-triple degenerate", degenerate)
    assert_true(
        abs(worst - 1.0) < 1e-9,
        name + ": a stored polygon normal has |n|^2 = " + String(worst)
        + ", so it is not a unit vector. Zero means the face can never match"
        " anything in `alignedFaces` and is silently unreachable; anything"
        " else means the direction is rounding noise that WILL match"
        " something, and the manifold gets clipped against a plane that is"
        " not the polygon's.",
    )
    return degenerate


def test_polygon_normals_are_unit_on_cad_meshes() raises:
    """Every stored polygon normal is a unit vector, on real CAD hulls.

    ⚠⚠ THE FIXTURES ABOVE CANNOT SEE THIS. `mc_cube`, `mc_hex` and the ankle
    mesh are small and well conditioned, so the first three vertices of every
    polygon path span a real triangle and the degenerate branch is never
    reached. The defect this pins lives on machined parts: a polygon whose
    boundary opens with a straight run of vertices makes
    `cross(p1 - p0, p2 - p0)` vanish, and the rule MuJoCo uses —
    `MakePolygonNormals` over the first three path vertices — has no answer.

    What shipped before was worse than no answer in BOTH directions. An exact
    zero was left un-normalised and stored as (0, 0, 0): `alignedFaces` dots it
    against every candidate, gets 0, and that face can never be chosen — a
    silently unreachable polygon. A cross product merely SHORTER than `mjEPS`
    was normalised anyway, turning pure rounding noise into a confident unit
    vector.

    ⚠⚠ THE FIXTURES MOVED ONCE ALREADY, AND THE NON-VACUITY ROW IS WHY THAT
    WAS NOTICED. When this was written the population was 2 759 polygons over
    the whole tree and the three so_arm fixtures carried 5 of them. The hull
    VERTEX REDUCTION then removed most of the collinear boundary runs that
    caused it: 36 polygons over 596 041 now, and those three fixtures carry
    ZERO. The assertion below caught that immediately rather than going quietly
    green on nothing.

    ⚠⚠ AND THEY MOVED A SECOND TIME, FOR THE SAME REASON AND CAUGHT THE SAME
    WAY. The hull is qhull's now (`convex_hull._qhull_hull`), which changes
    which boundary runs survive the merge, and the four survivors above went to
    ZERO again. The fixtures are now aloha's aluminium extrusions, chosen from
    the REFERENCE side rather than ours: MuJoCo stores `mjuu_makenormal`'s
    `(1, 0, 0)` placeholder for exactly 88 polygons across 37 Menagerie scenes,
    and nine aloha meshes carry two each. Picking them from `mesh_polynormal`
    means the population cannot quietly drain away when OUR hull moves again —
    it can only move if the REFERENCE's does.
    """
    print("=== polygon normals are unit vectors ===")
    var total = 0
    total += _normals_are_unit(
        "aloha extrusion_150",
        "references/mujoco_menagerie-main/aloha/assets/extrusion_150.stl",
    )
    total += _normals_are_unit(
        "aloha extrusion_600",
        "references/mujoco_menagerie-main/aloha/assets/extrusion_600.stl",
    )
    total += _normals_are_unit(
        "aloha angled_extr  ",
        "references/mujoco_menagerie-main/aloha/assets/angled_extrusion.stl",
    )
    total += _normals_are_unit(
        "aloha corner_brckt ",
        "references/mujoco_menagerie-main/aloha/assets/corner_bracket.stl",
    )
    print("   polygons whose first triple is degenerate, in total:", total)

    # ⚠⚠ THE NON-VACUITY CHECK IS SYNTHETIC NOW, AND THAT IS THE FIX, NOT A
    # WEAKENING. It used to demand that the CORPUS supply polygons with a
    # degenerate first triple, and the population drained twice — once when the
    # hull vertex reduction landed (2 759 -> 36) and again when the hull became
    # qhull's, each time taking the chosen fixtures to ZERO. Both times the
    # assertion caught it, and both times the answer was to hunt for new
    # fixtures. That hunt cannot end: whether a machined part's boundary run is
    # degenerate is a KNIFE EDGE at `mjEPS = 1e-14` on the cross product of
    # three nearly-collinear vertices, so it flips on the frame the normal is
    # computed in. MuJoCo stores the `(1, 0, 0)` placeholder for 88 polygons
    # across 37 Menagerie scenes; we agree on the RULE and not on which 88.
    #
    # A hand-built collinear polygon tests the branch directly and cannot drain
    # away. `polygon_normal` with `first_three=True` is the path qhull's cycle
    # start selects, and MuJoCo's answer there is `mjuu_makenormal`'s literal
    # `(1, 0, 0)`.
    var cv = List[Scalar[DTYPE]]()
    for k in range(4):
        cv.append(Scalar[DTYPE](Float64(k)))   # (0,0,0) (1,1,1) (2,2,2) (3,3,3)
        cv.append(Scalar[DTYPE](Float64(k)))
        cv.append(Scalar[DTYPE](Float64(k)))
    var cpv = List[Int]()
    for k in range(4):
        cpv.append(k)
    var dn = polygon_normal[DTYPE](cv, 0, cpv, 0, 4, True)
    print(
        "   collinear polygon ->", dn[0], dn[1], dn[2],
        " (MuJoCo's `mjuu_makenormal` placeholder is 1 0 0)",
    )
    assert_true(
        Float64(dn[0]) == 1.0
        and Float64(dn[1]) == 0.0
        and Float64(dn[2]) == 0.0,
        "a polygon whose first three path vertices are COLLINEAR must return"
        " `mjuu_makenormal`'s placeholder (1, 0, 0) when the cycle start is"
        " qhull's, because that is what `mesh_polynormal` holds for the 88"
        " polygons of the reference that hit it — and `alignedFaces` compares"
        " against the reference's stored value. Got ("
        + String(dn[0]) + ", " + String(dn[1]) + ", " + String(dn[2]) + ").",
    )
    _ = cv^
    _ = cpv^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
