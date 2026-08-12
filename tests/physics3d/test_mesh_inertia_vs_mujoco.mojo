"""Mesh volume / centre of mass / principal axes vs MuJoCo 3.10.0 — gap A core.

Gates `mesh_legacy_inertia` on its own, against MuJoCo compiling the same STL:
`mesh_pos`, `mesh_quat` and the density-scaled principal moments. Wiring it into
`fields_build` is a separate step, and this file is what makes a failure there
attributable to the wiring rather than to the integral.

⚠ WHAT MAKES THIS TEST WORTH HAVING: three plausible readings of "integrate the
mesh" disagree, and two of them are wrong by amounts that look like rounding.
On Jaco's `base` mesh:

    raw triangles, apex at origin   com z = 0.08845968     off by 6.7e-03
    convex hull,   apex at origin   com z = 0.08047642     off by 1.5e-03
    LEGACY (apex at facecen, abs)   com z = 0.0817722373   MuJoCo, to 2.8e-17

The middle one is the trap: it is a defensible algorithm, it is what
`inertia="convex"` does, and it is off by a millimetre and a half.

The meshes come from `manipulation_ref.bake('reach_site_features')`, i.e. the
real Jaco arm — 9 meshes spanning 62 to 504 vertices, three of them reused
across bodies.

Run: pixi run mojo run -I . tests/physics3d/test_mesh_inertia_vs_mujoco.mojo
"""

from std.math import abs as math_abs, sqrt
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.model.mesh_inertia import (
    MeshInertia,
    mesh_legacy_inertia,
    transform_verts_to_principal_frame,
)

comptime DTYPE = DType.float64


def _bake_dir() raises -> String:
    """Bake `reach_site_features` into a temp dir and return it."""
    var sys = Python.import_module("sys")
    _ = sys.path.insert(0, "tests/dm_control")
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    var tempfile = Python.import_module("tempfile")
    var refmod = Python.import_module("manipulation_ref")
    var d = String(tempfile.mkdtemp(prefix="jaco_mesh_"))
    _ = refmod.bake("reach_site_features", d)
    return d


def _read_stl_tris(path: String) raises -> Tuple[List[Scalar[DTYPE]], Int]:
    """Raw triangle soup, 9 scalars per face — the input legacy inertia wants.

    Read here with `struct` rather than through our own STL loader so this
    gates the INTEGRAL and not the loader.
    """
    var builtins = Python.import_module("builtins")
    var struct_m = Python.import_module("struct")
    var f = builtins.open(path, "rb")
    var head = f.read(84)
    var n = Int(py=struct_m.unpack("<I", head[80:84])[0])
    var out = List[Scalar[DTYPE]]()
    for _ in range(n):
        var b = f.read(50)
        var v = struct_m.unpack("<12f", b[0:48])
        for k in range(9):
            out.append(Scalar[DTYPE](Float64(py=v[3 + k])))
    _ = f.close()
    return (out^, n)


def _basename(s: String) -> String:
    """Last path component. ⚠ Written as a FUNCTION because
    `s = String(s[byte=a:b])` is an aliasing error in Mojo — the slice borrows
    the very String being assigned."""
    var slash = s.rfind("/")
    if slash < 0:
        return s
    return String(s[byte = slash + 1 : s.byte_length()])


def test_mesh_inertia_vs_mujoco() raises:
    print("=== mesh legacy inertia vs MuJoCo 3.10.0 ===")
    var mujoco = Python.import_module("mujoco")
    var os = Python.import_module("os")
    var glob = Python.import_module("glob")
    var d = _bake_dir()
    var m = mujoco.MjModel.from_xml_path(String(os.path.join(d, "model.xml")))

    var n_mesh = Int(py=m.nmesh)
    print("  meshes:", n_mesh)
    assert_true(n_mesh == 9, "expected Jaco's 9 meshes, got " + String(n_mesh))

    var worst_pos = Float64(0)
    var worst_quat = Float64(0)
    var failures = 0

    for mi in range(n_mesh):
        # MuJoCo's mesh name is the asset name; the FILE is content-hashed, so
        # find it by the stem PyMJCF wrote.
        var mname = String(
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mi)
        )
        var stem = _basename(mname)
        var hits = glob.glob(String(os.path.join(d, stem + "-*.stl")))
        assert_true(Int(py=builtin_len(hits)) > 0, "no STL for " + stem)
        var path = String(hits[0])

        var tris = _read_stl_tris(path)
        var res = mesh_legacy_inertia[DTYPE](tris[0], tris[1])

        var mj_px = Float64(py=m.mesh_pos[mi][0])
        var mj_py = Float64(py=m.mesh_pos[mi][1])
        var mj_pz = Float64(py=m.mesh_pos[mi][2])
        var dp = math_abs(Float64(res.com_x) - mj_px)
        var t1 = math_abs(Float64(res.com_y) - mj_py)
        var t2 = math_abs(Float64(res.com_z) - mj_pz)
        if t1 > dp:
            dp = t1
        if t2 > dp:
            dp = t2

        # MuJoCo stores (w, x, y, z); ours is (x, y, z, w). The sign of a
        # quaternion is free, so compare both.
        var mw = Float64(py=m.mesh_quat[mi][0])
        var mx = Float64(py=m.mesh_quat[mi][1])
        var my = Float64(py=m.mesh_quat[mi][2])
        var mz = Float64(py=m.mesh_quat[mi][3])
        var dq_p = Float64(0)
        var dq_n = Float64(0)
        var ours = InlineArray[Float64, 4](fill=0.0)
        ours[0] = Float64(res.qw)
        ours[1] = Float64(res.qx)
        ours[2] = Float64(res.qy)
        ours[3] = Float64(res.qz)
        var theirs = InlineArray[Float64, 4](fill=0.0)
        theirs[0] = mw
        theirs[1] = mx
        theirs[2] = my
        theirs[3] = mz
        for k in range(4):
            var a = math_abs(ours[k] - theirs[k])
            var b = math_abs(ours[k] + theirs[k])
            if a > dq_p:
                dq_p = a
            if b > dq_n:
                dq_n = b
        var dq = dq_p if dq_p < dq_n else dq_n

        if dp > worst_pos:
            worst_pos = dp
        if dq > worst_quat:
            worst_quat = dq

        var ok = dp < 1e-12 and dq < 1e-9
        if not ok:
            failures += 1
        print(
            "   mesh",
            mi,
            stem,
            " V=",
            Float64(res.volume),
            " |dpos|=",
            dp,
            " |dquat|=",
            dq,
            " -> ",
            "PASS" if ok else "FAIL",
        )

    print("  WORST |dpos|", worst_pos, " |dquat|", worst_quat)

    # ── the scaled moments, which is what actually reaches the dynamics ─────
    #
    # Every Jaco body has exactly ONE mass-bearing geom, and it is a mesh, so
    # `body_inertia == eigval * (body_mass / volume)` with no parallel-axis
    # term. That makes this a direct check of BOTH the eigenvalues and the
    # recomputed volume — the two quantities a wrong pass would corrupt in
    # compensating directions.
    print("  --- density-scaled principal moments vs body_inertia ---")
    var nbody = Int(py=m.nbody)
    var worst_rel = Float64(0)
    var checked = 0
    for b in range(1, nbody):
        # ⚠ `MjModel` exposes no per-geom mass, so the mass-bearing geom is
        # identified by its FRAME: for a single-contributing-geom body MuJoCo
        # sets `body_ipos`/`body_iquat` to that geom's frame, which here is the
        # mesh's own `mesh_pos`. A body's zero-mass mesh geom (Jaco's
        # `link_6_insert`, the rings) has a different mesh frame and is skipped.
        var mesh_id = -1
        for g in range(Int(py=m.ngeom)):
            if Int(py=m.geom_bodyid[g]) != b:
                continue
            if Int(py=m.geom_type[g]) != Int(py=mujoco.mjtGeom.mjGEOM_MESH):
                continue
            var did = Int(py=m.geom_dataid[g])
            var e = Float64(0)
            for k in range(3):
                var dd = math_abs(
                    Float64(py=m.mesh_pos[did][k])
                    - Float64(py=m.body_ipos[b][k])
                )
                if dd > e:
                    e = dd
            if e < 1e-12:
                mesh_id = did
        if mesh_id < 0:
            continue
        var mname = String(
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        )
        var stem = _basename(mname)
        var hits = glob.glob(String(os.path.join(d, stem + "-*.stl")))
        var tris = _read_stl_tris(String(hits[0]))
        var res = mesh_legacy_inertia[DTYPE](tris[0], tris[1])

        var mass = Float64(py=m.body_mass[b])
        var dens = mass / Float64(res.volume)
        var pred = InlineArray[Float64, 3](fill=0.0)
        pred[0] = Float64(res.eig0) * dens
        pred[1] = Float64(res.eig1) * dens
        pred[2] = Float64(res.eig2) * dens
        var scale = Float64(0)
        var rel = Float64(0)
        for k in range(3):
            var mjv = Float64(py=m.body_inertia[b][k])
            if math_abs(mjv) > scale:
                scale = math_abs(mjv)
        for k in range(3):
            var mjv = Float64(py=m.body_inertia[b][k])
            var e = math_abs(pred[k] - mjv) / scale
            if e > rel:
                rel = e
        if rel > worst_rel:
            worst_rel = rel
        checked += 1
        if rel >= 1e-12:
            failures += 1
            print("   body", b, " REL", rel, " pred", pred[0], pred[1], pred[2])

    print("  bodies checked:", checked, " WORST rel:", worst_rel)
    assert_true(checked == 13, "expected 13 mesh-bearing bodies")
    assert_true(failures == 0, String(failures) + " mesh(es) failed")


def builtin_len(o: PythonObject) raises -> PythonObject:
    var builtins = Python.import_module("builtins")
    return builtins.len(o)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
