"""`.obj` polygon triangulation vs MuJoCo 3.10.0 — the split, not the soup.

MuJoCo loads every `.obj` through its `obj_decoder` plugin, which hands the
file to **`tinyobjloader` with the default `triangulate = true`**. So the faces
a mesh ends up with are tinyobjloader's, and tinyobjloader does NOT fan a
polygon from its first corner:

  * a QUAD splits along the **shorter diagonal** — `[0,1,2] + [0,2,3]` when
    `|v2-v0|^2 < |v3-v1|^2` and `[0,1,3] + [1,2,3]` otherwise, the comparison
    being strict so an exact TIE takes the second form;
  * anything larger is **ear-clipped** in a 2D projection, and the clipper
    GIVES UP after a bounded number of unproductive passes, dropping whatever
    polygon is left — so an n-gon can yield fewer than n-2 triangles.

⚠⚠ WHY THIS IS WORTH A TEST OF ITS OWN. A fan is a defensible triangulation
and it is exact for a planar convex face, so it looks right and it reads right.
It is wrong on every non-planar face, and Menagerie is full of them:

    google_robot  link_finger_base_v.obj   1 quad   volume off 7.5e-04 relative
    stretch_3     link_SG3_gripper_body    5 n-gons centre of mass off 1.4e-04

The google_robot error reached `mj_fullM` as 2.7e-05 of the whole mass matrix
and the stretch_3 one is a 0.14 mm centre-of-mass shift — both small enough to
be filed as "mesh-volume numerics" and left alone, which is exactly what
happened until the model-time mass-matrix column made them visible.

⚠ The tie case is not a curiosity. Menagerie's collision meshes are full of
mirrored quads whose two diagonals are equal to the last bit, and the branch
taken there is decided by float32 rounding WITH FMA CONTRACTION — plain
float32 reproduces MuJoCo on 62 of the library's 66 non-triangular `.obj`
meshes, float64 on 64, the contracted float32 on all 66.

Run: pixi run mojo run -I . \
        tests/physics3d/test_obj_polygon_triangulation_vs_mujoco.mojo
"""

from std.math import abs as math_abs
from std.python import Python
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.model.mesh_inertia import mesh_inertia_from_file
from mojo_rl.render.obj_loader import load_obj

comptime DTYPE = DType.float64

comptime MENAGERIE = String("references/mujoco_menagerie-main/")

# The fan's answers, measured. Present so the assertions below cannot pass by
# accident: each one is checked to be FAR from the number a fan produces.
comptime FAN_GOOGLE_VOL: Float64 = 6.248581627180497e-05
comptime FAN_GOOGLE_COM_Z: Float64 = 4.06440716e-02
comptime FAN_STRETCH_COM_Z: Float64 = 0.04513378
comptime FAN_STRETCH_NTRI: Int = 23788


def _mesh_ref(scene: String, mesh: String) raises -> Tuple[List[Float64], Int]:
    """MuJoCo's own `mesh_pos` and face count for one mesh of one scene."""
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_path(scene)
    var mid = Int(py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, mesh))
    assert_true(
        mid >= 0,
        String("no mesh '") + mesh + "' in " + scene
        + " — the model changed, not the loader",
    )
    var pos = List[Float64]()
    for k in range(3):
        pos.append(Float64(py=m.mesh_pos[mid][k]))
    return (pos^, Int(py=m.mesh_facenum[mid]))


def test_quad_splits_on_the_shorter_diagonal() raises:
    """`google_robot`'s one quad, and the 7.5e-04 it puts in the volume."""
    var scene = MENAGERIE + "google_robot/scene.xml"
    var mref = _mesh_ref(scene, "link_finger_base_v")
    var mj_pos = mref[0].copy()
    var mj_nface = mref[1]

    var path = MENAGERIE + "google_robot/assets/link_finger_base_v.obj"
    var mi = mesh_inertia_from_file[DTYPE](path)
    var md = load_obj(path)
    var ntri = len(md.vertices) // 3

    print("--- google_robot/link_finger_base_v.obj")
    print("    triangles ours", ntri, " MuJoCo", mj_nface)
    print("    volume", Float64(mi.volume), " (a fan gives", FAN_GOOGLE_VOL, ")")
    print(
        "    com   ", Float64(mi.com_x), Float64(mi.com_y), Float64(mi.com_z)
    )
    print("    mesh_pos", mj_pos[0], mj_pos[1], mj_pos[2])

    assert_true(
        ntri == mj_nface,
        String("triangle count ") + String(ntri) + " != MuJoCo "
        + String(mj_nface),
    )
    # `mesh_pos` IS the legacy centre of mass — MuJoCo bakes it into the
    # vertices and records it, so it is a direct read of the integral.
    for k in range(3):
        var d = math_abs(
            Float64(mi.com_x if k == 0 else (mi.com_y if k == 1 else mi.com_z))
            - mj_pos[k]
        )
        assert_true(
            d < 1e-9,
            String("com[") + String(k) + "] off MuJoCo's mesh_pos by "
            + String(d) + " — the quad took the wrong diagonal",
        )
    # ⚠ NON-VACUITY: the fan's centre of mass differs in z at 1.2e-05, which
    # is what this assertion has to be able to see.
    var gap = math_abs(Float64(mi.com_z) - FAN_GOOGLE_COM_Z)
    assert_true(
        gap > 1e-06,
        String("our com_z is within ") + String(gap) + " of the FAN's answer"
        " — this test is not measuring anything",
    )
    var vgap = math_abs(Float64(mi.volume) - FAN_GOOGLE_VOL) / FAN_GOOGLE_VOL
    assert_true(
        vgap > 1e-04,
        String("our volume is within ") + String(vgap) + " relative of the"
        " FAN's — this test is not measuring anything",
    )


def test_ngon_ear_clipping_including_the_give_up_rule() raises:
    """`stretch_3`'s gripper: five n-gons, and twelve triangles MuJoCo drops.

    ⚠ THE FACE COUNT IS THE POINT. Ear clipping a simple polygon yields n-2
    triangles; tinyobjloader's clipper bounds its own iterations and abandons
    whatever is left, so MuJoCo keeps 23,776 faces where n-2 everywhere gives
    23,788. Reproducing the give-up is what makes the centre of mass land.
    """
    var scene = MENAGERIE + "hello_robot_stretch_3/scene.xml"
    var mref = _mesh_ref(scene, "link_SG3_gripper_body")
    var mj_pos = mref[0].copy()
    var mj_nface = mref[1]

    var path = (
        MENAGERIE + "hello_robot_stretch_3/assets/link_SG3_gripper_body.obj"
    )
    var mi = mesh_inertia_from_file[DTYPE](path)
    var md = load_obj(path)
    var ntri = len(md.vertices) // 3

    print("--- hello_robot_stretch_3/link_SG3_gripper_body.obj")
    print(
        "    triangles ours", ntri, " MuJoCo", mj_nface,
        " (a fan gives", FAN_STRETCH_NTRI, ")",
    )
    print(
        "    com   ", Float64(mi.com_x), Float64(mi.com_y), Float64(mi.com_z)
    )
    print("    mesh_pos", mj_pos[0], mj_pos[1], mj_pos[2])

    assert_true(
        ntri == mj_nface,
        String("triangle count ") + String(ntri) + " != MuJoCo "
        + String(mj_nface) + ". " + String(FAN_STRETCH_NTRI) + " means the"
        " polygons were fanned; anything between means the clipper gave up in"
        " a different place",
    )
    for k in range(3):
        var d = math_abs(
            Float64(mi.com_x if k == 0 else (mi.com_y if k == 1 else mi.com_z))
            - mj_pos[k]
        )
        assert_true(
            d < 1e-8,
            String("com[") + String(k) + "] off MuJoCo's mesh_pos by "
            + String(d),
        )
    var gap = math_abs(Float64(mi.com_z) - FAN_STRETCH_COM_Z)
    assert_true(
        gap > 1e-05,
        String("our com_z is within ") + String(gap) + " of the FAN's answer"
        " — this test is not measuring anything",
    )


def test_an_exact_tie_takes_the_second_split() raises:
    """A mirrored quad whose two diagonals are equal, corner by corner.

    `trossen_wxai/link_3_collision.001.obj`'s first face is `0 1 6 4`, and its
    two diagonals are the same length to the last bit of float64. MuJoCo emits
    `[0,1,3] + [1,2,3]`, i.e. the `else` branch, because the comparison is a
    strict `<`. Asserted here on the POSITIONS our loader emits, so it pins the
    branch and not merely the volume — a mirrored quad encloses the same
    volume either way, which is exactly why this case needs its own assertion.
    """
    var path = MENAGERIE + "trossen_wxai/assets/meshes/link_3_collision.001.obj"
    var md = load_obj(path)
    assert_true(len(md.vertices) >= 6, "loader returned no triangles")

    # The four corners of face `f 1 2 7 5`, straight out of the file.
    var v = List[Float32]()
    v.append(Float32(-0.007197)); v.append(Float32(-0.038797)); v.append(Float32(-0.014201))
    v.append(Float32(-0.014201)); v.append(Float32(-0.038797)); v.append(Float32(-0.007197))
    v.append(Float32(0.022794));  v.append(Float32(-0.022799)); v.append(Float32(0.045794))
    v.append(Float32(0.045794));  v.append(Float32(-0.022799)); v.append(Float32(0.022794))

    # else-branch: [0,1,3] then [1,2,3].
    var want = List[Int]()
    want.append(0); want.append(1); want.append(3)
    want.append(1); want.append(2); want.append(3)

    print("--- trossen_wxai/link_3_collision.001.obj, face 0")
    for i in range(6):
        var g = md.vertices[i]
        var w = want[i] * 3
        print(
            "    corner", i, "got", g.px, g.py, g.pz,
            " want v", want[i], "=", v[w], v[w + 1], v[w + 2],
        )
        assert_true(
            g.px == v[w] and g.py == v[w + 1] and g.pz == v[w + 2],
            String("corner ") + String(i) + " is not v" + String(want[i])
            + ". A fan would give [0,1,2] + [0,2,3] here; so would any rule"
            " that resolves the tie the other way.",
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
