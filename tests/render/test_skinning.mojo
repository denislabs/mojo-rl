"""Gate for linear blend skinning, on dm_control's dog skin.

    pixi run mojo run -I . tests/render/test_skinning.mojo

THE GATE IS THE IDENTITY CASE. Put every body at the bind pose its bone
records, and LBS must hand back the REST MESH — each bone's rotation collapses
to `bindquat * conj(bindquat)` = identity, its translation to
`bindpos - bindpos` = 0, and the per-vertex weights sum to 1.

⚠ THAT ONE CASE IS SENSITIVE TO EVERYTHING AT ONCE: the (w,x,y,z) quaternion
order, the conjugate, whether the rotation matrix is built row- or
column-major, and whether the bone->body map is right. Get any of them wrong
and the posed mesh moves. It is also the only correctness statement about
skinning that can be made without a window and a pair of eyes.

⚠ NEEDS THE REFERENCE TREE, so it SKIPS when `references/dm_control-main/` is
absent rather than going red on a checkout without it.
"""

from std.testing import assert_true, assert_equal, TestSuite
from std.math import abs, sqrt

from mojo_rl.render.skn_loader import load_skn
from mojo_rl.render.skinning import (
    resolve_skin_bones, skin_pose, bind_pose_transforms,
)


comptime SKN_PATH = String(
    "references/dm_control-main/dm_control/suite/dog_assets/dog_skin.skn"
)


def _available() -> Bool:
    try:
        var f = open(SKN_PATH, "r")
        f.close()
        return True
    except:
        return False


def _identity_map(n: Int) -> List[Int]:
    """bone i -> body i. Lets the identity gate run without a dog model."""
    var m = List[Int]()
    for i in range(n):
        m.append(i)
    return m^


def test_bind_pose_reproduces_the_rest_mesh() raises:
    """The load-bearing one — see the header."""
    if not _available():
        print("SKIP: no", SKN_PATH)
        return

    var skin = load_skn(SKN_PATH)
    var nb = len(skin.bones)
    var bone_body = _identity_map(nb)

    var xpos = List[Float32]()
    var xquat = List[Float32]()
    bind_pose_transforms(skin, bone_body, nb, xpos, xquat)

    var posed = List[Float32]()
    var normals = List[Float32]()
    skin_pose(skin, bone_body, xpos, xquat, posed, normals)

    var worst = Float32(0)
    var worst_v = -1
    for i in range(3 * skin.nvert):
        var d = abs(posed[i] - skin.vert[i])
        if d > worst:
            worst = d
            worst_v = i // 3
    print("  max |posed - rest| at bind pose =", worst, " (vertex", worst_v, ")")

    # Float32 LBS over ~2.6 weighted terms per vertex; the mesh spans about a
    # metre, so this is round-off and nothing else.
    assert_true(
        worst < Float32(1e-5),
        "bind pose did not reproduce the rest mesh — check the quaternion"
        " convention, the conjugate, or the matrix layout",
    )


def test_bind_pose_normals_are_unit() raises:
    """Every normal is normalized, and none came out NaN."""
    if not _available():
        print("SKIP: no", SKN_PATH)
        return

    var skin = load_skn(SKN_PATH)
    var nb = len(skin.bones)
    var bone_body = _identity_map(nb)

    var xpos = List[Float32]()
    var xquat = List[Float32]()
    bind_pose_transforms(skin, bone_body, nb, xpos, xquat)

    var posed = List[Float32]()
    var normals = List[Float32]()
    skin_pose(skin, bone_body, xpos, xquat, posed, normals)

    var worst = Float32(0)
    for v in range(skin.nvert):
        var nx = normals[3 * v]
        var ny = normals[3 * v + 1]
        var nz = normals[3 * v + 2]
        var l = sqrt(nx * nx + ny * ny + nz * nz)
        # NaN fails both comparisons, so this catches it too.
        assert_true(l > Float32(0.5), "vertex " + String(v) + " has no normal")
        var d = abs(l - Float32(1.0))
        if d > worst:
            worst = d
    print("  max |‖n‖ - 1| =", worst)
    assert_true(worst < Float32(1e-5), "normals are not unit length")


def test_a_moved_body_moves_only_its_own_vertices() raises:
    """Translating ONE bone must move its vertices and leave the rest alone.

    ⚠ THE SECOND HALF IS THE POINT. A skin that moves everything when one body
    moves would still pass the bind-pose gate — that case is symmetric in the
    bones, so it cannot see a map that sends every bone to the same body.
    """
    if not _available():
        print("SKIP: no", SKN_PATH)
        return

    var skin = load_skn(SKN_PATH)
    var nb = len(skin.bones)
    var bone_body = _identity_map(nb)

    var xpos = List[Float32]()
    var xquat = List[Float32]()
    bind_pose_transforms(skin, bone_body, nb, xpos, xquat)

    var rest_posed = List[Float32]()
    var rest_nrm = List[Float32]()
    skin_pose(skin, bone_body, xpos, xquat, rest_posed, rest_nrm)

    # Shove bone 0's body 1 m along +X. Bone 0 is `torso`.
    comptime SHIFT = Float32(1.0)
    xpos[0] += SHIFT

    var moved = List[Float32]()
    var moved_nrm = List[Float32]()
    skin_pose(skin, bone_body, xpos, xquat, moved, moved_nrm)

    # Vertices bone 0 influences at full weight must have moved by SHIFT;
    # vertices it does not touch at all must not have moved.
    var touched = List[Bool]()
    for _ in range(skin.nvert):
        touched.append(False)
    for k in range(len(skin.bones[0].vert_ids)):
        touched[Int(skin.bones[0].vert_ids[k])] = True

    var n_moved = 0
    var untouched_worst = Float32(0)
    var max_move = Float32(0)
    for v in range(skin.nvert):
        var dx = abs(moved[3 * v] - rest_posed[3 * v])
        if dx > max_move:
            max_move = dx
        if touched[v]:
            if dx > Float32(1e-6):
                n_moved += 1
        else:
            if dx > untouched_worst:
                untouched_worst = dx

    print("  bone 0 binds", len(skin.bones[0].vert_ids), "vertices;",
          n_moved, "moved")
    print("  max displacement =", max_move,
          " max on unbound vertices =", untouched_worst)

    assert_true(n_moved > 0, "moving a bone moved nothing")
    assert_true(
        max_move <= SHIFT + Float32(1e-5),
        "a vertex moved further than the body did",
    )
    assert_true(
        untouched_worst < Float32(1e-6),
        "moving one bone displaced vertices it does not influence",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
