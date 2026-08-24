"""SO-ARM101 `wrist_cam` — our composed camera world pose vs `mjData.cam_xpos`.

    pixi run mojo run -I . tests/robots/test_so_arm101_camera_vs_mujoco.mojo

WHAT THIS GATES, AND WHY IT DID NOT EXIST BEFORE
================================================
`mj_camlight` composes every camera's world pose from its PARENT BODY before it
dispatches on `cam_mode`: `cam_xpos = xpos[b] + xmat[b]*cam_pos`,
`cam_xmat = xmat[b]*cam_quat`. We never did. The parent body id was DROPPED at
the `CameraData` -> `RenderFields` boundary, so the render path composed
nothing and drew each camera at its LOCAL pose read as a world pose.

⚠⚠ THAT DEFECT WAS INVISIBLE TO EVERY GATE AND EVERY SCREENSHOT, because every
`<camera>` in every model ported so far sits in `<worldbody>`, where the parent
transform is the identity and the wrong answer equals the right one. SO-101's
`wrist_cam` is the first body-attached camera in the tree, and it is where the
two separate: without the composition the camera sits at (0, 0.04, -0.04) in
WORLD coordinates — under the table, pointing at nothing — while the wrist it
is bolted to swings through the workspace.

⚠ THE CONTROL IS THE POINT. `test_camera_world_pose_vs_mujoco` would pass on a
model whose arm happens to sit near the identity even with the fix reverted, so
it computes the PRE-FIX answer alongside and asserts it is wrong by a wide
margin. A drop in `WORST PRE-FIX` toward zero means the poses stopped
discriminating, not that the engine improved —
`feedback_degenerate-test-pose-gates-nothing`.

⚠ Both sides read OUR XML: MuJoCo compiles `assets/so_arm101.xml`, and the
camera constants come from OUR parser via `make_render_fields`. A defect in the
XML text is therefore invisible here BY CONSTRUCTION —
`tests/robots/so_arm_ref.py::check_camera` is the layer-1 gate that pins the
camera against what was authored. Run it first.

⚠ RUN FROM THE REPO ROOT: `<compiler meshdir>` resolves against the model file,
and MuJoCo is handed that file by path.
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.math3d import Vec3 as Vec3Generic, Quat as QuatGeneric
from mojo_rl.envs.robots.so_arm101 import SoArm101Reach
from mojo_rl.envs.robots.so_arm101_xml import SoArm101Model
from mojo_rl.physics3d.kinematics.camera_frame import (
    camera_world_pos,
    camera_world_quat,
)

comptime Vec3 = Vec3Generic[DType.float64]
comptime Quat = QuatGeneric[DType.float64]

comptime NQ = SoArm101Model.NQ

# Five poses, chosen so the camera's parent body is somewhere different in each.
# ⚠ Joint 4 is `wrist_roll`, the LAST joint above the `gripper` body the camera
# is mounted on, so it rotates the camera without moving the rest of the arm —
# it is the only axis that separates "composed the position" from "composed the
# orientation too". Pose 0 is all-zeros on purpose: it is the one pose where a
# position-only bug can still hide, and it is included so the control number
# below is honest about that rather than averaged away.
comptime NPOSE = 5


def _pose(p: Int, i: Int) -> Float64:
    if p == 0:
        return 0.0
    if p == 1:
        return 0.35 if i == 0 else (-1.10 if i == 1 else 0.90)
    if p == 2:
        return -0.80 if i == 0 else (0.60 if i == 4 else 0.20)
    if p == 3:
        # wrist_roll alone, near its limit: pure camera rotation.
        return 2.50 if i == 4 else 0.0
    return 1.20 if i == 1 else (-0.90 if i == 2 else 0.45)


def _cam_local() raises -> Tuple[Int, Vec3, Quat]:
    """The camera's parent body and its pose in that body's frame, OUR parse."""
    var rf = SoArm101Model.make_render_fields()
    assert_true(
        len(rf.cam_body) == 1,
        "expected exactly one camera in the SO-101 model, got "
        + String(len(rf.cam_body)),
    )
    return (
        rf.cam_body[0],
        Vec3(rf.cam_pos_x[0], rf.cam_pos_y[0], rf.cam_pos_z[0]),
        Quat(rf.cam_quat_w[0], rf.cam_quat_x[0], rf.cam_quat_y[0],
             rf.cam_quat_z[0]),
    )


def _body_quat(env: SoArm101Reach[DType.float64], b: Int) -> Quat:
    """`Data.xquat` is packed (x, y, z, w); `Quat` takes (w, x, y, z)."""
    return Quat(
        Float64(env.d.xquat.data[b * 4 + 3]),
        Float64(env.d.xquat.data[b * 4 + 0]),
        Float64(env.d.xquat.data[b * 4 + 1]),
        Float64(env.d.xquat.data[b * 4 + 2]),
    )


def _body_pos(env: SoArm101Reach[DType.float64], b: Int) -> Vec3:
    return Vec3(
        Float64(env.d.xpos.data[b * 3 + 0]),
        Float64(env.d.xpos.data[b * 3 + 1]),
        Float64(env.d.xpos.data[b * 3 + 2]),
    )


def test_camera_parent_body_survives_the_parse() raises:
    """`cam_body` must name the wrist, not the worldbody.

    ⚠ This is the half that was missing rather than wrong: `CameraData` carried
    `body_id` all along and `RenderFields` had no column to put it in, so the
    value was read and thrown away. A regression here reads as "the camera does
    not move" and nothing else.
    """
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_path(
        "mojo_rl/envs/robots/assets/so_arm101.xml"
    )
    var got = _cam_local()
    var ours = got[0]
    var theirs = Int(py=m.cam_bodyid[0])
    var name = String(
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, ours)
    )
    assert_true(
        ours == theirs,
        "cam_body: ours " + String(ours) + " != MuJoCo " + String(theirs),
    )
    assert_true(
        ours > 0,
        "cam_body is the WORLDBODY — a camera on body 0 cannot move, which is"
        " exactly the bug this file exists for",
    )
    assert_true(
        name == "gripper",
        "cam_body names " + name + ", expected 'gripper'",
    )
    print("  cam_body =", ours, "(" + name + ") — matches MuJoCo")


def test_camera_local_quat_vs_mujoco() raises:
    """Our `euler=` -> quaternion vs `mjModel.cam_quat`, before any composing.

    ⚠ THIS TEST EXISTS BECAUSE AN ATTRIBUTION CAME OUT WRONG. The world-pose
    gate below lands `cam_xmat` at ~1e-11 while `xmat[b]` — the body the camera
    hangs off, same comparison one level down — agrees at ~1e-15. Four orders
    apart means the residual is NOT forward kinematics, which is what it would
    have been comfortable to assume; it enters between the body and the camera,
    and the only thing there is this constant. So it gets its own gate rather
    than an explanation.

    MuJoCo's `eulerseq="xyz"` default is INTRINSIC — `Rx*Ry*Rz`, each rotation
    about the axis the previous ones left. Composing that by hand in float64
    reproduces `cam_quat` at EXACTLY 0.0, so the reference value is not itself
    noisy and the tolerance below is measuring us alone.

    MEASURED 8.5e-12, AND IT IS NOT A CAMERA DEFECT. `_parse_euler_to_quat`
    goes through `xml_parser._sin_cos_f64`, a hand-rolled range-reduced Taylor
    series that exists because the COMPTIME parser cannot call `std.math`. Its
    own docstring already records this floor from the other direction —
    "reducing by 4 instead of 8 leaves ~4e-11 at cheetah's `euler='0 -218 0'`".
    `euler="... 6.28"` is a half-angle of 3.14, i.e. the far end of the
    reduction where three double-angle climbs amplify the truncation most, so
    8.5e-12 is that series at its worst rather than anything about cameras.

    ⚠ IT IS A REAL, WIDER GAP AND IT IS NOT FIXED HERE. Every `euler=` in the
    tree — bodies, geoms, sites, cameras — pays it, on BOTH parser paths: the
    runtime parser imports the same helper from `xml_parser` and so inherits a
    comptime restriction it does not have. The honest fix is a runtime spelling
    backed by `std.math.sin/cos`, which is a change to a shared helper with
    every `euler=` model downstream of it, and it belongs in its own commit
    with its own sweep. Tolerance here is 1e-10: tight enough to catch a
    convention error (degrees, wrong `eulerseq`, dropped attribute — all of
    which move the answer by 1e-2 or more, see `so_arm_ref.check_camera`) and
    loose enough not to re-fail on a known floor it does not own.
    """
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_path(
        "mojo_rl/envs/robots/assets/so_arm101.xml"
    )
    var cam = _cam_local()
    var q = cam[2]
    # `mjModel.cam_quat` is (w, x, y, z) — MJCF's order, not the records'.
    var worst = 0.0
    worst = max(worst, abs(q.w - Float64(py=m.cam_quat[0][0])))
    worst = max(worst, abs(q.x - Float64(py=m.cam_quat[0][1])))
    worst = max(worst, abs(q.y - Float64(py=m.cam_quat[0][2])))
    worst = max(worst, abs(q.z - Float64(py=m.cam_quat[0][3])))
    print("  worst |d cam_quat| ", worst)
    assert_true(
        worst < 1e-10,
        "our euler->quat differs from MuJoCo's by " + String(worst),
    )


def test_camera_world_pose_vs_mujoco() raises:
    """Composed `cam_xpos`/`cam_xmat` vs MuJoCo, over five arm poses."""
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_path(
        "mojo_rl/envs/robots/assets/so_arm101.xml"
    )
    var d = mujoco.MjData(m)

    var cam = _cam_local()
    var cbody = cam[0]
    var clocal_pos = cam[1]
    var clocal_quat = cam[2]

    var env = SoArm101Reach[DType.float64]()
    _ = env.reset()

    var worst_pos = 0.0
    var worst_axis = 0.0
    var worst_prefix = 0.0
    # ⚠ ATTRIBUTION, not decoration, and it CHANGED THE ANSWER. `cam_xmat`
    # lands a few orders looser than `cam_xpos`, and the comfortable reading —
    # "that is just forward kinematics" — is wrong: `xmat[b]`, the same
    # comparison one level down on the very body the camera hangs off, agrees
    # at ~1e-15 while the camera is at ~1e-11. Four orders apart means the
    # residual enters BETWEEN the body and the camera, which sent the hunt to
    # `test_camera_local_quat_vs_mujoco` and found the euler series. Keep this
    # line: without it the number reads as FK noise and the real cause stays
    # hidden.
    var worst_body_axis = 0.0

    for p in range(NPOSE):
        var qp = List[Float64]()
        var qv = List[Float64]()
        mujoco.mj_resetData(m, d)
        for i in range(NQ):
            var v = _pose(p, i)
            qp.append(v)
            qv.append(0.0)
            d.qpos[i] = v
        mujoco.mj_forward(m, d)
        env.set_state(qp, qv)

        var bq = _body_quat(env, cbody)
        var cx = camera_world_pos(_body_pos(env, cbody), bq, clocal_pos)
        var cq = camera_world_quat(bq, clocal_quat)

        for k in range(3):
            var e = abs(
                (cx.x if k == 0 else (cx.y if k == 1 else cx.z))
                - Float64(py=d.cam_xpos[0][k])
            )
            worst_pos = max(worst_pos, e)
            # The control: what the renderer produced before the composition
            # existed — the LOCAL pose read as a world pose.
            var pre = abs(
                (
                    clocal_pos.x if k == 0
                    else (clocal_pos.y if k == 1 else clocal_pos.z)
                )
                - Float64(py=d.cam_xpos[0][k])
            )
            worst_prefix = max(worst_prefix, pre)

        # `cam_xmat` is row-major 3x3; column j is the camera's own j-th axis
        # expressed in world, which is what rotating the unit vector gives.
        for j in range(3):
            var axis = cq.rotate_vec(
                Vec3(
                    1.0 if j == 0 else 0.0,
                    1.0 if j == 1 else 0.0,
                    1.0 if j == 2 else 0.0,
                )
            )
            for r in range(3):
                var want = Float64(py=d.cam_xmat[0][r * 3 + j])
                var got = axis.x if r == 0 else (axis.y if r == 1 else axis.z)
                worst_axis = max(worst_axis, abs(got - want))

        # Same comparison one level down, on the body the camera hangs off.
        for j in range(3):
            var baxis = bq.rotate_vec(
                Vec3(
                    1.0 if j == 0 else 0.0,
                    1.0 if j == 1 else 0.0,
                    1.0 if j == 2 else 0.0,
                )
            )
            for r in range(3):
                var bw = Float64(py=d.xmat[cbody][r * 3 + j])
                var bg = (
                    baxis.x if r == 0 else (baxis.y if r == 1 else baxis.z)
                )
                worst_body_axis = max(worst_body_axis, abs(bg - bw))

    print("  worst |d cam_xpos| ", worst_pos)
    print("  worst |d cam_xmat| ", worst_axis)
    print("  worst |d xmat[b]|  ", worst_body_axis, "(the same, one level down)")
    print("  WORST PRE-FIX      ", worst_prefix, "(the control)")

    # Tolerance is the float32/float64 FK fold, not an epsilon chosen to pass:
    # the composition itself is exact, so anything above this is the body pose
    # disagreeing, which is a different gate's problem.
    assert_true(
        worst_pos < 1e-9,
        "cam_xpos differs from MuJoCo by " + String(worst_pos),
    )
    assert_true(
        worst_axis < 1e-9,
        "cam_xmat differs from MuJoCo by " + String(worst_axis),
    )
    assert_true(
        worst_prefix > 1e-2,
        "THE CONTROL COLLAPSED: reading the camera's local pose as a world"
        " pose is only " + String(worst_prefix) + " away from the right"
        " answer, so this gate would pass with the fix reverted. Pick poses"
        " that move the wrist.",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
