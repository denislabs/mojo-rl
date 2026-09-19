"""Render the `so101_tower` rig's two cameras from a recorded arm state — PNGs.

    pixi run mojo run -I . examples/so101/tower_camera_preview.mojo
    pixi run mojo run -I . examples/so101/tower_camera_preview.mojo \\
        --state 24.4,-105.8,95.8,77.1,59.2,25.1 --bowl 0.30,0.09 --brick 0.33,-0.07 \\
        --width 640 --out-dir /tmp/tower_preview

WHAT IT IS FOR. The family's two `<camera>`s are CAD-derived
(`docs/camera-rig.md`, `props/so101_tower_stand.xml`). Before any calibration
session, the cheapest check of "is the camera roughly where the rig's is" is
to put the arm where a recording had it and look at the sim frame beside the
real one — the tower's foot at the bottom edge of the overhead view, the jaws
at the bottom of the wrist view, the bowl near the overhead centre. This
renders those two frames with the batched tracer's host leg
(`raytrace/host_render.render_lane_cpu` — the SAME `render_pixel` the GPU
kernel runs) and writes `wrist_cam.png` and `overhead_cam.png`.

`--state` is in LeRobot units, as `observation.state` is recorded: five
joints in DEGREES relative to the calibration middle, the gripper 0..100.
The mapping to the model's radians is `robot/so101/sim_map.mojo`'s
(`qpos = deg * pi / 180`, gripper by FRACTION of the joint range); a joint
past the model's range is CLAMPED and reported, because the real
calibration's span disagrees with the model's on some joints
(`sim_map.range_report`).

⚠ THIS IS A POSE, NOT A SIMULATION: nothing is stepped. The bowl and the
brick are put on the mat where `--bowl` / `--brick` say (world x, y), at
their resting heights, and the arm is put at the state. Physics would settle
them by a fraction of a millimetre.

⚠ 320 px WIDE BY DEFAULT, 4:3, ONE SAMPLE, NO SHADOWS: ~2 s per frame on the
host. `--width 640` is the capture resolution and takes four times longer.
The overhead frame's background is the sky colour the tracer uses where no
geom is hit; the desk is the mat plus the floor plane.
"""

from std.math import pi
from std.os import makedirs
from std.sys import argv

from mojo_rl.io.png import save_png
from mojo_rl.math3d import Vec3 as Vec3Generic
from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from mojo_rl.physics3d.raytrace.host_render import render_lane_cpu
from mojo_rl.physics3d.raytrace.visual import build_visual_model
from mojo_rl.tasks.family import scene_path
from mojo_rl.tasks.spec import load_family
from mojo_rl.tasks.so101_tower_xml import (
    SO101_TOWER_MAX_CONTACTS, SO101_TOWER_NMESH_VERTS,
)

comptime DT = DType.float64
comptime Vec3 = Vec3Generic[DT]
comptime FAMILY = "mojo_rl/tasks/families/so101_tower.family"

# The visual group mask: group 0 (props, untagged) and 2 (the arm's `visual`
# class, the stand's meshes). Group 3 — the arm's collision meshes and the
# stand's collision boxes — is NOT drawn, as MuJoCo's viewer defaults.
comptime VISUAL_GROUP_MASK: Int = (1 << 0) | (1 << 2)

# The recorded rest state of `cube-in-bowl` episode 0, frame 0.
comptime DEFAULT_STATE = "24.4,-105.8,95.8,77.1,59.2,25.1"


def _floats(s: String, n: Int, what: String) raises -> List[Float64]:
    var out = List[Float64]()
    for p in s.split(","):
        out.append(Float64(String(String(p).strip())))
    if len(out) != n:
        raise Error(what + " needs " + String(n) + " numbers, got " + s)
    return out^


def _to_byte(x: Float64) -> UInt8:
    var v = x
    if v < 0.0:
        v = 0.0
    if v > 1.0:
        v = 1.0
    return UInt8(Int(v * 255.0 + 0.5))


def main() raises:
    var args = argv()
    var state_s = String(DEFAULT_STATE)
    var bowl_s = String("0.30,0.09")
    var brick_s = String("0.33,-0.07")
    var width = 320
    var out_dir = String("/tmp/tower_preview")
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if i + 1 < len(args):
            var v = String(args[i + 1])
            if a == "--state":
                state_s = v
            elif a == "--bowl":
                bowl_s = v
            elif a == "--brick":
                brick_s = v
            elif a == "--width":
                width = Int(v)
            elif a == "--out-dir":
                out_dir = v
            else:
                raise Error("unknown flag " + a)
            i += 2
        else:
            raise Error("flag " + a + " needs a value")
    var height = (width * 3) // 4
    var state = _floats(state_s, 6, "--state")
    var bowl = _floats(bowl_s, 2, "--bowl")
    var brick = _floats(brick_s, 2, "--brick")

    var f = load_family(String(FAMILY))
    var fmd = parse_model_runtime(scene_path(f))
    var dims = dims_from_flat(
        fmd, max_contacts=SO101_TOWER_MAX_CONTACTS,
        nmesh_verts=SO101_TOWER_NMESH_VERTS,
    )
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var d = Data[DT, DynDims, 1](dims)
    var nq = dims.get_nq()
    for k in range(nq):
        d.qpos.data[k] = Scalar[DT](0)

    # ── the arm: LeRobot units -> the model's radians ────────────────────
    print("arm state (LeRobot units):", state_s)
    for j in range(6):
        ref jt = fmd.joints[j]
        var q = 0.0
        if j == 5:
            var frac = state[5] / 100.0
            q = jt.range_min + frac * (jt.range_max - jt.range_min)
        else:
            q = state[j] * pi / 180.0
        var clamped = q
        if clamped < jt.range_min:
            clamped = jt.range_min
        if clamped > jt.range_max:
            clamped = jt.range_max
        if clamped != q:
            print(
                "  ⚠", String(fmd.joint_names[j]), "=", q,
                "rad is outside the model's [", jt.range_min, ",",
                jt.range_max, "] — clamped (sim_map.range_report)",
            )
        d.qpos.data[j] = Scalar[DT](clamped)

    # ── the props: on the mat, at rest ────────────────────────────────────
    var mat_top = 0.002
    for j in range(len(fmd.joints)):
        var name = String(fmd.joint_names[j])
        if fmd.joints[j].nq != 7:
            continue
        var adr = 0
        for k in range(j):
            adr += fmd.joints[k].nq
        var x = 0.0
        var y = 0.0
        var z = 0.0
        if name == "bowl_free":
            x = bowl[0]
            y = bowl[1]
            z = mat_top  # the bowl's origin is its underside
        elif name == "brick_free":
            x = brick[0]
            y = brick[1]
            z = mat_top + 0.0096
        else:
            continue
        d.qpos.data[adr] = Scalar[DT](x)
        d.qpos.data[adr + 1] = Scalar[DT](y)
        d.qpos.data[adr + 2] = Scalar[DT](z)
        d.qpos.data[adr + 3] = Scalar[DT](1)
        d.qpos.data[adr + 4] = Scalar[DT](0)
        d.qpos.data[adr + 5] = Scalar[DT](0)
        d.qpos.data[adr + 6] = Scalar[DT](0)
        print("  placed", name, "at (", x, y, z, ")")
    forward_kinematics["cpu", DT, DynDims, 1](d, m)

    # ── the tracer ─────────────────────────────────────────────────────────
    var vis = build_visual_model[DT, DynDims](
        fmd, m, group_mask=VISUAL_GROUP_MASK
    )
    print("  " + vis.describe())
    makedirs(out_dir, exist_ok=True)
    var rgb = List[Scalar[DT]]()
    var depth = List[Scalar[DT]]()
    var seg = List[Scalar[DT]]()
    var refl = List[Scalar[DT]]()
    var bg = Vec3(0.82, 0.86, 0.90)
    for cam in range(len(fmd.camera_names)):
        var cname = String(fmd.camera_names[cam])
        render_lane_cpu[DT, DynDims, 1, False, True, 1](
            d, m, vis, cam, 0, width, height, bg, rgb, depth, seg, refl,
        )
        var px = List[UInt8](length=width * height * 3, fill=UInt8(0))
        for k in range(width * height * 3):
            px[k] = _to_byte(Float64(rgb[k]))
        var short = cname
        if cname.startswith("robot_") or cname.startswith("tower_"):
            var trimmed = String(cname[byte = 6 :])
            short = trimmed
        var path = out_dir + "/" + short + ".png"
        save_png(path, px, width, height, 3)
        print("  wrote", path, "(", width, "x", height, ")")
