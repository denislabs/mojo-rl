"""The overhead pose estimator on the TRACER's frames of the so101-tower scene.

    pixi run mojo run -I . examples/so101/tower_pose_sim_check.mojo
    pixi run mojo run -I . examples/so101/tower_pose_sim_check.mojo \\
        --n 16 --seed 7 --out-dir /tmp/tower_pose

WHAT IT MEASURES. `vision/tabletop_pose.mojo` is gated on flat-shaded
synthetic prisms (`tests/vision/test_tabletop_pose.mojo`). This puts it on
the frames the simulator actually renders: the tower scene's lights and
materials, the bowl's printed MESH (hollow, with an inner wall, not a solid
prism), the arm folded at the family's rest pose, the stand at the image's
bottom edge. Per placement the brick is drawn in `desk_right` with a yaw and
the bowl in `desk_left` — the task's own regions, read from the family — and
the overhead camera is rendered at W x H by `render_lane_cpu` (the GPU
kernel's own pixel function, float64 host leg). The estimator reads the frame
through `Pinhole.sim`, the tracer's convention (`vision/fisheye.mojo`).

Printed per placement and summarised: brick xy error (mm) and yaw error
(deg, modulo 90), bowl xy error (mm), and the confidence (`coverage`,
`residual`). The object-pose plan's target: brick <= 3 mm / 5 deg, bowl
<= 5 mm. The process exits 1 above that, so it can sit in a manifest.

⚠ A POSE, NOT A SIMULATION: nothing is stepped; the props are placed at their
resting heights (as `tower_camera_preview.mojo` does).
⚠ ~8 s per frame at 640x480 on an M1 host; `--width 320` is 4x faster and
halves the brick's pixels.
"""

from std.math import pi, cos, sin, sqrt
from std.os import makedirs
from std.sys import argv, exit

from noeira.io.png import save_png
from noeira.math3d import Vec3 as Vec3Generic
from noeira.physics3d.fields import Data, Model, DynDims
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from noeira.physics3d.raytrace.host_render import render_lane_cpu
from noeira.physics3d.raytrace.visual import build_visual_model
from noeira.tasks.family import scene_path
from noeira.tasks.spec import load_family
from noeira.tasks.so101_tower_camera_pose import tower_sim_camera
from noeira.tasks.so101_tower_xml import (
    SO101_TOWER_MAX_CONTACTS, SO101_TOWER_NMESH_VERTS,
)
from noeira.vision.fisheye import Pinhole
from noeira.vision.tabletop_pose import (
    RigCamera, ColorClass, PrismModel, DeskROI, estimate_prism_pose,
)

comptime DT = DType.float64
comptime Vec3 = Vec3Generic[DT]
comptime FAMILY = "noeira/tasks/families/so101_tower.family"
comptime VISUAL_GROUP_MASK: Int = (1 << 0) | (1 << 2)
comptime FOVY = 73.7398
comptime BRICK_HALF = 0.0125

comptime TARGET_BRICK_MM = 3.0
comptime TARGET_BRICK_DEG = 5.0
comptime TARGET_BOWL_MM = 5.0


struct Rng:
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def u(mut self) -> Float64:
        self.s += UInt64(0x9E3779B97F4A7C15)
        var z = self.s
        z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
        z = z ^ (z >> 31)
        return Float64(z >> 11) / Float64(UInt64(1) << 53)


def _to_byte(x: Float64) -> UInt8:
    var v = x
    if v < 0.0:
        v = 0.0
    if v > 1.0:
        v = 1.0
    return UInt8(Int(v * 255.0 + 0.5))


def _yaw_err_deg(a: Float64, b: Float64, period: Float64) -> Float64:
    var d = a - b
    d = d - period * Float64(Int((d / period) + 1000.5) - 1000)
    return abs(d) * 180.0 / pi


def _mm(ax: Float64, ay: Float64, bx: Float64, by: Float64) -> Float64:
    return sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by)) * 1000.0


def main() raises:
    var args = argv()
    var n = 8
    var seed = 1
    var width = 640
    var out_dir = String("")
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if i + 1 >= len(args):
            raise Error("flag " + a + " needs a value")
        var v = String(args[i + 1])
        if a == "--n":
            n = Int(v)
        elif a == "--seed":
            seed = Int(v)
        elif a == "--width":
            width = Int(v)
        elif a == "--out-dir":
            out_dir = v
        else:
            raise Error("unknown flag " + a)
        i += 2
    var height = (width * 3) // 4

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
    for j in range(len(f.base_qpos)):
        d.qpos.data[j] = Scalar[DT](f.base_qpos[j])

    # free-joint qpos addresses, and a unit quaternion on every free joint
    var brick_adr = -1
    var bowl_adr = -1
    var adr = 0
    for j in range(len(fmd.joints)):
        var name = String(fmd.joint_names[j])
        if fmd.joints[j].nq == 7:
            d.qpos.data[adr + 3] = Scalar[DT](1)
            if name == "brick_free":
                brick_adr = adr
            elif name == "bowl_free":
                bowl_adr = adr
        adr += fmd.joints[j].nq
    if brick_adr < 0 or bowl_adr < 0:
        raise Error("the tower scene has no brick_free / bowl_free joint")

    # the desk surface site and the two strips the task places in
    forward_kinematics["cpu", DT, DynDims, 1](d, m)
    var site = -1
    for s in range(len(fmd.site_names)):
        if String(fmd.site_names[s]).endswith("desk_surface"):
            site = s
    if site < 0:
        raise Error("no desk_surface site")
    var sx = Float64(d.site_xpos.data[site * 3])
    var sy = Float64(d.site_xpos.data[site * 3 + 1])
    var sz = Float64(d.site_xpos.data[site * 3 + 2])
    var ri = f.region_index("desk_right")
    var li = f.region_index("desk_left")
    if ri < 0 or li < 0:
        raise Error("the family has no desk_right / desk_left region")
    var rr = f.regions[ri]
    var rl = f.regions[li]
    print("desk surface at (", sx, sy, sz, ")")
    print(
        "brick strip x [", sx + rr.x_min, sx + rr.x_max, "] y [", sy + rr.y_min,
        sy + rr.y_max, "]; bowl strip x [", sx + rl.x_min, sx + rl.x_max,
        "] y [", sy + rl.y_min, sy + rl.y_max, "]",
    )

    # the camera the tracer renders, as the estimator sees it
    var sim_cam = tower_sim_camera("overhead_cam")
    if not sim_cam.found:
        raise Error("no overhead_cam")
    var cam_idx = -1
    for c in range(len(fmd.camera_names)):
        if String(fmd.camera_names[c]) == sim_cam.name:
            cam_idx = c
    var cam = RigCamera(Pinhole.sim(FOVY, width, height), sim_cam.pos, sim_cam.rot)
    var roi = DeskROI(sz, 0.10, 0.50, -0.25, 0.25)
    var brick = PrismModel.tower_brick()
    var bowl = PrismModel.tower_bowl()
    var cls_brick = ColorClass.tower_brick_sim()
    var cls_bowl = ColorClass.tower_bowl_sim()
    print("camera", sim_cam.name, "at", sim_cam.pos, "->", width, "x", height, "pinhole")
    print("roi", roi)

    var vis = build_visual_model[DT, DynDims](fmd, m, group_mask=VISUAL_GROUP_MASK)
    var rgb = List[Scalar[DT]]()
    var depth = List[Scalar[DT]]()
    var seg = List[Scalar[DT]]()
    var refl = List[Scalar[DT]]()
    var bg = Vec3(0.82, 0.86, 0.90)
    if out_dir != "":
        makedirs(out_dir, exist_ok=True)

    var rng = Rng(UInt64(seed))
    var sum_b = 0.0
    var sum_y = 0.0
    var sum_o = 0.0
    var max_b = 0.0
    var max_y = 0.0
    var max_o = 0.0
    var n_ok = 0
    var n_miss = 0
    for k in range(n):
        var bx = sx + rr.x_min + rng.u() * (rr.x_max - rr.x_min)
        var by = sy + rr.y_min + rng.u() * (rr.y_max - rr.y_min)
        var byaw = rng.u() * 2.0 * pi
        var ox = sx + rl.x_min + rng.u() * (rl.x_max - rl.x_min)
        var oy = sy + rl.y_min + rng.u() * (rl.y_max - rl.y_min)
        var oyaw = rng.u() * 2.0 * pi
        d.qpos.data[brick_adr] = Scalar[DT](bx)
        d.qpos.data[brick_adr + 1] = Scalar[DT](by)
        d.qpos.data[brick_adr + 2] = Scalar[DT](sz + BRICK_HALF)
        d.qpos.data[brick_adr + 3] = Scalar[DT](cos(0.5 * byaw))
        d.qpos.data[brick_adr + 6] = Scalar[DT](sin(0.5 * byaw))
        d.qpos.data[bowl_adr] = Scalar[DT](ox)
        d.qpos.data[bowl_adr + 1] = Scalar[DT](oy)
        d.qpos.data[bowl_adr + 2] = Scalar[DT](sz)
        d.qpos.data[bowl_adr + 3] = Scalar[DT](cos(0.5 * oyaw))
        d.qpos.data[bowl_adr + 6] = Scalar[DT](sin(0.5 * oyaw))
        forward_kinematics["cpu", DT, DynDims, 1](d, m)
        render_lane_cpu[DT, DynDims, 1, False, True, 1](
            d, m, vis, cam_idx, 0, width, height, bg, rgb, depth, seg, refl,
        )
        var px = List[UInt8](length=width * height * 3, fill=UInt8(0))
        for q in range(width * height * 3):
            px[q] = _to_byte(Float64(rgb[q]))
        if out_dir != "":
            save_png(out_dir + "/overhead_" + String(k) + ".png", px, width, height, 3)
        var eb = estimate_prism_pose(px, cam, cls_brick, brick, roi)
        var eo = estimate_prism_pose(px, cam, cls_bowl, bowl, roi)
        if not eb.found or not eo.found:
            n_miss += 1
            print("  [", k, "] MISSED  brick:", eb, " bowl:", eo)
            continue
        var e_b = _mm(eb.x, eb.y, bx, by)
        var e_y = _yaw_err_deg(eb.yaw, byaw, brick.period)
        var e_o = _mm(eo.x, eo.y, ox, oy)
        print(
            "  [", k, "] brick (", bx, by, byaw * 180.0 / pi, "deg ) err", e_b,
            "mm", e_y, "deg, px", eb.n_px, "cov", eb.coverage, "res",
            eb.residual, "| bowl err", e_o, "mm, px", eo.n_px, "cov",
            eo.coverage, "res", eo.residual,
        )
        n_ok += 1
        sum_b += e_b
        sum_y += e_y
        sum_o += e_o
        max_b = max(max_b, e_b)
        max_y = max(max_y, e_y)
        max_o = max(max_o, e_o)

    print()
    if n_ok > 0:
        print(
            "brick xy  mean", sum_b / Float64(n_ok), "max", max_b, "mm (target <=",
            TARGET_BRICK_MM, ")",
        )
        print(
            "brick yaw mean", sum_y / Float64(n_ok), "max", max_y, "deg (target <=",
            TARGET_BRICK_DEG, ")",
        )
        print(
            "bowl xy   mean", sum_o / Float64(n_ok), "max", max_o, "mm (target <=",
            TARGET_BOWL_MM, ")",
        )
    print("found", n_ok, "of", n, "placements")
    if (
        n_miss > 0 or max_b > TARGET_BRICK_MM or max_y > TARGET_BRICK_DEG
        or max_o > TARGET_BOWL_MM
    ):
        print("FAIL")
        exit(1)
    print("PASS")
