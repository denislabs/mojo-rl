"""`tasks/so101_tower_overhead.mojo`: the overhead camera's pose sources and
the arm calibration's file round trip.

  1. No `--extrinsics`: the pose IS the asset's `overhead_cam`.
  2. The correction `tower_pose_arm_calib.mojo` writes — the camera moved by a
     rotation R about world z and a horizontal shift d, stored as calibration
     extrinsics (camera -> BASE, OpenCV axes, origin in the BASE frame) — read
     back through `tower_overhead_pose`, moves every desk-plane point by
     exactly R p + d. That is the claim the tool's fit rests on, and the file
     path crosses two frame changes (MuJoCo <-> OpenCV axes, world <-> base)
     where a sign or an offset could hide.
  3. CONTROL: the same test with the correction's rotation sign flipped in
     the written file fails by > 5 mm (the check is not vacuous).
  4. A calibration file whose name is not `*_armcal` (the lens file's own
     stale extrinsics) is REFUSED.
"""

from std.math import cos, sin, sqrt, pi
from std.sys import exit

from noeira.math3d import Mat3 as Mat3Generic, Vec3 as Vec3Generic
from noeira.tasks.so101_tower_camera_pose import tower_sim_camera
from noeira.tasks.so101_tower_overhead import tower_overhead_pose, ARMCAL_SUFFIX, DESK_Z
from noeira.vision.calib_file import CameraCalib, write_calib
from noeira.vision.fisheye import Pinhole
from noeira.vision.tabletop_pose import RigCamera

comptime Vec3d = Vec3Generic[DType.float64]
comptime Mat3d = Mat3Generic[DType.float64]


def check(mut fails: Int, name: String, ok: Bool, detail: String):
    if ok:
        print("  PASS  " + name + "  " + detail)
    else:
        fails += 1
        print("  FAIL  " + name + "  " + detail)


def write_armcal(path: String, name: String, th: Float64, d: Vec3d) raises:
    var sim = tower_sim_camera("overhead_cam")
    var c = cos(th)
    var s = sin(th)
    var R = Mat3d.from_cols(Vec3d(c, s, 0.0), Vec3d(-s, c, 0.0), Vec3d(0.0, 0.0, 1.0))
    var pos2 = R * sim.pos + d
    var rot2 = R @ sim.rot
    var cal = CameraCalib(name, -1, 640, 480, 320.0, 320.0, 319.5, 239.5)
    cal.has_extrinsics = True
    cal.rot = Mat3d.from_cols(rot2.col(0), -rot2.col(1), -rot2.col(2))
    cal.trans = pos2 - sim.base_off
    cal.poses = 5
    write_calib(path, cal)


def worst_mm(extr: String, th: Float64, d: Vec3d) raises -> Float64:
    var pin = Pinhole.sim(73.7398, 640, 480)
    var sim = tower_sim_camera("overhead_cam")
    var cam0 = RigCamera(pin, sim.pos, sim.rot)
    var p = tower_overhead_pose(extr)
    var cam1 = RigCamera(pin, p.pos, p.rot_mj)
    var c = cos(th)
    var s = sin(th)
    var worst = 0.0
    for v in range(40, 480, 80):
        for u in range(40, 640, 80):
            var a = cam0.plane_point(Float64(u), Float64(v), DESK_Z)
            var b = cam1.plane_point(Float64(u), Float64(v), DESK_Z)
            if not (a[2] and b[2]):
                continue
            var ex = c * a[0] - s * a[1] + Float64(d.x)
            var ey = s * a[0] + c * a[1] + Float64(d.y)
            worst = max(worst, sqrt((ex - b[0]) ** 2 + (ey - b[1]) ** 2) * 1000.0)
    return worst


def main() raises:
    var fails = 0
    var sim = tower_sim_camera("overhead_cam")
    var p0 = tower_overhead_pose()
    var dp = (p0.pos - sim.pos).length()
    check(fails, "no extrinsics = the asset", dp == 0.0 and p0.source.startswith("asset"), String(dp))

    var th = 0.9 * pi / 180.0
    var d = Vec3d(-0.0045, 0.0021, 0.0)
    var path = String("/tmp/test_so101_tower_overhead_armcal.txt")
    write_armcal(path, String("overhead") + ARMCAL_SUFFIX, th, d)
    var w = worst_mm(path, th, d)
    check(fails, "armcal moves desk points by R p + d", w < 1e-6, "worst " + String(w) + " mm")

    write_armcal(path, String("overhead") + ARMCAL_SUFFIX, -th, d)
    var wc = worst_mm(path, th, d)
    check(fails, "control: flipped rotation is caught", wc > 5.0, "worst " + String(wc) + " mm")

    write_armcal(path, String("overhead"), th, d)
    var refused = False
    try:
        _ = tower_overhead_pose(path)
    except:
        refused = True
    check(fails, "non-armcal extrinsics refused", refused, "")

    if fails > 0:
        print("FAILED:", fails)
        exit(1)
    print("ALL PASS")
