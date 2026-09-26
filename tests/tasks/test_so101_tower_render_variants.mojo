"""The tower renderer's speed-ups must not move a pixel — on the tower scene itself.

    pixi run -e apple mojo run -I . tests/tasks/test_so101_tower_render_variants.mojo

Two changes made the rig's tracer faster without being meant to change its
pictures (`noeira-docs/SO101_RENDER_SPEED.md`, levers 1 and 4):

- `make_tower_renderer` compiles the REFLECTION pass out by default (the tower
  scene has no reflective geom, and the factory refuses one that does);
- the mesh trees are built by a binned SAH instead of the reference's median
  split (`bvh_sah`).

So the old configuration (REFLECT on, median trees) and the new default are
rendered here on the same posed lanes — the cube-in-bowl placements the eval
draws, the arm at varied joint angles — and compared EXACTLY: colour, planar
depth and the geom id, both cameras, at the store's pixels (320x240, 4
samples) and at a pixel-RL observation's (128x128, 1 sample). Any difference
is a defect: both legs run the same `ray_triangle` on the same triangles, and
`tests/physics3d/test_ray_bvh_sah.mojo` gates the tree walks on their own.

The best-hit cut (`ray_model` -> `ray_mesh_bvh(tcut)`) and the triangle-index
tie rule are in BOTH legs here (they are not switchable); they are gated
against the linear sweep by `test_ray_bvh_sah.mojo` and, through the renderer,
by `tests/physics3d/test_ray_bvh_matches_linear.mojo`.

Vacuity guards: most pixels must hit geometry, and MESH geoms (the arm, the
stand, the camera mount) must be in both cameras' frames — a picture of an
empty desk would pass while testing no tree.
"""

from std.random import random_float64, seed as seed_rng
from std.sys import exit, has_accelerator
from max.gpu.host import DeviceContext

from noeira.physics3d.constants import GEOM_MESH
from noeira.physics3d.fields import Data, Model
from noeira.physics3d.gpu.constants import MODEL_GEOM_SIZE, GEOM_IDX_TYPE
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.tasks.family import scene_path
from noeira.tasks.family_config import So101TowerConfig
from noeira.tasks.placement.so101_tower import So101TowerPlacement
from noeira.tasks.posed_reset import posed_qpos
from noeira.tasks.so101_tower_rig import (
    RIG_DT, TOWER_MD, RIG_CAM_W, RIG_CAM_H, RIG_SAMPLES, TowerRendererSized,
    make_tower_model, make_tower_renderer, tower_cameras,
)
from noeira.tasks.so101_tower_xml import So101TowerModel
from noeira.tasks.spec import load_family

comptime L = 8
comptime NQ = So101TowerModel.NQ
comptime ARM_JOINTS = 6
comptime TASK = "so101_tower_cube_in_bowl"
comptime FAMILY = "so101_tower"
comptime FAMILY_PATH = "noeira/tasks/families/so101_tower.family"


def check(mut fails: Int, name: String, ok: Bool, detail: String = ""):
    if ok:
        print("  PASS  " + name + "  " + detail)
    else:
        fails += 1
        print("  FAIL  " + name + "  " + detail)


def _pose(ctx: DeviceContext, mut rm: Model[RIG_DT, TOWER_MD],
          mut rd: Data[RIG_DT, TOWER_MD, L]) raises:
    """The eval's placements for seeds 30000.., the arm's six joints drawn
    around its rest so the gripper is over the desk in some lanes."""
    seed_rng(7)
    for e in range(L):
        var q = posed_qpos[So101TowerPlacement](
            String(TASK), String(FAMILY), So101TowerConfig.SLOT_RADIUS,
            UInt64(30000 + e),
        )
        for k in range(NQ):
            rd.qpos.data[e * NQ + k] = Scalar[RIG_DT](q[k])
        for k in range(ARM_JOINTS):
            rd.qpos.data[e * NQ + k] = Scalar[RIG_DT](random_float64(-0.9, 0.9))
    forward_kinematics["cpu", RIG_DT, TOWER_MD, L](rd, rm)
    rd.qpos.upload_resident(ctx)
    rd.xpos.upload_resident(ctx)
    rd.xquat.upload_resident(ctx)
    ctx.synchronize()


def _grab[W: Int, H: Int, S: Int, R: Bool](
    ctx: DeviceContext, mut r: TowerRendererSized[L, W, H, S, R],
    mut rd: Data[RIG_DT, TOWER_MD, L], mut rm: Model[RIG_DT, TOWER_MD],
    cam: Int,
) raises -> Tuple[List[Scalar[RIG_DT]], List[Scalar[RIG_DT]], List[Scalar[RIG_DT]]]:
    comptime NPIX = W * H
    r.render(ctx, rd, rm, cam)
    var hr = ctx.enqueue_create_host_buffer[RIG_DT](L * NPIX * 3)
    var hd = ctx.enqueue_create_host_buffer[RIG_DT](L * NPIX)
    var hs = ctx.enqueue_create_host_buffer[RIG_DT](L * NPIX)
    ctx.enqueue_copy(hr, r.rgb)
    ctx.enqueue_copy(hd, r.depth)
    ctx.enqueue_copy(hs, r.seg)
    ctx.synchronize()
    var rgb = List[Scalar[RIG_DT]](length=L * NPIX * 3, fill=0)
    var dep = List[Scalar[RIG_DT]](length=L * NPIX, fill=0)
    var seg = List[Scalar[RIG_DT]](length=L * NPIX, fill=0)
    for i in range(L * NPIX * 3):
        rgb[i] = hr[i]
    for i in range(L * NPIX):
        dep[i] = hd[i]
        seg[i] = hs[i]
    return (rgb^, dep^, seg^)


def _compare[W: Int, H: Int, S: Int](
    mut fails: Int, ctx: DeviceContext, fmd_path: String,
    mut rm: Model[RIG_DT, TOWER_MD], mut rd: Data[RIG_DT, TOWER_MD, L],
    is_mesh: List[Bool], label: String,
) raises:
    var fmd = parse_model_runtime(fmd_path)
    var cams = tower_cameras(fmd)
    var old = make_tower_renderer[L, W, H, S, True](ctx, fmd, rm, bvh_sah=False)
    var new = make_tower_renderer[L, W, H, S](ctx, fmd, rm)
    for ci in range(2):
        var name = label + (" overhead" if ci == 0 else " wrist")
        var a = _grab[W, H, S, True](ctx, old, rd, rm, cams[ci])
        var b = _grab[W, H, S, False](ctx, new, rd, rm, cams[ci])
        var n = L * W * H
        var ds = 0
        var dd = 0
        var dc = 0
        var hit = 0
        var arm = 0
        for i in range(n):
            var g = Int(a[2][i])
            if g >= 0:
                hit += 1
            if g >= 0 and g < len(is_mesh) and is_mesh[g]:
                arm += 1
            if Int(a[2][i]) != Int(b[2][i]):
                ds += 1
            if a[1][i] != b[1][i]:
                dd += 1
            for c in range(3):
                if a[0][i * 3 + c] != b[0][i * 3 + c]:
                    dc += 1
                    break
        check(fails, name + ": pixels hit geometry", hit > n // 2,
              String(hit) + " / " + String(n))
        check(fails, name + ": meshes are in frame", arm > n // 200,
              String(arm) + " mesh pixels")
        check(fails, name + ": same bytes (REFLECT on + median vs default)",
              ds == 0 and dd == 0 and dc == 0,
              "seg " + String(ds) + ", depth " + String(dd) + ", colour "
              + String(dc) + " differ")


def main() raises:
    comptime if not has_accelerator():
        print("no accelerator — skipped")
        return
    var fails = 0
    var f = load_family(String(FAMILY_PATH))
    var path = scene_path(f)
    var fmd = parse_model_runtime(path)
    var ctx = DeviceContext()
    var rm = make_tower_model(ctx)
    var rd = Data[RIG_DT, TOWER_MD, L]()
    rd.upload_all(ctx)
    _pose(ctx, rm, rd)
    # Which visual rows are MESHES (the arm, the stand, the camera mount): the
    # vacuity guard below needs mesh pixels in frame, not just desk.
    var probe = make_tower_renderer[L, 16, 16, 1](ctx, fmd, rm)
    var is_mesh = List[Bool](length=probe.vis.ngeom + probe.vis.ncond, fill=False)
    var nmesh_geoms = 0
    for g in range(probe.vis.ngeom):
        if Int(probe.vis.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_TYPE]) == GEOM_MESH:
            is_mesh[g] = True
            nmesh_geoms += 1
    print("  visual geoms", probe.vis.ngeom, "| mesh geoms", nmesh_geoms,
          "| meshes", probe.vis.nmesh, "| triangles", probe.vis.ntri)
    check(fails, "the visual set carries meshes", nmesh_geoms > 5,
          String(nmesh_geoms))
    _compare[RIG_CAM_W, RIG_CAM_H, RIG_SAMPLES](fails, ctx, path, rm, rd, is_mesh, "rig 320x240x4")
    _compare[128, 128, 1](fails, ctx, path, rm, rd, is_mesh, "rl 128x128x1")
    if fails > 0:
        print("FAILED:", fails)
        exit(1)
    print("ALL PASS")
