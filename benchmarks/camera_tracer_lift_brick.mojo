"""What the batched camera tracer costs on a MESH-HEAVY scene, and what it looks like.

    pixi run mojo run -I . benchmarks/camera_tracer_lift_brick.mojo

The manipulation half of the pair. Run `camera_tracer_cheetah.mojo` for the
mesh-free control; the two print the same table so the numbers line up.

⚠⚠ WHAT THIS PAIR IS MEASURING IS THE MISSING BVH, NOT "locomotion vs
manipulation". `ray/mesh.mojo` is a LINEAR sweep over a mesh's triangle soup
behind an AABB reject — its own docstring says the BVH is the acceleration
structure it does not have. dm_control's suite is entirely primitive-based
(`meshes=0` in all 33 assets); manipulation is the only mesh-heavy family, and
all 13 of its tasks share the same nine Jaco meshes totalling 8 000 triangles.
So this file is the control and `lift_brick` is the treatment.

⚠⚠ THE CROSS-SCENE RATIO IS CONFOUNDED AND THIS FILE DOES NOT RELY ON IT.
cheetah is 9 geoms, lift_brick is 62, so a ~7x difference is explained by geom
count alone with no meshes involved. The measurement that actually isolates the
triangle sweep is INSIDE this file: render the scene, then zero every mesh's
`TRINUM` and render it again. Same geoms, same camera, same pixels, same
binary — `ray_mesh` simply returns NO HIT, so the difference is the sweep and
nothing else.

⚠ THE SOUP-OFF LEG IS ALSO A PICTURE YOU SHOULD NOT SHIP. With `TRINUM = 0` the
arm is INVISIBLE — rays pass through it to the floor. That is exactly what
every env in this tree rendered before `NMESH_TRI` was plumbed through
`ModelDims` on 2026-08-26, and it is why the control is worth running once:
the failure is a clean picture of an empty workspace, not an error.
"""

from std.time import perf_counter_ns
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.manipulation_lift_brick import DMLiftBrick
from mojo_rl.physics3d.dynamics.subtree_com import compute_subtree_com
from mojo_rl.physics3d.gpu.constants import (
    MAX_GPU_MESHES,
    MODEL_MESH_META_SIZE,
    MESH_META_IDX_TRINUM,
)
from mojo_rl.physics3d.raytrace import (
    BatchedCameraRenderer,
    init_camera_reference,
)
from mojo_rl.render.video_recorder import VideoRecorder

# ⚠ float32. Metal rejects `double`, so the whole env runs at float32 — the
# renderer reads `Data` in place and cannot convert.
comptime DT = DType.float32

# 84x84 is dm_control's own camera observable size
# (`manipulation/shared/observations.py`, `_DISABLED_CAMERA`), so both halves of
# this pair render exactly what an agent would be handed.
comptime W = 84
comptime H = 84
comptime FRAMES = 60
comptime CAM = 0  # the only camera: "front_close", a worldbody FIXED camera
comptime UPSCALE = 4  # host-side nearest neighbour, for the video only

comptime E = DMLiftBrick[DT]
comptime R = BatchedCameraRenderer[DT, E.MD, 1, W, H]
comptime R_NOSHADOW = BatchedCameraRenderer[
    DT, E.MD, 1, W, H, SHADOWS=False
]


def main() raises:
    comptime if not has_accelerator():
        print("no accelerator — this benchmark measures the GPU leg")
        return

    var ctx = DeviceContext()
    var env = E()
    _ = env.reset()
    # ⚠ `front_close` is a `<worldbody>` FIXED camera, so it reads no
    # reference pose at all — but the call is kept because it costs nothing and
    # because a task that later adds a tracking camera would otherwise fail at
    # the renderer's constructor with no clue why.
    compute_subtree_com["cpu", DT, E.MD, 1](env.d, env.mf)
    init_camera_reference(env.d, env.mf)
    env.mf.upload_all(ctx)
    env.d.upload_all(ctx)
    ctx.synchronize()

    print("scene   : dm_control manipulation/lift_brick")
    print("geoms   :", E.MD.NGEOM)
    print("mesh tri:", E.MD.NMESH_TRI, " (0 = no triangle soup)")
    print("camera  :", CAM, " resolution", W, "x", H, " lanes", 1)

    var r = R(ctx, env.mf, CAM)

    # ── one CPU frame, for the same-code reference ────────────────────────
    var crgb = List[Scalar[DT]]()
    var cdep = List[Scalar[DT]]()
    var cseg = List[Scalar[DT]]()
    var t0 = perf_counter_ns()
    r.render_cpu(env.d, env.mf, crgb, cdep, cseg)
    var cpu_ms = Float64(perf_counter_ns() - t0) / 1.0e6
    var hits = 0
    for i in range(len(cseg)):
        if Int(cseg[i]) >= 0:
            hits += 1
    print("cpu 1 frame :", cpu_ms, "ms  (", hits, "of", W * H, "px hit)")

    # ⚠ THE HIT COUNT IS PRINTED BESIDE THE TIMES. A camera pointed at empty
    # sky renders very fast and measures nothing.
    if hits == 0:
        print("!! nothing in frame — the timings below are meaningless")

    # ── the three timed legs ──────────────────────────────────────────────
    #
    # ⚠ ONE VARIABLE EACH, IN ORDER: shadows on/off with the soup unchanged,
    # then the soup off with shadows back on. Anything else moving between two
    # rows would make the difference unattributable.
    comptime REPS = 20

    r.render(ctx, env.d, env.mf)
    ctx.synchronize()
    var t = perf_counter_ns()
    for _i in range(REPS):
        r.render(ctx, env.d, env.mf)
    ctx.synchronize()
    var soup_shadow = Float64(perf_counter_ns() - t) / Float64(REPS) / 1.0e6

    var rn = R_NOSHADOW(ctx, env.mf, CAM)
    rn.render(ctx, env.d, env.mf)
    ctx.synchronize()
    t = perf_counter_ns()
    for _i in range(REPS):
        rn.render(ctx, env.d, env.mf)
    ctx.synchronize()
    var soup_noshadow = Float64(perf_counter_ns() - t) / Float64(REPS) / 1.0e6

    # ⚠⚠ THE CONTROL. `TRINUM = 0` makes `ray_mesh` return NO HIT without
    # touching a geom, a pose, a pixel or the kernel — so the delta against
    # `soup_shadow` is the triangle sweep and nothing else.
    for m in range(MAX_GPU_MESHES):
        env.mf.mesh_meta.data[
            m * MODEL_MESH_META_SIZE + MESH_META_IDX_TRINUM
        ] = Scalar[DT](0)
    env.mf.upload_all(ctx)
    ctx.synchronize()
    r.render(ctx, env.d, env.mf)
    ctx.synchronize()
    t = perf_counter_ns()
    for _i in range(REPS):
        r.render(ctx, env.d, env.mf)
    ctx.synchronize()
    var nosoup_shadow = Float64(perf_counter_ns() - t) / Float64(REPS) / 1.0e6

    # And the picture that proves the control did what it says.
    var nrgb = List[Scalar[DT]]()
    var ndep = List[Scalar[DT]]()
    var nseg = List[Scalar[DT]]()
    r.render_cpu(env.d, env.mf, nrgb, ndep, nseg)
    var nhits = 0
    for i in range(len(nseg)):
        if Int(nseg[i]) >= 0:
            nhits += 1

    print("")
    print("  soup ON , shadows ON  :", soup_shadow, "ms/frame")
    print("  soup ON , shadows OFF :", soup_noshadow, "ms/frame")
    print("  soup OFF, shadows ON  :", nosoup_shadow, "ms/frame  <- control")
    print("  => the triangle sweep is", soup_shadow / nosoup_shadow, "x")
    print("  => the shadow ray is   ", soup_shadow / soup_noshadow, "x")
    print("  control frame hits", nhits, "of", W * H, "px (arm invisible)")

    # ── restore the soup and record the video ─────────────────────────────
    var env2 = E()
    _ = env2.reset()
    compute_subtree_com["cpu", DT, E.MD, 1](env2.d, env2.mf)
    init_camera_reference(env2.d, env2.mf)
    env2.mf.upload_all(ctx)
    env2.d.upload_all(ctx)
    ctx.synchronize()

    var rec = VideoRecorder()
    rec.start(String("camera_tracer_lift_brick.mp4"), fps=30)
    var frame = List[UInt8]()

    var total_ns = 0
    var act = E.ActionType()
    for _f in range(FRAMES):
        _ = env2.step(act)
        compute_subtree_com["cpu", DT, E.MD, 1](env2.d, env2.mf)
        env2.d.upload_all(ctx)
        ctx.synchronize()

        var st = perf_counter_ns()
        r.render(ctx, env2.d, env2.mf)
        ctx.synchronize()
        total_ns += perf_counter_ns() - st

        r.frame_bgra(ctx, 0, frame, UPSCALE)
        rec.add_frame_bgra(
            Int(frame.unsafe_ptr()), W * UPSCALE, H * UPSCALE
        )
    rec.stop()

    var per_frame_ms = Float64(total_ns) / Float64(FRAMES) / 1.0e6
    print("gpu 1 lane  :", per_frame_ms, "ms/frame")
    print(
        "            :",
        Float64(total_ns) / Float64(FRAMES) / Float64(W * H),
        "ns/pixel",
    )
