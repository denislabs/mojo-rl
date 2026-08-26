"""What the batched camera tracer costs on a MESH-FREE scene, and what it looks like.

    pixi run mojo run -I . benchmarks/camera_tracer_cheetah.mojo

The locomotion half of the pair. Run `camera_tracer_lift_brick.mojo` for the
manipulation half; the two print the same table so the numbers line up.

⚠⚠ WHAT THIS PAIR IS MEASURING IS THE MISSING BVH, NOT "locomotion vs
manipulation". `ray/mesh.mojo` is a LINEAR sweep over a mesh's triangle soup
behind an AABB reject — its own docstring says the BVH is the acceleration
structure it does not have. dm_control's suite is entirely primitive-based
(`meshes=0` in all 33 assets); manipulation is the only mesh-heavy family, and
all 13 of its tasks share the same nine Jaco meshes totalling 8 000 triangles.
So this file is the control and `lift_brick` is the treatment.

⚠ THE TWO SCENES DIFFER IN MORE THAN MESHES, and the number should be read with
that in mind: cheetah is 9 geoms, lift_brick is 62. A ratio of ~7 would be
explained by geom count alone. Anything far larger is the triangle sweep. If the
answer lands ambiguously, the clean single-variable control is this same
manipulation scene with `LiftBrickConfig.NMESH_TRI = 0` — identical pixels,
identical geoms, meshes invisible to the ray.

⚠ SEPARATE BINARIES ON PURPOSE. Each `BatchedCameraRenderer[...]` instantiation
is its own Metal kernel compile, and that compile is minutes (see `batch.mojo`).
Two scenes in one file would be two compiles before the first number.
"""

from std.time import perf_counter_ns
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.cheetah import DMCheetahRun
from mojo_rl.physics3d.dynamics.subtree_com import compute_subtree_com
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
comptime CAM = 0  # cheetah's "side" camera — mode="trackcom"
comptime UPSCALE = 4  # host-side nearest neighbour, for the video only

comptime E = DMCheetahRun[DT]
comptime R = BatchedCameraRenderer[DT, E.MD, 1, W, H]


def main() raises:
    comptime if not has_accelerator():
        print("no accelerator — this benchmark measures the GPU leg")
        return

    var ctx = DeviceContext()
    var env = E()
    _ = env.reset()
    # ⚠ `subtree_com` BEFORE `init_camera_reference`: cheetah's cameras are
    # BOTH `mode="trackcom"`, so the reference pose is taken relative to the
    # subtree CoM and the renderer's constructor refuses a camera whose
    # reference was never filled.
    compute_subtree_com["cpu", DT, E.MD, 1](env.d, env.mf)
    init_camera_reference(env.d, env.mf)
    env.mf.upload_all(ctx)
    env.d.upload_all(ctx)
    ctx.synchronize()

    print("scene   : dm_control cheetah")
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

    # ── warm up, then time the GPU leg ────────────────────────────────────
    r.render(ctx, env.d, env.mf)
    ctx.synchronize()

    var rec = VideoRecorder()
    rec.start(String("camera_tracer_cheetah.mp4"), fps=30)
    var frame = List[UInt8]()

    var total_ns = 0
    var act = E.ActionType()
    for _f in range(FRAMES):
        # Physics on the CPU, then upload — this env has a batched twin but
        # the manipulation half does not, so both run at one lane on the CPU
        # to keep the comparison honest. The upload is OUTSIDE the timer.
        _ = env.step(act)
        compute_subtree_com["cpu", DT, E.MD, 1](env.d, env.mf)
        env.d.upload_all(ctx)
        ctx.synchronize()

        var s = perf_counter_ns()
        r.render(ctx, env.d, env.mf)
        ctx.synchronize()
        total_ns += perf_counter_ns() - s

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
