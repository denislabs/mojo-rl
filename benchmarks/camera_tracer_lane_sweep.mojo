"""How the batched camera tracer SCALES with lanes, and whether the BVH gap holds.

    pixi run -e nvidia mojo run -I . benchmarks/camera_tracer_lane_sweep.mojo

⚠⚠ WHY THIS FILE EXISTS: `camera_tracer_lift_brick.mojo` runs at ONE lane, and
one lane is the least representative point on a large GPU. 84x84 is 7 056
threads; an RTX 5090 has 21 760 CUDA cores, so a single-lane launch puts about
a third of the machine under one thread each with no occupancy to hide memory
latency. BOTH arms of the soup ablation are latency-bound at that size, so
neither the ms/frame nor the 76x ratio measured there is a throughput number.
This file sweeps the lane count so the ratio can be read at the batch size
training would actually use.

⚠⚠ THE RATIO IS THE POINT, NOT THE THROUGHPUT. The 5090 measured the triangle
sweep at 76.1x against Apple's 34.9x — the sweep did not get slower (it got
3.75x faster), the PRIMITIVE path got 8.3x faster and pulled the denominator
down. Whether that gap survives at saturation decides how much a BVH is worth,
and it cannot be answered from one lane. Read the `sweep_x` COLUMN, not any
single cell.

⚠ EVERY LANE RENDERS THE SAME POSE, and that is a BEST CASE this file does not
hide. Identical lanes read identical triangles in identical order, so the L2
sees one working set and every warp diverges the same way. Real training lanes
hold different poses and will do worse. `JITTER` below translates lane `i`'s
scene rigidly to break the lockstep; it is off by default because the clean
number is the reproducible one, and the two together bracket the truth.

⚠ THE FULL `Data` IS ALLOCATED PER LANE COUNT even though the renderer reads
only four of its fields (`xpos`, `xquat`, `subtree_com`, `hfield_data`).
`render` takes a `Data`, so this is what the type costs. At 1 024 lanes that is
the whole physics state 1 024 times over — if a box runs out of memory, drop
`LANES_1024` rather than trimming the sweep from the small end, where the
interesting curvature is.

⚠ `REPS` FALLS AS LANES RISE. A 1 024-lane frame is ~1 024x the pixels of a
single-lane one; holding reps at 20 would make the tail of this sweep take
longer than the rest of it combined. The counts are per-row and printed.
"""

from std.time import perf_counter_ns
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.manipulation_lift_brick import DMLiftBrick
from mojo_rl.physics3d.dynamics.subtree_com import compute_subtree_com
from mojo_rl.physics3d.fields import Data
from mojo_rl.physics3d.gpu.constants import (
    MAX_GPU_MESHES,
    MODEL_MESH_META_SIZE,
    MESH_META_IDX_TRINUM,
)
from mojo_rl.physics3d.raytrace import (
    BatchedCameraRenderer,
    init_camera_reference,
)

# float32: Metal rejects `double` and the renderer reads `Data` in place.
comptime DT = DType.float32

# dm_control's own camera observable size — the pixels an agent is handed.
comptime W = 84
comptime H = 84
comptime CAM = 0  # "front_close", a worldbody FIXED camera

comptime E = DMLiftBrick[DT]

# ⚠ SET FALSE ON A SMALL BOARD. 1 024 lanes allocates the full `Data` 1 024
# times plus ~145 MB of pixel buffers; everything below it is cheap.
comptime LANES_1024: Bool = True

# ⚠ RIGID PER-LANE TRANSLATION, OFF BY DEFAULT. See the header. Every body
# EXCEPT worldbody (index 0) is shifted, so the scene moves and the fixed
# camera does not — a valid geometry, just a different view per lane. Leaving
# body 0 alone is the whole trick: shifting it too would move the camera with
# the scene and change nothing.
comptime JITTER: Bool = False
comptime JITTER_M = 0.02  # metres of shift between adjacent lanes


def _r3(x: Float64) -> Float64:
    """Three decimals, so a row is readable as a table."""
    return Float64(Int(x * 1000.0 + 0.5)) / 1000.0


def bench_lanes[
    BATCH: Int
](ctx: DeviceContext, mut env: E, tri_backup: List[Scalar[DT]]) raises:
    """One row of the sweep: three timed legs at `BATCH` lanes.

    The legs are the same three `camera_tracer_lift_brick.mojo` runs and in the
    same order — shadows on/off with the soup unchanged, then the soup off with
    shadows back on — so the two files' rows are directly comparable at
    BATCH = 1.
    """
    comptime RL = BatchedCameraRenderer[DT, E.MD, BATCH, W, H]
    comptime RL_NS = BatchedCameraRenderer[
        DT, E.MD, BATCH, W, H, SHADOWS=False
    ]
    # Enough repetitions to average out launch jitter, few enough that the wide
    # rows do not dominate the wall clock.
    comptime REPS = 20 if BATCH <= 16 else (10 if BATCH <= 64 else 4)

    comptime NB3 = E.MD.NBODY * 3
    comptime NB4 = E.MD.NBODY * 4
    comptime NHF = E.MD.NHFIELD_DATA

    # ── a `Data` this wide, every lane a copy of the env's single lane ────
    var d = Data[DT, E.MD, BATCH]()
    for b in range(BATCH):
        for i in range(NB3):
            d.xpos.data[b * NB3 + i] = env.d.xpos.data[i]
            d.subtree_com.data[b * NB3 + i] = env.d.subtree_com.data[i]
        for i in range(NB4):
            d.xquat.data[b * NB4 + i] = env.d.xquat.data[i]

    # ⚠ POSITIVE BRANCH, so the guard carries the evidence that `NHF > 0` and
    # the index below is in range. A `comptime if NHF == 0: pass` would not.
    comptime if NHF > 0:
        for b in range(BATCH):
            for i in range(NHF):
                d.hfield_data.data[b * NHF + i] = env.d.hfield_data.data[i]

    comptime if JITTER:
        for b in range(BATCH):
            var off = Scalar[DT](Float64(b) * JITTER_M)
            for body in range(1, E.MD.NBODY):
                d.xpos.data[b * NB3 + body * 3 + 0] += off

    d.upload_all(ctx)
    ctx.synchronize()

    # ── leg 1: soup ON, shadows ON ────────────────────────────────────────
    var r = RL(ctx, env.mf, CAM)
    r.render(ctx, d, env.mf)
    ctx.synchronize()
    var t = perf_counter_ns()
    for _i in range(REPS):
        r.render(ctx, d, env.mf)
    ctx.synchronize()
    var soup_shadow = Float64(perf_counter_ns() - t) / Float64(REPS) / 1.0e6

    # ── leg 2: soup ON, shadows OFF (a different kernel, same data) ───────
    var rn = RL_NS(ctx, env.mf, CAM)
    rn.render(ctx, d, env.mf)
    ctx.synchronize()
    t = perf_counter_ns()
    for _i in range(REPS):
        rn.render(ctx, d, env.mf)
    ctx.synchronize()
    var soup_noshadow = Float64(perf_counter_ns() - t) / Float64(REPS) / 1.0e6

    # ── leg 3: soup OFF, shadows ON — the control ─────────────────────────
    # `TRINUM = 0` makes `ray_mesh` return NO HIT without touching a geom, a
    # pose, a pixel or the kernel, so the delta is the triangle sweep alone.
    for m in range(MAX_GPU_MESHES):
        env.mf.mesh_meta.data[
            m * MODEL_MESH_META_SIZE + MESH_META_IDX_TRINUM
        ] = Scalar[DT](0)
    env.mf.upload_all(ctx)
    ctx.synchronize()
    r.render(ctx, d, env.mf)
    ctx.synchronize()
    t = perf_counter_ns()
    for _i in range(REPS):
        r.render(ctx, d, env.mf)
    ctx.synchronize()
    var nosoup_shadow = Float64(perf_counter_ns() - t) / Float64(REPS) / 1.0e6

    # ⚠ RESTORE, OR EVERY LATER ROW MEASURES AN EMPTY SCENE. The model is
    # shared across the whole sweep; this is the one piece of global state the
    # rows can corrupt for each other.
    for m in range(MAX_GPU_MESHES):
        env.mf.mesh_meta.data[
            m * MODEL_MESH_META_SIZE + MESH_META_IDX_TRINUM
        ] = tri_backup[m]
    env.mf.upload_all(ctx)
    ctx.synchronize()

    var us_lane = soup_shadow * 1000.0 / Float64(BATCH)
    var fps = Float64(BATCH) / (soup_shadow / 1000.0)

    print(
        "  ",
        BATCH,
        "\t",
        _r3(soup_shadow),
        "\t",
        _r3(us_lane),
        "\t",
        _r3(fps),
        "\t",
        _r3(soup_shadow / nosoup_shadow),
        "\t",
        _r3(soup_shadow / soup_noshadow),
        "\t",
        _r3(nosoup_shadow),
        "\t",
        REPS,
    )


def main() raises:
    comptime if not has_accelerator():
        print("no accelerator — this benchmark measures the GPU leg")
        return

    var ctx = DeviceContext()
    var env = E()
    _ = env.reset()
    compute_subtree_com["cpu", DT, E.MD, 1](env.d, env.mf)
    init_camera_reference(env.d, env.mf)
    env.mf.upload_all(ctx)
    env.d.upload_all(ctx)
    ctx.synchronize()

    print("scene   : dm_control manipulation/lift_brick")
    print("geoms   :", E.MD.NGEOM)
    print("mesh tri:", E.MD.NMESH_TRI, " (0 = no triangle soup)")
    print("camera  :", CAM, " resolution", W, "x", H)
    print("jitter  :", JITTER)

    # ⚠⚠ THE VACUITY CHECK, ONCE. A camera pointed at empty sky renders very
    # fast and measures nothing, so the hit count is established before any
    # timing. It holds for EVERY row because every lane is a copy of this one
    # — which is exactly why `JITTER` is off by default.
    var crgb = List[Scalar[DT]]()
    var cdep = List[Scalar[DT]]()
    var cseg = List[Scalar[DT]]()
    var r1 = BatchedCameraRenderer[DT, E.MD, 1, W, H](ctx, env.mf, CAM)
    r1.render_cpu(env.d, env.mf, crgb, cdep, cseg)
    var hits = 0
    for i in range(len(cseg)):
        if Int(cseg[i]) >= 0:
            hits += 1
    print("hits    :", hits, "of", W * H, "px (per lane)")
    if hits == 0:
        print("!! nothing in frame — every row below is meaningless")
        return

    # The soup is global state the control leg zeroes; back it up once.
    var tri_backup = List[Scalar[DT]]()
    for m in range(MAX_GPU_MESHES):
        tri_backup.append(
            env.mf.mesh_meta.data[
                m * MODEL_MESH_META_SIZE + MESH_META_IDX_TRINUM
            ]
        )

    print("")
    print(
        "   lanes \t ms/frame \t us/lane \t frames/s \t sweep_x \t"
        " shadow_x \t control_ms \t reps"
    )
    bench_lanes[1](ctx, env, tri_backup)
    bench_lanes[16](ctx, env, tri_backup)
    bench_lanes[64](ctx, env, tri_backup)
    bench_lanes[256](ctx, env, tri_backup)
    comptime if LANES_1024:
        bench_lanes[1024](ctx, env, tri_backup)

    print("")
    print("read the sweep_x COLUMN: if it falls as lanes rise, the 76x at one")
    print("lane was an occupancy artefact; if it holds, the BVH is worth that")
    print("much at training batch size.")
