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

⚠⚠ FOUR LEGS, A 2x2 OVER (soup, shadows), BECAUSE `shadow_x` ALONE CANNOT BE
READ. The shadow ray is a second full `ray_model` against the SAME triangle
soup, and it starts INSIDE the scene aimed at the light, so it enters mesh
AABBs more often than a camera ray grazing them from outside. That makes
`shadow_x` on a mesh-heavy scene partly a measurement of the SWEEP. The fourth
cell — soup OFF, shadows OFF — is what separates them:

    shadow_x_free ~= 1   the shadow cost IS the triangle sweep; a BVH fixes
                         shadows as a side effect and `SHADOWS=False` stops
                         being the interesting lever.
    shadow_x_free ~= shadow_x   shadows cost what they cost, independently.

⚠⚠ THE BVH LANDED, AND THIS FILE MEASURES IT WITHOUT A BISECT. `bvh_x` is a
FIFTH leg that zeroes `MESH_META_IDX_BVHNUM` — the dispatch `ray_model` already
carries — so the tree and the linear sweep are timed in the same binary, on the
same board, in the same run. Apple at 1 024 lanes: **11.2x**, 72 fps -> 810 fps,
and `sweep_x` (the headroom that REMAINS to a triangle-free scene) fell from
38-46x to **3.34x**. The tree took about 92% of what was there.

⚠⚠ AND IT REORDERS THE TWO LEVERS. With meshes at 40x, `SHADOWS=False`'s ~3x
was the small one; with meshes at 3.3x it is the LARGER remaining lever on this
scene. Read `bvh_x` and `shadow_x` side by side before optimising anything else.

⚠ `sweep_x` IS NOT A BVH's YIELD AND NEVER WAS. It is the ratio to a scene with
no triangles at all, which no acceleration structure reaches: a walk still costs
its ~13 node visits. That distinction is why `bvh_x` exists as its own leg —
sizing the port against `sweep_x` would have promised 94x and delivered 11.

⚠⚠ ANSWERED, AND IT IS THE SECOND BRANCH ON BOTH BOARDS. 5090 at 1 024 lanes:
`shadow_x` 2.99 vs `shadow_x_free` **3.37** — mesh-free is HIGHER, so removing
the soup does not make the shadow ray cheap. Apple: 1.91 vs 2.37, same verdict
at a smaller multiplier. **SHADOWS AND MESHES ARE INDEPENDENT COSTS; a BVH will
not touch shadows, and `SHADOWS=False` stays a real ~3x lever on NVIDIA that no
BVH will hand you.**

⚠ AND THE SHADOW COST RISES WITH OCCUPANCY: `shadow_x_free` goes 1.74 -> 2.21
-> 2.73 -> 3.29 -> 3.37 across the sweep. A second ray should cost ~2x. At one
lane it costs LESS (1.74 — it hides in the same memory latency); at saturation
it costs 3.4. That is the signature of REGISTER PRESSURE — two inlined
`ray_model` copies roughly double live registers and cut resident warps exactly
when occupancy is what carries the kernel. HYPOTHESIS, not measured: check it
against the compiler's register report before acting on it.

⚠ THE FOURTH LEG IS FREE. Legs 1 and 2 already instantiate both kernels and the
soup is DATA (`TRINUM`), not a kernel — so the cell that discriminates costs one
timing loop and no compile. It was missing from the first version of this file
for no better reason than that three legs matched the bench it grew out of.

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
    MESH_META_IDX_BVHNUM,
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

# ⚠ EVERY LEG GETS THE SAME WALL-CLOCK WINDOW, not the same rep count. See
# `_reps_for`. 200 ms is long enough that a stray kernel from another process
# cannot dominate it and short enough that the whole sweep stays minutes.
comptime MIN_WINDOW_MS = 200.0
comptime MIN_REPS = 3
comptime MAX_REPS = 2000


def _reps_for(probe_ms: Float64) -> Int:
    """Repetitions enough to fill `MIN_WINDOW_MS`, from a measured probe.

    ⚠⚠ THIS EXISTS BECAUSE A PER-ROW `REPS` IS A BUG. The four legs of a row
    differ by up to 100x in cost — at 1 024 lanes the soup leg is ~832 ms and
    the mesh-free leg ~3 ms on a 5090 — so one rep count sized for the
    expensive leg gives the cheap one a window of a few milliseconds, which
    interference dominates. The symptom is not noise that averages out: it is
    `shadow_x_free` coming back as **0.631**, i.e. shadows OFF timing SLOWER
    than shadows ON, which is impossible and was printed as data.
    """
    if probe_ms <= 0.0:
        return MAX_REPS
    var n = Int(MIN_WINDOW_MS / probe_ms) + 1
    if n < MIN_REPS:
        return MIN_REPS
    if n > MAX_REPS:
        return MAX_REPS
    return n


def _r3(x: Float64) -> Float64:
    """Three decimals, so a row is readable as a table."""
    return Float64(Int(x * 1000.0 + 0.5)) / 1000.0


def bench_lanes[
    BATCH: Int
](
    ctx: DeviceContext,
    mut env: E,
    tri_backup: List[Scalar[DT]],
    bvh_backup: List[Scalar[DT]],
) raises:
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
    r.render(ctx, d, env.mf)
    ctx.synchronize()
    var n1 = _reps_for(Float64(perf_counter_ns() - t) / 1.0e6)
    t = perf_counter_ns()
    for _i in range(n1):
        r.render(ctx, d, env.mf)
    ctx.synchronize()
    var soup_shadow = Float64(perf_counter_ns() - t) / Float64(n1) / 1.0e6

    # ── leg 2: soup ON, shadows OFF (a different kernel, same data) ───────
    var rn = RL_NS(ctx, env.mf, CAM)
    rn.render(ctx, d, env.mf)
    ctx.synchronize()
    t = perf_counter_ns()
    rn.render(ctx, d, env.mf)
    ctx.synchronize()
    var n2 = _reps_for(Float64(perf_counter_ns() - t) / 1.0e6)
    t = perf_counter_ns()
    for _i in range(n2):
        rn.render(ctx, d, env.mf)
    ctx.synchronize()
    var soup_noshadow = Float64(perf_counter_ns() - t) / Float64(n2) / 1.0e6

    # ── leg 5: soup ON, shadows ON, BVH ABLATED — what the tree bought ───
    #
    # ⚠⚠ THE ONLY LEG THAT ANSWERS "WHAT DID THE BVH DO", AND IT HAS TO BE
    # HERE RATHER THAN IN A CHANGELOG. `sweep_x` is the headroom to a scene
    # with NO triangles, which no acceleration structure reaches; the number a
    # reader actually wants is the linear sweep against the tree, on THIS
    # board, in THIS run. `MESH_META_IDX_BVHNUM = 0` is the dispatch
    # `ray_model` already has — the same data ablation
    # `test_ray_bvh_matches_linear.mojo` uses as its control leg, so the two
    # files agree on what "BVH off" means and the parity of the two answers is
    # gated there rather than assumed here.
    #
    # ⚠ IT RUNS BEFORE THE SOUP ABLATION because both write `mesh_meta`, and a
    # leg that measured "no BVH" over "no triangles" would report the control
    # twice and look like a 1.0x.
    for m in range(MAX_GPU_MESHES):
        env.mf.mesh_meta.data[
            m * MODEL_MESH_META_SIZE + MESH_META_IDX_BVHNUM
        ] = Scalar[DT](0)
    env.mf.upload_all(ctx)
    ctx.synchronize()
    r.render(ctx, d, env.mf)
    ctx.synchronize()
    t = perf_counter_ns()
    r.render(ctx, d, env.mf)
    ctx.synchronize()
    var n5 = _reps_for(Float64(perf_counter_ns() - t) / 1.0e6)
    t = perf_counter_ns()
    for _i in range(n5):
        r.render(ctx, d, env.mf)
    ctx.synchronize()
    var linear_shadow = Float64(perf_counter_ns() - t) / Float64(n5) / 1.0e6
    for m in range(MAX_GPU_MESHES):
        env.mf.mesh_meta.data[
            m * MODEL_MESH_META_SIZE + MESH_META_IDX_BVHNUM
        ] = bvh_backup[m]
    env.mf.upload_all(ctx)
    ctx.synchronize()

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
    r.render(ctx, d, env.mf)
    ctx.synchronize()
    var n3 = _reps_for(Float64(perf_counter_ns() - t) / 1.0e6)
    t = perf_counter_ns()
    for _i in range(n3):
        r.render(ctx, d, env.mf)
    ctx.synchronize()
    var nosoup_shadow = Float64(perf_counter_ns() - t) / Float64(n3) / 1.0e6

    # ── leg 4: soup OFF, shadows OFF — the 2x2's fourth cell ──────────────
    #
    # ⚠⚠ THIS LEG COSTS NO COMPILE. Both kernels are already instantiated by
    # legs 1 and 2; the soup is DATA (`TRINUM`), not a kernel. So the only
    # cell that tells `shadow_x` apart from `sweep_x` is also the cheapest.
    rn.render(ctx, d, env.mf)
    ctx.synchronize()
    t = perf_counter_ns()
    rn.render(ctx, d, env.mf)
    ctx.synchronize()
    var n4 = _reps_for(Float64(perf_counter_ns() - t) / 1.0e6)
    t = perf_counter_ns()
    for _i in range(n4):
        rn.render(ctx, d, env.mf)
    ctx.synchronize()
    var nosoup_noshadow = (
        Float64(perf_counter_ns() - t) / Float64(n4) / 1.0e6
    )

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
        _r3(linear_shadow / soup_shadow),
        "\t",
        _r3(soup_shadow / soup_noshadow),
        "\t",
        _r3(nosoup_shadow / nosoup_noshadow),
        "\t",
        _r3(nosoup_shadow),
        "\t",
        String(n1) + "/" + String(n2) + "/" + String(n3) + "/"
        + String(n4) + "/" + String(n5),
    )

    # ⚠⚠ THE ROW IS CHECKED AGAINST PHYSICS BEFORE IT IS BELIEVED. Turning
    # shadows OFF cannot make a kernel slower, and removing the triangle soup
    # cannot either — both legs run strictly less work in the same kernel or a
    # smaller one. A violation is not a slow GPU, it is a measurement window
    # too short to contain the work, and it MUST NOT be read as data: an
    # earlier version of this file printed `shadow_x_free` = 0.631 and 10.107
    # from a 12 ms window and they went into a report as findings.
    if linear_shadow < soup_shadow:
        print("      !! the LINEAR sweep timed FASTER than the BVH — a tree can"
              " only remove triangle tests, so this is a short window (or the"
              " ablation did not take)")
    if soup_noshadow > soup_shadow:
        print("      !! shadows OFF timed SLOWER than ON (soup) — window too short")
    if nosoup_noshadow > nosoup_shadow:
        print("      !! shadows OFF timed SLOWER than ON (free) — window too short")
    if nosoup_shadow > soup_shadow:
        print("      !! soup OFF timed SLOWER than ON — window too short")


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

    # The BVH is the other piece of `mesh_meta` a leg zeroes.
    var bvh_backup = List[Scalar[DT]]()
    var trees = 0
    for m in range(MAX_GPU_MESHES):
        var v = env.mf.mesh_meta.data[
            m * MODEL_MESH_META_SIZE + MESH_META_IDX_BVHNUM
        ]
        bvh_backup.append(v)
        if Int(v) > 0:
            trees += 1
    print("bvh     :", trees, "meshes carry a tree")
    if trees == 0:
        print(
            "!! NO TREE ON ANY MESH — `bvh_x` below will be 1.0 and it will"
            " not mean the BVH is worthless, it will mean the model was built"
            " without one. Check `nmesh_tri` and `build_mesh_bvh`."
        )

    print("")
    print(
        "   lanes \t ms/frame \t us/lane \t frames/s \t sweep_x \t bvh_x \t"
        " shadow_x \t shadow_x_free \t control_ms \t reps(1/2/3/4/5)"
    )
    bench_lanes[1](ctx, env, tri_backup, bvh_backup)
    bench_lanes[16](ctx, env, tri_backup, bvh_backup)
    bench_lanes[64](ctx, env, tri_backup, bvh_backup)
    bench_lanes[256](ctx, env, tri_backup, bvh_backup)
    comptime if LANES_1024:
        bench_lanes[1024](ctx, env, tri_backup, bvh_backup)

    print("")
    print("bvh_x: what the tree bought, measured in THIS run against the")
    print("linear sweep in the SAME binary. Apple: 9.5x at one lane rising to")
    print("11.2x at 1 024 — 72 fps -> 810 fps on lift_brick.")
    print("")
    print("sweep_x: the mesh cost that REMAINS. Apple fell 38-46x -> 3.34x, so")
    print("the tree took ~92% of the headroom and meshes are no longer what")
    print("this kernel is made of. The rest is not another tree: it is the")
    print("~13 node visits a walk costs, which is the floor.")
    print("")
    print("shadow_x vs shadow_x_free: THEY AGREE (5090 2.99 vs 3.37 at 1 024;")
    print("Apple 1.91 vs 2.37), so the shadow cost is NOT the triangle sweep.")
    print("Shadows and meshes are INDEPENDENT: `SHADOWS=False` is a real ~3x")
    print("on NVIDIA that no BVH gives you — and now that meshes cost 3.3x")
    print("instead of 40x, shadows are the LARGER of the two levers.")
