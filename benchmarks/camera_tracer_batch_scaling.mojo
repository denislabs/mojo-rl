"""How the camera tracer's cost amortises across lanes.

    pixi run mojo run -I . benchmarks/camera_tracer_batch_scaling.mojo

`camera_tracer_cheetah.mojo` and `camera_tracer_lift_brick.mojo` measure ONE
lane, which is the right shape for comparing scenes and the wrong shape for
answering "can we train on this". A single 84x84 frame is 7 056 threads — far
too few to fill a GPU — so the one-lane number is dominated by launch overhead
and understates the throughput a batch would get.

This sweeps lanes on the MESH-FREE scene, so the number is the tracer's own
scaling and not the missing BVH (`camera_tracer_lift_brick.mojo` measures that
separately, and isolates it with a soup-off control).

⚠ `us/lane/frame` IS THE COLUMN TO READ. Total time per frame necessarily grows
with lanes; the question is whether the per-lane cost FALLS, and where it stops
falling. Where it flattens is where the GPU is saturated and the tracer has
become genuinely compute-bound.
"""

from std.time import perf_counter_ns
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.cheetah import DMCheetahRunBatched
from mojo_rl.physics3d.raytrace import BatchedCameraRenderer

comptime DT = DType.float32
comptime W = 84
comptime H = 84
comptime CAM = 0
comptime REPS = 20


def _leg[N: Int](ctx: DeviceContext) raises:
    comptime E = DMCheetahRunBatched[N]
    comptime R = BatchedCameraRenderer[DT, E.MD, N, W, H]
    var env = E(ctx)
    env.reset_batch[N](ctx, UInt64(42))
    ctx.synchronize()
    # ⚠ cheetah's cameras are `mode="trackcom"`, which reads a reference pose.
    # The batched env has no CPU-side FK to take it from here, so this leg uses
    # the FIXED path by construction: `init_camera_reference` on a fresh model
    # writes the identity reference, and the number below is a THROUGHPUT
    # measurement, not a picture. See the scene benchmarks for correct frames.
    for c in range(2):
        env.mf.cameras.data[
            c * 24 + 22
        ] = Scalar[DT](1)  # CAM_IDX_REF_SET
    env.mf.upload_all(ctx)
    ctx.synchronize()

    var r = R(ctx, env.mf, CAM)
    r.render(ctx, env.d, env.mf)
    ctx.synchronize()

    var t = perf_counter_ns()
    for _i in range(REPS):
        r.render(ctx, env.d, env.mf)
    ctx.synchronize()
    var ns = Float64(perf_counter_ns() - t) / Float64(REPS)
    print(
        "  lanes", N,
        "  ", ns / 1.0e6, "ms/frame  ",
        ns / 1.0e3 / Float64(N), "us/lane/frame  ",
        Float64(N) * 1.0e9 / ns, "frames/s total",
    )


def main() raises:
    comptime if not has_accelerator():
        print("no accelerator")
        return
    var ctx = DeviceContext()
    print("scene: dm_control cheetah (mesh-free)  ", W, "x", H)
    _leg[1](ctx)
    _leg[16](ctx)
    _leg[64](ctx)
    _leg[256](ctx)
