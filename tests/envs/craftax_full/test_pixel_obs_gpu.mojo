"""Phase-7H gate: GPU pixel obs renders the same RGB as CPU.

Strategy: reset a CPU env, copy its state into the GPU buffer, render
pixel obs both ways, compare per-pixel within tight tolerance. The
atlas is built on host then uploaded to the shared workspace; the
render kernel samples it via `_render_pixel_rgb_from_state`.

Run:
  pixi run -e apple  mojo run -I . tests/envs/craftax_full/test_pixel_obs_gpu.mojo
  pixi run -e nvidia mojo run -I . tests/envs/craftax_full/test_pixel_obs_gpu.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.envs.craftax_full import (
    CraftaxFullPixelEnv,
    CraftaxFullAction,
    PIXEL_OBS_DIM,
    OBS_PIX_H,
    OBS_PIX_W,
    STATE_SIZE,
)
from mojo_rl.envs.craftax_full.craftax_full_pixel import ATLAS_FLOATS
from mojo_rl.nn2.constants import DT as dtype


comptime BATCH: Int = 1


@always_inline
def check(mut counts: List[Int], name: String, ok: Bool):
    if ok:
        counts[0] += 1
        print("  PASS", name)
    else:
        counts[1] += 1
        print("  FAIL", name)


@always_inline
def approx(a: Float64, b: Float64, eps: Float64 = 0.001) -> Bool:
    var d = a - b
    if d < 0.0:
        d = -d
    return d < eps


def test_gpu_pixel_obs_matches_cpu(mut counts: List[Int]) raises:
    print("test_gpu_pixel_obs_matches_cpu")
    with DeviceContext() as ctx:
        var env = CraftaxFullPixelEnv[dtype]()
        _ = env.reset_with_seed(UInt64(0xBEEF))
        var cpu_obs = env.get_obs_list()

        # Upload state and atlas to GPU.
        var states = ctx.enqueue_create_buffer[dtype](BATCH * STATE_SIZE)
        var obs = ctx.enqueue_create_buffer[dtype](BATCH * PIXEL_OBS_DIM)
        var workspace = ctx.enqueue_create_buffer[dtype](ATLAS_FLOATS)

        var host_state = ctx.enqueue_create_host_buffer[dtype](STATE_SIZE)
        for i in range(STATE_SIZE):
            host_state[i] = env.inner.state[i]
        ctx.enqueue_copy(states, host_state)

        env.init_step_workspace_gpu_with_atlas[BATCH](ctx, workspace)
        ctx.synchronize()

        # Render via the GPU kernel.
        CraftaxFullPixelEnv[dtype]._render_kernel[BATCH, STATE_SIZE](
            ctx, states, workspace.unsafe_ptr(), obs,
        )

        var host_obs = ctx.enqueue_create_host_buffer[dtype](
            BATCH * PIXEL_OBS_DIM
        )
        ctx.enqueue_copy(host_obs, obs)
        ctx.synchronize()

        check(counts, "GPU pixel obs has same length",
              len(cpu_obs) == PIXEL_OBS_DIM)
        var all_in_range = True
        var any_nonzero = False
        for i in range(PIXEL_OBS_DIM):
            var v = Float32(host_obs[i])
            if v < Float32(0.0) or v > Float32(1.0):
                all_in_range = False
            if v != Float32(0.0):
                any_nonzero = True
        check(counts, "GPU pixel obs in [0,1]", all_in_range)
        check(counts, "GPU pixel obs not all zero", any_nonzero)

        var mismatch = 0
        var first_bad = -1
        for i in range(PIXEL_OBS_DIM):
            if not approx(
                Float64(host_obs[i]), Float64(cpu_obs[i])
            ):
                if mismatch == 0:
                    first_bad = i
                mismatch += 1
        if first_bad >= 0:
            print("    first pixel mismatch at idx", first_bad,
                  " cpu=", Float32(cpu_obs[first_bad]),
                  " gpu=", Float32(host_obs[first_bad]))
        check(counts, "GPU pixel obs matches CPU (≤1e-3)",
              mismatch == 0)


def main() raises:
    print("Craftax-Full Phase-7H GPU pixel obs gate")
    print("=" * 50)
    var counts = [0, 0]
    test_gpu_pixel_obs_matches_cpu(counts)
    print()
    print("=" * 50)
    print("Passed:", counts[0], "Failed:", counts[1])
    if counts[1] > 0:
        raise Error("Phase-7H GPU pixel obs gate FAILED")
    print("Phase-7H GPU pixel obs gate PASS")
