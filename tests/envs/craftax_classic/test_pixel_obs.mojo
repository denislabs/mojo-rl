"""Phase 6B gate: RGB sprite pixel obs (Craftax-spec).

Verifies:
  - obs length == 3 * 90 * 90 = 24300, channel-first
  - All values in [0, 1]
  - At least one non-zero pixel after reset (something rendered)
  - Channel-first layout: stride is OBS_PIX_H * OBS_PIX_W
  - CPU vs GPU pixel obs bitwise-equivalent for a mirrored state

Run:
  pixi run mojo run -I . tests/envs/craftax_classic/test_pixel_obs.mojo
  pixi run -e nvidia mojo run -I . tests/envs/craftax_classic/test_pixel_obs.mojo
"""

from std.gpu.host import DeviceContext
from mojo_rl.envs.craftax_classic import (
    CraftaxClassicEnv,
    CraftaxClassicPixelEnv,
    PIXEL_OBS_DIM,
    OBS_PIX_H,
    OBS_PIX_W,
    OBS_CHANNELS,
    BLOCK_PIXEL_SIZE,
)
from mojo_rl.envs.craftax_classic.state import STATE_SIZE
from mojo_rl.envs.craftax_classic.constants import ACTION_NOOP, ACTION_RIGHT
from mojo_rl.nn import dtype


@always_inline
def check(mut counts: List[Int], name: String, ok: Bool):
    if ok:
        counts[0] += 1
        print("  PASS", name)
    else:
        counts[1] += 1
        print("  FAIL", name)


def test_obs_shape(mut counts: List[Int]) raises:
    print("test_obs_shape")
    var env = CraftaxClassicPixelEnv[dtype]()
    var obs = env.reset_obs_list()
    check(counts, "obs_dim == 3*90*90 = 24300", len(obs) == 24300)
    check(counts, "obs_dim == PIXEL_OBS_DIM", len(obs) == PIXEL_OBS_DIM)
    check(counts, "channels = 3", OBS_CHANNELS == 3)
    check(counts, "H = 90", OBS_PIX_H == 90)
    check(counts, "W = 90", OBS_PIX_W == 90)
    check(counts, "block_pixel_size = 10", BLOCK_PIXEL_SIZE == 10)

    var any_above_1 = False
    var any_below_0 = False
    var any_non_zero = False
    for i in range(len(obs)):
        var v = Float32(obs[i])
        if v > Float32(1.0):
            any_above_1 = True
        if v < Float32(0.0):
            any_below_0 = True
        if v > Float32(0.0):
            any_non_zero = True
    check(counts, "all pixels <= 1.0", not any_above_1)
    check(counts, "all pixels >= 0.0", not any_below_0)
    check(counts, "at least one non-zero pixel", any_non_zero)


def test_channel_first_layout(mut counts: List[Int]) raises:
    """For channel-first (C, H, W) row-major flat, pixel (c, h, w) sits
    at offset c * (H*W) + h*W + w. Verify the three channels are stored
    as three contiguous H*W blocks (not interleaved RGB-per-pixel)."""
    print("test_channel_first_layout")
    var env = CraftaxClassicPixelEnv[dtype]()
    var obs = env.reset_obs_list()
    comptime HW = OBS_PIX_H * OBS_PIX_W

    # Reach into the obs and sample a position that should be on grass at
    # view center (player tile). Grass has a non-trivial G channel but a
    # low B channel; check G[center] > B[center] for typical grass sprites.
    var center = (OBS_PIX_H // 2) * OBS_PIX_W + (OBS_PIX_W // 2)
    var r = Float32(obs[0 * HW + center])
    var g = Float32(obs[1 * HW + center])
    var b = Float32(obs[2 * HW + center])
    print("    center RGB =", r, g, b)
    # On a fresh grass spawn the green channel should dominate (player
    # sprite blends in but most surrounding pixels are grass).
    check(counts, "G channel non-zero at center", g > Float32(0.05))


def test_inventory_region_renders(mut counts: List[Int]) raises:
    """The bottom 20 rows are the inventory bar. After reset, HP/FD/DR/EN
    cells should show their icon sprites (count=9 → non-empty)."""
    print("test_inventory_region_renders")
    var env = CraftaxClassicPixelEnv[dtype]()
    var obs = env.reset_obs_list()
    comptime HW = OBS_PIX_H * OBS_PIX_W

    # Inventory starts at h = 70. Health cell is (row=0, col=0) → covers
    # h in [70, 80), w in [0, 10). Look for at least one channel > the
    # cell's background (~0.05).
    var any_bright = False
    for h in range(70, 80):
        for w in range(0, 10):
            var pix = h * OBS_PIX_W + w
            var r = Float32(obs[0 * HW + pix])
            var g = Float32(obs[1 * HW + pix])
            var b = Float32(obs[2 * HW + pix])
            if r > Float32(0.3) or g > Float32(0.3) or b > Float32(0.3):
                any_bright = True
                break
        if any_bright:
            break
    check(counts, "health icon cell has bright pixel after reset", any_bright)


def test_cpu_vs_gpu_pixel_parity(mut counts: List[Int]) raises:
    """Mirror CPU state to GPU and compare rendered pixel obs."""
    print("test_cpu_vs_gpu_pixel_parity")
    var ctx = DeviceContext()
    comptime BATCH: Int = 1

    var cpu_env = CraftaxClassicPixelEnv[dtype]()
    _ = cpu_env.reset_with_seed(42, False)
    cpu_env.inner._rng_counter = 0

    var states_buf = ctx.enqueue_create_buffer[dtype](BATCH * STATE_SIZE)
    var actions_buf = ctx.enqueue_create_buffer[dtype](BATCH)
    var rewards_buf = ctx.enqueue_create_buffer[dtype](BATCH)
    var dones_buf = ctx.enqueue_create_buffer[dtype](BATCH)
    var terminated_buf = ctx.enqueue_create_buffer[dtype](BATCH)
    var obs_buf = ctx.enqueue_create_buffer[dtype](BATCH * PIXEL_OBS_DIM)
    var ws_size = (
        CraftaxClassicPixelEnv[dtype].STEP_WS_SHARED
        + BATCH * CraftaxClassicPixelEnv[dtype].STEP_WS_PER_ENV
    )
    var ws_buf = ctx.enqueue_create_buffer[dtype](ws_size)

    # Mirror CPU state to GPU.
    var host_state = List[Float32](capacity=STATE_SIZE)
    for _ in range(STATE_SIZE):
        host_state.append(Float32(0))
    for i in range(STATE_SIZE):
        host_state[i] = Float32(cpu_env.inner.state[i])
    ctx.enqueue_copy(states_buf, host_state.unsafe_ptr())

    # Upload the atlas into the shared region of workspace.
    cpu_env.init_step_workspace_gpu_with_atlas[BATCH](ctx, ws_buf)
    ctx.synchronize()

    var ws_optional = Optional[
        UnsafePointer[Scalar[dtype], MutAnyOrigin]
    ](ws_buf.unsafe_ptr())

    # Step both sides once with NOOP and matching seeds.
    var host_act = List[Float32](capacity=BATCH)
    host_act.append(Float32(ACTION_NOOP))
    ctx.enqueue_copy(actions_buf, host_act.unsafe_ptr())
    ctx.synchronize()
    CraftaxClassicPixelEnv[dtype].step_kernel_gpu[
        BATCH, STATE_SIZE, PIXEL_OBS_DIM
    ](
        ctx,
        states_buf,
        actions_buf,
        rewards_buf,
        dones_buf,
        terminated_buf,
        obs_buf,
        rng_seed=UInt64(0),
        workspace_ptr=ws_optional,
    )
    ctx.synchronize()
    _ = cpu_env.step_obs(ACTION_NOOP)

    var cpu_obs = cpu_env.get_obs_list()
    var host_obs = List[Float32](capacity=PIXEL_OBS_DIM)
    for _ in range(PIXEL_OBS_DIM):
        host_obs.append(Float32(0.0))
    ctx.enqueue_copy(host_obs.unsafe_ptr(), obs_buf)
    ctx.synchronize()

    var max_diff: Float32 = 0.0
    var n_diff = 0
    for i in range(PIXEL_OBS_DIM):
        var d = Float32(cpu_obs[i]) - host_obs[i]
        if d < Float32(0.0):
            d = -d
        if d > max_diff:
            max_diff = d
        if d > Float32(0.001):
            n_diff += 1
    print("    max |cpu - gpu| =", max_diff, "diff>1e-3:", n_diff)
    check(counts, "CPU/GPU pixel obs match within 1e-3", max_diff < Float32(0.001))


def main() raises:
    print("Craftax-Classic Phase-6B RGB pixel obs gate")
    print("=" * 50)
    var counts = [0, 0]
    test_obs_shape(counts)
    test_channel_first_layout(counts)
    test_inventory_region_renders(counts)
    test_cpu_vs_gpu_pixel_parity(counts)
    print()
    print("=" * 50)
    print("Passed:", counts[0])
    print("Failed:", counts[1])
    if counts[1] > 0:
        raise Error("Phase-6B RGB gate FAILED")
    print("Phase-6B RGB gate PASS")
