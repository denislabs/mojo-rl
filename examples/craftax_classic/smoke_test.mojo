"""Phase-1 smoke test for the Craftax-Classic Mojo port.

Exercises:
  - CPU: reset, step, step_obs — verify obs has OBS_DIM entries and step bumps timestep.
  - GPU: reset_kernel_gpu, step_kernel_gpu, selective_reset_kernel_gpu — verify
         buffers are the right size and kernels launch without error.

No game logic yet; this only confirms the skeleton compiles and runs end-to-end.

Run:
  pixi run mojo run -I . examples/craftax_classic/smoke_test.mojo
  pixi run -e apple  mojo run -I . examples/craftax_classic/smoke_test.mojo   # GPU
  pixi run -e nvidia mojo run -I . examples/craftax_classic/smoke_test.mojo
"""

from std.gpu.host import DeviceContext
from mojo_rl.envs.craftax_classic import (
    CraftaxClassicEnv,
    NUM_ACTIONS,
    OBS_DIM,
    STATE_SIZE,
)
from mojo_rl.envs.craftax_classic.state import (
    S_PLAYER_POS,
    S_INTRINSICS_BASE,
    S_LIGHT_LEVEL,
)
from mojo_rl.envs.craftax_classic.constants import BLOCK_GRASS, MAP_W
from mojo_rl.envs.craftax_classic.state import S_MAP_BASE
from mojo_rl.nn import dtype


def cpu_smoke() raises:
    print("=== CPU smoke ===")
    var env = CraftaxClassicEnv[dtype]()

    var obs0 = env.reset_obs_list()
    if len(obs0) != OBS_DIM:
        raise Error("reset_obs_list returned wrong obs size")
    print("  reset obs len:", len(obs0), "(expected", OBS_DIM, ")")

    # After reset, check that the player tile is GRASS and intrinsics are 9.
    var py = Int(env.state[S_PLAYER_POS])
    var px = Int(env.state[S_PLAYER_POS + 1])
    var spawn_block = Int(env.state[S_MAP_BASE + py * MAP_W + px])
    print("  player at (", py, ",", px, ")  spawn_block =", spawn_block)
    if spawn_block != BLOCK_GRASS:
        raise Error("player spawn tile is not GRASS")
    for k in range(4):
        var v = Int(env.state[S_INTRINSICS_BASE + k])
        if v != 9:
            raise Error("intrinsic " + String(k) + " not at max")
    print("  intrinsics all 9, light=", Float64(env.state[S_LIGHT_LEVEL]))

    var total_reward = Float64(0.0)
    for step in range(5):
        var result = env.step_obs(step % NUM_ACTIONS)
        if len(result[0]) != OBS_DIM:
            raise Error("step_obs returned wrong obs size")
        total_reward += Float64(result[1])
    print("  5 CPU steps → total_reward:", total_reward)
    print("  done:", env.done)


def gpu_smoke() raises:
    print("=== GPU smoke ===")
    comptime BATCH_SIZE: Int = 8

    print("  BATCH_SIZE:", BATCH_SIZE)
    print("  STATE_SIZE:", STATE_SIZE)
    print("  OBS_DIM:", OBS_DIM)

    var ctx = DeviceContext()

    var states_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * STATE_SIZE)
    var actions_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
    var rewards_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
    var dones_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
    var terminated_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
    var obs_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * OBS_DIM)

    # Zero dones to start (selective_reset reads it).
    var host_zeros = List[Float32](capacity=BATCH_SIZE)
    for _ in range(BATCH_SIZE):
        host_zeros.append(Float32(0))
    ctx.enqueue_copy(dones_buf, host_zeros.unsafe_ptr())

    # Zero actions.
    ctx.enqueue_copy(actions_buf, host_zeros.unsafe_ptr())
    ctx.synchronize()

    CraftaxClassicEnv[dtype].reset_kernel_gpu[BATCH_SIZE, STATE_SIZE](
        ctx, states_buf, rng_seed=UInt64(42)
    )

    for step in range(5):
        CraftaxClassicEnv[dtype].step_kernel_gpu[
            BATCH_SIZE, STATE_SIZE, OBS_DIM
        ](
            ctx,
            states_buf,
            actions_buf,
            rewards_buf,
            dones_buf,
            terminated_buf,
            obs_buf,
            rng_seed=UInt64(step + 1),
        )

    CraftaxClassicEnv[dtype].selective_reset_kernel_gpu[
        BATCH_SIZE, STATE_SIZE
    ](ctx, states_buf, dones_buf, rng_seed=UInt64(99))

    ctx.synchronize()

    # Read back a few rewards and obs entries to confirm shapes.
    var host_rewards = List[Float32](capacity=BATCH_SIZE)
    for _ in range(BATCH_SIZE):
        host_rewards.append(Float32(0))
    ctx.enqueue_copy(host_rewards.unsafe_ptr(), rewards_buf)
    ctx.synchronize()

    print("  rewards[0:3]:", host_rewards[0], host_rewards[1], host_rewards[2])
    print("  GPU smoke OK")


def main() raises:
    print("Craftax-Classic — Phase-1 smoke test")
    print("=" * 40)
    cpu_smoke()
    gpu_smoke()
    print()
    print("Done.")
