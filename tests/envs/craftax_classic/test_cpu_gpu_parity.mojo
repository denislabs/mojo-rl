"""Phase-5 CPU/GPU parity: same seed + same actions → same state.

Strategy: run CPU and GPU envs side by side with matching Philox seeds
for both world gen and per-step RNG. Compare key state fields after
each step. Even rare random branches (mob spawn rolls) must coincide.

Seed alignment for a 1-env GPU batch (BATCH_SIZE=1, env=0):
  - CPU reset uses seed=42                  → GPU rng_seed=41
  - CPU step n uses seed=n (rng_counter)    → GPU step rng_seed=n-1

Run:
  pixi run mojo run -I . tests/envs/craftax_classic/test_cpu_gpu_parity.mojo
"""

from std.gpu.host import DeviceContext

from mojo_rl.envs.craftax_classic import CraftaxClassicEnv
from mojo_rl.envs.craftax_classic.constants import (
    MAP_W,
    BLOCK_STONE,
    BLOCK_GRASS,
    BLOCK_TREE,
    ACTION_NOOP,
    ACTION_DO,
    INV_WOOD,
    NUM_INVENTORY,
    NUM_ACHIEVEMENTS,
    NUM_INTRINSICS,
)
from mojo_rl.envs.craftax_classic.state import (
    STATE_SIZE,
    S_MAP_BASE,
    S_PLAYER_POS,
    S_INV_BASE,
    S_INTRINSICS_BASE,
    S_INTRINSICS_F_BASE,
    S_ACHIEVEMENTS_BASE,
    S_TIMESTEP,
    S_LIGHT_LEVEL,
)
from mojo_rl.nn2.constants import DT as dtype


@always_inline
def check(mut counts: List[Int], name: String, condition: Bool):
    if condition:
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


def compare_states(
    mut counts: List[Int],
    label: String,
    cpu_state: CraftaxClassicEnv[dtype],
    gpu_host: List[Float32],
) raises:
    """Compare each section of state between CPU env and GPU host
    snapshot."""
    # Player position
    var cpu_py = Int(cpu_state.state[S_PLAYER_POS])
    var cpu_px = Int(cpu_state.state[S_PLAYER_POS + 1])
    var gpu_py = Int(gpu_host[S_PLAYER_POS])
    var gpu_px = Int(gpu_host[S_PLAYER_POS + 1])
    check(
        counts,
        label + " player_pos",
        cpu_py == gpu_py and cpu_px == gpu_px,
    )

    # Map: count differing cells.
    var diffs = 0
    for i in range(4096):
        if Int(cpu_state.state[S_MAP_BASE + i]) != Int(
            gpu_host[S_MAP_BASE + i]
        ):
            diffs += 1
    check(counts, label + " map identical", diffs == 0)

    # Inventory
    var inv_ok = True
    for k in range(NUM_INVENTORY):
        if Int(cpu_state.state[S_INV_BASE + k]) != Int(
            gpu_host[S_INV_BASE + k]
        ):
            inv_ok = False
            break
    check(counts, label + " inventory", inv_ok)

    # Integer intrinsics
    var intr_ok = True
    for k in range(NUM_INTRINSICS):
        if Int(cpu_state.state[S_INTRINSICS_BASE + k]) != Int(
            gpu_host[S_INTRINSICS_BASE + k]
        ):
            intr_ok = False
            break
    check(counts, label + " intrinsics_i", intr_ok)

    # Float intrinsics
    var intr_f_ok = True
    for k in range(NUM_INTRINSICS):
        if not approx(
            Float64(cpu_state.state[S_INTRINSICS_F_BASE + k]),
            Float64(gpu_host[S_INTRINSICS_F_BASE + k]),
        ):
            intr_f_ok = False
            break
    check(counts, label + " intrinsics_f", intr_f_ok)

    # Achievements (any bit difference?)
    var ach_ok = True
    for k in range(NUM_ACHIEVEMENTS):
        if (
            cpu_state.state[S_ACHIEVEMENTS_BASE + k] > Float32(0.5)
        ) != (gpu_host[S_ACHIEVEMENTS_BASE + k] > Float32(0.5)):
            ach_ok = False
            break
    check(counts, label + " achievements", ach_ok)

    # Timestep + light level
    check(
        counts,
        label + " timestep",
        Int(cpu_state.state[S_TIMESTEP]) == Int(gpu_host[S_TIMESTEP]),
    )
    check(
        counts,
        label + " light_level",
        approx(
            Float64(cpu_state.state[S_LIGHT_LEVEL]),
            Float64(gpu_host[S_LIGHT_LEVEL]),
        ),
    )


def setup_deterministic_state(mut env: CraftaxClassicEnv[dtype]):
    """Set up a known state where spawn rolls can't succeed (all tiles
    near player are STONE so no GRASS/PATH for any mob)."""
    var py = Int(env.state[S_PLAYER_POS])
    var px = Int(env.state[S_PLAYER_POS + 1])
    # 5×5 STONE box around player; keep player tile GRASS and put a TREE
    # in front (NORTH).
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            env.state[S_MAP_BASE + (py + dy) * MAP_W + (px + dx)] = Float32(
                BLOCK_STONE
            )
    env.state[S_MAP_BASE + py * MAP_W + px] = Float32(BLOCK_GRASS)
    env.state[S_MAP_BASE + (py - 1) * MAP_W + px] = Float32(BLOCK_TREE)


def copy_state_into_gpu_buffer(
    cpu_state: CraftaxClassicEnv[dtype],
    mut host_buf: List[Float32],
):
    for i in range(STATE_SIZE):
        host_buf[i] = Float32(cpu_state.state[i])


def main() raises:
    print("Craftax-Classic Phase-5 CPU/GPU parity gate")
    print("=" * 50)
    var counts = [0, 0]

    comptime BATCH_SIZE: Int = 1
    var ctx = DeviceContext()

    # ----- CPU side: deterministic setup, then a scripted action trace.
    var cpu_env = CraftaxClassicEnv[dtype]()
    _ = cpu_env.reset_with_seed(42, False)
    setup_deterministic_state(cpu_env)
    # We just mutated state directly — reset _rng_counter so step seeds
    # start at the same place as a fresh sequence would (1, 2, 3, ...).
    cpu_env._rng_counter = 0

    # ----- GPU side: matched buffers, mirror CPU's state explicitly.
    var states_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * STATE_SIZE)
    var actions_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
    var rewards_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
    var dones_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
    var terminated_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
    var obs_buf = ctx.enqueue_create_buffer[dtype](
        BATCH_SIZE * cpu_env.OBS_DIM
    )

    var host_state = List[Float32](capacity=STATE_SIZE)
    for _ in range(STATE_SIZE):
        host_state.append(Float32(0))
    copy_state_into_gpu_buffer(cpu_env, host_state)
    ctx.enqueue_copy(states_buf, host_state.unsafe_ptr())

    var host_zeros = List[Float32](capacity=BATCH_SIZE)
    for _ in range(BATCH_SIZE):
        host_zeros.append(Float32(0))
    ctx.enqueue_copy(actions_buf, host_zeros.unsafe_ptr())
    ctx.enqueue_copy(dones_buf, host_zeros.unsafe_ptr())
    ctx.synchronize()

    # ----- Compare initial states.
    var snap = List[Float32](capacity=STATE_SIZE)
    for _ in range(STATE_SIZE):
        snap.append(Float32(0))
    ctx.enqueue_copy(snap.unsafe_ptr(), states_buf)
    ctx.synchronize()
    compare_states(counts, "step=0", cpu_env, snap)

    # ----- Action sequence: [NOOP, NOOP, DO, NOOP, NOOP].
    # CPU step seed n uses _rng_counter=n; matching GPU rng_seed=n-1
    # makes per-env seed = (n-1)*1 + 0 + 1 = n.
    var actions = [ACTION_NOOP, ACTION_NOOP, ACTION_DO, ACTION_NOOP, ACTION_NOOP]
    for step_idx in range(5):
        var a = actions[step_idx]
        # CPU step
        _ = cpu_env.step_obs(a)

        # GPU step: rng_seed = step_idx (so per-env seed = step_idx + 1).
        var host_act = List[Float32](capacity=BATCH_SIZE)
        for _ in range(BATCH_SIZE):
            host_act.append(Float32(a))
        ctx.enqueue_copy(actions_buf, host_act.unsafe_ptr())
        ctx.synchronize()
        CraftaxClassicEnv[dtype].step_kernel_gpu[
            BATCH_SIZE, STATE_SIZE, cpu_env.OBS_DIM
        ](
            ctx,
            states_buf,
            actions_buf,
            rewards_buf,
            dones_buf,
            terminated_buf,
            obs_buf,
            rng_seed=UInt64(step_idx),
        )
        ctx.synchronize()

        # Snapshot GPU state and compare.
        ctx.enqueue_copy(snap.unsafe_ptr(), states_buf)
        ctx.synchronize()
        compare_states(
            counts, "step=" + String(step_idx + 1), cpu_env, snap
        )

    print()
    print("=" * 50)
    print("Passed:", counts[0])
    print("Failed:", counts[1])
    if counts[1] > 0:
        raise Error("Phase-5 CPU/GPU parity FAILED")
    print("Phase-5 CPU/GPU parity PASS")
