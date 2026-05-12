"""Phase-2 gate: GPU world generation, 1024 envs in <100 ms.

Generates `BATCH_SIZE` worlds via `reset_kernel_gpu`, copies the state
buffer back, and verifies:
  - Wall-clock time for the reset (excluding host alloc/copy)
  - Player tile is GRASS in every env
  - Map has at least one diamond reachable from the player tile (via the
    `always_diamond=False` path the diamond may be 0; we just check the
    spawn block invariant)

Run:
  pixi run -e apple  mojo run -I . examples/craftax_classic/gpu_world_gen_bench.mojo
  pixi run -e nvidia mojo run -I . examples/craftax_classic/gpu_world_gen_bench.mojo
"""

from std.gpu.host import DeviceContext
from std.time import perf_counter

from mojo_rl.envs.craftax_classic import (
    CraftaxClassicEnv,
    STATE_SIZE,
)
from mojo_rl.envs.craftax_classic.constants import (
    BLOCK_GRASS,
    BLOCK_WATER,
    BLOCK_LAVA,
    MAP_W,
    MAP_SIZE,
)
from mojo_rl.envs.craftax_classic.state import S_MAP_BASE, S_PLAYER_POS
from mojo_rl.nn import dtype


def main() raises:
    comptime BATCH_SIZE: Int = 1024

    print("Craftax-Classic — Phase-2 GPU world-gen benchmark")
    print("=" * 50)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("STATE_SIZE:", STATE_SIZE)

    var ctx = DeviceContext()
    var states_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE * STATE_SIZE)

    # Warm-up: first launch JIT-compiles the kernel — exclude from timing.
    CraftaxClassicEnv[dtype].reset_kernel_gpu[BATCH_SIZE, STATE_SIZE](
        ctx, states_buf, rng_seed=UInt64(1)
    )
    ctx.synchronize()

    # Timed run.
    var t0 = perf_counter()
    CraftaxClassicEnv[dtype].reset_kernel_gpu[BATCH_SIZE, STATE_SIZE](
        ctx, states_buf, rng_seed=UInt64(42)
    )
    ctx.synchronize()
    var elapsed_ms = (perf_counter() - t0) * 1000.0

    print()
    print("Wall time:", elapsed_ms, "ms")
    print("Per env:  ", elapsed_ms / Float64(BATCH_SIZE), "ms")
    if elapsed_ms < 100.0:
        print("GATE: PASS (<100 ms)")
    else:
        print("GATE: FAIL (>=100 ms)")

    # Pull back state and verify invariants.
    var host = List[Float32](capacity=BATCH_SIZE * STATE_SIZE)
    for _ in range(BATCH_SIZE * STATE_SIZE):
        host.append(Float32(0))
    ctx.enqueue_copy(host.unsafe_ptr(), states_buf)
    ctx.synchronize()

    var bad_spawn = 0
    var diamond_envs = 0
    var water_count_total = 0
    var stone_count_total = 0
    var lava_count_total = 0
    for e in range(BATCH_SIZE):
        var base = e * STATE_SIZE
        var py = Int(host[base + S_PLAYER_POS])
        var px = Int(host[base + S_PLAYER_POS + 1])
        var spawn_block = Int(host[base + S_MAP_BASE + py * MAP_W + px])
        if spawn_block != BLOCK_GRASS:
            bad_spawn += 1

        var has_diamond = False
        var w = 0
        var l = 0
        var s = 0
        for i in range(MAP_SIZE):
            var b = Int(host[base + S_MAP_BASE + i])
            if b == 10:  # BLOCK_DIAMOND
                has_diamond = True
            if b == BLOCK_WATER:
                w += 1
            if b == BLOCK_LAVA:
                l += 1
            if b == 4:  # BLOCK_STONE
                s += 1
        water_count_total += w
        lava_count_total += l
        stone_count_total += s
        if has_diamond:
            diamond_envs += 1

    print()
    print("Invariants over", BATCH_SIZE, "worlds:")
    print(
        "  bad spawn (player not on GRASS):",
        bad_spawn,
        "/",
        BATCH_SIZE,
    )
    print("  envs with at least 1 diamond  :", diamond_envs, "/", BATCH_SIZE)
    print(
        "  avg WATER tiles:",
        Float64(water_count_total) / Float64(BATCH_SIZE),
    )
    print(
        "  avg STONE tiles:",
        Float64(stone_count_total) / Float64(BATCH_SIZE),
    )
    print(
        "  avg LAVA tiles :",
        Float64(lava_count_total) / Float64(BATCH_SIZE),
    )

    if bad_spawn != 0:
        raise Error("INVARIANT FAILED: some envs spawn off-grass")

    print()
    print("Done.")
