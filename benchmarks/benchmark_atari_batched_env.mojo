"""Multi-core BatchedCpuDiscreteEnv scaling benchmark (Atari pixel mode).

Stage-1 measurement for `docs/ATARI_AUDIT.md` §2: aggregate env steps/s
of `BatchedCpuDiscreteEnv[AtariEnv[1], N]` for N = 1, 2, 4, 8 on Pong
pixel mode (the Rainbow-Atari training configuration: 4 emulated frames
per step + max-pool/gray/resize/obs-build pipeline). Random actions; no
agent or GPU in the loop — this isolates the env side of a training
iteration.

Run (add `-D ASSERT=none` for the +15% recommended training build):
    pixi run mojo run -I . benchmarks/benchmark_atari_batched_env.mojo

Requires `roms/pong.bin` (run from the repo root).
"""

from std.time import perf_counter_ns

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.training.batched_env import BatchedCpuDiscreteEnv
from mojo_rl.envs.atari import AtariEnv, load_rom
from mojo_rl.envs.atari.games.registry import AtariGame


comptime PongPixel = AtariEnv[1]
comptime OBS = 4 * 84 * 84
comptime ITERS = 250


def bench[N: Int](
    rom: UnsafePointer[UInt8, MutAnyOrigin], rom_size: Int
) raises -> Float64:
    var envs = List[PongPixel]()
    for _ in range(N):
        envs.append(PongPixel(AtariGame.PONG, rom, rom_size))
    var batched = BatchedCpuDiscreteEnv[PongPixel, N, OBS](
        envs^, noop_max=30
    )
    batched.reset_batch[N](ctx=None, rng_seed=UInt64(42))

    var act = batched.action_ptr()
    # Warm-up iterations (thread pool spin-up, caches).
    var rng: UInt64 = 1
    for _ in range(10):
        for i in range(N):
            rng = rng * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            act[i] = Scalar[DT]((rng >> 33) % 6)
        batched.step_batch[N](ctx=None, rng_seed=rng)
        batched.selective_reset_batch[N](ctx=None, rng_seed=rng)

    var t0 = perf_counter_ns()
    for it in range(ITERS):
        for i in range(N):
            rng = rng * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            act[i] = Scalar[DT]((rng >> 33) % 6)
        batched.step_batch[N](ctx=None, rng_seed=UInt64(it))
        batched.selective_reset_batch[N](ctx=None, rng_seed=UInt64(it) * 7)
    var elapsed = Float64(perf_counter_ns() - t0) / 1e9
    return Float64(N * ITERS) / elapsed


def main() raises:
    var rom = load_rom("roms/pong.bin")
    print("Atari Pong pixel-mode batched CPU env scaling (steps/s aggregate)")
    var s1 = bench[1](rom.data.value(), rom.size)
    print("N_ENVS=1 :", s1, " (1.00x)")
    var s2 = bench[2](rom.data.value(), rom.size)
    print("N_ENVS=2 :", s2, " (", s2 / s1, "x)")
    var s4 = bench[4](rom.data.value(), rom.size)
    print("N_ENVS=4 :", s4, " (", s4 / s1, "x)")
    var s8 = bench[8](rom.data.value(), rom.size)
    print("N_ENVS=8 :", s8, " (", s8 / s1, "x)")
