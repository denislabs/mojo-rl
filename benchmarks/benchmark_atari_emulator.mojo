"""Atari emulator throughput benchmark — Pong, headless + pixel pipeline.

Measures the two paths that matter for RL training throughput:
  1. RAM-mode step (frame_skip=4, RENDER=False frames only)
  2. Pixel-mode step (2 headless + 2 rendered frames + maxpool/gray/resize
     + obs-list build)

Run from the project root:
    pixi run mojo run -I . benchmarks/benchmark_atari_emulator.mojo

Requires roms/pong.bin.
"""

from std.time import perf_counter_ns

from mojo_rl.envs.atari import AtariEnv, load_rom
from mojo_rl.envs.atari.games.registry import AtariGame


comptime RAM_STEPS = 2_000
comptime PIXEL_STEPS = 1_000


def main() raises:
    var rom = load_rom("roms/pong.bin")
    print("ROM loaded:", rom.size, "bytes")

    # ---- RAM mode (headless, frame_skip=4) ----
    var env = AtariEnv[0](AtariGame.PONG, rom.data.value(), rom.size)
    _ = env.reset_obs_list()
    # Warmup (let the game leave the boot screen)
    for _ in range(100):
        _ = env.step_obs(0)

    var t0 = perf_counter_ns()
    var ep = 0
    for i in range(RAM_STEPS):
        var r = env.step_obs(i % 6)
        if r[2]:
            ep += 1
            _ = env.reset_obs_list()
    var dt = Float64(perf_counter_ns() - t0) / 1e9
    print(
        "RAM-mode  :",
        Int(Float64(RAM_STEPS) / dt),
        "steps/s |",
        Int(Float64(RAM_STEPS * 4) / dt),
        "frames/s | episodes:",
        ep,
    )

    # ---- Pixel mode (2 headless + 2 rendered frames per step) ----
    var envp = AtariEnv[1](AtariGame.PONG, rom.data.value(), rom.size)
    _ = envp.reset_obs_list()
    for _ in range(50):
        _ = envp.step_obs(0)

    var t1 = perf_counter_ns()
    var ep_p = 0
    var checksum: Float64 = 0.0
    for i in range(PIXEL_STEPS):
        var r = envp.step_obs(i % 6)
        # Touch the obs so the pipeline can't be optimized away.
        checksum += Float64(r[0][0]) + Float64(r[0][14000])
        if r[2]:
            ep_p += 1
            _ = envp.reset_obs_list()
    var dtp = Float64(perf_counter_ns() - t1) / 1e9
    print(
        "Pixel-mode:",
        Int(Float64(PIXEL_STEPS) / dtp),
        "steps/s |",
        Int(Float64(PIXEL_STEPS * 4) / dtp),
        "frames/s | episodes:",
        ep_p,
        "| checksum:",
        checksum,
    )
