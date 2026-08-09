"""GPU Atari emulation — scaling / throughput sweep on the local accelerator.

Measures aggregate emulation throughput as N_ENVS grows, one env per GPU thread
(the CuLE shape). The point of the port is *aggregate* frames/s at scale, and
the audit's #1 risk is the per-thread local-memory wall (952 B AtariState ×
resident threads) — this sweep is where that wall shows up as a throughput
collapse or an allocation failure.

Kernel-only timing: states/rom/table are uploaded ONCE per N; a warmup launch
pays the one-time pipeline JIT; then REPS single launches, each with its own
synchronize, are timed (enqueuing many launches without an intervening sync
overflows Metal's command buffer and hangs).

**Intended for NVIDIA — this benchmark is unreliable on Apple Metal and may hang
outright there.** Compiling this enormous kernel (the whole 6502+TIA+RIOT
emulator) to GPU machine code on first launch costs ~30 s of pipeline JIT, and
the repeated large device allocations across the N sweep + that JIT make Metal
stall unpredictably (observed: the same MEAS=2 launch returns in ~0.02 s on one
run and hangs >200 s on the next). It is NOT a logic bug — the emulator runs
correctly and bit-identically on Metal (see atari_gpu_spike.mojo, which is
reliable). But aggregate FPS at scale + the local-mem wall must be measured on
CUDA, where CuLE-class emulation is handled and the number reflects the target.

    pixi run -e nvidia mojo run -I . benchmarks/benchmark_atari_gpu_scaling.mojo
    pixi run -e apple  mojo run -I . benchmarks/benchmark_atari_gpu_scaling.mojo  # rough only

Requires roms/pong.bin.
"""

from std.sys import has_accelerator
from std.sys.info import size_of
from std.time import perf_counter_ns
from std.gpu import global_idx
from max.gpu.host import DeviceContext

from mojo_rl.envs.atari.environment import AtariEnvironment, load_rom
from mojo_rl.envs.atari.atari_state import AtariState
from mojo_rl.envs.atari.cpu6502 import run_frame, run_frame_cycle_accurate
from mojo_rl.envs.atari.opcodes import OpcodeEntry, OPCODE_TABLE
from mojo_rl.envs.atari.riot import set_action


comptime F = 4  # frames per "step" (ALE frame_skip)


def atari_frames_kernel(
    states: Pointer[AtariState, MutAnyOrigin],
    rom: Pointer[UInt8, MutAnyOrigin],
    rom_size_arg: Int64,
    op_table: Pointer[OpcodeEntry, MutAnyOrigin],
    actions: Pointer[UInt8, MutAnyOrigin],
    n_envs_arg: Int64,
    n_frames_arg: Int64,
):
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var rom_size = Int(rom_size_arg)
    var n_envs = Int(n_envs_arg)
    var n_frames = Int(n_frames_arg)
    var i = Int(global_idx.x)
    if i < n_envs:
        var st = states[i].copy()
        var act = actions[i]
        var dummy = InlineArray[UInt8, 4](fill=0)
        for _ in range(n_frames):
            set_action(st, act)
            run_frame_cycle_accurate[RENDER=False](
                st, rom, rom_size, dummy.unsafe_ptr(), op_table
            )
        states[i] = st^


def bench_n(
    ctx: DeviceContext,
    n_envs: Int,
    s0: AtariState,
    rom: Pointer[UInt8, MutAnyOrigin],
    rom_size: Int,
) raises:
    comptime SB = size_of[AtariState]()
    comptime OTB = 256 * size_of[OpcodeEntry]()

    # ---- upload once ----
    var h_states = ctx.enqueue_create_host_buffer[DType.uint8](n_envs * SB)
    var hsp = h_states.unsafe_ptr().bitcast[AtariState]()
    for i in range(n_envs):
        hsp[i] = s0.copy()

    var h_rom = ctx.enqueue_create_host_buffer[DType.uint8](rom_size)
    for i in range(rom_size):
        h_rom.unsafe_ptr()[i] = rom[i]

    var h_act = ctx.enqueue_create_host_buffer[DType.uint8](n_envs)
    for i in range(n_envs):
        # Alternate RIGHT/LEFT so lanes diverge (worst case for warp divergence).
        h_act.unsafe_ptr()[i] = UInt8(3) if (i % 2 == 0) else UInt8(4)

    var optab = materialize[OPCODE_TABLE]()
    var h_opt = ctx.enqueue_create_host_buffer[DType.uint8](OTB)
    var hop = h_opt.unsafe_ptr().bitcast[OpcodeEntry]()
    for i in range(256):
        hop[i] = optab[i]

    var d_states = ctx.enqueue_create_buffer[DType.uint8](n_envs * SB)
    var d_rom = ctx.enqueue_create_buffer[DType.uint8](rom_size)
    var d_act = ctx.enqueue_create_buffer[DType.uint8](n_envs)
    var d_opt = ctx.enqueue_create_buffer[DType.uint8](OTB)
    ctx.enqueue_copy(d_states, h_states)
    ctx.enqueue_copy(d_rom, h_rom)
    ctx.enqueue_copy(d_act, h_act)
    ctx.enqueue_copy(d_opt, h_opt)

    var sp = d_states.unsafe_ptr().bitcast[AtariState]()
    var op = d_opt.unsafe_ptr().bitcast[OpcodeEntry]()
    var rp = d_rom.unsafe_ptr()
    var ap = d_act.unsafe_ptr()

    comptime TPB = 64
    var blocks = (n_envs + TPB - 1) // TPB

    # Clean timing: a warmup launch pays the one-time Metal pipeline JIT; then
    # REPS single launches each followed by its OWN synchronize (enqueuing many
    # launches without an intervening sync overflows Metal's command buffer and
    # hangs). MEAS frames per launch amortizes per-launch overhead.
    # Frames per launch. Small for Apple (the giant-kernel JIT dominates and
    # larger counts stall there); raise it substantially on NVIDIA for a
    # steady-state throughput number.
    comptime MEAS = 2
    comptime REPS = 3
    print("  N=", n_envs, "warmup ...")
    ctx.enqueue_function[atari_frames_kernel](
        sp, rp, Int64(rom_size), op, ap, Int64(n_envs), Int64(MEAS), grid_dim=blocks, block_dim=TPB
    )
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(REPS):
        ctx.enqueue_function[atari_frames_kernel](
            sp, rp, Int64(rom_size), op, ap, Int64(n_envs), Int64(MEAS), grid_dim=blocks, block_dim=TPB
        )
        ctx.synchronize()
    var dt = Float64(perf_counter_ns() - t0) / 1e9

    var frames = Float64(REPS * n_envs * MEAS)
    print(
        "  N=",
        n_envs,
        "|",
        Int(frames / dt),
        "frames/s |",
        Int(frames / dt / Float64(F)),
        "steps/s |",
        Int(frames / dt / Float64(n_envs)),
        "fps/env | wall",
        dt,
        "s |",
        Int(Float64(n_envs * SB) / 1e6),
        "MB states",
    )


def main() raises:
    comptime assert has_accelerator(), "scaling bench requires a GPU"

    var rom = load_rom("roms/pong.bin")
    var rom_ptr = rom.data.value()
    var rom_size = rom.size

    var env = AtariEnvironment(rom_ptr, rom_size)
    env.reset()
    var s0 = env.state.copy()

    # CPU single-env baseline (context for the per-env vs aggregate gap).
    comptime CPU_FRAMES = 2000
    var cs = s0.copy()
    var c0 = perf_counter_ns()
    for _ in range(CPU_FRAMES):
        set_action(cs, UInt8(3))
        run_frame(cs, rom_ptr, rom_size)
    var cdt = Float64(perf_counter_ns() - c0) / 1e9
    print(
        "CPU 1-env baseline:",
        Int(Float64(CPU_FRAMES) / cdt),
        "frames/s (",
        Int(Float64(CPU_FRAMES) / cdt / Float64(F)),
        "steps/s ) | size_of(AtariState) =",
        size_of[AtariState](),
        "B/env",
    )
    print("--- GPU scaling (one env per thread, divergent actions) ---")

    var ctx = DeviceContext()
    var sweep = [256, 1024, 4096, 16384, 65536]
    for n in sweep:
        bench_n(ctx, n, s0, rom_ptr, rom_size)
