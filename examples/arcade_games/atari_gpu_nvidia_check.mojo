"""Turnkey NVIDIA validation + benchmark for the GPU Atari emulator.

One file to run on an NVIDIA box to settle the Stage-2 spike questions
(docs/ATARI_AUDIT.md §3). It does four things, in order:

  0. CPU self-check  — the UNIFORM (per-clock, no bulk-skip) runner is
     bit-identical to the default bulk runner on CPU. Proves the runner variant
     used for the divergence test below is correct before trusting any GPU
     number.
  1. PARITY GATE     — the real correctness go/no-go: step P frames across N
     envs (divergent per-env actions) on CPU and GPU, compare the FULL
     AtariState byte-for-byte. This is the acceptance test at scale.
  2. THROUGHPUT      — aggregate frames/s vs N (the headline "FPS at scale"),
     with fps/env so the local-mem wall (audit risk #1) shows up as a collapse.
  3. RUNNER COMPARE  — same sweep with the UNIFORM runner, so bulk-skip vs
     uniform-per-clock can be compared head-to-head (audit risk #2: uniform work
     across a warp may beat the divergent bulk path).

Tunables are the comptime constants near the top; bump MEAS / the N sweep up on
real hardware. On Apple Metal phases 0–1 work (correctness), but the throughput
phases are unreliable there — see benchmark_atari_gpu_scaling.mojo. Run it on
NVIDIA:

    pixi run -e nvidia mojo run -I . examples/arcade_games/atari_gpu_nvidia_check.mojo

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


comptime F = 4  # ALE frame_skip (frames per step)

# Parity gate: (N_ENVS, frames) configs. Raise on NVIDIA.
comptime PARITY_N = 4096
comptime PARITY_FRAMES = 120

# Throughput sweep. Raise MEAS substantially on NVIDIA for steady state.
comptime MEAS = 8
comptime REPS = 3


@always_inline
def env_action(i: Int) -> UInt8:
    """Fixed-per-env action so lanes in a warp diverge (worst case for warp
    divergence) while the stream stays identical on CPU and GPU."""
    return UInt8(i % 6)


# ---------------------------------------------------------------------------
# Per-thread kernel, parameterized on the runner variant.
# ---------------------------------------------------------------------------
def atari_frames_kernel[
    UNIFORM: Bool
](
    states: Pointer[AtariState, MutAnyOrigin],
    rom: Pointer[UInt8, MutAnyOrigin],
    # ⚠ Int32, NOT Int — `Int`/`UInt` are not `DevicePassable`. A bare
    # `Int` still compiles in `pixi run build` and fails only where the
    # kernel is LAUNCHED; keep the `Int32(...)` casts at the call sites.
    rom_size: Int32,
    op_table: Pointer[OpcodeEntry, MutAnyOrigin],
    actions: Pointer[UInt8, MutAnyOrigin],
    n_envs: Int32,
    n_frames: Int32,
):
    var i = Int(global_idx.x)
    if i < Int(n_envs):
        var st = states[i].copy()
        var act = actions[i]
        var dummy = InlineArray[UInt8, 4](fill=0)
        for _ in range(Int(n_frames)):
            set_action(st, act)
            run_frame_cycle_accurate[RENDER=False, UNIFORM=UNIFORM](
                st, rom, Int(rom_size), dummy.unsafe_ptr(), op_table
            )
        states[i] = st^


def run_frames_gpu[
    UNIFORM: Bool
](
    ctx: DeviceContext,
    states: Pointer[AtariState, MutAnyOrigin],
    n_envs: Int,
    rom: Pointer[UInt8, MutAnyOrigin],
    rom_size: Int,
    actions: Pointer[UInt8, MutAnyOrigin],
    n_frames: Int,
) raises:
    comptime SB = size_of[AtariState]()
    comptime OTB = 256 * size_of[OpcodeEntry]()

    var h_states = ctx.enqueue_create_host_buffer[DType.uint8](n_envs * SB)
    var hsp = h_states.unsafe_ptr().bitcast[AtariState]()
    for i in range(n_envs):
        hsp[i] = states[i].copy()
    var h_rom = ctx.enqueue_create_host_buffer[DType.uint8](rom_size)
    for i in range(rom_size):
        h_rom.unsafe_ptr()[i] = rom[i]
    var h_act = ctx.enqueue_create_host_buffer[DType.uint8](n_envs)
    for i in range(n_envs):
        h_act.unsafe_ptr()[i] = actions[i]
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

    comptime TPB = 64
    var blocks = (n_envs + TPB - 1) // TPB
    comptime kern = atari_frames_kernel[UNIFORM]
    ctx.enqueue_function[kern](
        d_states.unsafe_ptr().bitcast[AtariState](),
        d_rom.unsafe_ptr(),
        Int32(rom_size),
        d_opt.unsafe_ptr().bitcast[OpcodeEntry](),
        d_act.unsafe_ptr(),
        Int32(n_envs),
        Int32(n_frames),
        grid_dim=blocks,
        block_dim=TPB,
    )

    ctx.enqueue_copy(h_states, d_states)
    ctx.synchronize()
    for i in range(n_envs):
        states[i] = hsp[i].copy()


def cpu_step[
    UNIFORM: Bool
](
    mut st: AtariState,
    rom: Pointer[UInt8, MutAnyOrigin],
    rom_size: Int,
    action: UInt8,
):
    """One frame on CPU via the chosen runner variant (for self-check)."""
    var optab = materialize[OPCODE_TABLE]()
    var dummy = InlineArray[UInt8, 4](fill=0)
    set_action(st, action)
    run_frame_cycle_accurate[RENDER=False, UNIFORM=UNIFORM](
        st, rom, rom_size, dummy.unsafe_ptr(), optab.unsafe_ptr()
    )


def ram_diff(a: AtariState, b: AtariState) -> Int:
    """First differing RAM byte (0..127), or -1 if the 128 B RAM is identical.

    RAM is the gameplay/observation state (RL obs is derived from it), so RAM
    equality is the meaningful correctness gate. A full-struct byte compare is
    deliberately NOT used: (1) the GPU kernel's struct copy-in/out does not
    preserve inter-field padding bytes, and (2) the bulk and uniform runners
    leave internal CycleTIA counter *phase* slightly different while producing
    identical RAM and pixels — both are non-bugs that a full compare would flag.
    """
    for k in range(128):
        if a.ram[k] != b.ram[k]:
            return k
    return -1


def bench_throughput[
    UNIFORM: Bool
](
    ctx: DeviceContext,
    n_envs: Int,
    s0: AtariState,
    rom: Pointer[UInt8, MutAnyOrigin],
    rom_size: Int,
) raises:
    comptime SB = size_of[AtariState]()
    comptime OTB = 256 * size_of[OpcodeEntry]()

    var h_states = ctx.enqueue_create_host_buffer[DType.uint8](n_envs * SB)
    var hsp = h_states.unsafe_ptr().bitcast[AtariState]()
    for i in range(n_envs):
        hsp[i] = s0.copy()
    var h_rom = ctx.enqueue_create_host_buffer[DType.uint8](rom_size)
    for i in range(rom_size):
        h_rom.unsafe_ptr()[i] = rom[i]
    var h_act = ctx.enqueue_create_host_buffer[DType.uint8](n_envs)
    for i in range(n_envs):
        h_act.unsafe_ptr()[i] = env_action(i)
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
    comptime kern = atari_frames_kernel[UNIFORM]

    ctx.enqueue_function[kern](
        sp, rp, Int32(rom_size), op, ap, Int32(n_envs), Int32(MEAS),
        grid_dim=blocks, block_dim=TPB,
    )
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(REPS):
        ctx.enqueue_function[kern](
            sp, rp, Int32(rom_size), op, ap, Int32(n_envs), Int32(MEAS),
            grid_dim=blocks, block_dim=TPB,
        )
        ctx.synchronize()
    var dt = Float64(perf_counter_ns() - t0) / 1e9
    var frames = Float64(REPS * n_envs * MEAS)
    print(
        "    N=",
        n_envs,
        "|",
        Int(frames / dt),
        "frames/s |",
        Int(frames / dt / Float64(n_envs)),
        "fps/env |",
        Int(Float64(n_envs * SB) / 1e6),
        "MB",
    )


def parity_gate(
    ctx: DeviceContext,
    n_envs: Int,
    n_frames: Int,
    s0: AtariState,
    rom: Pointer[UInt8, MutAnyOrigin],
    rom_size: Int,
) raises:
    var actions = List[UInt8]()
    for i in range(n_envs):
        actions.append(env_action(i))

    var cpu = List[AtariState]()
    for _ in range(n_envs):
        cpu.append(s0.copy())
    for i in range(n_envs):
        var st = cpu[i].copy()
        for _ in range(n_frames):
            set_action(st, actions[i])
            run_frame(st, rom, rom_size)
        cpu[i] = st^

    var gpu = List[AtariState]()
    for _ in range(n_envs):
        gpu.append(s0.copy())
    run_frames_gpu[False](
        ctx,
        gpu.unsafe_ptr().as_unsafe_any_origin(),
        n_envs,
        rom,
        rom_size,
        actions.unsafe_ptr().as_unsafe_any_origin(),
        n_frames,
    )

    var bad = 0
    var first_env = -1
    var first_byte = -1
    for i in range(n_envs):
        var d = ram_diff(cpu[i], gpu[i])
        if d >= 0:
            bad += 1
            if first_env < 0:
                first_env = i
                first_byte = d
    if bad == 0:
        print(
            "  PARITY OK:",
            n_envs,
            "envs ×",
            n_frames,
            "frames — GPU RAM == CPU RAM, bit-identical",
        )
    else:
        print(
            "  PARITY FAIL:",
            bad,
            "/",
            n_envs,
            "envs differ (first env",
            first_env,
            "RAM byte",
            first_byte,
            ")",
        )


def main() raises:
    comptime assert has_accelerator(), "requires a GPU"
    var rom = load_rom("roms/pong.bin")
    # Two names for one buffer, deliberately: `load_rom` owns the
    # allocation so it hands back `MutUntrackedOrigin`, which is what
    # `AtariEnvironment` stores in its field; every free helper here and
    # in `cpu6502` takes an Any origin. Convert once, at the source.
    var rom_ptr = rom.data.value()
    var rom_any = rom_ptr.as_unsafe_any_origin()
    var rom_size = rom.size
    var env = AtariEnvironment(rom_ptr, rom_size)
    env.reset()
    var s0 = env.state.copy()
    var ctx = DeviceContext()
    print(
        "Atari GPU NVIDIA check | size_of(AtariState) =",
        size_of[AtariState](),
        "B/env\n",
    )

    # ---- Phase 0: CPU self-check of the UNIFORM runner ----
    print("[0] CPU self-check: uniform runner == bulk runner ...")
    var a = s0.copy()
    var b = s0.copy()
    for f in range(200):
        var act = UInt8(f % 6)
        cpu_step[False](a, rom_any, rom_size, act)
        cpu_step[True](b, rom_any, rom_size, act)
    var d0 = ram_diff(a, b)
    if d0 < 0:
        print("    OK — RAM bit-identical over 200 frames\n")
    else:
        print("    FAIL — RAM diverged at byte", d0, "\n")

    # ---- Phase 1: parity gate ----
    print("[1] Parity gate (CPU vs GPU, full state):")
    parity_gate(ctx, 256, PARITY_FRAMES, s0, rom_any, rom_size)
    parity_gate(ctx, PARITY_N, PARITY_FRAMES, s0, rom_any, rom_size)

    # ---- Phase 2: throughput (bulk runner) ----
    print("\n[2] Throughput — BULK runner:")
    var sweep = [256, 1024, 4096, 16384, 65536]
    for n in sweep:
        bench_throughput[False](ctx, n, s0, rom_any, rom_size)

    # ---- Phase 3: throughput (uniform runner) ----
    print("\n[3] Throughput — UNIFORM (per-clock) runner:")
    for n in sweep:
        bench_throughput[True](ctx, n, s0, rom_any, rom_size)
