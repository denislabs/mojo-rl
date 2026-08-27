"""GPU Atari emulation spike — one env per GPU thread (CuLE-style AoS).

Scaffolding for the Stage-2 spike in docs/ATARI_AUDIT.md §3. Headless RAM-mode
Pong: N `AtariState`s live in a device byte buffer (array-of-structs), each GPU
thread runs F frames of its own env via the existing `run_frame`, and the result
is checked for bit-parity against the CPU path.

This file is ALSO a Metal-compilability probe. The opcode-dispatch divergence
historically crashed the Metal backend; the §1 outlining + the prereq-3 direct
opcode-table index may have moved that wall. It is kept OUT of the mojo_rl
package so a backend crash here cannot break other Atari imports.

    pixi run -e apple  mojo run -I . examples/arcade_games/atari_gpu_spike.mojo
    pixi run -e nvidia mojo run -I . examples/arcade_games/atari_gpu_spike.mojo

Requires roms/pong.bin.
"""

from std.sys import has_accelerator
from std.sys.info import size_of
from std.gpu import global_idx
from max.gpu.host import DeviceContext

from mojo_rl.envs.atari.environment import AtariEnvironment, load_rom
from mojo_rl.envs.atari.atari_state import AtariState
from mojo_rl.envs.atari.cpu6502 import run_frame, run_frame_cycle_accurate
from mojo_rl.envs.atari.opcodes import OpcodeEntry, OPCODE_TABLE
from mojo_rl.envs.atari.riot import set_action


# ---------------------------------------------------------------------------
# Per-thread kernel: one env per thread, run F frames applying a fixed action.
# `run_frame` / `set_action` are non-raising and heap-free (prereqs 1 & 3), so
# the headless path is kernel-legal. The ~1.2 KB `AtariState` is copied into a
# thread-local — the per-thread local-memory footprint the audit flags as the
# likely scaling wall (narrow the counters before pushing N_ENVS high).
# ---------------------------------------------------------------------------
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
        # Headless runner directly (NOT run_frame, which materializes the
        # comptime OPCODE_TABLE global — unavailable in the device module). The
        # opcode table is uploaded by the host and passed in via `op_table`.
        var dummy = InlineArray[UInt8, 4](fill=0)
        for _ in range(n_frames):
            set_action(st, act)
            run_frame_cycle_accurate[RENDER=False](
                st, rom, rom_size, dummy.unsafe_ptr(), op_table
            )
        states[i] = st^


# ---------------------------------------------------------------------------
# Host driver: stage states/rom/actions to device (AoS via a uint8 buffer
# reinterpreted as AtariState), launch one thread per env, copy states back.
# Operates in place on a host array of `n_envs` AtariStates.
# ---------------------------------------------------------------------------
def run_frames_gpu(
    ctx: DeviceContext,
    states: Pointer[AtariState, MutAnyOrigin],
    n_envs: Int,
    rom: Pointer[UInt8, MutAnyOrigin],
    rom_size: Int,
    actions: Pointer[UInt8, MutAnyOrigin],
    n_frames: Int,
) raises:
    comptime SB = size_of[AtariState]()
    var n_state_bytes = n_envs * SB

    # Host staging buffers (page-locked / device-mappable).
    var h_states = ctx.enqueue_create_host_buffer[DType.uint8](n_state_bytes)
    var hsp = h_states.unsafe_ptr().bitcast[AtariState]()
    for i in range(n_envs):
        hsp[i] = states[i].copy()

    var h_rom = ctx.enqueue_create_host_buffer[DType.uint8](rom_size)
    var hrp = h_rom.unsafe_ptr()
    for i in range(rom_size):
        hrp[i] = rom[i]

    var h_act = ctx.enqueue_create_host_buffer[DType.uint8](n_envs)
    var hap = h_act.unsafe_ptr()
    for i in range(n_envs):
        hap[i] = actions[i]

    # Opcode table → device (the §3 "table in device memory" answer): the
    # comptime OPCODE_TABLE global cannot be referenced from the device module,
    # so materialize it on the host and upload its 256 entries.
    comptime OTB = 256 * size_of[OpcodeEntry]()
    var optab = materialize[OPCODE_TABLE]()
    var h_opt = ctx.enqueue_create_host_buffer[DType.uint8](OTB)
    var hop = h_opt.unsafe_ptr().bitcast[OpcodeEntry]()
    for i in range(256):
        hop[i] = optab[i]

    # Device buffers + uploads.
    var d_states = ctx.enqueue_create_buffer[DType.uint8](n_state_bytes)
    var d_rom = ctx.enqueue_create_buffer[DType.uint8](rom_size)
    var d_act = ctx.enqueue_create_buffer[DType.uint8](n_envs)
    var d_opt = ctx.enqueue_create_buffer[DType.uint8](OTB)
    ctx.enqueue_copy(d_states, h_states)
    ctx.enqueue_copy(d_rom, h_rom)
    ctx.enqueue_copy(d_act, h_act)
    ctx.enqueue_copy(d_opt, h_opt)

    # One thread per env.
    comptime TPB = 64
    var blocks = (n_envs + TPB - 1) // TPB
    ctx.enqueue_function[atari_frames_kernel](
        d_states.unsafe_ptr().bitcast[AtariState](),
        d_rom.unsafe_ptr(),
        Int64(rom_size),
        d_opt.unsafe_ptr().bitcast[OpcodeEntry](),
        d_act.unsafe_ptr(),
        Int64(n_envs),
        Int64(n_frames),
        grid_dim=blocks,
        block_dim=TPB,
    )

    # Download the stepped states.
    ctx.enqueue_copy(h_states, d_states)
    ctx.synchronize()
    for i in range(n_envs):
        states[i] = hsp[i].copy()


def main() raises:
    comptime assert has_accelerator(), "spike requires a GPU"

    var rom = load_rom("roms/pong.bin")
    # Two names for one buffer, deliberately: `load_rom` owns the
    # allocation so it hands back `MutUntrackedOrigin`, which is what
    # `AtariEnvironment` stores in its field; every free helper here and
    # in `cpu6502` takes an Any origin. Convert once, at the source.
    var rom_ptr = rom.data.value()
    var rom_any = rom_ptr.as_unsafe_any_origin()
    var rom_size = rom.size
    print(
        "ROM:",
        rom_size,
        "bytes | size_of(AtariState) =",
        size_of[AtariState](),
        "bytes/env",
    )

    # A valid post-reset Pong state on the CPU; clone it across all envs.
    var env = AtariEnvironment(rom_ptr, rom_size)
    env.reset()
    var s0 = env.state.copy()

    comptime N = 256
    comptime F = 30  # frames stepped per env

    # Per-env action stream (alternate RIGHT/LEFT so lanes diverge — a real
    # test of independent per-thread emulation, not N copies of one trajectory).
    var actions = List[UInt8]()
    for i in range(N):
        actions.append(UInt8(3) if (i % 2 == 0) else UInt8(4))

    # ---- CPU reference ----
    var cpu_states = List[AtariState]()
    for _ in range(N):
        cpu_states.append(s0.copy())
    for i in range(N):
        var st = cpu_states[i].copy()
        for _ in range(F):
            set_action(st, actions[i])
            run_frame(st, rom_any, rom_size)
        cpu_states[i] = st^

    # ---- GPU ----
    var gpu_states = List[AtariState]()
    for _ in range(N):
        gpu_states.append(s0.copy())
    var ctx = DeviceContext()
    run_frames_gpu(
        ctx,
        gpu_states.unsafe_ptr().as_unsafe_any_origin(),
        N,
        rom_any,
        rom_size,
        actions.unsafe_ptr().as_unsafe_any_origin(),
        F,
    )

    # ---- Bit-parity on RAM (128 bytes) per env ----
    var mismatched_envs = 0
    var first_bad = -1
    for i in range(N):
        var diff = False
        for b in range(128):
            if cpu_states[i].ram[b] != gpu_states[i].ram[b]:
                diff = True
                break
        if diff:
            mismatched_envs += 1
            if first_bad < 0:
                first_bad = i

    if mismatched_envs == 0:
        print(
            "PARITY OK:",
            N,
            "envs ×",
            F,
            "frames — GPU RAM == CPU RAM, bit-identical",
        )
    else:
        print(
            "PARITY FAIL:",
            mismatched_envs,
            "/",
            N,
            "envs differ (first =",
            first_bad,
            ")",
        )
