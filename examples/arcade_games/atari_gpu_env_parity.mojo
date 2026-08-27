"""Parity test for AtariGpuBatchedEnv (RAM-mode) vs a CPU reference.

Steps N Pong envs on the GPU through the BatchedEnv interface (reset + step with
divergent per-env actions) and checks reward / done / RAM against a CPU reference
that replays the identical action streams. NOOP_MAX=0 so reset is deterministic
(pure s0 broadcast), making the CPU reference exact.

    pixi run -e apple  mojo run -I . examples/arcade_games/atari_gpu_env_parity.mojo
    pixi run -e nvidia mojo run -I . examples/arcade_games/atari_gpu_env_parity.mojo

Requires roms/pong.bin.
"""

from std.sys import has_accelerator
from std.sys.info import size_of
from std.math import abs
from max.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.training.atari_gpu_env import AtariGpuBatchedEnv
from mojo_rl.envs.atari.environment import AtariEnvironment, load_rom
from mojo_rl.envs.atari.atari_state import AtariState
from mojo_rl.envs.atari.cpu6502 import run_frame_cycle_accurate
from mojo_rl.envs.atari.opcodes import OpcodeEntry, OPCODE_TABLE
from mojo_rl.envs.atari.riot import set_action
from mojo_rl.envs.atari.games import PongDef


comptime N = 64
comptime T = 20
comptime FRAME_SKIP = 4
comptime MAX_FRAMES = 108_000


def cpu_step(
    mut st: AtariState,
    action_idx: Int,
    rom: Pointer[UInt8, MutAnyOrigin],
    rom_size: Int,
    op_table: Pointer[OpcodeEntry, MutAnyOrigin],
) -> Tuple[Int, Int]:
    """One env step on CPU (mirrors the GPU step kernel). Returns (reward, done)."""
    var ale = PongDef.map_action(action_idx)
    var prev = Int(st.score)
    var dummy = InlineArray[UInt8, 4](fill=0)
    for _ in range(FRAME_SKIP):
        set_action(st, ale)
        run_frame_cycle_accurate[RENDER=False](
            st, rom, rom_size, dummy.unsafe_ptr(), op_table
        )
    var ns = PongDef.get_score(st.ram)
    st.score = Int32(ns)
    var term = PongDef.is_terminal(st.ram)
    var trunc = Int(st.frame_number) >= MAX_FRAMES
    return (ns - prev, 1 if (term or trunc) else 0)


def main() raises:
    comptime assert has_accelerator(), "requires a GPU"
    var rom = load_rom("roms/pong.bin")
    var rom_ptr = rom.data.value()
    var rom_size = rom.size
    var ctx = DeviceContext()

    comptime Env = AtariGpuBatchedEnv[PongDef, N, FRAME_SKIP, 0, MAX_FRAMES]
    var env = Env(ctx, rom_ptr, rom_size)
    print("AtariGpuBatchedEnv | N =", N, "| OBS_DIM =", Env.OBS_DIM,
          "| NUM_ACTIONS =", Env.NUM_ACTIONS,
          "| STATE_FLOATS =", Env.STATE_FLOATS)

    # --- CPU reference base state: same host boot + same rebase as reset kernel ---
    var henv = AtariEnvironment(rom_ptr, rom_size)
    henv.reset()
    var base = henv.state.copy()
    base.score = Int32(PongDef.get_score(base.ram))
    base.frame_number = 0
    var optab = materialize[OPCODE_TABLE]()

    var cpu = List[AtariState]()
    for _ in range(N):
        cpu.append(base.copy())

    # per-env action stream (fixed per env so lanes diverge)
    var actions = List[Int]()
    for e in range(N):
        actions.append(e % 6)

    # --- GPU: reset then T steps ---
    env.reset_batch[N](ctx, UInt64(0))

    var act_h = ctx.enqueue_create_host_buffer[DT](N)
    for e in range(N):
        act_h.unsafe_ptr()[e] = Scalar[DT](actions[e])
    var act_dev = DeviceBuffer[DT](ctx, env.action_ptr(), N, owning=False)
    ctx.enqueue_copy(act_dev, act_h)

    var obs_h = ctx.enqueue_create_host_buffer[DT](N * 128)
    var rew_h = ctx.enqueue_create_host_buffer[DT](N)
    var done_h = ctx.enqueue_create_host_buffer[DT](N)

    var reward_mismatch = 0
    var done_mismatch = 0
    var ram_mismatch = 0

    for _t in range(T):
        # GPU step
        env.step_batch[N](ctx, UInt64(0))
        var obs_dev = DeviceBuffer[DT](ctx, env.obs_ptr(), N * 128, owning=False)
        var rew_dev = DeviceBuffer[DT](ctx, env.reward_ptr(), N, owning=False)
        var done_dev = DeviceBuffer[DT](ctx, env.done_ptr(), N, owning=False)
        ctx.enqueue_copy(obs_h, obs_dev)
        ctx.enqueue_copy(rew_h, rew_dev)
        ctx.enqueue_copy(done_h, done_dev)
        ctx.synchronize()

        # CPU reference step + compare
        for e in range(N):
            # `.as_unsafe_any_origin()` on both: `rom_ptr` is
            # `MutUntrackedOrigin` (it comes from `load_rom`) and
            # `optab.unsafe_ptr()` is tracked. Neither converts implicitly
            # to the `MutAnyOrigin` this helper shares with the GPU kernel.
            var res = cpu_step(
                cpu[e],
                actions[e],
                rom_ptr.as_unsafe_any_origin(),
                rom_size,
                optab.unsafe_ptr().as_unsafe_any_origin(),
            )
            var c_reward = res[0]
            var c_done = res[1]
            if abs(Float64(rew_h.unsafe_ptr()[e]) - Float64(c_reward)) > 0.5:
                reward_mismatch += 1
            if Int(done_h.unsafe_ptr()[e] + 0.5) != c_done:
                done_mismatch += 1
            for b in range(128):
                var gpu_byte = Int(Float64(obs_h.unsafe_ptr()[e * 128 + b]) * 255.0 + 0.5)
                if gpu_byte != Int(cpu[e].ram[b]):
                    ram_mismatch += 1
                    break

    print("steps:", T, "× envs:", N)
    print("  reward mismatches:", reward_mismatch)
    print("  done   mismatches:", done_mismatch)
    print("  RAM    mismatches:", ram_mismatch)
    if reward_mismatch == 0 and done_mismatch == 0 and ram_mismatch == 0:
        print("PARITY OK — GPU env == CPU reference (reward/done/RAM)")
    else:
        print("PARITY FAIL")
