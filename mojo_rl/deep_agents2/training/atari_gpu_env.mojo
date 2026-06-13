"""GPU-batched Atari 2600 environment (RAM-mode), one env per GPU thread.

`AtariGpuBatchedEnv[GAME, N_ENVS]` conforms to `BatchedEnv`, so it plugs into the
existing GPU off-policy discrete driver + Rainbow/C51 agent unchanged (the same
path that converged on Pong clean-obs). See docs/ATARI_AUDIT.md §3 — the spike
proved the emulator runs on the GPU bit-identically (~184K steps/s on NVIDIA);
this wraps it as a trainable env.

Design:
- States are `[N_ENVS, STATE_FLOATS]` `float32` reinterpreted to `AtariState*`
  (the 952-byte struct = 238 float32, the float-buffer the driver expects).
- The env OWNS device buffers for the ROM, the opcode table, and a canonical
  booted `s0` (the host boot sequence — 60 NOOP + RESET + … — can't run cheaply
  per-thread, so it runs once on the host and is broadcast on the GPU). This is
  why it conforms to `BatchedEnv` directly rather than via `BatchedGpuDiscreteEnv`
  (whose `reset_kernel_gpu` gets no workspace to read the ROM-dependent `s0`).
- Obs = the 128-byte console RAM / 255. Reward = per-game score delta
  (`GameDef.get_score` — a GPU-legal `@staticmethod`). Terminated =
  `GameDef.is_terminal`; done = terminated OR frame-budget truncation.
- Random no-op starts (ALE-standard, `NOOP_MAX`) decorrelate the N lanes.
"""

from std.sys.info import size_of
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import mptr
from mojo_rl.nn2.core.target_storage import require_ctx
from mojo_rl.deep_agents2.training.batched_env import BatchedEnv

from mojo_rl.envs.atari.environment import AtariEnvironment, GameDef
from mojo_rl.envs.atari.atari_state import AtariState
from mojo_rl.envs.atari.cpu6502 import run_frame_cycle_accurate
from mojo_rl.envs.atari.opcodes import OpcodeEntry, OPCODE_TABLE
from mojo_rl.envs.atari.riot import set_action
from mojo_rl.envs.atari.flags import ACTION_NOOP


comptime _TPB = 64


@always_inline
def _splitmix(x: UInt64) -> UInt64:
    """SplitMix64 finalizer — per-(seed,env) reproducible no-op counts."""
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


@always_inline
def _write_obs(
    st: AtariState, obs: UnsafePointer[Scalar[DT], MutAnyOrigin], i: Int
):
    """RAM (128 bytes) → obs[i*128 : i*128+128], normalized to [0,1]."""
    for b in range(128):
        obs[i * 128 + b] = st.ram[b].cast[DT]() / 255.0


# ---------------------------------------------------------------------------
# Reset kernel (shared by full reset and selective reset via comptime flag).
# Copies the canonical booted s0 into each (done) lane, runs a per-env random
# number of NOOP frames to decorrelate, rebases score/frame counters, writes obs.
# ---------------------------------------------------------------------------
def _atari_reset_kernel[
    GAME: GameDef, NOOP_MAX: Int, SELECTIVE: Bool
](
    states: UnsafePointer[AtariState, MutAnyOrigin],
    s0: UnsafePointer[AtariState, MutAnyOrigin],
    rom: UnsafePointer[UInt8, MutAnyOrigin],
    rom_size: Int,
    op_table: UnsafePointer[OpcodeEntry, MutAnyOrigin],
    dones: UnsafePointer[Scalar[DT], MutAnyOrigin],
    obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
    n_envs: Int,
    seed: UInt64,
):
    var i = Int(global_idx.x)
    if i >= n_envs:
        return
    comptime if SELECTIVE:
        if dones[i] <= 0.5:
            return  # lane still running — leave it
    var st = s0[0].copy()
    comptime if NOOP_MAX > 0:
        var k = Int(_splitmix(seed ^ UInt64(i)) % UInt64(NOOP_MAX + 1))
        var dummy = InlineArray[UInt8, 4](fill=0)
        for _ in range(k):
            set_action(st, ACTION_NOOP)
            run_frame_cycle_accurate[RENDER=False](
                st, rom, rom_size, dummy.unsafe_ptr(), op_table
            )
    # Rebase: episode reward baseline + frame budget start fresh post-decorrelation.
    st.score = Int32(GAME.get_score(st.ram))
    st.frame_number = 0
    comptime if SELECTIVE:
        dones[i] = 0.0
    _write_obs(st, obs, i)
    states[i] = st^


# ---------------------------------------------------------------------------
# Step kernel: frame_skip frames, then reward (score delta) / terminated /
# done (terminated OR truncation) / obs.
# ---------------------------------------------------------------------------
def _atari_step_kernel[
    GAME: GameDef, FRAME_SKIP: Int, MAX_FRAMES: Int
](
    states: UnsafePointer[AtariState, MutAnyOrigin],
    rom: UnsafePointer[UInt8, MutAnyOrigin],
    rom_size: Int,
    op_table: UnsafePointer[OpcodeEntry, MutAnyOrigin],
    actions: UnsafePointer[Scalar[DT], MutAnyOrigin],
    rewards: UnsafePointer[Scalar[DT], MutAnyOrigin],
    dones: UnsafePointer[Scalar[DT], MutAnyOrigin],
    terminated: UnsafePointer[Scalar[DT], MutAnyOrigin],
    obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
    n_envs: Int,
):
    var i = Int(global_idx.x)
    if i >= n_envs:
        return
    var st = states[i].copy()
    var ale = GAME.map_action(Int(actions[i]))
    var prev_score = Int(st.score)
    var dummy = InlineArray[UInt8, 4](fill=0)
    for _ in range(FRAME_SKIP):
        set_action(st, ale)
        run_frame_cycle_accurate[RENDER=False](
            st, rom, rom_size, dummy.unsafe_ptr(), op_table
        )
    var new_score = GAME.get_score(st.ram)
    st.score = Int32(new_score)
    var term = GAME.is_terminal(st.ram)
    var trunc = Int(st.frame_number) >= MAX_FRAMES
    rewards[i] = Scalar[DT](new_score - prev_score)
    terminated[i] = 1.0 if term else 0.0
    dones[i] = 1.0 if (term or trunc) else 0.0
    _write_obs(st, obs, i)
    states[i] = st^


struct AtariGpuBatchedEnv[
    GAME: GameDef,
    N_ENVS: Int,
    FRAME_SKIP: Int = 4,
    NOOP_MAX: Int = 30,
    MAX_FRAMES: Int = 108_000,
](BatchedEnv):
    comptime ENV_TARGET: StaticString = "gpu"
    comptime OBS_DIM: Int = 128
    comptime ACT_DIM: Int = 1
    comptime NUM_ACTIONS: Int = Self.GAME.NUM_ACTIONS
    # 952-byte AtariState as float32 lanes (238); the [N, STATE_FLOATS] float
    # buffer the driver expects, reinterpreted to AtariState* in the kernels.
    comptime STATE_FLOATS: Int = (
        size_of[AtariState]() + size_of[Scalar[DT]]() - 1
    ) // size_of[Scalar[DT]]()

    var _states: DeviceBuffer[DT]
    var _obs: DeviceBuffer[DT]
    var _action: DeviceBuffer[DT]
    var _reward: DeviceBuffer[DT]
    var _done: DeviceBuffer[DT]
    var _terminated: DeviceBuffer[DT]
    var _rom: DeviceBuffer[DType.uint8]
    var _optab: DeviceBuffer[DType.uint8]
    var _s0: DeviceBuffer[DType.uint8]
    var rom_size: Int

    def __init__(
        out self,
        ctx: DeviceContext,
        rom_ptr: UnsafePointer[UInt8, MutAnyOrigin],
        rom_size: Int,
    ) raises:
        self.rom_size = rom_size

        # Boot one canonical state on the host (the GPU can't cheaply run the
        # 60-NOOP + RESET boot per thread). reset() is game-agnostic and enough
        # for Pong; games needing ALE starting-actions would extend this.
        var henv = AtariEnvironment(rom_ptr, rom_size)
        henv.reset()
        var s0 = henv.state.copy()

        comptime SB = size_of[AtariState]()
        comptime OTB = 256 * size_of[OpcodeEntry]()

        # --- upload rom / opcode table / s0 to device ---
        var h_rom = ctx.enqueue_create_host_buffer[DType.uint8](rom_size)
        for i in range(rom_size):
            h_rom.unsafe_ptr()[i] = rom_ptr[i]
        self._rom = ctx.enqueue_create_buffer[DType.uint8](rom_size)
        ctx.enqueue_copy(self._rom, h_rom)

        var optab = materialize[OPCODE_TABLE]()
        var h_opt = ctx.enqueue_create_host_buffer[DType.uint8](OTB)
        var hop = h_opt.unsafe_ptr().bitcast[OpcodeEntry]()
        for i in range(256):
            hop[i] = optab[i]
        self._optab = ctx.enqueue_create_buffer[DType.uint8](OTB)
        ctx.enqueue_copy(self._optab, h_opt)

        var h_s0 = ctx.enqueue_create_host_buffer[DType.uint8](SB)
        h_s0.unsafe_ptr().bitcast[AtariState]()[0] = s0.copy()
        self._s0 = ctx.enqueue_create_buffer[DType.uint8](SB)
        ctx.enqueue_copy(self._s0, h_s0)

        # --- env buffers ---
        self._states = ctx.enqueue_create_buffer[DT](
            Self.N_ENVS * Self.STATE_FLOATS
        )
        self._obs = ctx.enqueue_create_buffer[DT](Self.N_ENVS * Self.OBS_DIM)
        self._action = ctx.enqueue_create_buffer[DT](Self.N_ENVS)
        self._reward = ctx.enqueue_create_buffer[DT](Self.N_ENVS)
        self._done = ctx.enqueue_create_buffer[DT](Self.N_ENVS)
        self._terminated = ctx.enqueue_create_buffer[DT](Self.N_ENVS)
        ctx.enqueue_memset(self._action, 0)
        ctx.enqueue_memset(self._reward, 0)
        ctx.enqueue_memset(self._done, 0)
        ctx.enqueue_memset(self._terminated, 0)

    @always_inline
    def _states_p(self) -> UnsafePointer[AtariState, MutAnyOrigin]:
        return self._states.unsafe_ptr().bitcast[AtariState]()

    @always_inline
    def _s0_p(self) -> UnsafePointer[AtariState, MutAnyOrigin]:
        return self._s0.unsafe_ptr().bitcast[AtariState]()

    @always_inline
    def _rom_p(self) -> UnsafePointer[UInt8, MutAnyOrigin]:
        return self._rom.unsafe_ptr()

    @always_inline
    def _opt_p(self) -> UnsafePointer[OpcodeEntry, MutAnyOrigin]:
        return self._optab.unsafe_ptr().bitcast[OpcodeEntry]()

    def reset_batch[BATCH: Int](
        mut self, ctx: Optional[DeviceContext], rng_seed: UInt64
    ) raises:
        comptime assert BATCH == Self.N_ENVS, "reset_batch BATCH mismatch"
        var c = require_ctx["AtariGpuBatchedEnv.reset_batch"](ctx)
        comptime blocks = (Self.N_ENVS + _TPB - 1) // _TPB
        comptime k = _atari_reset_kernel[Self.GAME, Self.NOOP_MAX, False]
        c.enqueue_function[k](
            self._states_p(),
            self._s0_p(),
            self._rom_p(),
            self.rom_size,
            self._opt_p(),
            mptr(self._done.unsafe_ptr()),
            mptr(self._obs.unsafe_ptr()),
            Self.N_ENVS,
            rng_seed,
            grid_dim=(blocks,),
            block_dim=(_TPB,),
        )

    def step_batch[BATCH: Int](
        mut self, ctx: Optional[DeviceContext], rng_seed: UInt64
    ) raises:
        comptime assert BATCH == Self.N_ENVS, "step_batch BATCH mismatch"
        _ = rng_seed
        var c = require_ctx["AtariGpuBatchedEnv.step_batch"](ctx)
        comptime blocks = (Self.N_ENVS + _TPB - 1) // _TPB
        comptime k = _atari_step_kernel[
            Self.GAME, Self.FRAME_SKIP, Self.MAX_FRAMES
        ]
        c.enqueue_function[k](
            self._states_p(),
            self._rom_p(),
            self.rom_size,
            self._opt_p(),
            mptr(self._action.unsafe_ptr()),
            mptr(self._reward.unsafe_ptr()),
            mptr(self._done.unsafe_ptr()),
            mptr(self._terminated.unsafe_ptr()),
            mptr(self._obs.unsafe_ptr()),
            Self.N_ENVS,
            grid_dim=(blocks,),
            block_dim=(_TPB,),
        )

    def selective_reset_batch[BATCH: Int](
        mut self, ctx: Optional[DeviceContext], rng_seed: UInt64
    ) raises:
        comptime assert BATCH == Self.N_ENVS, "selective_reset_batch BATCH mismatch"
        var c = require_ctx["AtariGpuBatchedEnv.selective_reset_batch"](ctx)
        comptime blocks = (Self.N_ENVS + _TPB - 1) // _TPB
        comptime k = _atari_reset_kernel[Self.GAME, Self.NOOP_MAX, True]
        c.enqueue_function[k](
            self._states_p(),
            self._s0_p(),
            self._rom_p(),
            self.rom_size,
            self._opt_p(),
            mptr(self._done.unsafe_ptr()),
            mptr(self._obs.unsafe_ptr()),
            Self.N_ENVS,
            rng_seed,
            grid_dim=(blocks,),
            block_dim=(_TPB,),
        )

    def obs_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._obs.unsafe_ptr())

    def action_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._action.unsafe_ptr())

    def reward_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._reward.unsafe_ptr())

    def done_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._done.unsafe_ptr())

    def terminated_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return mptr(self._terminated.unsafe_ptr())
