"""Online sim-backed Pong sampler — generates LeWM windows on the fly.

SKETCH (2026-06-06). A drop-in alternative to ``PongOfflineBuffer`` that
conforms to the SAME ``mojo_rl.core.offline_buffer.OfflineBuffer`` contract,
but instead of replaying frames from disk it *generates* them by stepping
``PongPixelEnv`` instances under a behavior policy. This is the "we own both
sides" path: no 350 GB dataset, no recording — the simulator IS the data
source, and the existing window-source / trainer stack consumes it unchanged.

Design decisions (see the design notes in the chat / DREAMER4_PORT_PLAN.md):

1. **Same trait, drop-in.** ``INPUT_LAYOUT_HWC == False`` and
   ``sample_batch_uint8`` have byte-identical semantics to ``PongOfflineBuffer``:
   pixels ``(B, T, PONG_FRAME_BYTES)`` uint8 CHW, actions ``(B, T, ACT)`` fp32
   one-hot. So a generic ``OfflineBuffer``-typed consumer (see the
   ``WindowSource`` generalization note below) swaps offline↔online for free.

2. **uint8 round-trip is deliberate, not waste.** ``PongPixelEnv`` emits fp32
   obs in [0, 1]; we quantize to uint8 here, and the downstream window source
   converts uint8→fp32 again. That round-trip is exactly the quantization a
   *stored* buffer would apply, so online and offline data are numerically
   comparable — the offline fixture stays a valid parity oracle for the online
   path. (If profiling ever shows it matters, add a parallel fp32 "live" seam;
   not needed at lighthouse scale.)

3. **B = pool of persistent envs.** Each ``sample_batch_uint8`` call advances
   ``B`` independent env instances by ``T`` steps, capturing one length-``T``
   window per env. Envs persist across calls (continuous play), so windows are
   drawn from mid-episode trajectories — matching the distribution of a
   continuously-collected offline buffer, not just episode openings.

4. **No reset-bridging windows.** Mirrors ``PongOfflineBuffer._window_is_valid``:
   a window must not straddle an episode boundary. We roll ``T`` steps into a
   scratch and, if a ``done`` fires before the final step, reset that env and
   re-roll (bounded retries). The final frame may be terminal (valid target).

5. **Pluggable policy = the data-recipe knob.** ``POLICY`` is the behavior that
   drives the env. The default ``ScriptedPongPolicy`` is the follow-the-ball +
   epsilon mix lifted from ``lewm_pong_collect_buffer.mojo``. Swap in a trained
   DQN/PPO agent (reading ``env.get_obs_list()``) + action noise to reproduce
   the Dreamer-4 "expert demos + mixed-quality noise" recipe natively.

TODO before production:
  - GPU target: today this is the CPU path (parallels ``PongWindowSource`` "cpu").
    A GPU variant should drive ``BatchedGpuDiscreteEnv`` + a device quantize
    kernel and write uint8 straight to a device staging buffer.
  - Generalize ``PongWindowSource`` to ``WindowSource[BUF: OfflineBuffer, ...]``
    (it currently hardcodes ``var buf: PongOfflineBuffer``) so this sampler is a
    literal drop-in. That ~3-line change is the real payoff of the trait.
  - Strict reproducibility: thread an explicit RNG seed through the policy and
    the env resets rather than relying on the global ``std.random`` stream.
"""

from std.random import random_float64

from mojo_rl.core.offline_buffer import OfflineBuffer
from mojo_rl.envs.arcade_games.pong import PongPixelEnv
from mojo_rl.envs.arcade_games.pong.pong import S_BALL_Y, S_PADDLE_Y
from mojo_rl.envs.arcade_games.pong.offline_buffer import (
    PONG_FRAME_BYTES,
    PONG_NUM_ACTIONS,
)

comptime PONG_DT = DType.float32


# ============================================================================
# Behavior-policy seam
# ============================================================================


trait PongActionPolicy(Movable & ImplicitlyDeletable):
    """Maps the current env to a discrete action in {0=NOOP, 1=UP, 2=DOWN}.

    Implementations may read ``env.inner.state`` (scripted) or
    ``env.get_obs_list()`` (a trained pixel agent). This is where the
    expert-vs-noise data recipe lives.
    """

    def select_action(mut self, env: PongPixelEnv[PONG_DT]) raises -> Int:
        ...


struct ScriptedPongPolicy(PongActionPolicy):
    """Follow-the-ball with per-instance epsilon-random exploration.

    Lifted from ``examples/lewm/lewm_pong_collect_buffer.mojo`` so online and
    offline data share a generator. ``eps`` is the probability of a uniform
    random action; vary it per env (below) for behavioral diversity.
    """

    var eps: Float64

    def __init__(out self, eps: Float64 = 0.3):
        self.eps = eps

    def select_action(mut self, env: PongPixelEnv[PONG_DT]) raises -> Int:
        if random_float64() < self.eps:
            return Int(random_float64() * 3.0) % 3
        var ball_y = env.inner.state[S_BALL_Y]
        var pad_y = env.inner.state[S_PADDLE_Y]
        var diff = ball_y - pad_y
        var dead = Scalar[PONG_DT](2.0)
        if diff > dead:
            return 2  # DOWN
        elif diff < -dead:
            return 1  # UP
        return 0  # NOOP


# ============================================================================
# OnlinePongSampler — OfflineBuffer conformer backed by live PongPixelEnv pool
# ============================================================================


struct OnlinePongSampler[
    POLICY: PongActionPolicy,
    B: Int,
    T: Int,
    MAX_RETRIES: Int = 16,
](Movable, OfflineBuffer):
    """Generate ``(B, T)`` Pong pixel windows on demand by stepping a pool of
    ``B`` persistent ``PongPixelEnv`` instances under ``POLICY``.

    Conforms to ``OfflineBuffer`` so it is interchangeable with
    ``PongOfflineBuffer`` everywhere the consumer is written against the trait.
    """

    # Pong pixel frames are channel-major (4, 84, 84) — same as PatchEmbed's
    # Conv2D expects, so the downstream kernel only normalises (no permute).
    comptime INPUT_LAYOUT_HWC: Bool = False

    # PongPixelEnv owns raw frame buffers + a custom __del__, so it is Movable
    # but NOT Copyable — we hold a List of freshly-constructed (moved, not
    # copied) envs rather than InlineArray(fill=...). One shared policy serves
    # the whole pool: select_action takes the env as an arg and reads global
    # RNG, so it carries no per-env state.
    var envs: List[PongPixelEnv[PONG_DT]]
    var policy: Self.POLICY

    def __init__(out self, var policy: Self.POLICY):
        self.envs = List[PongPixelEnv[PONG_DT]](capacity=Self.B)
        for _ in range(Self.B):
            var env = PongPixelEnv[PONG_DT]()
            _ = env.reset()  # independent initial conditions per env
            self.envs.append(env^)
        self.policy = policy^

    def __init__(out self, *, deinit move: Self):
        self.envs = move.envs^
        self.policy = move.policy^

    @staticmethod
    def make(var policy: Self.POLICY) raises -> Self:
        return Self(policy^)

    # ------------------------------------------------------------------
    # Quantize one fp32 obs frame in [0, 1] → uint8 CHW at a dst offset.
    # Same formula as PongOfflineBuffer.add_step_fp32 so quantization is
    # bit-identical to the stored path.
    # ------------------------------------------------------------------
    @always_inline
    def _quantize_into(
        self,
        obs: List[Scalar[PONG_DT]],
        dst: UnsafePointer[Scalar[DType.uint8], MutAnyOrigin],
    ):
        for i in range(PONG_FRAME_BYTES):
            var v = obs[i] * 255.0 + 0.5
            if v < 0.0:
                v = 0.0
            elif v > 255.0:
                v = 255.0
            dst[i] = UInt8(Int(v))

    # ------------------------------------------------------------------
    # OfflineBuffer contract.
    # ------------------------------------------------------------------
    def sample_batch_uint8(
        mut self,
        batch: Int,
        seq: Int,
        pixels_out: UnsafePointer[Scalar[DType.uint8], MutAnyOrigin],
        actions_out: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    ) raises:
        """Roll the env pool forward ``T`` steps, one clean window per env.

        Output layouts (identical to ``PongOfflineBuffer``):
          pixels_out:  (B, T, PONG_FRAME_BYTES) uint8 CHW in [0, 255].
          actions_out: (B, T, PONG_NUM_ACTIONS) fp32 one-hot.

        ``batch`` / ``seq`` are the caller's runtime (B, T); they must match
        the comptime ``Self.B`` / ``Self.T`` that sized the env pool.
        """
        if batch != Self.B or seq != Self.T:
            raise Error(
                "OnlinePongSampler: runtime (B, T) must match comptime params"
            )

        var pix_stride = Self.T * PONG_FRAME_BYTES
        var act_stride = Self.T * PONG_NUM_ACTIONS

        for b in range(Self.B):
            var pix_dst = pixels_out + b * pix_stride
            var act_dst = actions_out + b * act_stride

            # Roll one window for env b, rejecting boundary-bridging windows.
            for _attempt in range(Self.MAX_RETRIES):
                var bridged = False
                # Capture obs_t alongside the action taken from it, then step
                # — matching the (s_t, a_t) → s_{t+1} convention of the
                # collect script.
                for t in range(Self.T):
                    var obs = self.envs[b].get_obs_list()
                    var a = self.policy.select_action(self.envs[b])

                    # Pixels: quantize obs_t into slot t.
                    self._quantize_into(obs, pix_dst + t * PONG_FRAME_BYTES)
                    # Actions: one-hot a_t into slot t.
                    for k in range(PONG_NUM_ACTIONS):
                        act_dst[t * PONG_NUM_ACTIONS + k] = 0.0
                    if a >= 0 and a < PONG_NUM_ACTIONS:
                        act_dst[t * PONG_NUM_ACTIONS + a] = 1.0

                    var result = self.envs[b].step_obs(a)
                    var done = result[2]
                    # A done before the final step would bridge an episode
                    # boundary mid-window (cf. _window_is_valid). Reset and
                    # re-roll. A done on the last step is a valid target.
                    if done and t < Self.T - 1:
                        _ = self.envs[b].reset()
                        bridged = True
                        break
                    if done:
                        _ = self.envs[b].reset()
                if not bridged:
                    break
            # NOTE: after MAX_RETRIES the last (possibly bridged) window is
            # committed rather than failing hard — acceptable for episodes
            # comfortably longer than T. Tighten if T approaches episode length.
