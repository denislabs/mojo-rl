"""LeWM Phase 3 — Collect Pong pixel trajectories.

Runs PongPixelEnv on CPU, alternating a "follow-the-ball" scripted policy
with random actions, and dumps frames + actions + dones to a flat binary
buffer (LWMP v1, see `mojo_rl/experimental/lewm/pong_buffer.mojo`).

Args (edit comptime constants below):
    NUM_EPISODES   — how many episodes to collect.
    MAX_STEPS      — per-episode step cap.
    EPS_RANDOM     — probability of taking a random action (vs scripted).

Run:
    pixi run mojo run -I . examples/lewm/lewm_pong_collect_buffer.mojo

Output:
    /tmp/lewm_pong_buffer.bin  (default, ~28KB × n_frames)
"""

from std.random import seed, random_float64
from std.time import perf_counter_ns

from mojo_rl.envs.arcade_games.pong import PongPixelEnv
from mojo_rl.envs.arcade_games.pong.pong import S_BALL_Y, S_PADDLE_Y
from mojo_rl.envs.arcade_games.core.gpu_env import ArcadeGameAction
from mojo_rl.experimental.lewm.pong_buffer import (
    PongBuffer,
    PONG_FRAME_BYTES,
)


# ============================================================================
# Config — tweak these
# ============================================================================

comptime dtype = DType.float32
comptime NUM_EPISODES: Int = 16
comptime MAX_STEPS: Int = 256
comptime BUFFER_CAPACITY: Int = NUM_EPISODES * MAX_STEPS
comptime EPS_RANDOM: Float64 = 0.3
comptime OUTPUT_PATH: String = "/tmp/lewm_pong_buffer.bin"
comptime SEED: Int = 0xC0DE


# ============================================================================
# Scripted policy: follow the ball
# ============================================================================


@always_inline
def _follow_ball_action(env: PongPixelEnv[dtype]) -> Int:
    """Move paddle toward ball's y. NOOP in a small dead-zone."""
    var ball_y = env.inner.state[S_BALL_Y]
    var pad_y = env.inner.state[S_PADDLE_Y]
    var diff = ball_y - pad_y
    var dead = Scalar[dtype](2.0)
    if diff > dead:
        return 2  # DOWN
    elif diff < -dead:
        return 1  # UP
    return 0  # NOOP


@always_inline
def _mixed_action(env: PongPixelEnv[dtype]) -> Int:
    if random_float64() < EPS_RANDOM:
        return Int(random_float64() * 3.0) % 3
    return _follow_ball_action(env)


# ============================================================================
# Collection loop
# ============================================================================


def main() raises:
    seed(SEED)
    print("=" * 70)
    print("LeWM — collect Pong pixel buffer")
    print("=" * 70)
    print("Episodes:     ", NUM_EPISODES)
    print("Max steps/ep: ", MAX_STEPS)
    print("Epsilon-rand: ", EPS_RANDOM)
    print("Output:       ", OUTPUT_PATH)
    print()

    var env = PongPixelEnv[dtype]()
    var buf = PongBuffer(capacity=BUFFER_CAPACITY)

    var t0 = perf_counter_ns()
    var total_steps: Int = 0
    var total_reward: Scalar[dtype] = 0.0

    for ep in range(NUM_EPISODES):
        _ = env.reset()
        var obs = env.get_obs_list()
        # Record the initial frame so that (s_t, a_t) → s_{t+1} pairs are
        # well-defined for predictor training. We record obs_t alongside
        # the action taken from that observation, then step the env.
        var ep_reward: Scalar[dtype] = 0.0
        var ep_steps: Int = 0

        for _ in range(MAX_STEPS):
            var a = _mixed_action(env)
            var result = env.step_obs(a)
            var reward = result[1]
            var done = result[2]

            # Store (obs_t, a_t, done_t). For JEPA we just need consecutive
            # frames; episode boundary marks where the latent dynamics
            # reset.
            buf.add_step_fp32_list(obs, a, done)
            total_steps += 1
            ep_reward += reward
            ep_steps += 1
            obs = result[0].copy()
            if done:
                break
            if ep_steps >= MAX_STEPS:
                # Mark a synthetic episode boundary at the truncated step
                # so the sampler doesn't bridge across resets.
                buf.dones[buf.n_frames - 1] = UInt8(1)
                break

        total_reward += ep_reward
        if (ep + 1) % 4 == 0 or ep == NUM_EPISODES - 1:
            print(
                "  ep",
                ep + 1,
                "/",
                NUM_EPISODES,
                "steps=",
                ep_steps,
                "reward=",
                Float64(ep_reward),
                "n_frames=",
                buf.n_frames,
            )

    var t1 = perf_counter_ns()
    var elapsed_s = Float64(t1 - t0) / 1e9
    print()
    print("Collected", buf.n_frames, "frames in", elapsed_s, "s")
    print(
        "Throughput:",
        Float64(total_steps) / elapsed_s,
        "steps/s",
    )
    print("Avg reward / episode:", Float64(total_reward) / Float64(NUM_EPISODES))

    # ------------------------------------------------------------------
    # Save buffer
    # ------------------------------------------------------------------
    buf.save(OUTPUT_PATH)
    var size_mb = Float64(
        buf.n_frames * (PONG_FRAME_BYTES + 2) + 64
    ) / (1024.0 * 1024.0)
    print("Wrote", OUTPUT_PATH, "(~", size_mb, "MB)")

    env.close()
