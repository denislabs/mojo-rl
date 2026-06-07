"""Dreamer 4 — collect a REWARD-BEARING Pong pixel buffer.

    pixi run mojo run -I . examples/dreamer4/pong_reward_collect_buffer.mojo

Like `examples/lewm/lewm_pong_collect_buffer.mojo`, but records the per-step
REWARD alongside frames / actions / dones (format LWMR, see
`mojo_rl/deep_agents2/dreamer4/pong_reward_buffer.mojo`). The Dreamer 4 agent's
reward head (eq. 9) + imagination RL (eq. 10/11) need this signal; the LeWM
LWMP buffer has none.

The scripted "follow-the-ball" policy makes the dataset reward-rich: Pong's
dense shaping (HIT_REWARD=0.1 on every ball return) plus ±1 on points means
most episodes carry a clear, learnable reward trace. A modest per-episode
epsilon keeps some exploration without drowning the signal — lower than the
LeWM JEPA buffer's [0, 1] so the reward/behavior is less random.

Output: /tmp/dreamer4_pong_reward_buffer.bin
"""

from std.random import seed, random_float64
from std.time import perf_counter_ns

from mojo_rl.envs.arcade_games.pong import PongPixelEnv
from mojo_rl.envs.arcade_games.pong.pong import S_BALL_Y, S_PADDLE_Y
from mojo_rl.deep_agents2.dreamer4.pong_reward_buffer import (
    Dreamer4PongRewardBuffer,
)
from mojo_rl.envs.arcade_games.pong.offline_buffer import PONG_FRAME_BYTES


comptime dtype = DType.float32
comptime NUM_EPISODES: Int = 96
comptime MAX_STEPS: Int = 256
comptime BUFFER_CAPACITY: Int = NUM_EPISODES * MAX_STEPS
# Lower epsilon than the JEPA buffer → mostly-scripted, reward-rich behavior.
comptime EPS_MIN: Float64 = 0.0
comptime EPS_MAX: Float64 = 0.3
comptime OUTPUT_PATH: String = "/tmp/dreamer4_pong_reward_buffer.bin"
comptime SEED: Int = 0xD4EA


@always_inline
def _follow_ball_action(env: PongPixelEnv[dtype]) -> Int:
    """Move paddle toward the ball's y; NOOP in a small dead-zone."""
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
def _mixed_action(env: PongPixelEnv[dtype], eps: Float64) -> Int:
    if random_float64() < eps:
        return Int(random_float64() * 3.0) % 3
    return _follow_ball_action(env)


def main() raises:
    seed(SEED)
    print("=" * 70)
    print("Dreamer 4 — collect REWARD-BEARING Pong pixel buffer")
    print("=" * 70)
    print("Episodes:     ", NUM_EPISODES)
    print("Max steps/ep: ", MAX_STEPS)
    print("Epsilon-rand: ", EPS_MIN, "to", EPS_MAX, "(per-episode)")
    print("Output:       ", OUTPUT_PATH)
    print()

    var env = PongPixelEnv[dtype]()
    var buf = Dreamer4PongRewardBuffer(capacity=BUFFER_CAPACITY)

    var t0 = perf_counter_ns()
    var total_steps: Int = 0
    var total_reward: Scalar[dtype] = 0.0
    var nonzero_r: Int = 0
    var pos_r: Int = 0
    var neg_r: Int = 0

    for ep in range(NUM_EPISODES):
        _ = env.reset()
        var obs = env.get_obs_list()
        var ep_eps = EPS_MIN + random_float64() * (EPS_MAX - EPS_MIN)
        var ep_reward: Scalar[dtype] = 0.0
        var ep_steps: Int = 0

        for _ in range(MAX_STEPS):
            var a = _mixed_action(env, ep_eps)
            var result = env.step_obs(a)
            var reward = result[1]
            var done = result[2]

            # Store (obs_t, a_t, done_t, r_t): the reward EARNED by taking a_t
            # from obs_t. Aligned with the action so the reward head learns
            # p(r_{t} | h_t) over the same window the policy sees.
            buf.add_step_fp32_list(obs, a, done, reward)
            total_steps += 1
            ep_reward += reward
            if reward != Scalar[dtype](0.0):
                nonzero_r += 1
                if reward > Scalar[dtype](0.0):
                    pos_r += 1
                else:
                    neg_r += 1
            ep_steps += 1
            obs = result[0].copy()
            if done:
                break
            if ep_steps >= MAX_STEPS:
                buf.dones[buf.n_frames - 1] = UInt8(1)
                break

        total_reward += ep_reward
        if (ep + 1) % 16 == 0 or ep == NUM_EPISODES - 1:
            print(
                "  ep", ep + 1, "/", NUM_EPISODES,
                "  eps=", ep_eps, "  steps=", ep_steps,
                "  reward=", Float64(ep_reward), "  n_frames=", buf.n_frames,
            )

    var t1 = perf_counter_ns()
    var elapsed_s = Float64(t1 - t0) / 1e9
    print()
    print("Collected", buf.n_frames, "frames in", elapsed_s, "s")
    print("Throughput:", Float64(total_steps) / elapsed_s, "steps/s")
    print("Avg reward / episode:", Float64(total_reward) / Float64(NUM_EPISODES))
    print(
        "Reward density: ", nonzero_r, "/", total_steps,
        " nonzero  (", pos_r, "pos /", neg_r, "neg )",
    )

    buf.save(OUTPUT_PATH)
    var size_mb = Float64(
        buf.n_frames * (PONG_FRAME_BYTES + 2 + 4) + 64
    ) / (1024.0 * 1024.0)
    print("Wrote", OUTPUT_PATH, "(~", size_mb, "MB)")

    env.close()
