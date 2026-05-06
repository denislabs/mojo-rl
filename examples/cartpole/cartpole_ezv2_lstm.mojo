"""EZ-V2 CartPole demo with the reward-prefix LSTM head wired in.

Same agent + sampling loop as `cartpole_ezv2.mojo`, but with
`USE_REWARD_PREFIX=True` so the dynamics-network's per-step reward CE
is replaced with the EZ-V1 reward-prefix head (paper App. G):

    (h_lstm[k+1], c_lstm[k+1]) = LSTMCell(hidden[k+1], h_lstm[k], c_lstm[k])
    reward_prefix_logits[k]    = MLP_head(h_lstm[k+1])
    target[k]                  = two_hot(scalar_transform(Σ_{j≤k} reward[j]))

Hidden states reset every `lstm_horizon_len` unroll steps; with
K_UNROLL=3 and LSTM_HORIZON_LEN=5 no within-unroll resets fire here.

Wall budget: ~2-4 minutes on Apple Silicon for the default 8000 env steps
+ ~2000 train_steps. The full 50k convergence run (matching
`cartpole_ezv2.mojo`) is left to a follow-up tuning effort — same
chicken-and-egg failure mode as the non-LSTM CartPole run; the LSTM
head changes the reward target shape, not the value/policy bootstrap.

Run:
    pixi run mojo run -I . examples/cartpole/cartpole_ezv2_lstm.mojo
"""

from std.math import abs
from std.random import seed
from std.time import perf_counter_ns
from mojo_rl.deep_agents.efficient_zero_v2 import (
    EZV2DiscreteMLPConfig,
    GenericEfficientZeroV2Agent,
)
from mojo_rl.envs.cartpole import CartPoleEnv
from mojo_rl.nn.constants import dtype


def _is_finite(x: Float64) -> Bool:
    if x != x:
        return False
    if x > 1.0e300 or x < -1.0e300:
        return False
    return True


def _mean(xs: List[Float64]) -> Float64:
    if len(xs) == 0:
        return 0.0
    var s = Float64(0.0)
    for i in range(len(xs)):
        s += xs[i]
    return s / Float64(len(xs))


def main():
    print("=== EZ-V2 CartPole demo — reward-prefix LSTM head ===")

    comptime NUM_ENV_STEPS = 8_000
    comptime TRAIN_INTERVAL = 4
    comptime LOG_EVERY = 500

    comptime Config = EZV2DiscreteMLPConfig[
        OBS=4,
        ACT=2,
        LATENT=64,
        HIDDEN=64,
        PROJ=128,
        PRED_BOTTLENECK=64,
        BINS=21,
        BS=16,
        K_UNROLL=3,
        N_TD=5,
        SIMS=8,
        NODES=32,
        K_GUMBEL=2,
        LR=Float64(5e-4),
        LAMBDA_G=Float64(1.0),
        # Reward-prefix LSTM head — paper App. G defaults.
        USE_REWARD_PREFIX=True,
        LSTM_HIDDEN=64,
        LSTM_HORIZON_LEN=5,
        LSTM_MLP_HIDDEN=64,
    ]

    seed(2026)
    var agent = GenericEfficientZeroV2Agent[Config](
        gamma=0.99,
        v_min=-15.0,
        v_max=15.0,
        temperature=1.0,
        temperature_decay_steps=10_000_000,
    )
    var env = CartPoleEnv[DType.float32]()

    print()
    print("--- Run config ---")
    print("    NUM_ENV_STEPS         =", NUM_ENV_STEPS)
    print("    TRAIN_INTERVAL        =", TRAIN_INTERVAL)
    print("    USE_REWARD_PREFIX     =", Config.use_reward_prefix)
    print("    LSTM_HIDDEN           =", Config.lstm_hidden)
    print("    LSTM_HORIZON_LEN      =", Config.lstm_horizon_len)
    print("    LSTM_MLP_HIDDEN       =", Config.lstm_mlp_hidden)
    print(
        "    Config: LATENT=", Config.latent_dim,
        " PROJ=", Config.proj_dim,
        " BINS=", Config.num_bins,
    )
    print(
        "    BS=", Config.batch_size,
        " K_UNROLL=", Config.unroll_steps,
        " N_TD=", Config.td_steps,
        " SIMS=", Config.num_simulations,
        " K_GUMBEL=", Config.num_root_candidates,
    )
    print()

    var ep_returns = List[Float64]()
    var ep_return = Float64(0.0)
    var obs = env.reset_obs_list()
    var num_train_calls = 0
    var any_nan_loss = False
    var last_L_R = Float64(0.0)
    var last_L_P = Float64(0.0)
    var last_L_V = Float64(0.0)
    var last_L_G = Float64(0.0)
    var best_ep_return = Float64(0.0)

    var t0 = perf_counter_ns()

    for env_step in range(NUM_ENV_STEPS):
        var result = agent.select_action(obs, training=True)
        var action = result[0]
        var policy = result[1]
        var root_value = result[2]
        var step_result = env.step_obs(action)
        var next_obs = step_result[0].copy()
        var reward = Float64(step_result[1])
        var done = step_result[2]
        agent.store_transition(
            obs, action, reward, policy, root_value, done
        )
        ep_return += reward

        if done:
            ep_returns.append(ep_return)
            if ep_return > best_ep_return:
                best_ep_return = ep_return
            ep_return = Float64(0.0)
            obs = env.reset_obs_list()
        else:
            obs = next_obs^

        if (
            agent.state.is_ready()
            and (env_step + 1) % TRAIN_INTERVAL == 0
        ):
            var t = agent.train_step()
            if not _is_finite(t[0]):
                any_nan_loss = True
            last_L_R = t[1]
            last_L_P = t[2]
            last_L_V = t[3]
            last_L_G = t[4]
            num_train_calls += 1

        if (env_step + 1) % LOG_EVERY == 0:
            var recent = List[Float64]()
            var n_recent = 30
            var start = (
                len(ep_returns) - n_recent
                if len(ep_returns) > n_recent
                else 0
            )
            for j in range(start, len(ep_returns)):
                recent.append(ep_returns[j])
            var elapsed_s = Float64(perf_counter_ns() - t0) / 1.0e9
            print(
                "step", env_step + 1,
                ": eps=", len(ep_returns),
                " best=", best_ep_return,
                " recent30=", _mean(recent),
                " | L_R(prefix)=", last_L_R,
                " L_P=", last_L_P,
                " L_V=", last_L_V,
                " L_G=", last_L_G,
                " | train=", num_train_calls,
                " | t=", elapsed_s, "s",
            )

    var elapsed_s = Float64(perf_counter_ns() - t0) / 1.0e9
    print()
    print("--- Final summary ---")
    print("    total env_steps    =", NUM_ENV_STEPS)
    print("    train_step calls   =", num_train_calls)
    print("    episodes           =", len(ep_returns))
    print("    best episode       =", best_ep_return)

    var last_n = 30
    var start_n = (
        len(ep_returns) - last_n
        if len(ep_returns) > last_n
        else 0
    )
    var tail = List[Float64]()
    for j in range(start_n, len(ep_returns)):
        tail.append(ep_returns[j])
    print("    last-", last_n, " mean   =", _mean(tail))
    print("    last loss components:")
    print("        L_R(prefix) =", last_L_R)
    print("        L_P         =", last_L_P)
    print("        L_V         =", last_L_V)
    print("        L_G         =", last_L_G)
    print("    wall time          =", elapsed_s, "s")

    if any_nan_loss:
        print("WARNING: at least one train_step produced a non-finite loss")
