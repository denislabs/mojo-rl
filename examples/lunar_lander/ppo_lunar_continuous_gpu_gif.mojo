"""Train PPO Continuous on LunarLander (GPU), then record a GIF.

Trains PPO with 512 parallel environments on GPU, then records
episodes of the trained agent landing to a GIF.

Run with:
    pixi run -e apple mojo run -I . examples/lunar_lander/ppo_lunar_continuous_gpu_gif.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.math import sqrt, log, cos, exp

from layout import LayoutTensor, Layout
from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.core.agents import DeepPPOContinuousAgent
from mojo_rl.envs.lunar_lander import LunarLander, LLConstants


comptime OBS_DIM = LLConstants.OBS_DIM_VAL  # 8
comptime ACTION_DIM = LLConstants.ACTION_DIM_VAL  # 2
comptime HIDDEN_DIM = 256
comptime ROLLOUT_LEN = 128
comptime N_ENVS = 512
comptime GPU_MINIBATCH_SIZE = 512
comptime NUM_UPDATES = 100
comptime dtype = DType.float32


def main() raises:
    seed(42)
    print("=" * 70)
    print("PPO Continuous LunarLander — GPU Train + GIF Export")
    print("=" * 70)
    print()

    # ==========================================================================
    # Phase 1: Train on GPU
    # ==========================================================================
    with DeviceContext() as ctx:
        var agent = DeepPPOContinuousAgent[
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=HIDDEN_DIM,
            rollout_len=ROLLOUT_LEN,
            n_envs=N_ENVS,
            gpu_minibatch_size=GPU_MINIBATCH_SIZE,
            actor_lr=0.0003,
            critic_lr=0.001,
        ](
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            entropy_coef=0.01,
            value_loss_coef=0.5,
            num_epochs=10,
            target_kl=0.1,
            max_grad_norm=0.5,
            clip_value=True,
            norm_adv_per_minibatch=True,
        )

        print("Training PPO on GPU (", N_ENVS, "envs,", NUM_UPDATES, "updates)...")
        print("-" * 70)
        var start_ns = perf_counter_ns()

        var metrics = agent.train_gpu[LunarLander[dtype]](
            ctx,
            num_updates=NUM_UPDATES,
            verbose=True,
            print_every=1,
        )

        var elapsed_s = Float64(perf_counter_ns() - start_ns) / 1e9
        var final_avg = metrics.mean_reward_last_n(100)
        print()
        print("Training done in", String(elapsed_s)[byte=:6], "seconds")
        print("Final avg reward (last 100):", String(final_avg)[byte=:10])
        print()

        # ======================================================================
        # Phase 2: Record GIF of trained agent (CPU eval with rendering)
        # ======================================================================
        print("-" * 70)
        print("Recording trained agent to GIF...")
        print("-" * 70)

        var env = LunarLander[dtype]()
        _ = env.init_renderer()
        env.start_recording("gifs/lunar_lander_ppo_trained.gif", fps=30, skip=3)

        var eval_state = agent.make_cpu_state()
        eval_state.actor.copy_params_from(agent.cpu_state.actor)

        comptime ACTOR_OUT = OBS_DIM  # PPO continuous: ACTION_DIM means + ACTION_DIM log_stds
        # Actually need to check the real ACTOR_OUT size
        comptime REAL_ACTOR_OUT = ACTION_DIM * 2  # mean + log_std

        comptime NUM_EVAL_EPISODES = 3

        for episode in range(NUM_EVAL_EPISODES):
            var obs_raw = env.reset_obs_list()
            var obs_arr = InlineArray[Scalar[dtype], OBS_DIM](
                uninitialized=True
            )
            for i in range(OBS_DIM):
                obs_arr[i] = Scalar[dtype](obs_raw[i])

            var episode_reward: Float64 = 0.0

            for _ in range(1000):
                var obs_t = LayoutTensor[
                    dtype, Layout.row_major(1, OBS_DIM), MutAnyOrigin
                ](obs_arr.unsafe_ptr())

                var actor_out = InlineArray[Scalar[dtype], REAL_ACTOR_OUT](
                    uninitialized=True
                )
                var actor_out_t = LayoutTensor[
                    dtype, Layout.row_major(1, REAL_ACTOR_OUT), MutAnyOrigin
                ](actor_out.unsafe_ptr())
                var p = eval_state.actor.params_view()
                var s = eval_state.actor.model_state_view()
                agent.ActorNet.forward[1](obs_t, actor_out_t, p, s)

                # Deterministic: use mean directly
                var action = List[Float64](capacity=ACTION_DIM)
                for j in range(ACTION_DIM):
                    action.append(Float64(actor_out[j]))

                var result = env.step_continuous_vec(action)
                episode_reward += Float64(result[1])

                env.render_frame()
                env.renderer_delay(16)

                if env.check_renderer_quit():
                    break

                if result[2]:
                    break

                for i in range(OBS_DIM):
                    obs_arr[i] = Scalar[dtype](result[0][i])

            print(
                "  Episode",
                episode + 1,
                "| Reward:",
                String(episode_reward)[byte=:10],
            )

        env.stop_recording()
        env.close_renderer()

        print()
        print("Saved: gifs/lunar_lander_ppo_trained.gif")
        print("Done!")
