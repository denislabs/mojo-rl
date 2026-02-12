"""Diagnostic: Test CPU evaluation pipeline for HalfCheetah.

Tests whether the CPU actor forward pass produces sensible actions
for known observations, and compares with a manual GPU forward pass.

Run with:
    pixi run -e apple mojo run tests/debug_cpu_eval_pipeline.mojo
"""

from random import seed
from collections import InlineArray
from math import tanh

from gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from deep_agents.ppo import DeepPPOContinuousAgent
from envs.half_cheetah import HalfCheetah, HalfCheetahParams
from deep_rl import dtype as gpu_dtype
from deep_rl.constants import dtype, TPB, TILE

comptime C = HalfCheetahParams[DType.float32]
comptime OBS_DIM = C.OBS_DIM  # 17
comptime ACTION_DIM = C.ACTION_DIM  # 6
comptime HIDDEN_DIM = 256
comptime ROLLOUT_LEN = 512
comptime N_ENVS = 256
comptime GPU_MINIBATCH_SIZE = 2048


fn main() raises:
    seed(42)
    print("=" * 70)
    print("DIAGNOSTIC: CPU Evaluation Pipeline Test")
    print("=" * 70)

    with DeviceContext() as ctx:
        var agent = DeepPPOContinuousAgent[
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=HIDDEN_DIM,
            rollout_len=ROLLOUT_LEN,
            n_envs=N_ENVS,
            gpu_minibatch_size=GPU_MINIBATCH_SIZE,
            clip_value=True,
        ](
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            actor_lr=0.0003,
            critic_lr=0.0003,
            entropy_coef=0.0,
            value_loss_coef=0.5,
            num_epochs=10,
            target_kl=0.0,
            max_grad_norm=0.5,
            anneal_lr=True,
            anneal_entropy=False,
            target_total_steps=0,
            norm_adv_per_minibatch=True,
            checkpoint_every=0,
            checkpoint_path="",
            normalize_rewards=True,
            obs_noise_std=0.0,
        )

        print()
        print("Agent created. Actor param count:", len(agent.actor.params))
        print(
            "Action scale:",
            agent.action_scale,
            "Action bias:",
            agent.action_bias,
        )
        print()

        # ================================================================
        # Test 1: Forward pass with zero observation (untrained agent)
        # ================================================================
        print("=" * 70)
        print("TEST 1: CPU forward pass with zero observation (untrained)")
        print("=" * 70)

        var zero_obs = InlineArray[Scalar[dtype], OBS_DIM](
            fill=Scalar[dtype](0.0)
        )

        var action_result = agent.select_action(zero_obs, training=False)
        var actions_det = action_result[0].copy()

        print("Deterministic actions from zero obs:")
        for j in range(ACTION_DIM):
            print(
                "  action["
                + String(j)
                + "] = "
                + String(Float64(actions_det[j]))
            )

        var action_result_stoch = agent.select_action(zero_obs, training=True)
        var actions_stoch = action_result_stoch[0].copy()

        print("Stochastic actions from zero obs:")
        for j in range(ACTION_DIM):
            print(
                "  action["
                + String(j)
                + "] = "
                + String(Float64(actions_stoch[j]))
            )

        # ================================================================
        # Test 2: Forward pass with realistic observation (after reset)
        # ================================================================
        print()
        print("=" * 70)
        print("TEST 2: CPU forward pass with realistic observation")
        print("=" * 70)

        # Create a realistic observation: rootz=0.7, rest=0, qvel=0
        var real_obs = InlineArray[Scalar[dtype], OBS_DIM](
            fill=Scalar[dtype](0.0)
        )
        real_obs[0] = Scalar[dtype](0.7)  # z_position = rootz

        var action_result2 = agent.select_action(real_obs, training=False)
        var actions_det2 = action_result2[0].copy()

        print("Deterministic actions from realistic obs (z=0.7):")
        for j in range(ACTION_DIM):
            print(
                "  action["
                + String(j)
                + "] = "
                + String(Float64(actions_det2[j]))
            )

        # ================================================================
        # Test 3: Manual CPU forward pass (layer by layer)
        # ================================================================
        print()
        print("=" * 70)
        print("TEST 3: Manual layer-by-layer forward pass verification")
        print("=" * 70)

        # Print first few weights of linear layer 1
        print("First 5 weights of actor (linear layer 1):")
        for i in range(5):
            print(
                "  params["
                + String(i)
                + "] = "
                + String(Float64(agent.actor.params[i]))
            )

        # Compute linear layer 1 manually: output = tanh(obs @ W1 + b1)
        # W1 is at offset 0 (17 * 256 = 4352 elements)
        # b1 is at offset 4352 (256 elements)
        comptime W1_SIZE = OBS_DIM * HIDDEN_DIM  # 17 * 256 = 4352
        comptime B1_OFFSET = W1_SIZE  # 4352
        comptime L1_PARAM_SIZE = W1_SIZE + HIDDEN_DIM  # 4608

        print()
        print("Layer 1 (LinearTanh[17, 256]):")
        print(
            "  W1 size:",
            W1_SIZE,
            "B1 offset:",
            B1_OFFSET,
            "Total L1 params:",
            L1_PARAM_SIZE,
        )

        # Compute first hidden output manually
        var hidden1 = InlineArray[Scalar[dtype], HIDDEN_DIM](
            fill=Scalar[dtype](0.0)
        )
        for j in range(HIDDEN_DIM):
            var acc: Float64 = Float64(
                agent.actor.params[B1_OFFSET + j]
            )  # bias
            for i in range(OBS_DIM):
                acc += Float64(real_obs[i]) * Float64(
                    agent.actor.params[i * HIDDEN_DIM + j]
                )
            hidden1[j] = Scalar[dtype](tanh(acc))

        print("  First 5 hidden1 values (manual):")
        for j in range(5):
            print("    h1[" + String(j) + "] = " + String(Float64(hidden1[j])))

        # Compare with actor forward pass output
        # The actor output is [mean(6), log_std(6)] = 12 values
        var actor_output = InlineArray[Scalar[dtype], ACTION_DIM * 2](
            uninitialized=True
        )
        for i in range(ACTION_DIM * 2):
            actor_output[i] = Scalar[dtype](0.0)
        agent.actor.forward[1](real_obs, actor_output)

        print()
        print("Actor forward pass output (12 values):")
        for i in range(ACTION_DIM * 2):
            var label = (
                "mean[" + String(i) + "]" if i
                < ACTION_DIM else "log_std[" + String(i - ACTION_DIM) + "]"
            )
            print("  " + label + " = " + String(Float64(actor_output[i])))

        # ================================================================
        # Test 4: GPU forward pass with same observation
        # ================================================================
        print()
        print("=" * 70)
        print("TEST 4: GPU forward pass with same observation")
        print("=" * 70)

        comptime ACTOR_PARAM_SIZE = agent.ACTOR_PARAM_SIZE
        comptime ACTOR_OUT = agent.ACTOR_OUT
        comptime WORKSPACE_PER_SAMPLE = 4 * HIDDEN_DIM

        var actor_params_buf = ctx.enqueue_create_buffer[dtype](
            ACTOR_PARAM_SIZE
        )
        ctx.enqueue_copy(actor_params_buf, agent.actor.params.unsafe_ptr())

        var obs_buf = ctx.enqueue_create_buffer[dtype](OBS_DIM)
        # Copy the same realistic observation to GPU
        var obs_host = List[Scalar[dtype]](capacity=OBS_DIM)
        for i in range(OBS_DIM):
            obs_host.append(real_obs[i])
        ctx.enqueue_copy(obs_buf, obs_host.unsafe_ptr())

        var actor_out_buf = ctx.enqueue_create_buffer[dtype](ACTOR_OUT)
        var workspace_buf = ctx.enqueue_create_buffer[dtype](
            WORKSPACE_PER_SAMPLE
        )
        ctx.synchronize()

        # Run GPU forward pass
        agent.ActorNetwork.MODEL.forward_gpu_no_cache[1](
            ctx,
            actor_out_buf,
            obs_buf,
            actor_params_buf,
            workspace_buf,
        )
        ctx.synchronize()

        # Read GPU output
        var gpu_actor_out = List[Scalar[dtype]](capacity=ACTOR_OUT)
        for _ in range(ACTOR_OUT):
            gpu_actor_out.append(Scalar[dtype](0.0))
        ctx.enqueue_copy(gpu_actor_out.unsafe_ptr(), actor_out_buf)
        ctx.synchronize()

        print("GPU actor forward pass output (12 values):")
        for i in range(ACTOR_OUT):
            var label = (
                "mean[" + String(i) + "]" if i
                < ACTION_DIM else "log_std[" + String(i - ACTION_DIM) + "]"
            )
            print("  " + label + " = " + String(Float64(gpu_actor_out[i])))

        # Compare CPU vs GPU
        print()
        print("CPU vs GPU actor output comparison:")
        var max_diff: Float64 = 0.0
        for i in range(ACTOR_OUT):
            var diff = abs(Float64(actor_output[i]) - Float64(gpu_actor_out[i]))
            if diff > max_diff:
                max_diff = diff
            var label = (
                "mean[" + String(i) + "]" if i
                < ACTION_DIM else "log_std[" + String(i - ACTION_DIM) + "]"
            )
            print(
                "  " + label,
                "CPU:" + String(Float64(actor_output[i]))[:12].ljust(12),
                "GPU:" + String(Float64(gpu_actor_out[i]))[:12].ljust(12),
                "diff:" + String(diff)[:12],
            )
        print("Max CPU/GPU difference:", max_diff)

        if max_diff > 0.01:
            print(
                "WARNING: Significant CPU/GPU forward pass difference detected!"
            )
            print("This is likely the root cause of the eval gap.")
        else:
            print("CPU and GPU forward passes match well.")

        # ================================================================
        # Test 5: Train briefly, then check if CPU forward pass changes
        # ================================================================
        print()
        print("=" * 70)
        print("TEST 5: Train 2000 episodes, then compare CPU/GPU forward")
        print("=" * 70)

        var metrics = agent.train_gpu[HalfCheetah[gpu_dtype]](
            ctx,
            num_episodes=2000,
            verbose=False,
            print_every=0,
        )
        print(
            "Training done. Mean reward (last 100):",
            metrics.mean_reward_last_n(100),
        )

        # Check CPU forward pass with same obs
        var actor_output_trained = InlineArray[Scalar[dtype], ACTION_DIM * 2](
            uninitialized=True
        )
        for i in range(ACTION_DIM * 2):
            actor_output_trained[i] = Scalar[dtype](0.0)
        agent.actor.forward[1](real_obs, actor_output_trained)

        # Check GPU forward pass
        ctx.enqueue_copy(actor_params_buf, agent.actor.params.unsafe_ptr())
        ctx.enqueue_copy(obs_buf, obs_host.unsafe_ptr())
        ctx.synchronize()

        agent.ActorNetwork.MODEL.forward_gpu_no_cache[1](
            ctx,
            actor_out_buf,
            obs_buf,
            actor_params_buf,
            workspace_buf,
        )
        ctx.synchronize()

        ctx.enqueue_copy(gpu_actor_out.unsafe_ptr(), actor_out_buf)
        ctx.synchronize()

        print()
        print("After training - CPU vs GPU actor output:")
        var max_diff_trained: Float64 = 0.0
        for i in range(ACTOR_OUT):
            var diff = abs(
                Float64(actor_output_trained[i]) - Float64(gpu_actor_out[i])
            )
            if diff > max_diff_trained:
                max_diff_trained = diff
            var label = (
                "mean[" + String(i) + "]" if i
                < ACTION_DIM else "log_std[" + String(i - ACTION_DIM) + "]"
            )
            print(
                "  " + label,
                "CPU:"
                + String(Float64(actor_output_trained[i]))[:12].ljust(12),
                "GPU:" + String(Float64(gpu_actor_out[i]))[:12].ljust(12),
                "diff:" + String(diff)[:12],
            )
        print("Max CPU/GPU difference after training:", max_diff_trained)

        if max_diff_trained > 0.01:
            print("WARNING: CPU/GPU forward pass diverged after training!")
            print(
                "The CPU evaluation uses different actions than GPU -> root"
                " cause found!"
            )
        else:
            print("CPU and GPU forward passes still match.")
            print(
                "The issue may be elsewhere (e.g., stochastic sampling or env"
                " interface)."
            )

        # ================================================================
        # Test 6: Run CPU eval with verbose first episode
        # ================================================================
        print()
        print("=" * 70)
        print("TEST 6: CPU evaluation - first 5 steps of first episode")
        print("=" * 70)

        var env = HalfCheetah[dtype]()
        var obs_list = env.reset_obs_list()
        var obs = InlineArray[Scalar[dtype], OBS_DIM](uninitialized=True)
        for i in range(OBS_DIM):
            obs[i] = Scalar[dtype](obs_list[i])

        print("Initial obs:")
        for i in range(OBS_DIM):
            print("  obs[" + String(i) + "] = " + String(Float64(obs[i])))

        for step in range(5):
            var act_res = agent.select_action(obs, training=True)
            var acts = act_res[0].copy()

            print()
            print("Step", step, ":")
            print("  Actions (from policy):")
            for j in range(ACTION_DIM):
                print("    a[" + String(j) + "] = " + String(Float64(acts[j])))

            # Apply action scaling and clipping (like evaluate does)
            var action_list = List[Scalar[dtype]]()
            for j in range(ACTION_DIM):
                var action_val = Float64(acts[j])
                action_val = action_val * agent.action_scale + agent.action_bias
                if action_val > 1.0:
                    action_val = 1.0
                elif action_val < -1.0:
                    action_val = -1.0
                action_list.append(Scalar[dtype](action_val))

            print("  Clipped actions (sent to env):")
            for j in range(ACTION_DIM):
                print(
                    "    a_clip["
                    + String(j)
                    + "] = "
                    + String(Float64(action_list[j]))
                )

            var result = env.step_continuous_vec(action_list)
            var next_obs = result[0].copy()
            var reward = result[1]

            print("  Reward:", Float64(reward))
            print(
                "  Next obs[0:3]:",
                Float64(next_obs[0]),
                Float64(next_obs[1]),
                Float64(next_obs[2]),
            )
            print(
                "  rootx:",
                Float64(env.data.qpos[0]),
                "x_vel:",
                Float64(env.data.qvel[0]),
            )

            for i in range(OBS_DIM):
                obs[i] = next_obs[i]

    print()
    print(">>> Diagnostic complete <<<")
