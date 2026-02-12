"""Diagnostic: run HalfCheetah with policy actions, detect instability.

Runs quietly until detecting anomaly (large penetration, velocity explosion,
or flying), then enables verbose for the next few steps.

Run with:
    cd mojo-rl && pixi run mojo run tests/test_physics_instability_diag.mojo 2>&1 | tee /tmp/instability_diag.txt
"""

from random import seed
from math import abs

from deep_agents.ppo import DeepPPOContinuousAgent
from envs.half_cheetah import HalfCheetah, HalfCheetahParams
from envs.half_cheetah.half_cheetah_def import (
    JOINT_ROOTX,
    JOINT_ROOTZ,
    JOINT_ROOTY,
    HalfCheetahJoints,
)


comptime C = HalfCheetahParams[DType.float32]
comptime OBS_DIM = C.OBS_DIM
comptime ACTION_DIM = C.ACTION_DIM
comptime HIDDEN_DIM = 256
comptime ROLLOUT_LEN = 512
comptime N_ENVS = 256
comptime GPU_MINIBATCH_SIZE = 2048
comptime dtype = DType.float32

# Anomaly thresholds
comptime MAX_PEN_THRESH: Float64 = 0.02  # 20mm penetration
comptime MAX_VEL_THRESH: Float64 = 9.0  # near MAX_QVEL clamp
comptime MIN_ROOTZ_THRESH: Float64 = -0.1  # below ground
comptime MAX_ROOTZ_THRESH: Float64 = 3.0  # flying


fn main() raises:
    seed(42)
    print("=" * 70)
    print("Physics Instability Diagnostic — HalfCheetah")
    print("=" * 70)

    # Create agent and load checkpoint
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
        anneal_lr=False,
    )

    var has_checkpoint = True
    try:
        agent.load_checkpoint("ppo_half_cheetah.ckpt")
        print("Checkpoint loaded.")
    except:
        print("WARNING: No checkpoint. Using random policy.")
        has_checkpoint = False

    var env = HalfCheetah()
    _ = env.reset()

    print("dt =", Float64(env.model.timestep))
    print("solref_contact =", env.model.solref_contact[0], env.model.solref_contact[1])
    print("solimp_contact =", env.model.solimp_contact[0], env.model.solimp_contact[1], env.model.solimp_contact[2])
    print()

    # Build initial obs
    var obs = InlineArray[Scalar[dtype], OBS_DIM](uninitialized=True)
    var obs_list = List[Float64](capacity=OBS_DIM)
    HalfCheetahJoints.extract_obs(env.data, obs_list)
    for i in range(OBS_DIM):
        obs[i] = Scalar[dtype](obs_list[i])

    # Run episode, monitoring for anomalies
    comptime MAX_STEPS = 1000
    var anomaly_found = False
    var anomaly_step = -1
    var verbose_remaining = 0

    for step in range(MAX_STEPS):
        # Get action from policy (deterministic)
        var action_result = agent.select_action(obs, training=False)
        var actions = action_result[0].copy()

        # Scale and clip actions (same as evaluate_renderable)
        var action_list = List[Scalar[dtype]]()
        for j in range(ACTION_DIM):
            var action_val = Float64(actions[j])
            action_val = action_val * Float64(agent.action_scale) + Float64(
                agent.action_bias
            )
            if action_val > 1.0:
                action_val = 1.0
            elif action_val < -1.0:
                action_val = -1.0
            action_list.append(Scalar[dtype](action_val))

        # Check if we should print verbose this step
        var do_verbose = verbose_remaining > 0
        if do_verbose:
            verbose_remaining -= 1
            print()
            print("=" * 60)
            print("STEP", step + 1, "(verbose)")
            print("  action:", end="")
            for j in range(ACTION_DIM):
                print(" ", Float64(action_list[j]), end="")
            print("")
            print("=" * 60)

        # Take step
        var result = env.step_continuous_vec[dtype](
            action_list, verbose=do_verbose
        )
        var next_obs_list = result[0].copy()
        var reward = result[1]
        var done = result[2]

        # Update obs
        for i in range(OBS_DIM):
            obs[i] = Scalar[dtype](next_obs_list[i])

        # Extract diagnostic values
        var rootz = Float64(env.data.qpos[JOINT_ROOTZ])
        var vz = Float64(env.data.qvel[JOINT_ROOTZ])
        var pitch = Float64(env.data.qpos[JOINT_ROOTY])
        var nc = Int(env.data.num_contacts)

        # Max penetration
        var max_pen: Float64 = 0
        for c in range(nc):
            var pen = -Float64(env.data.contacts[c].dist)
            if pen > max_pen:
                max_pen = pen

        # Max absolute velocity
        var max_vel: Float64 = 0
        for i in range(C.NV):
            var v = abs(Float64(env.data.qvel[i]))
            if v > max_vel:
                max_vel = v

        # Print summary every 25 steps
        if (step + 1) % 25 == 0:
            print(
                "  step", step + 1,
                ": rootz=", rootz,
                " vz=", vz,
                " pitch=", pitch,
                " contacts=", nc,
                " max_pen=", max_pen,
                " max_vel=", max_vel,
                " reward=", Float64(reward),
            )

        # Detect anomaly
        if not anomaly_found:
            var anomaly_type = String("")
            if max_pen > MAX_PEN_THRESH:
                anomaly_type = "LARGE_PENETRATION=" + String(max_pen)
            elif max_vel > MAX_VEL_THRESH:
                anomaly_type = "VELOCITY_EXPLOSION=" + String(max_vel)
            elif rootz < MIN_ROOTZ_THRESH:
                anomaly_type = "BELOW_GROUND rootz=" + String(rootz)
            elif rootz > MAX_ROOTZ_THRESH:
                anomaly_type = "FLYING rootz=" + String(rootz)

            if anomaly_type != "":
                anomaly_found = True
                anomaly_step = step + 1
                print()
                print("!" * 60)
                print(
                    "ANOMALY DETECTED at step",
                    anomaly_step,
                    ":",
                    anomaly_type,
                )
                print("!" * 60)
                print()
                print("  rootz=", rootz, " vz=", vz, " pitch=", pitch)
                print(
                    "  contacts=", nc, " max_pen=", max_pen, " max_vel=",
                    max_vel,
                )
                print("  qpos:", end="")
                for i in range(C.NQ):
                    print(" ", Float64(env.data.qpos[i]), end="")
                print("")
                print("  qvel:", end="")
                for i in range(C.NV):
                    print(" ", Float64(env.data.qvel[i]), end="")
                print("")
                print("  qfrc:", end="")
                for i in range(C.NV):
                    print(" ", Float64(env.data.qfrc[i]), end="")
                print("")
                print("  action:", end="")
                for j in range(len(action_list)):
                    print(" ", Float64(action_list[j]), end="")
                print("")

                # Enable verbose for the next 3 steps
                verbose_remaining = 3

        if done:
            print("  Episode ended at step", step + 1)
            break

    if not anomaly_found:
        print()
        print("No anomaly detected in", MAX_STEPS, "steps — physics is stable!")
    else:
        print()
        print("Anomaly first detected at step", anomaly_step)

    print()
    print("DIAGNOSTIC COMPLETE")
