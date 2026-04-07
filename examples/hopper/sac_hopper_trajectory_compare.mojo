"""Compare Hopper trajectory: our physics vs MuJoCo step-by-step.

Loads the 1000k checkpoint (peak policy), runs one episode with deterministic
actions, and at each env step compares our qpos/qvel against MuJoCo running
the SAME actions from the SAME initial state.

This reveals exactly WHERE our physics diverges from MuJoCo during the
"exploit" gait that scores 1780 in our engine but only 226 in MuJoCo.

Run with:
    pixi run mojo run -I . examples/hopper/sac_hopper_trajectory_compare.mojo
"""

from std.random import seed
from std.math import abs
from std.python import Python, PythonObject

from mojo_rl.deep_agents.core.agents import DeepSACAgent
from mojo_rl.core.logger import NoOpLogger
from mojo_rl.envs.hopper import Hopper, HopperConfig
from mojo_rl.envs.hopper.hopper_xml import HopperModel
from mojo_rl.physics3d.types import Model, Data
from mojo_rl.physics3d.kinematics import forward_kinematics


comptime OBS_DIM = HopperConfig.OBS_DIM
comptime ACTION_DIM = HopperConfig.ACTION_DIM
comptime HIDDEN_DIM = 256
comptime BUFFER_CAPACITY = 1_000_000
comptime BATCH_SIZE = 256
comptime MAX_N_ENVS = 4
comptime NQ = HopperModel.NQ
comptime NV = HopperModel.NV
comptime FRAME_SKIP = HopperConfig.FRAME_SKIP  # 4

comptime MAX_STEPS = 200  # Compare first 200 env steps

from layout import Layout, LayoutTensor
from std.math import tanh
from mojo_rl.nn.constants import dtype as nn_dtype
from mojo_rl.nn.model.stochastic_actor import get_deterministic_action
from mojo_rl.deep_agents.core.configs.offpolicy_config import SACConfig
from mojo_rl.deep_agents.core.utils import obs_to_inline
from mojo_rl.nn.training import Network

comptime ActorModel = SACConfig[OBS_DIM, ACTION_DIM].ActorModel
comptime ActorOpt = SACConfig[OBS_DIM, ACTION_DIM].ActorOpt


comptime AgentType = DeepSACAgent[
    OBS_DIM, ACTION_DIM, HIDDEN_DIM, BUFFER_CAPACITY, BATCH_SIZE,
    0.0003, 0.0003, 0, NoOpLogger, MAX_N_ENVS,
]


def _get_greedy_action(
    agent: AgentType,
    obs: List[Float64],
) -> List[Float64]:
    """Get deterministic action: tanh(mean) from actor network."""
    var obs_arr = obs_to_inline[OBS_DIM, DType.float64](obs)
    var obs_f32 = InlineArray[Scalar[nn_dtype], OBS_DIM](uninitialized=True)
    for i in range(OBS_DIM):
        obs_f32[i] = Scalar[nn_dtype](obs_arr[i])
    var obs_t = LayoutTensor[
        nn_dtype, Layout.row_major(1, OBS_DIM), MutAnyOrigin
    ](obs_f32.unsafe_ptr())

    comptime ACTOR_OUT = ActorModel.OUT_DIM
    var out_arr = InlineArray[Scalar[nn_dtype], ACTOR_OUT](
        uninitialized=True
    )
    var out_t = LayoutTensor[
        nn_dtype, Layout.row_major(1, ACTOR_OUT), MutAnyOrigin
    ](out_arr.unsafe_ptr())

    # Access raw params pointer to avoid comptime type unification issue
    comptime PS = ActorModel.PARAM_SIZE
    var p = LayoutTensor[nn_dtype, Layout.row_major(PS), MutAnyOrigin](
        agent.state.actor.online.params
    )
    Network[ActorModel, ActorOpt].forward[1](obs_t, out_t, p)

    var result = List[Float64](capacity=ACTION_DIM)
    for i in range(ACTION_DIM):
        var mean = Float64(out_arr[i])
        var a = tanh(mean) * agent.action_scale
        result.append(a)
    return result^


def main() raises:
    seed(42)
    print("=" * 70)
    print("Hopper Trajectory Comparison: Our Physics vs MuJoCo")
    print("=" * 70)

    # Load trained agent (peak checkpoint)
    var agent = AgentType(
        gamma=0.99,
        tau=0.005,
        action_scale=1.0,
        alpha=0.2,
        auto_alpha=False,
        target_entropy=-3.0,
    )
    agent.load_checkpoint("sac_hopper_1000.ckpt")
    print("Loaded checkpoint: sac_hopper_1000.ckpt")

    # Create our environment
    var env = Hopper[DType.float64, TERMINATE_ON_UNHEALTHY=True]()
    _ = env.reset()

    # Create MuJoCo environment
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")
    var xml_path = "./references/Gymnasium-main/gymnasium/envs/mujoco/assets/hopper.xml"
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_model.opt.integrator = 1  # RK4
    mj_model.opt.solver = 2  # Newton
    var mj_data = mujoco.MjData(mj_model)

    # Sync initial state: copy our env's qpos/qvel to MuJoCo
    for i in range(NQ):
        mj_data.qpos[i] = env.get_qpos(i)
    for i in range(NV):
        mj_data.qvel[i] = env.get_qvel(i)
    mujoco.mj_forward(mj_model, mj_data)

    print()
    print(
        "Step | OurReward | MJ_Reward | Qpos_MaxErr | Qvel_MaxErr"
        " | Our_z  | MJ_z   | Our_angle | MJ_angle"
    )
    print("-" * 100)

    var our_total_reward: Float64 = 0.0
    var mj_total_reward: Float64 = 0.0
    var diverged_step = -1

    for step in range(MAX_STEPS):
        # Get observation from our env and select greedy action
        # Use the eval overload on a dummy single-episode eval
        var obs_raw = List[Scalar[DType.float64]](capacity=OBS_DIM)
        for i in range(OBS_DIM):
            obs_raw.append(Scalar[DType.float64](0))
        # Read obs from our env state
        # obs = qpos[1:6] + clip(qvel[0:6], -10, 10)
        for i in range(5):
            obs_raw[i] = Scalar[DType.float64](env.get_qpos(i + 1))
        for i in range(6):
            var v = env.get_qvel(i)
            if Float64(v) > 10.0:
                v = Scalar[DType.float64](10.0)
            elif Float64(v) < -10.0:
                v = Scalar[DType.float64](-10.0)
            obs_raw[5 + i] = v
        var obs_f64 = List[Float64](capacity=OBS_DIM)
        for i in range(OBS_DIM):
            obs_f64.append(Float64(obs_raw[i]))

        # Get deterministic action via evaluate-style forward pass
        var action = _get_greedy_action(agent, obs_f64)


        # === Step our environment ===
        var result = env.step_continuous_vec(action)
        var our_reward = Float64(result[1])
        var our_done = result[2]
        our_total_reward += our_reward

        # === Step MuJoCo with same actions ===
        for i in range(ACTION_DIM):
            mj_data.ctrl[i] = action[i]
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(mj_model, mj_data)

        # Compute MuJoCo reward (same formula)
        var mj_x_vel = Float64(
            py=(mj_data.qpos[0] - Float64(py=mj_data.qpos[0]))
        )
        # Actually we need prev_x. Let's just compute from qvel for rough comparison
        var mj_xvel = Float64(py=mj_data.qvel[0])
        var mj_z = Float64(py=mj_data.qpos[1])
        var mj_angle = Float64(py=mj_data.qpos[2])
        var mj_healthy: Float64 = 1.0
        if mj_z < 0.7 or abs(mj_angle) > 0.2:
            mj_healthy = 0.0
        var mj_ctrl_cost: Float64 = 0.0
        for i in range(ACTION_DIM):
            mj_ctrl_cost += action[i] * action[i]
        mj_ctrl_cost *= 0.001
        var mj_reward = mj_xvel + mj_healthy - mj_ctrl_cost
        mj_total_reward += mj_reward

        # === Compare qpos/qvel ===
        var qpos_max_err: Float64 = 0.0
        var qvel_max_err: Float64 = 0.0

        for i in range(NQ):
            var our_q = Float64(env.get_qpos(i))
            var mj_q = Float64(py=mj_data.qpos[i])
            var err = abs(our_q - mj_q)
            if err > qpos_max_err:
                qpos_max_err = err

        for i in range(NV):
            var our_v = Float64(env.get_qvel(i))
            var mj_v = Float64(py=mj_data.qvel[i])
            var err = abs(our_v - mj_v)
            if err > qvel_max_err:
                qvel_max_err = err

        var our_z = Float64(env.get_qpos(1))
        var our_angle = Float64(env.get_qpos(2))

        # Print every 5 steps or when error is large
        if step % 5 == 0 or qpos_max_err > 0.1 or qvel_max_err > 1.0:
            print(
                String(step)[byte=:4]
                + " | "
                + String(our_reward)[byte=:9]
                + " | "
                + String(mj_reward)[byte=:9]
                + " | "
                + String(qpos_max_err)[byte=:11]
                + " | "
                + String(qvel_max_err)[byte=:11]
                + " | "
                + String(our_z)[byte=:6]
                + " | "
                + String(mj_z)[byte=:6]
                + " | "
                + String(our_angle)[byte=:9]
                + " | "
                + String(mj_angle)[byte=:9]
            )

        if diverged_step < 0 and qpos_max_err > 0.5:
            diverged_step = step
            print("  >>> SIGNIFICANT DIVERGENCE at step " + String(step) + " <<<")

        if our_done:
            print("  >>> Our env terminated at step " + String(step) + " <<<")
            break

        # Check if MuJoCo would have terminated
        if mj_z < 0.7 or abs(mj_angle) > 0.2:
            print(
                "  >>> MuJoCo would terminate at step "
                + String(step)
                + " (z="
                + String(mj_z)[byte=:6]
                + " angle="
                + String(mj_angle)[byte=:6]
                + ") <<<"
            )

    print("-" * 100)
    print("Our total reward:    " + String(our_total_reward)[byte=:10])
    print("MuJoCo total reward: " + String(mj_total_reward)[byte=:10])
    if diverged_step >= 0:
        print(
            "First major divergence at step: " + String(diverged_step)
        )

    env.close()
