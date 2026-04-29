"""Dump per-step Hopper state during deterministic SAC rollout.

Loads sac_hopper.ckpt, runs N episodes with the deterministic mean policy,
and writes one CSV row per env step capturing torso z, torso pitch, x velocity,
reward, and the 3 actions. Also prints per-episode early/mid/late thirds so the
"first hop tilts torso forward, drift compounds, fall mid-episode" hypothesis
can be verified without plotting.

Run with:
    pixi run mojo run -I . examples/hopper/sac_hopper_trajectory_dump.mojo
Output: hopper_trajectory.csv  (open in any spreadsheet / pandas)
"""

from std.random import seed
from std.math import tanh
from std.python import Python, PythonObject

from layout import Layout, LayoutTensor

from mojo_rl.core.logger import NoOpLogger
from mojo_rl.deep_agents.core.agents import DeepSACAgent
from mojo_rl.deep_agents.core.configs.offpolicy_config import SACConfig
from mojo_rl.deep_agents.core.utils import obs_to_inline
from mojo_rl.envs.hopper import Hopper, HopperConfig
from mojo_rl.nn.constants import dtype as nn_dtype
from mojo_rl.nn.training import Network


comptime OBS_DIM = HopperConfig.OBS_DIM  # 11
comptime ACTION_DIM = HopperConfig.ACTION_DIM  # 3
comptime HIDDEN_DIM = 256
comptime BUFFER_CAPACITY = 1_000_000
comptime BATCH_SIZE = 256
comptime MAX_N_ENVS = 4

comptime NUM_EPISODES = 5
comptime MAX_STEPS = 1000
comptime CKPT_PATH = "sac_hopper.ckpt"
comptime CSV_PATH = "hopper_trajectory.csv"

comptime ActorModel = SACConfig[OBS_DIM, ACTION_DIM].ActorModel
comptime ActorOpt = SACConfig[OBS_DIM, ACTION_DIM].ActorOpt

comptime AgentType = DeepSACAgent[
    OBS_DIM,
    ACTION_DIM,
    HIDDEN_DIM,
    BUFFER_CAPACITY,
    BATCH_SIZE,
    0.0003,
    0.001,
    0,
    NoOpLogger,
    MAX_N_ENVS,
]


def _get_greedy_action(
    agent: AgentType,
    obs: List[Float64],
) -> List[Float64]:
    """Deterministic SAC action: tanh(actor_mean) * action_scale.

    Mirrors the working pattern in sac_hopper_trajectory_compare.mojo to
    sidestep trait-method type unification with `agent.select_greedy_action`.
    """
    var obs_arr = obs_to_inline[OBS_DIM, DType.float64](obs)
    var obs_f32 = InlineArray[Scalar[nn_dtype], OBS_DIM](uninitialized=True)
    for i in range(OBS_DIM):
        obs_f32[i] = Scalar[nn_dtype](obs_arr[i])
    var obs_t = LayoutTensor[
        nn_dtype, Layout.row_major(1, OBS_DIM), MutAnyOrigin
    ](obs_f32.unsafe_ptr())

    comptime ACTOR_OUT = ActorModel.OUT_DIM
    var out_arr = InlineArray[Scalar[nn_dtype], ACTOR_OUT](uninitialized=True)
    var out_t = LayoutTensor[
        nn_dtype, Layout.row_major(1, ACTOR_OUT), MutAnyOrigin
    ](out_arr.unsafe_ptr())

    comptime PS = ActorModel.PARAM_SIZE
    comptime SS = ActorModel.STATE_SIZE
    var p = LayoutTensor[nn_dtype, Layout.row_major(PS), MutAnyOrigin](
        agent.state.actor.online.params
    )
    var s = LayoutTensor[nn_dtype, Layout.row_major(SS), MutAnyOrigin](
        agent.state.actor.online.model_state
    )
    Network[ActorModel, ActorOpt].forward[1](obs_t, out_t, p, s)

    var result = List[Float64](capacity=ACTION_DIM)
    for i in range(ACTION_DIM):
        var mean = Float64(out_arr[i])
        var a = tanh(mean) * agent.action_scale
        result.append(a)
    return result^


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC Hopper trajectory dump (deterministic rollout)")
    print("=" * 70)

    var agent = AgentType(
        gamma=0.99,
        tau=0.005,
        action_scale=1.0,
        alpha=0.2,
        auto_alpha=True,
        alpha_lr=0.0003,
        target_entropy=-3.0,
    )

    print("Loading checkpoint:", CKPT_PATH)
    try:
        agent.load_checkpoint(CKPT_PATH)
    except:
        print("Could not load", CKPT_PATH)
        print("Train first via examples/hopper/sac_hopper_training_gpu.mojo")
        return

    var env = Hopper[DType.float64, TERMINATE_ON_UNHEALTHY=True]()

    var builtins = Python.import_module("builtins")
    var csv = builtins.open(PythonObject(CSV_PATH), PythonObject("w"))
    csv.write(
        PythonObject("episode,step,z,angle,x_vel,reward,a0,a1,a2,done\n")
    )

    print()
    print(
        "Episode | Steps |  Reward |  early_z early_ang |  mid_z   mid_ang |"
        "  late_z  late_ang"
    )
    print("-" * 92)

    for ep in range(NUM_EPISODES):
        var obs_raw = env.reset_obs_list()
        var obs = List[Float64]()
        for i in range(len(obs_raw)):
            obs.append(Float64(obs_raw[i]))

        var zs = List[Float64]()
        var angs = List[Float64]()
        var ep_reward: Float64 = 0.0
        var ep_steps = 0

        for step in range(MAX_STEPS):
            # obs layout (custom_extract_obs_cpu): qpos[1:6] then qvel[0:6]
            # so obs[0]=z, obs[1]=torso_angle, obs[5]=x_vel (clipped to ±10).
            var z = Float64(obs[0])
            var angle = Float64(obs[1])
            var x_vel = Float64(obs[5])

            var action = _get_greedy_action(agent, obs)
            var result = env.step_continuous_vec(action)
            var reward = Float64(result[1])
            var done = result[2]

            ep_reward += reward
            ep_steps += 1
            zs.append(z)
            angs.append(angle)

            var row = (
                String(ep)
                + ","
                + String(step)
                + ","
                + String(z)[byte=:8]
                + ","
                + String(angle)[byte=:8]
                + ","
                + String(x_vel)[byte=:8]
                + ","
                + String(reward)[byte=:8]
                + ","
                + String(Float64(action[0]))[byte=:8]
                + ","
                + String(Float64(action[1]))[byte=:8]
                + ","
                + String(Float64(action[2]))[byte=:8]
                + ","
                + (String("1") if done else String("0"))
                + "\n"
            )
            csv.write(PythonObject(row))

            var next_obs = List[Float64]()
            for i in range(len(result[0])):
                next_obs.append(Float64(result[0][i]))
            obs = next_obs^

            if done:
                break

        var n = len(zs)
        var third = n // 3
        if third < 1:
            third = 1

        var ez: Float64 = 0.0
        var ea: Float64 = 0.0
        for i in range(third):
            ez += zs[i]
            ea += angs[i]
        ez /= Float64(third)
        ea /= Float64(third)

        var mid_lo = third
        var mid_hi = 2 * third
        if mid_hi > n:
            mid_hi = n
        var mid_n = mid_hi - mid_lo
        var mz: Float64 = 0.0
        var ma: Float64 = 0.0
        if mid_n > 0:
            for i in range(mid_lo, mid_hi):
                mz += zs[i]
                ma += angs[i]
            mz /= Float64(mid_n)
            ma /= Float64(mid_n)

        var late_lo = mid_hi
        var late_n = n - late_lo
        var lz: Float64 = 0.0
        var la: Float64 = 0.0
        if late_n > 0:
            for i in range(late_lo, n):
                lz += zs[i]
                la += angs[i]
            lz /= Float64(late_n)
            la /= Float64(late_n)

        print(
            "  " + String(ep)
            + "    | " + String(ep_steps)[byte=:5]
            + " | " + String(ep_reward)[byte=:7]
            + " |  " + String(ez)[byte=:6] + "  " + String(ea)[byte=:7]
            + "  |  " + String(mz)[byte=:6] + "  " + String(ma)[byte=:7]
            + "  |  " + String(lz)[byte=:6] + "  " + String(la)[byte=:7]
        )

    csv.close()
    print()
    print("CSV written to:", CSV_PATH)
    print(
        "Termination: z must stay > 0.7 and |angle| < 0.2 (see HopperConfig)."
    )
    print("If late_angle drifts toward +/- 0.2 across episodes, the lean theory holds.")
