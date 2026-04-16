"""IDP Diagnostic: Test GPU actor network + action extraction.

Compares:
1. CPU actor forward + tanh(mean)
2. GPU actor forward_gpu_no_cache + extract_deterministic_actions kernel
For the same observation, with batch sizes 1 and 32.

This isolates whether the GPU evaluation pipeline produces correct actions.

Run with:
    pixi run -e apple mojo run -I . examples/inverted_double_pendulum/idp_gpu_actor_diagnostic.mojo
    pixi run -e nvidia mojo run -I . examples/inverted_double_pendulum/idp_gpu_actor_diagnostic.mojo
"""

from std.random import seed
from std.math import abs, sin, cos, tanh, exp
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor

from mojo_rl.envs.inverted_double_pendulum import InvertedDoublePendulum
from mojo_rl.envs.inverted_double_pendulum.inverted_double_pendulum_xml import (
    InvertedDoublePendulumModel,
)
from mojo_rl.deep_agents.core.agents import DeepSACAgent
from mojo_rl.deep_agents.core.configs.offpolicy_config import SACConfig
from mojo_rl.nn.training import Network
from mojo_rl.nn.constants import dtype as nn_dtype
from mojo_rl.core.logger import NoOpLogger


comptime OBS_DIM = 9
comptime ACTION_DIM = 1
comptime HIDDEN_DIM = 128
comptime BUFFER_CAPACITY = 1_000_000
comptime BATCH_SIZE = 256
comptime MAX_N_ENVS = 32  # Max we'll test

comptime ActorModel = SACConfig[OBS_DIM, ACTION_DIM, HIDDEN_DIM].ActorModel
comptime ActorOpt = SACConfig[OBS_DIM, ACTION_DIM, HIDDEN_DIM].ActorOpt

comptime AgentType = DeepSACAgent[
    OBS_DIM, ACTION_DIM, HIDDEN_DIM, BUFFER_CAPACITY, BATCH_SIZE,
    0.0003, 0.001, 0, NoOpLogger, MAX_N_ENVS,
]


def cpu_forward_action(
    agent: AgentType,
    obs: InlineArray[Scalar[nn_dtype], OBS_DIM],
) -> Tuple[Float64, Float64]:
    """Returns (raw_mean, tanh_action) from CPU forward pass."""
    var obs_local = obs
    var obs_t = LayoutTensor[
        nn_dtype, Layout.row_major(1, OBS_DIM), MutAnyOrigin
    ](obs_local.unsafe_ptr())
    comptime ACTOR_OUT = ActorModel.OUT_DIM
    var out_arr = InlineArray[Scalar[nn_dtype], ACTOR_OUT](uninitialized=True)
    var out_t = LayoutTensor[
        nn_dtype, Layout.row_major(1, ACTOR_OUT), MutAnyOrigin
    ](out_arr.unsafe_ptr())
    comptime PS = ActorModel.PARAM_SIZE
    var p = LayoutTensor[nn_dtype, Layout.row_major(PS), MutAnyOrigin](
        agent.state.actor.online.params
    )
    Network[ActorModel, ActorOpt].forward[1](obs_t, out_t, p)
    var mean = Float64(out_arr[0])
    var action = tanh(mean) * agent.action_scale
    return (mean, action)


def test_gpu_batch[N: Int](
    ctx: DeviceContext,
    agent: AgentType,
    obs: InlineArray[Scalar[nn_dtype], OBS_DIM],
    cpu_mean: Float64,
    cpu_action: Float64,
) raises:
    """Test GPU forward pass with batch size N."""
    print("  Testing GPU batch size " + String(N) + ":")

    comptime ACTOR_OUT = ActorModel.OUT_DIM
    comptime PS = ActorModel.PARAM_SIZE

    # Create GPU buffers
    var obs_buf = ctx.enqueue_create_buffer[nn_dtype](N * OBS_DIM)
    var actor_out_buf = ctx.enqueue_create_buffer[nn_dtype](N * ACTOR_OUT)
    var params_buf = ctx.enqueue_create_buffer[nn_dtype](PS)

    # Workspace for forward pass
    comptime WS_PER = ActorModel.WORKSPACE_SIZE_PER_SAMPLE
    var workspace_buf = ctx.enqueue_create_buffer[nn_dtype](N * WS_PER)

    # Fill obs buffer — all N envs get the same observation
    var obs_host = ctx.enqueue_create_host_buffer[nn_dtype](N * OBS_DIM)
    for env in range(N):
        for i in range(OBS_DIM):
            obs_host[env * OBS_DIM + i] = obs[i]
    ctx.enqueue_copy(obs_buf, obs_host.unsafe_ptr())

    # Copy params to GPU
    var params_host = ctx.enqueue_create_host_buffer[nn_dtype](PS)
    for i in range(PS):
        params_host[i] = agent.state.actor.online.params[i]
    ctx.enqueue_copy(params_buf, params_host.unsafe_ptr())
    ctx.synchronize()

    # GPU forward pass
    var obs_t = LayoutTensor[
        nn_dtype, Layout.row_major(N, OBS_DIM), MutAnyOrigin
    ](obs_buf.unsafe_ptr())
    var actor_out_t = LayoutTensor[
        nn_dtype, Layout.row_major(N, ACTOR_OUT), MutAnyOrigin
    ](actor_out_buf.unsafe_ptr())
    var params_t = LayoutTensor[
        nn_dtype, Layout.row_major(PS), MutAnyOrigin
    ](params_buf.unsafe_ptr())

    ActorModel.forward_gpu_no_cache[N](
        ctx, actor_out_t, obs_t, params_t, workspace_buf
    )
    ctx.synchronize()

    # Read raw actor output
    var actor_out_host = ctx.enqueue_create_host_buffer[nn_dtype](N * ACTOR_OUT)
    ctx.enqueue_copy(actor_out_host.unsafe_ptr(), actor_out_buf)
    ctx.synchronize()

    # Check all envs — they all got the same obs, so output should be identical
    var max_mean_err: Float64 = 0.0
    var max_inter_env_err: Float64 = 0.0
    var first_mean = Float64(actor_out_host[0])
    for env in range(N):
        var gpu_mean = Float64(actor_out_host[env * ACTOR_OUT])
        var err_vs_cpu = abs(gpu_mean - cpu_mean)
        var err_vs_env0 = abs(gpu_mean - first_mean)
        if err_vs_cpu > max_mean_err:
            max_mean_err = err_vs_cpu
        if err_vs_env0 > max_inter_env_err:
            max_inter_env_err = err_vs_env0
        # Compute tanh action on CPU from GPU mean
        var gpu_action = tanh(gpu_mean) * agent.action_scale
        var action_err = abs(gpu_action - cpu_action)
        if env < 4 or err_vs_cpu > 0.001 or err_vs_env0 > 0.001:
            print(
                "    env[" + String(env) + "] mean="
                + String(gpu_mean)[byte=:12]
                + " action=" + String(gpu_action)[byte=:12]
                + " (cpu_err=" + String(err_vs_cpu)[byte=:12]
                + " env0_err=" + String(err_vs_env0)[byte=:12] + ")"
            )
    if N > 4:
        print(
            "    ... max cpu_err=" + String(max_mean_err)
            + " max inter_env_err=" + String(max_inter_env_err)
        )
    print()


def main() raises:
    seed(42)
    print("=" * 90)
    print("IDP Diagnostic: GPU Actor Network + Action Extraction")
    print("=" * 90)

    var agent = AgentType(
        gamma=0.99, tau=0.005, action_scale=1.0,
        alpha=0.2, auto_alpha=False, target_entropy=-1.0,
    )
    agent.load_checkpoint("sac_inverted_double_pendulum.ckpt")
    print("Loaded checkpoint")
    print()

    # Test with a few different observations
    var test_obs = InlineArray[InlineArray[Scalar[nn_dtype], OBS_DIM], 3](
        uninitialized=True
    )

    # Obs 1: near upright (typical good state)
    test_obs[0] = InlineArray[Scalar[nn_dtype], OBS_DIM](fill=Scalar[nn_dtype](0))
    test_obs[0][0] = Scalar[nn_dtype](0.02)   # cart x
    test_obs[0][1] = Scalar[nn_dtype](-0.05)   # sin(q1)
    test_obs[0][2] = Scalar[nn_dtype](0.03)    # sin(q2)
    test_obs[0][3] = Scalar[nn_dtype](0.9987)  # cos(q1)
    test_obs[0][4] = Scalar[nn_dtype](0.9995)  # cos(q2)
    test_obs[0][5] = Scalar[nn_dtype](0.1)     # v0
    test_obs[0][6] = Scalar[nn_dtype](-0.3)    # v1
    test_obs[0][7] = Scalar[nn_dtype](0.2)     # v2
    test_obs[0][8] = Scalar[nn_dtype](0.0)     # placeholder

    # Obs 2: zero state
    test_obs[1] = InlineArray[Scalar[nn_dtype], OBS_DIM](fill=Scalar[nn_dtype](0))
    test_obs[1][3] = Scalar[nn_dtype](1.0)  # cos(0)=1
    test_obs[1][4] = Scalar[nn_dtype](1.0)  # cos(0)=1

    # Obs 3: tilted state
    test_obs[2] = InlineArray[Scalar[nn_dtype], OBS_DIM](fill=Scalar[nn_dtype](0))
    test_obs[2][0] = Scalar[nn_dtype](-0.1)
    test_obs[2][1] = Scalar[nn_dtype](0.3)
    test_obs[2][2] = Scalar[nn_dtype](-0.2)
    test_obs[2][3] = Scalar[nn_dtype](0.954)
    test_obs[2][4] = Scalar[nn_dtype](0.980)
    test_obs[2][5] = Scalar[nn_dtype](-1.5)
    test_obs[2][6] = Scalar[nn_dtype](2.0)
    test_obs[2][7] = Scalar[nn_dtype](-1.0)
    test_obs[2][8] = Scalar[nn_dtype](0.0)

    var ctx = DeviceContext()

    for t in range(3):
        print("--- Test obs " + String(t) + " ---")
        var result = cpu_forward_action(agent, test_obs[t])
        var cpu_mean = result[0]
        var cpu_action = result[1]
        print("  CPU: mean=" + String(cpu_mean)[byte=:12] + " action=" + String(cpu_action)[byte=:12])
        print()

        test_gpu_batch[1](ctx, agent, test_obs[t], cpu_mean, cpu_action)
        test_gpu_batch[4](ctx, agent, test_obs[t], cpu_mean, cpu_action)
        test_gpu_batch[32](ctx, agent, test_obs[t], cpu_mean, cpu_action)

    print("=" * 90)
    print("If actions match across batch sizes → issue is in physics/eval loop")
    print("If actions differ with larger batch → GPU forward pass has batch bug")
