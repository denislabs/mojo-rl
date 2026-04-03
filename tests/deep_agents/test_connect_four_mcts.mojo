"""Diagnostic: Can MCTS beat Random on ConnectFour with a fresh network?

If this fails, the issue is in env/MCTS, not training.
If this passes, training is corrupting the network.
"""

from std.gpu.host import DeviceContext
from std.memory import UnsafePointer
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.deep_agents.alphazero import (
    GenericAlphaZeroAgent,
    AlphaZeroConnectFourCNNConfig,
    AlphaZeroConnectFourConfig,
)
from mojo_rl.deep_agents.muzero.evaluators import RandomOpponent
from mojo_rl.deep_agents.muzero.gpu_mcts import GPUMCTSState
from mojo_rl.envs.board_games.connect_four import ConnectFourEnv


def _run_eval[
    Config: __type_of(AlphaZeroConnectFourConfig[]),
    AgentType: __type_of(GenericAlphaZeroAgent[AlphaZeroConnectFourConfig[], 64]),
    E: __type_of(ConnectFourEnv[DType.float32]),
](
    ctx: DeviceContext,
    mut agent: AgentType,
    mut gpu: AgentType.GPUStateType,
    rng_offset: Int,
) raises -> Tuple[Int, Int, Int]:
    """Helper to run gpu_eval with pre-allocated buffers."""
    comptime ACT = Config.action_dim
    comptime OBS = Config.obs_dim
    comptime MAX_NODES = Config.max_nodes
    comptime GS = E.STATE_SIZE
    comptime N_ENVS = 64
    comptime BATCH_SIMS = 8
    comptime WS = Config.PredModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime WS_SIZE = N_ENVS * BATCH_SIMS * WS if WS > 0 else 1
    comptime TOTAL_EXPAND = N_ENVS * BATCH_SIMS

    var gpu_mcts = GPUMCTSState[N_ENVS, MAX_NODES, ACT, OBS, 1, GS](ctx)
    var mcts_ws = ctx.enqueue_create_buffer[dtype](WS_SIZE)
    var states_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * GS)
    var obs_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS)
    var acts_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var rews_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var dones_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var term_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var legal_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * ACT)
    var exp_rews = ctx.enqueue_create_buffer[dtype](TOTAL_EXPAND)
    var exp_dones = ctx.enqueue_create_buffer[dtype](TOTAL_EXPAND)
    var exp_term = ctx.enqueue_create_buffer[dtype](TOTAL_EXPAND)
    var exp_obs = ctx.enqueue_create_buffer[dtype](TOTAL_EXPAND * OBS)
    var rews_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    var dones_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS)

    return agent.gpu_eval[E, RandomOpponent](
        ctx, gpu, gpu_mcts, mcts_ws,
        states_buf, obs_buf, acts_buf, rews_buf, dones_buf, term_buf, legal_buf,
        exp_rews, exp_dones, exp_term, exp_obs,
        rews_host, dones_host, rng_offset=rng_offset,
    )


def main() raises:
    var ctx = DeviceContext()
    comptime C4 = ConnectFourEnv[DType.float32]

    # Test 1: MLP with fresh (untrained) network
    print("=" * 60)
    print("TEST 1: Fresh MLP network + MCTS(25) vs Random")
    print("=" * 60)
    comptime MLPConfig = AlphaZeroConnectFourConfig[]
    var mlp_agent = GenericAlphaZeroAgent[MLPConfig, 64]()
    var mlp_gpu = mlp_agent.GPUStateType(ctx)
    mlp_gpu.upload_from(mlp_agent.state, ctx)

    comptime MLP_ACT = MLPConfig.action_dim
    comptime MLP_OBS = MLPConfig.obs_dim
    comptime MLP_MAX_NODES = MLPConfig.max_nodes
    comptime MLP_GS = C4.STATE_SIZE
    comptime MLP_BATCH_SIMS = 8
    comptime MLP_WS = MLPConfig.PredModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime MLP_WS_SIZE = 64 * MLP_BATCH_SIMS * MLP_WS if MLP_WS > 0 else 1
    comptime MLP_TOTAL_EXPAND = 64 * MLP_BATCH_SIMS

    var mlp_mcts = GPUMCTSState[64, MLP_MAX_NODES, MLP_ACT, MLP_OBS, 1, MLP_GS](ctx)
    var mlp_ws = ctx.enqueue_create_buffer[dtype](MLP_WS_SIZE)
    var mlp_states = ctx.enqueue_create_buffer[dtype](64 * MLP_GS)
    var mlp_obs = ctx.enqueue_create_buffer[dtype](64 * MLP_OBS)
    var mlp_acts = ctx.enqueue_create_buffer[dtype](64)
    var mlp_rews = ctx.enqueue_create_buffer[dtype](64)
    var mlp_dones = ctx.enqueue_create_buffer[dtype](64)
    var mlp_term = ctx.enqueue_create_buffer[dtype](64)
    var mlp_legal = ctx.enqueue_create_buffer[dtype](64 * MLP_ACT)
    var mlp_exp_rews = ctx.enqueue_create_buffer[dtype](MLP_TOTAL_EXPAND)
    var mlp_exp_dones = ctx.enqueue_create_buffer[dtype](MLP_TOTAL_EXPAND)
    var mlp_exp_term = ctx.enqueue_create_buffer[dtype](MLP_TOTAL_EXPAND)
    var mlp_exp_obs = ctx.enqueue_create_buffer[dtype](MLP_TOTAL_EXPAND * MLP_OBS)
    var mlp_rews_host = ctx.enqueue_create_host_buffer[dtype](64)
    var mlp_dones_host = ctx.enqueue_create_host_buffer[dtype](64)

    var r1 = mlp_agent.gpu_eval[C4, RandomOpponent](
        ctx, mlp_gpu, mlp_mcts, mlp_ws,
        mlp_states, mlp_obs, mlp_acts, mlp_rews, mlp_dones, mlp_term, mlp_legal,
        mlp_exp_rews, mlp_exp_dones, mlp_exp_term, mlp_exp_obs,
        mlp_rews_host, mlp_dones_host,
        rng_offset=42,
    )
    print("  W", r1[0], "D", r1[1], "L", r1[2])
    if r1[0] > r1[2]:
        print("  PASS: Fresh MLP+MCTS beats Random")
    elif r1[0] == r1[2]:
        print("  NEUTRAL: Fresh MLP+MCTS ties Random")
    else:
        print("  FAIL: Fresh MLP+MCTS LOSES to Random!")

    # Run again with different seed to check consistency
    var r1b = mlp_agent.gpu_eval[C4, RandomOpponent](
        ctx, mlp_gpu, mlp_mcts, mlp_ws,
        mlp_states, mlp_obs, mlp_acts, mlp_rews, mlp_dones, mlp_term, mlp_legal,
        mlp_exp_rews, mlp_exp_dones, mlp_exp_term, mlp_exp_obs,
        mlp_rews_host, mlp_dones_host, rng_offset=9999,
    )
    print("  (seed 2) W", r1b[0], "D", r1b[1], "L", r1b[2])

    print()

    # Test 2: CNN with fresh network
    print("=" * 60)
    print("TEST 2: Fresh CNN network + MCTS(25) vs Random")
    print("=" * 60)
    comptime CNNConfig = AlphaZeroConnectFourCNNConfig[]
    var cnn_agent = GenericAlphaZeroAgent[CNNConfig, 64]()
    var cnn_gpu = cnn_agent.GPUStateType(ctx)
    cnn_gpu.upload_from(cnn_agent.state, ctx)

    comptime CNN_ACT = CNNConfig.action_dim
    comptime CNN_OBS = CNNConfig.obs_dim
    comptime CNN_MAX_NODES = CNNConfig.max_nodes
    comptime CNN_GS = C4.STATE_SIZE
    comptime CNN_BATCH_SIMS = 8
    comptime CNN_WS = CNNConfig.PredModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime CNN_WS_SIZE = 64 * CNN_BATCH_SIMS * CNN_WS if CNN_WS > 0 else 1
    comptime CNN_TOTAL_EXPAND = 64 * CNN_BATCH_SIMS

    var cnn_mcts = GPUMCTSState[64, CNN_MAX_NODES, CNN_ACT, CNN_OBS, 1, CNN_GS](ctx)
    var cnn_ws = ctx.enqueue_create_buffer[dtype](CNN_WS_SIZE)
    var cnn_states = ctx.enqueue_create_buffer[dtype](64 * CNN_GS)
    var cnn_obs = ctx.enqueue_create_buffer[dtype](64 * CNN_OBS)
    var cnn_acts = ctx.enqueue_create_buffer[dtype](64)
    var cnn_rews = ctx.enqueue_create_buffer[dtype](64)
    var cnn_dones = ctx.enqueue_create_buffer[dtype](64)
    var cnn_term = ctx.enqueue_create_buffer[dtype](64)
    var cnn_legal = ctx.enqueue_create_buffer[dtype](64 * CNN_ACT)
    var cnn_exp_rews = ctx.enqueue_create_buffer[dtype](CNN_TOTAL_EXPAND)
    var cnn_exp_dones = ctx.enqueue_create_buffer[dtype](CNN_TOTAL_EXPAND)
    var cnn_exp_term = ctx.enqueue_create_buffer[dtype](CNN_TOTAL_EXPAND)
    var cnn_exp_obs = ctx.enqueue_create_buffer[dtype](CNN_TOTAL_EXPAND * CNN_OBS)
    var cnn_rews_host = ctx.enqueue_create_host_buffer[dtype](64)
    var cnn_dones_host = ctx.enqueue_create_host_buffer[dtype](64)

    print(" Test beginning")
    var r2 = cnn_agent.gpu_eval[C4, RandomOpponent](
        ctx, cnn_gpu, cnn_mcts, cnn_ws,
        cnn_states, cnn_obs, cnn_acts, cnn_rews, cnn_dones, cnn_term, cnn_legal,
        cnn_exp_rews, cnn_exp_dones, cnn_exp_term, cnn_exp_obs,
        cnn_rews_host, cnn_dones_host, rng_offset=42,
    )
    print(" Test ending")
    print("  W", r2[0], "D", r2[1], "L", r2[2])
    if r2[0] > r2[2]:
        print("  PASS: Fresh CNN+MCTS beats Random")
    elif r2[0] == r2[2]:
        print("  NEUTRAL: Fresh CNN+MCTS ties Random")
    else:
        print("  FAIL: Fresh CNN+MCTS LOSES to Random!")

    var r2b = cnn_agent.gpu_eval[C4, RandomOpponent](
        ctx, cnn_gpu, cnn_mcts, cnn_ws,
        cnn_states, cnn_obs, cnn_acts, cnn_rews, cnn_dones, cnn_term, cnn_legal,
        cnn_exp_rews, cnn_exp_dones, cnn_exp_term, cnn_exp_obs,
        cnn_rews_host, cnn_dones_host, rng_offset=9999,
    )
    print("  (seed 2) W", r2b[0], "D", r2b[1], "L", r2b[2])

    print()
    print("If FRESH network + MCTS can't beat Random → env/MCTS bug")
    print("If FRESH network + MCTS beats Random → training is corrupting")
