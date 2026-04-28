"""Synthetic-overfit test for AlphaZero training pipeline.

Bypasses self-play and MCTS: fills the CPU replay buffer with a small
fixed set of (obs, policy_target, value_target) tuples, then runs
`train_step_gpu` in a tight loop. A correct training pipeline must drive
policy CE → 0 and value MSE → 0 on this fixed dataset within a few
hundred steps.

If the loss does NOT go down here, the bug is in
forward / loss / backward / optimizer / param-state plumbing,
NOT in MCTS, self-play, replay sampling, or arena.

Usage:
    pixi run -e nvidia mojo run -I . examples/board_games/tictactoe_alphazero_overfit.mojo
    pixi run -e apple  mojo run -I . examples/board_games/tictactoe_alphazero_overfit.mojo
"""

from std.memory import UnsafePointer
from std.gpu.host import DeviceContext
from std.random import random_float64, seed
from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.alphazero import (
    GenericAlphaZeroAgent,
    AlphaZeroTicTacToeConfig,
    AlphaZeroTicTacToeCNNConfig,
    AlphaZeroTicTacToeResNetConfig,
)
from mojo_rl.nn.constants import dtype


def main() raises:
    print("=== AlphaZero Synthetic-Overfit Test ===")
    print()

    var env_vars = load_dotenv()
    var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
    var url = env_vars.get("RL_MONITOR_URL", "")

    var logger = RemoteLogger(
        server_url=url,
        run_name="AlphaZero TTT Overfit",
        buffer_size=13,
        api_key=api_key,
    )

    # Toggle the architecture under test:
    # comptime Config = AlphaZeroTicTacToeConfig[]          # MLP
    # comptime Config = AlphaZeroTicTacToeCNNConfig[]       # CNN
    comptime Config = AlphaZeroTicTacToeResNetConfig[]      # ResNet (default)

    logger.set_config("test", "synthetic-overfit")
    logger.set_config("network", Config.NAME)
    logger.set_config("batch_size", String(Config.batch_size))

    var ctx = DeviceContext()
    var agent = GenericAlphaZeroAgent[Config, 64, 64, RemoteLogger]()
    agent.logger = UnsafePointer(to=logger)
    agent.diag_every = 5

    comptime OBS = Config.obs_dim          # 27 for TTT
    comptime ACT = Config.action_dim       # 9 for TTT
    comptime N = Config.batch_size         # populate exactly one batch worth

    # ── Build a tiny fixed dataset directly in the CPU replay buffer ──
    # Each sample: random sparse 3-plane obs + one-hot policy + value in {-1,0,+1}
    seed(0)
    for s in range(N):
        for p in range(3):
            for c in range(9):
                var v: Scalar[dtype] = 0.0
                if random_float64() < 0.25:
                    v = 1.0
                agent.state.buf_obs[s * OBS + p * 9 + c] = v

        var best = s % 9
        for a in range(ACT):
            agent.state.buf_policy[s * ACT + a] = 0.0
        agent.state.buf_policy[s * ACT + best] = 1.0

        var vt: Scalar[dtype] = -1.0
        var r = s % 3
        if r == 0:
            vt = 1.0
        elif r == 1:
            vt = 0.0
        agent.state.buf_value[s] = vt
    agent.state.buf_size = N

    print("Dataset:", N, "fixed samples, OBS=", OBS, "ACT=", ACT)
    print("Network:", Config.NAME)
    print()

    # ── Allocate GPU state + diagnostic host buffers ─────────────────
    # Mirrors train_selfplay_gpu's setup so train_step_gpu's diagnostic
    # logging works.
    var gpu = agent.GPUStateType(ctx)
    gpu.upload_from(agent.state, ctx)

    comptime _DIAG_BATCH = Config.batch_size
    comptime _DIAG_POUT = Config.PredModel.OUT_DIM
    comptime _DIAG_PS = Config.PredModel.PARAM_SIZE
    var diag_pred_host = ctx.enqueue_create_host_buffer[dtype](
        _DIAG_BATCH * _DIAG_POUT
    )
    var diag_go_host = ctx.enqueue_create_host_buffer[dtype](
        _DIAG_BATCH * _DIAG_POUT
    )
    var diag_params_host = ctx.enqueue_create_host_buffer[dtype](_DIAG_PS)
    var diag_grads_host = ctx.enqueue_create_host_buffer[dtype](_DIAG_PS)
    ctx.synchronize()

    # ── Train ────────────────────────────────────────────────────────
    var STEPS = 1000
    print("Training for", STEPS, "steps...")
    print()
    for step in range(STEPS):
        agent.train_step_gpu(
            ctx,
            gpu,
            diag_pred_host,
            diag_go_host,
            diag_params_host,
            diag_grads_host,
        )
        if (step + 1) % 100 == 0:
            print("  step", step + 1, "/", STEPS)

    ctx.synchronize()
    logger.close()
    print()
    print("=== Done ===")
    print(
        "Inspect policy_ce / value_mse / policy_entropy / param_norm in the"
        " logger UI."
    )
    print(
        "Expected: policy_ce → ~0, value_mse → ~0, policy_entropy → ~0, "
        "param_norm bounded."
    )
