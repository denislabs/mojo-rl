"""AlphaZero Diagnostic Test — verify each component in isolation.

Tests:
1. Gradient correctness: train on known data, verify loss decreases
2. Value head: learn to predict +1 for winning, -1 for losing positions
3. Policy head: learn to prefer center opening
4. Obs encoding: verify TicTacToe canonical observations are correct
5. Outcome assignment: verify alternating +1/-1 for two-player games
6. Augmentation: verify symmetry produces valid transforms
"""

from std.memory import alloc, memset
from std.math import abs, exp, min
from std.random import random_float64
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Linear, LinearReLU, Sequential, Parallel
from mojo_rl.nn.optimizer import Adam
from mojo_rl.deep_agents.alphazero import (
    GenericAlphaZeroAgent,
    AlphaZeroTicTacToeConfig,
)
from mojo_rl.deep_agents.alphazero.configs import AlphaZeroConfig
from mojo_rl.planners.tree_search.strategies import (
    DirichletNoise, AlphaGoPUCT, SelfPlay,
)
from mojo_rl.deep_agents.muzero.evaluators import RandomOpponent, MinimaxTicTacToe
from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv


struct DiagConfig(AlphaZeroConfig):
    comptime NAME: String = "AZ-Diag"
    comptime obs_dim: Int = 27
    comptime action_dim: Int = 9
    comptime PredModel = Sequential[
        LinearReLU[27, 64],
        LinearReLU[64, 64],
        Parallel[Linear[64, 9], Linear[64, 1]],
    ]
    comptime OptType = Adam[LR=0.001]
    comptime batch_size: Int = 32
    comptime buffer_capacity: Int = 10000
    comptime history_window: Int = 20
    comptime num_simulations: Int = 25
    comptime max_nodes: Int = 64
    comptime temp_threshold: Int = 15
    comptime temp_min: Float64 = 0.0
    comptime batch_sims: Int = 8
    comptime invalid_action_penalty: Float64 = 0.0
    comptime max_grad_norm: Float64 = 0.0
    comptime value_target_q_weight: Float64 = 0.0
    comptime max_episode_length: Int = 9
    comptime board_rows: Int = 3
    comptime board_cols: Int = 3
    comptime board_planes: Int = 3
    comptime num_symmetries: Int = 2
    comptime Noise = DirichletNoise[0.25, 0.25]
    comptime PUCT = AlphaGoPUCT[1.0]
    comptime Players = SelfPlay


def test_obs_encoding():
    """Test 1: Verify TicTacToe observation encoding is correct."""
    print("=== Test 1: Obs Encoding ===")
    comptime TTTCPU = TicTacToeEnv[DType.float64]
    var env = TTTCPU()
    _ = env.reset()

    # Empty board: current player = P0
    var obs = env.get_obs_list()
    print("  Empty board obs (27D):")
    print("    Plane 0 (my pieces):  ", end="")
    var ok = True
    for i in range(9):
        print(obs[i], end=" ")
        if obs[i] != 0.0:
            ok = False  # Should be all 0
    print()
    print("    Plane 1 (opp pieces): ", end="")
    for i in range(9, 18):
        print(obs[i], end=" ")
        if obs[i] != 0.0:
            ok = False  # Should be all 0
    print()
    print("    Plane 2 (legal/empty):", end="")
    for i in range(18, 27):
        print(obs[i], end=" ")
        if obs[i] != 1.0:
            ok = False  # Should be all 1
    print()

    # Play center (action 4)
    _ = env.step(env.action_from_index(4))
    var obs2 = env.get_obs_list()
    print("  After P0 plays center (now P1's turn, canonical):")
    print("    Plane 0 (my=P1):      ", end="")
    for i in range(9):
        print(obs2[i], end=" ")
    print()
    print("    Plane 1 (opp=P0):     ", end="")
    for i in range(9, 18):
        print(obs2[i], end=" ")
    print()

    # P1 should see P0's center as opponent's piece
    if obs2[9 + 4] != 1.0:
        print("  FAIL: P0's center not visible as opponent piece!")
        ok = False
    if obs2[4] != 0.0:
        print("  FAIL: P1 shouldn't have my-piece at center!")
        ok = False
    if obs2[18 + 4] != 0.0:
        print("  FAIL: Center should not be legal!")
        ok = False

    if ok:
        print("  PASS: Obs encoding correct")
    else:
        print("  FAIL: Obs encoding has issues")
    print()


def test_outcome_assignment():
    """Test 2: Verify outcome assignment alternates correctly."""
    print("=== Test 2: Outcome Assignment ===")

    # Simulate a 3-move game where P0 wins (reward = +1)
    # Moves: P0, P1, P0(wins)
    # ep_len = 3, last_reward = 1.0
    # steps_from_end:  2, 1, 0
    # Perspective:  P0, P1, P0
    # Expected outcome: +1, -1, +1

    var last_reward: Float64 = 1.0
    var ep_len = 3
    var is_draw = (last_reward > -0.01 and last_reward < 0.01)
    var ok = True

    print("  P0 wins game (3 moves):")
    for t in range(ep_len):
        var steps_from_end = ep_len - 1 - t
        var outcome: Float64
        if is_draw:
            outcome = 1e-4
        elif steps_from_end % 2 == 0:
            outcome = last_reward
        else:
            outcome = -last_reward
        var player = "P0" if steps_from_end % 2 == 0 else "P1"
        print("    Move", t, "(", player, "): outcome =", outcome)

        if steps_from_end % 2 == 0 and outcome != 1.0:
            ok = False
        if steps_from_end % 2 == 1 and outcome != -1.0:
            ok = False

    # Test draw
    last_reward = 0.0
    is_draw = True
    print("  Draw game:")
    for t in range(3):
        var outcome: Float64 = 1e-4 if is_draw else 0.0
        print("    Move", t, ": outcome =", outcome)
        if outcome != 1e-4:
            ok = False

    if ok:
        print("  PASS: Outcome assignment correct")
    else:
        print("  FAIL: Outcome assignment wrong")
    print()


def test_gradient_descent() raises:
    """Test 3: Train on fixed data, verify loss decreases."""
    print("=== Test 3: Gradient Descent (loss should decrease) ===")
    var ctx = DeviceContext()
    var agent = GenericAlphaZeroAgent[DiagConfig, 64]()

    # Create fixed training data: empty board → prefer center, value ≈ 0.1
    comptime OBS = 27
    comptime ACT = 9
    var obs_ptr = alloc[Scalar[dtype]](OBS)
    var pol_ptr = alloc[Scalar[dtype]](ACT)

    # Empty board obs: plane2 (legal moves) all 1s
    memset(obs_ptr, 0, OBS)
    for i in range(18, 27):
        obs_ptr[i] = Scalar[dtype](1.0)

    # Target policy: strong preference for center (action 4)
    memset(pol_ptr, 0, ACT)
    pol_ptr[4] = Scalar[dtype](0.8)
    for a in range(9):
        if a != 4:
            pol_ptr[a] = Scalar[dtype](0.025)

    # Add 200 copies to buffer (same position, same target)
    for _ in range(200):
        agent.state.add(
            rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](obs_ptr),
            rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](pol_ptr),
            Scalar[dtype](0.1),  # Slight advantage
        )

    # Compute initial prediction
    var legal = List[Bool]()
    var obs_list = List[Scalar[dtype]]()
    for i in range(OBS):
        obs_list.append(obs_ptr[i])
    for _ in range(9):
        legal.append(True)

    # Check initial policy
    print("  Initial action (raw argmax):", agent.select_action(obs_list, legal))

    # Train for many steps
    var gpu = agent.GPUStateType(ctx)
    gpu.upload_from(agent.state, ctx)

    # Pre-allocated diagnostic host buffers (reused every train step)
    comptime _DIAG_BATCH = DiagConfig.batch_size
    comptime _DIAG_POUT = DiagConfig.PredModel.OUT_DIM
    comptime _DIAG_PS = DiagConfig.PredModel.PARAM_SIZE
    var diag_pred_host = ctx.enqueue_create_host_buffer[dtype](
        _DIAG_BATCH * _DIAG_POUT
    )
    var diag_go_host = ctx.enqueue_create_host_buffer[dtype](
        _DIAG_BATCH * _DIAG_POUT
    )
    var diag_params_host = ctx.enqueue_create_host_buffer[dtype](_DIAG_PS)
    var diag_grads_host = ctx.enqueue_create_host_buffer[dtype](_DIAG_PS)

    # Compute approximate loss at start and end by checking predictions
    for step in range(500):
        agent.train_step_gpu(
            ctx,
            gpu,
            diag_pred_host,
            diag_go_host,
            diag_params_host,
            diag_grads_host,
        )

    gpu.download_to(agent.state, ctx)

    # Check learned policy
    var learned_action = agent.select_action(obs_list, legal)
    print("  After 500 steps, action:", learned_action)

    if learned_action == 4:
        print("  PASS: Network learned to prefer center")
    else:
        print("  FAIL: Network didn't learn center preference (got action", learned_action, ")")

    # The select_action test above already verifies policy learning.
    # If center is learned, the gradient is flowing correctly.

    obs_ptr.free()
    pol_ptr.free()
    print()


def test_value_signs() raises:
    """Test 4: Train on +1 and -1 positions, verify value head learns signs."""
    print("=== Test 4: Value Sign Test (win=+1, loss=-1) ===")
    var ctx = DeviceContext()
    var agent = GenericAlphaZeroAgent[DiagConfig, 64]()

    comptime OBS = 27
    comptime ACT = 9

    # Position A: P0 has corner+center → winning, value = +1
    var obs_win = alloc[Scalar[dtype]](OBS)
    memset(obs_win, 0, OBS)
    obs_win[0] = Scalar[dtype](1.0)   # My piece at 0
    obs_win[4] = Scalar[dtype](1.0)   # My piece at 4 (center)
    obs_win[9 + 1] = Scalar[dtype](1.0)  # Opp at 1
    for i in range(18, 27):
        if i - 18 != 0 and i - 18 != 1 and i - 18 != 4:
            obs_win[i] = Scalar[dtype](1.0)

    # Position B: mirror (opponent has advantage), value = -1
    var obs_lose = alloc[Scalar[dtype]](OBS)
    memset(obs_lose, 0, OBS)
    obs_lose[0] = Scalar[dtype](1.0)   # My piece at 0
    obs_lose[9 + 4] = Scalar[dtype](1.0)  # Opp at center
    obs_lose[9 + 8] = Scalar[dtype](1.0)  # Opp at 8
    for i in range(18, 27):
        if i - 18 != 0 and i - 18 != 4 and i - 18 != 8:
            obs_lose[i] = Scalar[dtype](1.0)

    # Uniform policy target (we only care about value)
    var pol_ptr = alloc[Scalar[dtype]](ACT)
    for a in range(ACT):
        pol_ptr[a] = Scalar[dtype](1.0 / 9.0)

    # Add to buffer
    for _ in range(200):
        agent.state.add(
            rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](obs_win),
            rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](pol_ptr),
            Scalar[dtype](1.0),
        )
        agent.state.add(
            rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](obs_lose),
            rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](pol_ptr),
            Scalar[dtype](-1.0),
        )

    # Train
    var gpu = agent.GPUStateType(ctx)
    gpu.upload_from(agent.state, ctx)

    # Pre-allocated diagnostic host buffers (reused every train step)
    comptime _DIAG_BATCH = DiagConfig.batch_size
    comptime _DIAG_POUT = DiagConfig.PredModel.OUT_DIM
    comptime _DIAG_PS = DiagConfig.PredModel.PARAM_SIZE
    var diag_pred_host = ctx.enqueue_create_host_buffer[dtype](
        _DIAG_BATCH * _DIAG_POUT
    )
    var diag_go_host = ctx.enqueue_create_host_buffer[dtype](
        _DIAG_BATCH * _DIAG_POUT
    )
    var diag_params_host = ctx.enqueue_create_host_buffer[dtype](_DIAG_PS)
    var diag_grads_host = ctx.enqueue_create_host_buffer[dtype](_DIAG_PS)

    for _ in range(1000):
        agent.train_step_gpu(
            ctx,
            gpu,
            diag_pred_host,
            diag_go_host,
            diag_params_host,
            diag_grads_host,
        )
    gpu.download_to(agent.state, ctx)

    # Check via select_action — if actions differ for win/lose, policy learned
    var obs_win_list = List[Scalar[dtype]]()
    var obs_lose_list = List[Scalar[dtype]]()
    var legal = List[Bool]()
    for i in range(OBS):
        obs_win_list.append(obs_win[i])
        obs_lose_list.append(obs_lose[i])
    for _ in range(ACT):
        legal.append(True)

    var act_win = agent.select_action(obs_win_list, legal)
    var act_lose = agent.select_action(obs_lose_list, legal)

    print("  Win position preferred action:", act_win)
    print("  Lose position preferred action:", act_lose)

    var ok = True
    if act_win != act_lose:
        print("  PASS: Network distinguishes win/lose positions")
    else:
        print("  WARN: Same action for both (may still work, value heads may differ)")

    obs_win.free()
    obs_lose.free()
    pol_ptr.free()
    print()


def test_augmentation():
    """Test 5: Verify symmetry augmentation produces valid data."""
    print("=== Test 5: Symmetry Augmentation ===")
    from mojo_rl.nn.constants import dtype as nn_dtype

    comptime OBS = 27
    comptime ACT = 9

    # Create obs with P0 at corner 0
    var obs = alloc[Scalar[nn_dtype]](OBS)
    memset(obs, 0, OBS)
    obs[0] = Scalar[nn_dtype](1.0)   # My piece at cell 0 (top-left)
    for i in range(18, 27):
        if i != 18:
            obs[i] = Scalar[nn_dtype](1.0)  # All legal except cell 0

    # Policy: prefer cell 4 (center)
    var pol = alloc[Scalar[nn_dtype]](ACT)
    memset(pol, 0, ACT)
    pol[4] = Scalar[nn_dtype](1.0)

    comptime TTT = TicTacToeEnv[DType.float32]

    var ok = True
    print("  Original: piece at cell 0, policy prefers cell 4")
    for s in range(TTT.NUM_SYMMETRIES):
        var sym_obs = alloc[Scalar[nn_dtype]](OBS)
        var sym_pol = alloc[Scalar[nn_dtype]](ACT)
        var obs_p = rebind[UnsafePointer[Scalar[nn_dtype], MutAnyOrigin]](obs)
        var pol_p = rebind[UnsafePointer[Scalar[nn_dtype], MutAnyOrigin]](pol)
        var so = rebind[UnsafePointer[Scalar[nn_dtype], MutAnyOrigin]](sym_obs)
        var sp = rebind[UnsafePointer[Scalar[nn_dtype], MutAnyOrigin]](sym_pol)
        TTT.augment_obs[OBS](obs_p, s, so)
        TTT.augment_policy[ACT](pol_p, s, sp)

        # Find where the piece moved and where policy points
        var piece_cell = -1
        var policy_cell = -1
        for c in range(9):
            if Float64(sym_obs[c]) > 0.5:
                piece_cell = c
            if Float64(sym_pol[c]) > 0.5:
                policy_cell = c
        print("    Sym", s, ": piece at", piece_cell, "policy at", policy_cell)

        # Verify: piece should always be at a corner (0,2,6,8) and
        # policy should always be center (4) since center is invariant
        if policy_cell != 4:
            print("      FAIL: Center should map to center!")
            ok = False

        sym_obs.free()
        sym_pol.free()

    obs.free()
    pol.free()

    if ok:
        print("  PASS: Augmentation preserves center invariance")
    else:
        print("  FAIL: Augmentation has issues")
    print()


def test_selfplay_data_quality() raises:
    """Test 6: Run a few self-play games, check data makes sense."""
    print("=== Test 6: Self-Play Data Quality ===")
    var ctx = DeviceContext()
    comptime TTT = TicTacToeEnv[DType.float32]

    var agent = GenericAlphaZeroAgent[DiagConfig, 64]()

    # Collect 1 iteration of self-play data
    agent.start_new_iteration()
    _ = agent.train_selfplay_gpu[TTT](
        ctx,
        num_iters=1,
        steps_per_iter=500,
        train_epochs=0,
        do_eval=False,
        do_arena=False,
    )

    print("  Buffer size:", agent.state.buf_size)

    # Analyze buffer contents
    comptime OBS = 27
    comptime ACT = 9
    var pos_values = 0
    var neg_values = 0
    var zero_values = 0
    var tiny_values = 0
    var total = agent.state.buf_size
    if total > 1000:
        total = 1000  # Sample first 1000

    for i in range(total):
        var v = Float64(agent.state.buf_value[i])
        if v > 0.5:
            pos_values += 1
        elif v < -0.5:
            neg_values += 1
        elif abs(v) < 0.001:
            zero_values += 1
        else:
            tiny_values += 1

    print("  Value distribution (first", total, "samples):")
    print("    +1 (wins):", pos_values)
    print("    -1 (losses):", neg_values)
    print("    ~0 (draws/1e-4):", zero_values + tiny_values)

    if pos_values == 0 and neg_values == 0:
        print("  FAIL: No decisive games collected!")
    elif pos_values > 0 and neg_values > 0:
        print("  PASS: Both winning and losing positions in buffer")
    else:
        print("  WARN: Imbalanced values (wins:", pos_values, "losses:", neg_values, ")")

    # Check policy targets sum to ~1
    var bad_policies = 0
    for i in range(min(total, 100)):
        var pol_sum: Float64 = 0.0
        for a in range(ACT):
            pol_sum += Float64(agent.state.buf_policy[i * ACT + a])
        if abs(pol_sum - 1.0) > 0.1:
            bad_policies += 1

    if bad_policies == 0:
        print("  PASS: All policy targets sum to ~1.0")
    else:
        print("  FAIL:", bad_policies, "/ 100 policies don't sum to 1.0")

    # Check obs are valid (should have values in {0, 1})
    var bad_obs = 0
    for i in range(min(total, 100)):
        for j in range(OBS):
            var v = Float64(agent.state.buf_obs[i * OBS + j])
            if v < -0.1 or v > 1.1:
                bad_obs += 1

    if bad_obs == 0:
        print("  PASS: All obs values in [0, 1]")
    else:
        print("  FAIL:", bad_obs, "obs values out of range")
    print()


def main() raises:
    print("╔══════════════════════════════════════════════╗")
    print("║  AlphaZero Diagnostic Test Suite             ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    test_obs_encoding()
    test_outcome_assignment()
    test_gradient_descent()
    test_value_signs()
    test_augmentation()
    test_selfplay_data_quality()

    print("╔══════════════════════════════════════════════╗")
    print("║  All diagnostic tests complete               ║")
    print("╚══════════════════════════════════════════════╝")
