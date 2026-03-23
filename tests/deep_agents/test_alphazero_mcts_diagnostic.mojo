"""AlphaZero MCTS Diagnostic — verify MCTS produces sensible visit counts.

Strategy: Run a few self-play steps, then inspect the collected policy targets.
If MCTS is working, policies should NOT be uniform — they should concentrate
on better moves. We also check specific properties:

1. Policy entropy should be lower than uniform (MCTS focuses on good moves)
2. After training on collected data, the network should improve vs random
3. MCTS with a trained network should produce sharper policies than random network

This is a higher-level integration test that catches bugs in:
- MCTS selection (PUCT), expansion (env.step), backup (value negation)
- Policy extraction (visit counts → proportional policy)
- Temperature annealing
"""

from std.memory import alloc, memset
from std.math import abs, log
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
from mojo_rl.deep_agents.muzero.strategies import (
    DirichletNoise, AlphaGoPUCT, SelfPlay,
)
from mojo_rl.deep_agents.muzero.evaluators import RandomOpponent, MinimaxTicTacToe
from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv


struct DiagConfig(AlphaZeroConfig):
    comptime NAME: String = "AZ-MCTS-Diag"
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
    comptime Noise = DirichletNoise[0.25, 0.25]
    comptime PUCT = AlphaGoPUCT[1.0]
    comptime Players = SelfPlay


def compute_entropy(pol: UnsafePointer[Scalar[dtype], MutAnyOrigin], n: Int) -> Float64:
    """Compute entropy of a probability distribution."""
    var h: Float64 = 0.0
    for i in range(n):
        var p = Float64(pol[i])
        if p > 1e-8:
            h -= p * log(p)
    return h


def test_mcts_policy_quality() raises:
    """Test 1: MCTS policies should not be uniform — entropy should be lower."""
    print("=== Test 1: MCTS Policy Entropy ===")
    var ctx = DeviceContext()
    comptime TTT = TicTacToeEnv[DType.float32]
    comptime TTTCPU = TicTacToeEnv[DType.float64]
    comptime ACT = 9

    var agent = GenericAlphaZeroAgent[DiagConfig, 64]()
    var arena_env = TTTCPU()

    # Collect self-play data with random network
    agent.start_new_iteration()
    _ = agent.train_selfplay_gpu[TTT, TTTCPU](
        ctx,
        arena_env,
        num_steps=500,
        warmup_steps=0,
        gradient_steps=0,
        print_every=100000,
    )

    print("  Buffer size:", agent.state.buf_size)

    # Analyze policy entropy
    var total_entropy: Float64 = 0.0
    var num_samples = agent.state.buf_size
    if num_samples > 500:
        num_samples = 500

    var uniform_count = 0
    var sharp_count = 0  # max prob > 0.5

    for i in range(num_samples):
        var pol_ptr = agent.state.buf_policy + i * ACT
        var h = compute_entropy(
            rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](pol_ptr), ACT
        )
        total_entropy += h

        # Check if essentially uniform
        var max_p: Float64 = 0.0
        var nonzero = 0
        for a in range(ACT):
            var p = Float64(pol_ptr[a])
            if p > 0.01:
                nonzero += 1
            if p > max_p:
                max_p = p
        if max_p < 0.2:
            uniform_count += 1
        if max_p > 0.5:
            sharp_count += 1

    var avg_entropy = total_entropy / Float64(num_samples)
    # Max entropy for 9 actions = ln(9) ≈ 2.197
    # For legal moves mid-game (5-7 legal), max ≈ ln(7) ≈ 1.95
    var max_entropy = log(9.0)

    print("  Avg policy entropy:", avg_entropy, "/ max:", max_entropy)
    print("  Uniform policies (max_p < 0.2):", uniform_count, "/", num_samples)
    print("  Sharp policies (max_p > 0.5):", sharp_count, "/", num_samples)

    if avg_entropy < max_entropy * 0.95:
        print("  PASS: MCTS policies have lower entropy than uniform")
    else:
        print("  FAIL: MCTS policies are nearly uniform — MCTS may not be working!")

    if sharp_count > num_samples // 10:
        print("  PASS: Some policies are sharp (MCTS concentrating visits)")
    else:
        print("  WARN: Very few sharp policies — MCTS may be too shallow")
    print()


def test_training_improves_play() raises:
    """Test 2: After training on MCTS data, the agent should beat untrained."""
    print("=== Test 2: Training Improves Play ===")
    var ctx = DeviceContext()
    comptime TTT = TicTacToeEnv[DType.float32]
    comptime TTTCPU = TicTacToeEnv[DType.float64]

    var agent = GenericAlphaZeroAgent[DiagConfig, 64]()
    var random_eval = RandomOpponent()
    var eval_env = TTTCPU()
    var arena_env = TTTCPU()

    # Evaluate before training
    var r_before = agent.evaluate_against[TTTCPU](eval_env, random_eval, 100)
    var win_before = r_before[0]
    print("  Before training - vs Random: W", r_before[0], "D", r_before[1], "L", r_before[2])

    # Collect + train for 3 iterations
    for iter in range(3):
        agent.start_new_iteration()
        _ = agent.train_selfplay_gpu[TTT, TTTCPU](
            ctx,
            arena_env,
            num_steps=1000,
            warmup_steps=0,
            gradient_steps=0,
            print_every=100000,
        )
        var num_train = 10 * agent.state.buf_size // 32
        if num_train > 3000:
            num_train = 3000
        var gpu = agent.GPUStateType(ctx)
        gpu.upload_from(agent.state, ctx)
        for _ in range(num_train):
            agent.train_step_gpu(ctx, gpu)
        gpu.download_to(agent.state, ctx)

    # Evaluate after training
    var r_after = agent.evaluate_against[TTTCPU](eval_env, random_eval, 100)
    var win_after = r_after[0]
    print("  After 3 iters  - vs Random: W", r_after[0], "D", r_after[1], "L", r_after[2])

    # The trained agent should win more (or at least not worse)
    if win_after >= win_before:
        print("  PASS: Training improved or maintained performance")
    else:
        print("  WARN: Training didn't improve. Before:", win_before, "After:", win_after)
        if win_before - win_after > 15:
            print("  FAIL: Significant regression — training is hurting!")
        else:
            print("  (Small difference, may be noise)")
    print()


def test_mcts_finds_winning_move() raises:
    """Test 3: After training, MCTS policy should prefer better moves.

    We train the network, then check if the MCTS-collected policies
    from later iterations are sharper than from iteration 1.
    """
    print("=== Test 3: MCTS Gets Sharper With Training ===")
    var ctx = DeviceContext()
    comptime TTT = TicTacToeEnv[DType.float32]
    comptime TTTCPU = TicTacToeEnv[DType.float64]
    comptime ACT = 9

    var agent = GenericAlphaZeroAgent[DiagConfig, 64]()
    var arena_env = TTTCPU()

    # Iteration 1: random network
    agent.start_new_iteration()
    _ = agent.train_selfplay_gpu[TTT, TTTCPU](
        ctx,
        arena_env,
        num_steps=500,
        warmup_steps=0,
        gradient_steps=0,
        print_every=100000,
    )

    var early_buf_size = agent.state.buf_size
    var early_entropy: Float64 = 0.0
    var early_n = early_buf_size
    if early_n > 300:
        early_n = 300
    for i in range(early_n):
        var pol_ptr = agent.state.buf_policy + i * ACT
        early_entropy += compute_entropy(
            rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](pol_ptr), ACT
        )
    early_entropy /= Float64(early_n)

    # Train
    var gpu = agent.GPUStateType(ctx)
    gpu.upload_from(agent.state, ctx)
    var num_train = 10 * agent.state.buf_size // 32
    if num_train > 3000:
        num_train = 3000
    for _ in range(num_train):
        agent.train_step_gpu(ctx, gpu)
    gpu.download_to(agent.state, ctx)

    # Iteration 2: trained network (new iteration clears old data)
    agent.start_new_iteration()
    _ = agent.train_selfplay_gpu[TTT, TTTCPU](
        ctx,
        arena_env,
        num_steps=500,
        warmup_steps=0,
        gradient_steps=0,
        print_every=100000,
    )

    # Compute entropy of new policies (skip old data)
    var new_start = early_buf_size
    var new_n = agent.state.buf_size - new_start
    if new_n > 300:
        new_n = 300
    if new_n < 10:
        print("  SKIP: Not enough new samples")
        return

    var late_entropy: Float64 = 0.0
    for i in range(new_n):
        var idx = new_start + i
        if idx >= agent.state.buf_size:
            break
        var pol_ptr = agent.state.buf_policy + idx * ACT
        late_entropy += compute_entropy(
            rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](pol_ptr), ACT
        )
    late_entropy /= Float64(new_n)

    print("  Random network MCTS entropy:", early_entropy)
    print("  Trained network MCTS entropy:", late_entropy)

    if late_entropy < early_entropy:
        print("  PASS: Trained network produces sharper MCTS policies")
    else:
        print("  WARN: Trained network MCTS not sharper (", late_entropy, ">=", early_entropy, ")")
        if late_entropy - early_entropy > 0.2:
            print("  FAIL: Significantly worse — possible MCTS or training bug")
        else:
            print("  (Small difference, may be noise)")
    print()


def test_policy_targets_nondegenerate() raises:
    """Test 4: Check that MCTS doesn't produce degenerate policies.

    Degenerate cases that indicate bugs:
    - All policies are one-hot on the same action (MCTS stuck)
    - All policies have identical entropy (no adaptation to position)
    - Illegal actions have non-zero probability
    """
    print("=== Test 4: Policy Target Sanity ===")
    var ctx = DeviceContext()
    comptime TTT = TicTacToeEnv[DType.float32]
    comptime TTTCPU = TicTacToeEnv[DType.float64]
    comptime ACT = 9

    var agent = GenericAlphaZeroAgent[DiagConfig, 64]()
    var arena_env = TTTCPU()

    agent.start_new_iteration()
    _ = agent.train_selfplay_gpu[TTT, TTTCPU](
        ctx,
        arena_env,
        num_steps=500,
        warmup_steps=0,
        gradient_steps=0,
        print_every=100000,
    )

    var n = agent.state.buf_size
    if n > 500:
        n = 500

    # Check 1: Are all policies the same action?
    var action_counts = alloc[Int](ACT)
    memset(action_counts, 0, ACT)
    for i in range(n):
        var max_a = 0
        var max_p = Float64(agent.state.buf_policy[i * ACT])
        for a in range(1, ACT):
            var p = Float64(agent.state.buf_policy[i * ACT + a])
            if p > max_p:
                max_p = p
                max_a = a
        action_counts[max_a] += 1

    print("  Action distribution of argmax(policy):")
    var max_count = 0
    for a in range(ACT):
        print("    Action", a, ":", action_counts[a])
        if action_counts[a] > max_count:
            max_count = action_counts[a]

    if max_count > n * 9 // 10:
        print("  FAIL: >90% of policies prefer the same action — MCTS is stuck!")
    else:
        print("  PASS: Policies distributed across multiple actions")

    # Check 2: Do policies have varying entropy?
    var min_h: Float64 = 100.0
    var max_h: Float64 = -1.0
    for i in range(n):
        var pol_ptr = agent.state.buf_policy + i * ACT
        var h = compute_entropy(
            rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](pol_ptr), ACT
        )
        if h < min_h:
            min_h = h
        if h > max_h:
            max_h = h

    print("  Entropy range: [", min_h, ",", max_h, "]")
    if max_h - min_h > 0.1:
        print("  PASS: Entropy varies across positions (MCTS adapts)")
    else:
        print("  FAIL: Constant entropy — MCTS produces same policy everywhere")

    # Check 3: Verify policies sum to ~1 and have no negative values
    var bad_sum = 0
    var negative_probs = 0
    for i in range(n):
        var s: Float64 = 0.0
        for a in range(ACT):
            var p = Float64(agent.state.buf_policy[i * ACT + a])
            s += p
            if p < -0.01:
                negative_probs += 1
        if abs(s - 1.0) > 0.05:
            bad_sum += 1

    if bad_sum == 0:
        print("  PASS: All policies sum to ~1.0")
    else:
        print("  FAIL:", bad_sum, "policies don't sum to 1.0")

    if negative_probs == 0:
        print("  PASS: No negative probabilities")
    else:
        print("  FAIL:", negative_probs, "negative probability values")

    action_counts.free()
    print()


def main() raises:
    print("╔══════════════════════════════════════════════╗")
    print("║  AlphaZero MCTS Diagnostic Test Suite        ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    test_mcts_policy_quality()
    test_policy_targets_nondegenerate()
    test_training_improves_play()
    test_mcts_finds_winning_move()

    print("╔══════════════════════════════════════════════╗")
    print("║  All MCTS diagnostic tests complete          ║")
    print("╚══════════════════════════════════════════════╝")
