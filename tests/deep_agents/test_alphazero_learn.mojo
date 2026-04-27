"""Test: Can the AlphaZero TicTacToe MLP learn from fixed data?

Isolates the network from the MCTS/self-play loop.
Creates 4 fixed board positions with known-good targets and trains
for 200 steps. If the network can't reduce loss on CPU, it confirms
a backward pass or architecture issue. Then tests GPU to check for
GPU-specific backward bugs.
"""

from std.math import exp, log, sqrt
from std.memory import alloc, memset
from std.random import random_float64
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.training import Network, NetworkState
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.model import (
    Linear,
    LinearReLU,
    Sequential,
    Parallel,
)
from mojo_rl.nn.optimizer import Adam

# Exact same architecture as AlphaZeroTicTacToeConfig
comptime PredModel = Sequential[
    LinearReLU[27, 128],
    LinearReLU[128, 128],
    Parallel[
        Linear[128, 9],   # Policy head
        Linear[128, 1],   # Value head
    ],
]
comptime OptType = Adam[LR=0.01]
comptime PredNet = Network[PredModel, OptType]
comptime ACT = 9
comptime OBS = 27
comptime PRED_OUT = PredModel.OUT_DIM  # 10 = 9 + 1
comptime BATCH = 4


def compute_loss(
    pred_host: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    target_policy: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    target_value: UnsafePointer[Scalar[dtype], MutAnyOrigin],
) -> Float64:
    """Compute CE + MSE loss for a batch of BATCH samples."""
    var total_ce: Float64 = 0.0
    var total_mse: Float64 = 0.0
    for b in range(BATCH):
        var pred_off = b * PRED_OUT
        # Softmax
        var max_l: Float64 = -1e18
        for a in range(ACT):
            var l = Float64(pred_host[pred_off + a])
            if l > max_l:
                max_l = l
        var sum_e: Float64 = 0.0
        for a in range(ACT):
            sum_e += exp(Float64(pred_host[pred_off + a]) - max_l)
        # CE
        var ce: Float64 = 0.0
        for a in range(ACT):
            var prob = exp(Float64(pred_host[pred_off + a]) - max_l) / sum_e
            var target = Float64(target_policy[b * ACT + a])
            if target > 1e-8 and prob > 1e-8:
                ce -= target * log(prob)
        total_ce += ce
        # MSE through tanh
        var raw_v = Float64(pred_host[pred_off + ACT])
        var tanh_v: Float64 = 0.0
        if raw_v > 15.0:
            tanh_v = 1.0
        elif raw_v < -15.0:
            tanh_v = -1.0
        else:
            var e2 = exp(2.0 * raw_v)
            tanh_v = (e2 - 1.0) / (e2 + 1.0)
        var tv = Float64(target_value[b])
        total_mse += (tanh_v - tv) * (tanh_v - tv)
    return total_ce / Float64(BATCH) + total_mse / Float64(BATCH)


def compute_policy_entropy(
    pred_host: UnsafePointer[Scalar[dtype], MutAnyOrigin],
) -> Float64:
    """Average entropy of softmax policy across batch."""
    var total: Float64 = 0.0
    for b in range(BATCH):
        var pred_off = b * PRED_OUT
        var max_l: Float64 = -1e18
        for a in range(ACT):
            var l = Float64(pred_host[pred_off + a])
            if l > max_l:
                max_l = l
        var sum_e: Float64 = 0.0
        for a in range(ACT):
            sum_e += exp(Float64(pred_host[pred_off + a]) - max_l)
        var ent: Float64 = 0.0
        for a in range(ACT):
            var prob = exp(Float64(pred_host[pred_off + a]) - max_l) / sum_e
            if prob > 1e-8:
                ent -= prob * log(prob)
        total += ent
    return total / Float64(BATCH)


def print_logits(
    pred_host: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    sample: Int,
):
    """Print raw logits for a single sample."""
    var pred_off = sample * PRED_OUT
    print("  Logits[", sample, "]: ", end="")
    for a in range(ACT):
        var v = Float64(pred_host[pred_off + a])
        # Print with 4 decimal places
        var iv = Int(v * 10000)
        print(Float64(iv) / 10000.0, end=" ")
    var raw_v = Float64(pred_host[pred_off + ACT])
    print("| val_raw=", Float64(Int(raw_v * 10000)) / 10000.0)


def test_cpu_training() raises:
    """Test: can the network learn on CPU with fixed data?"""
    print("=" * 60)
    print("TEST: CPU Training on Fixed Data")
    print("=" * 60)
    print("Architecture: LinearReLU[27,128] -> LinearReLU[128,128] -> Parallel[Linear[128,9], Linear[128,1]]")
    print("PARAM_SIZE:", PredModel.PARAM_SIZE)
    print("CACHE_SIZE:", PredModel.CACHE_SIZE)
    print("OUT_DIM:", PredModel.OUT_DIM)
    print()

    # Create network state with Kaiming init
    var state = NetworkState[PredModel, OptType]()
    state.initialize[Kaiming[]]()

    # Fixed training data: 4 board positions with clear targets
    # Obs: 27D = 3 planes × 9 (mine, opponent, empty)
    var obs_data = alloc[Scalar[dtype]](BATCH * OBS)
    var policy_data = alloc[Scalar[dtype]](BATCH * ACT)
    var value_data = alloc[Scalar[dtype]](BATCH)
    memset(obs_data, 0, BATCH * OBS)
    memset(policy_data, 0, BATCH * ACT)

    # Sample 0: Empty board, X to play center (action 4)
    # Plane 2 (empty) = all 1s
    for i in range(9):
        obs_data[0 * OBS + 18 + i] = Scalar[dtype](1.0)
    policy_data[0 * ACT + 4] = Scalar[dtype](1.0)  # Center
    value_data[0] = Scalar[dtype](1.0)  # X wins

    # Sample 1: X in center, O's turn, play corner (action 0)
    obs_data[1 * OBS + 4] = Scalar[dtype](1.0)  # X in center (opponent plane for O)
    for i in range(9):
        if i != 4:
            obs_data[1 * OBS + 18 + i] = Scalar[dtype](1.0)  # empty
    policy_data[1 * ACT + 0] = Scalar[dtype](1.0)  # Corner
    value_data[1] = Scalar[dtype](-1.0)  # O loses (from O's perspective)

    # Sample 2: X center + corner, O has 1, X to play
    obs_data[2 * OBS + 0] = Scalar[dtype](1.0)  # X corner (mine)
    obs_data[2 * OBS + 4] = Scalar[dtype](1.0)  # X center (mine)
    obs_data[2 * OBS + 9 + 2] = Scalar[dtype](1.0)  # O at 2 (opponent)
    for i in range(9):
        if i != 0 and i != 4 and i != 2:
            obs_data[2 * OBS + 18 + i] = Scalar[dtype](1.0)
    # Spread policy: prefer 8 (opposite corner)
    policy_data[2 * ACT + 8] = Scalar[dtype](0.7)
    policy_data[2 * ACT + 6] = Scalar[dtype](0.3)
    value_data[2] = Scalar[dtype](1.0)

    # Sample 3: Almost full board, draw
    for i in range(4):
        obs_data[3 * OBS + i] = Scalar[dtype](1.0)
    for i in range(4):
        obs_data[3 * OBS + 9 + i + 4] = Scalar[dtype](1.0)
    obs_data[3 * OBS + 18 + 8] = Scalar[dtype](1.0)  # only cell 8 empty
    policy_data[3 * ACT + 8] = Scalar[dtype](1.0)  # forced move
    value_data[3] = Scalar[dtype](0.0)  # draw

    # Forward pass to check initial logits
    var pred_data = alloc[Scalar[dtype]](BATCH * PRED_OUT)
    var cache_data = alloc[Scalar[dtype]](BATCH * PredModel.CACHE_SIZE)
    memset(pred_data, 0, BATCH * PRED_OUT)
    memset(cache_data, 0, BATCH * PredModel.CACHE_SIZE)

    var obs_t = LayoutTensor[dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin](obs_data)
    var pred_t = LayoutTensor[dtype, Layout.row_major(BATCH, PRED_OUT), MutAnyOrigin](pred_data)
    var cache_t = LayoutTensor[dtype, Layout.row_major(BATCH, PredModel.CACHE_SIZE), MutAnyOrigin](cache_data)

    # Print first few weights to check init diversity
    var params_ptr = state.params
    print("First 16 weights (layer 0, MatMul[27,128]):")
    print("  ", end="")
    for i in range(16):
        var v = Float64(params_ptr[i])
        print(Float64(Int(v * 10000)) / 10000.0, end=" ")
    print()
    # Check weight diversity: count unique values in first 128 weights
    var unique_count = 0
    for i in range(128):
        var is_unique = True
        for j in range(i):
            var diff = Float64(params_ptr[i]) - Float64(params_ptr[j])
            if diff < 1e-6 and diff > -1e-6:
                is_unique = False
                break
        if is_unique:
            unique_count += 1
    print("Unique values in first 128 weights:", unique_count, "(expect ~128, got", unique_count, ")")

    PredModel.forward[BATCH](obs_t, pred_t, state.params_view(), state.model_state_view(), cache_t)

    # Print hidden features for samples 0 and 1 to check feature collapse
    print("First 8 hidden features (layer 0 ReLU output):")
    print("  Sample 0:", end="")
    for j in range(8):
        var v = Float64(cache_data[0 * PredModel.CACHE_SIZE + 27 + j])
        print(Float64(Int(v * 10000)) / 10000.0, end=" ")
    print()
    print("  Sample 1:", end="")
    for j in range(8):
        var v = Float64(cache_data[1 * PredModel.CACHE_SIZE + 27 + j])
        print(Float64(Int(v * 10000)) / 10000.0, end=" ")
    print()

    var init_loss = compute_loss(pred_data, policy_data, value_data)
    var init_entropy = compute_policy_entropy(pred_data)
    print("Initial loss:", Float64(Int(init_loss * 10000)) / 10000.0)
    print("Initial entropy:", Float64(Int(init_entropy * 10000)) / 10000.0, "(ln9 =", Float64(Int(log(9.0) * 10000)) / 10000.0, ")")
    print_logits(pred_data, 0)
    print_logits(pred_data, 1)

    # Check if logits are literally identical
    var logit_range: Float64 = 0.0
    for b in range(BATCH):
        var min_l: Float64 = 1e18
        var max_l: Float64 = -1e18
        for a in range(ACT):
            var v = Float64(pred_data[b * PRED_OUT + a])
            if v < min_l:
                min_l = v
            if v > max_l:
                max_l = v
        logit_range += max_l - min_l
    logit_range /= Float64(BATCH)
    print("Avg logit range:", Float64(Int(logit_range * 10000)) / 10000.0)

    # Check hidden activations (fraction of alive neurons)
    # Cache for FusedMBR[27,128] = 27 (input) + 128 (relu output)
    # L0 = FusedMBR[27,128] -> CACHE = 27 + 128 = 155
    # L1 = FusedMBR[128,128] -> CACHE = 128 + 128 = 256
    comptime L0_CS = 155
    comptime L1_CS = 256
    var alive_count = 0
    var total_neurons = 0
    for b in range(BATCH):
        for j in range(128):
            total_neurons += 1
            # ReLU output cached at offset 27 within L0's cache region
            var val = Float64(rebind[Scalar[dtype]](cache_data[b * PredModel.CACHE_SIZE + 27 + j]))
            if val > 0:
                alive_count += 1
    print("Layer 0 alive neurons:", alive_count, "/", total_neurons,
          "=", Float64(Int(Float64(alive_count) / Float64(total_neurons) * 10000)) / 100.0, "%")

    alive_count = 0
    total_neurons = 0
    for b in range(BATCH):
        for j in range(128):
            total_neurons += 1
            # L1 cache starts at L0_CS, input part is 128, then ReLU output is 128
            var val = Float64(rebind[Scalar[dtype]](cache_data[b * PredModel.CACHE_SIZE + L0_CS + 128 + j]))
            if val > 0:
                alive_count += 1
    print("Layer 1 alive neurons:", alive_count, "/", total_neurons,
          "=", Float64(Int(Float64(alive_count) / Float64(total_neurons) * 10000)) / 100.0, "%")
    print()

    # Train for 200 steps
    print("Training 200 steps (CPU)...")
    var grad_data = alloc[Scalar[dtype]](BATCH * PRED_OUT)
    for step in range(200):
        memset(pred_data, 0, BATCH * PRED_OUT)
        memset(cache_data, 0, BATCH * PredModel.CACHE_SIZE)

        # Forward with cache
        PredModel.forward[BATCH](obs_t, pred_t, state.params_view(), state.model_state_view(), cache_t)

        # Compute output gradient (same as az_policy_value_grad_kernel)
        var inv_batch = Scalar[dtype](1.0) / Scalar[dtype](BATCH)
        for b in range(BATCH):
            var pred_off = b * PRED_OUT
            var pol_off = b * ACT
            # Policy: softmax - target
            var max_logit = pred_data[pred_off]
            for a in range(1, ACT):
                if pred_data[pred_off + a] > max_logit:
                    max_logit = pred_data[pred_off + a]
            var sum_exp = Scalar[dtype](0.0)
            for a in range(ACT):
                sum_exp += exp(pred_data[pred_off + a] - max_logit)
            for a in range(ACT):
                var prob = exp(pred_data[pred_off + a] - max_logit) / sum_exp
                var target = policy_data[pol_off + a]
                grad_data[pred_off + a] = (prob - target) * inv_batch
            # Value: MSE through tanh
            var raw_v = pred_data[pred_off + ACT]
            var tv = value_data[b]
            var ev_p = exp(raw_v)
            var ev_n = exp(-raw_v)
            var tanh_v = (ev_p - ev_n) / (ev_p + ev_n)
            var dtanh = Scalar[dtype](1.0) - tanh_v * tanh_v
            grad_data[pred_off + ACT] = Scalar[dtype](2.0) * (tanh_v - tv) * dtanh * inv_batch

        # Backward
        state.zero_grads()
        var grad_out_t = LayoutTensor[dtype, Layout.row_major(BATCH, PRED_OUT), MutAnyOrigin](grad_data)
        var grad_in_data = alloc[Scalar[dtype]](BATCH * OBS)
        memset(grad_in_data, 0, BATCH * OBS)
        var grad_in_t = LayoutTensor[dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin](grad_in_data)
        var grads_v = state.grads_view()
        PredModel.backward[BATCH](grad_out_t, grad_in_t, state.params_view(), state.model_state_view(), cache_t, grads_v)
        grad_in_data.free()

        # Check grad norms per layer
        if step == 0 or step == 99 or step == 199:
            # Layer 0: params 0 .. 27*128+128 = 3584
            comptime L0_PS = 27 * 128 + 128  # 3584
            comptime L1_PS = 128 * 128 + 128  # 16512
            var grads_ptr = state.grads
            var l0_gnorm: Float64 = 0.0
            for i in range(L0_PS):
                var g = Float64(grads_ptr[i])
                l0_gnorm += g * g
            l0_gnorm = sqrt(l0_gnorm)

            # Layer 1 starts after layer 0 (with alignment)
            # _seq_align4(3584) = 3584 (already aligned)
            comptime L1_OFF = 3584
            var l1_gnorm: Float64 = 0.0
            for i in range(L1_PS):
                var g = Float64(grads_ptr[L1_OFF + i])
                l1_gnorm += g * g
            l1_gnorm = sqrt(l1_gnorm)

            # Head params after L1
            comptime HEAD_OFF = L1_OFF + L1_PS  # Need aligned offset
            var head_gnorm: Float64 = 0.0
            comptime HEAD_PS = 128 * 9 + 9 + 128 * 1 + 1  # 1290
            for i in range(HEAD_PS):
                var g = Float64(grads_ptr[HEAD_OFF + i])
                head_gnorm += g * g
            head_gnorm = sqrt(head_gnorm)

            print("  Step", step, "| L0 grad_norm:", Float64(Int(l0_gnorm * 10000)) / 10000.0,
                  "| L1 grad_norm:", Float64(Int(l1_gnorm * 10000)) / 10000.0,
                  "| Head grad_norm:", Float64(Int(head_gnorm * 10000)) / 10000.0)

        # Optimizer step
        state.optimizer_step()

        if step % 50 == 0 or step == 199:
            var loss = compute_loss(pred_data, policy_data, value_data)
            var entropy = compute_policy_entropy(pred_data)
            print("  Step", step, "| loss:", Float64(Int(loss * 10000)) / 10000.0,
                  "| entropy:", Float64(Int(entropy * 10000)) / 10000.0)
            if step == 199:
                print_logits(pred_data, 0)
                print_logits(pred_data, 1)

    # Final check
    memset(pred_data, 0, BATCH * PRED_OUT)
    PredModel.forward[BATCH](obs_t, pred_t, state.params_view(), state.model_state_view())
    var final_loss = compute_loss(pred_data, policy_data, value_data)
    var final_entropy = compute_policy_entropy(pred_data)
    print()
    print("Initial loss:", Float64(Int(init_loss * 10000)) / 10000.0,
          "| Final loss:", Float64(Int(final_loss * 10000)) / 10000.0)
    print("Initial entropy:", Float64(Int(init_entropy * 10000)) / 10000.0,
          "| Final entropy:", Float64(Int(final_entropy * 10000)) / 10000.0)

    if final_loss < init_loss * 0.5:
        print("PASS: Network learned (loss decreased by >50%)")
    elif final_loss < init_loss * 0.9:
        print("PARTIAL: Network learned somewhat (loss decreased by >10%)")
    else:
        print("FAIL: Network did not learn (loss barely decreased)")

    print_logits(pred_data, 0)
    print_logits(pred_data, 1)

    # Cleanup
    grad_data.free()
    obs_data.free()
    policy_data.free()
    value_data.free()
    pred_data.free()
    cache_data.free()


def main() raises:
    test_cpu_training()
