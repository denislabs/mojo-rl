"""Test full SAC actor loss expressed via autodiff composition.

This test verifies that the entire SAC actor-critic loss graph can be
composed from primitives and produces correct gradients end-to-end.
"""

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import (
    Sequential,
    Linear,
    LinearReLU,
    LinearTanh,
    RSample,
    Min,
    Slice,
    Negate,
    Parallel,
    SkipConcat,
    DualPath,
    SplitApply,
)
from mojo_rl.nn.training import Network, NetworkState
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Xavier
from layout import Layout, LayoutTensor


fn test_sac_graph_shapes() raises:
    """Verify all compile-time shapes compose correctly."""
    comptime OBS = 17
    comptime ACT = 6
    comptime H = 64

    # Actor: obs → [mean || tanh(log_std)]
    comptime ActorModel = Sequential[
        LinearReLU[OBS, H],
        LinearReLU[H, H],
        Parallel[Linear[H, ACT], LinearTanh[H, ACT]],
    ]

    # Actor + RSample: obs → [action || log_prob]
    comptime ActorRSample = Sequential[ActorModel, RSample[ACT]]

    # SkipConcat: obs → [obs || action || log_prob]
    comptime ActorSkip = SkipConcat[ActorRSample]

    # Critic: [obs, action] → Q
    comptime CriticModel = Sequential[
        LinearReLU[OBS + ACT, H],
        LinearReLU[H, H],
        Linear[H, 1],
    ]

    # TwinCritic + Min: [obs, action] → min(Q1, Q2)
    comptime TwinCriticMin = Sequential[DualPath[CriticModel, CriticModel], Min[1]]

    # SplitApply: [obs(17), action(6), log_prob(1)] →
    #   Left([obs, action](23)) → TwinCriticMin → min_Q(1)
    #   Right([log_prob](1))    → Identity       → log_prob(1)
    # Output: [min_Q(1), log_prob(1)]
    comptime LogProbPass = Slice[1, 0, 1]  # Identity for dim=1
    comptime SACOutput = SplitApply[TwinCriticMin, LogProbPass, OBS + ACT]

    # Full graph: obs → ActorSkip → SACOutput → [min_Q, log_prob]
    comptime SACGraph = Sequential[ActorSkip, SACOutput]

    print("  SACGraph: IN=", SACGraph.IN_DIM, "OUT=", SACGraph.OUT_DIM,
          "PARAMS=", SACGraph.PARAM_SIZE)
    # Expected: IN=17 (obs), OUT=2 (min_Q + log_prob)

    # Verify shapes
    if SACGraph.IN_DIM != OBS:
        print("  [FAIL] IN_DIM should be", OBS, "got", SACGraph.IN_DIM)
        return
    if SACGraph.OUT_DIM != 2:
        print("  [FAIL] OUT_DIM should be 2 got", SACGraph.OUT_DIM)
        return

    print("  [PASS] SACGraph shapes correct: IN=17, OUT=2")


fn test_sac_graph_forward_backward() raises:
    """Test forward + backward through the full SAC graph."""
    comptime OBS = 4
    comptime ACT = 2
    comptime H = 16
    comptime BS = 8

    # Build the graph (smaller dims for testing)
    comptime ActorModel = Sequential[
        LinearReLU[OBS, H],
        LinearReLU[H, H],
        Parallel[Linear[H, ACT], LinearTanh[H, ACT]],
    ]
    comptime ActorRSample = Sequential[ActorModel, RSample[ACT]]
    comptime ActorSkip = SkipConcat[ActorRSample]
    comptime CriticModel = Sequential[
        LinearReLU[OBS + ACT, H],
        LinearReLU[H, H],
        Linear[H, 1],
    ]
    comptime TwinCriticMin = Sequential[DualPath[CriticModel, CriticModel], Min[1]]
    comptime LogProbPass = Slice[1, 0, 1]
    comptime SACOutput = SplitApply[TwinCriticMin, LogProbPass, OBS + ACT]
    comptime SACGraph = Sequential[ActorSkip, SACOutput]

    print("  SACGraph PARAM_SIZE:", SACGraph.PARAM_SIZE,
          "CACHE_SIZE:", SACGraph.CACHE_SIZE)

    # Initialize state
    var state = NetworkState[SACGraph, Adam[]]()
    state.initialize[Xavier[]]()

    var params = state.params_view()
    var grads = state.grads_view()

    # Create input (observations)
    var obs_arr = InlineArray[Scalar[dtype], BS * OBS](uninitialized=True)
    for i in range(BS * OBS):
        obs_arr[i] = Scalar[dtype](0.1 * Float64(i % 7) - 0.3)
    var obs_t = LayoutTensor[
        dtype, Layout.row_major(BS, OBS), MutAnyOrigin
    ](obs_arr.unsafe_ptr())

    # Forward
    var output_arr = InlineArray[Scalar[dtype], BS * 2](uninitialized=True)
    var output_t = LayoutTensor[
        dtype, Layout.row_major(BS, 2), MutAnyOrigin
    ](output_arr.unsafe_ptr())

    var cache_arr = InlineArray[Scalar[dtype], BS * SACGraph.CACHE_SIZE](
        uninitialized=True
    )
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BS, SACGraph.CACHE_SIZE), MutAnyOrigin
    ](cache_arr.unsafe_ptr())

    SACGraph.forward[BS](obs_t, output_t, params, cache_t)

    # Check output: [min_Q, log_prob] per sample
    var fwd_ok = True
    for b in range(BS):
        var min_q = Float64(output_arr[b * 2])
        var log_prob = Float64(output_arr[b * 2 + 1])
        if min_q != min_q or log_prob != log_prob:  # NaN check
            fwd_ok = False
            print("  [FAIL] NaN in output at b=", b, "min_q=", min_q, "lp=", log_prob)

    if fwd_ok:
        print("  [PASS] Forward: all outputs finite")

    # Backward with SAC-style gradient seed:
    # grad_output[:, 0] = -1/BS (maximize Q → minimize negative Q)
    # grad_output[:, 1] = alpha/BS (entropy regularization)
    var alpha: Float64 = 0.2
    var grad_out_arr = InlineArray[Scalar[dtype], BS * 2](uninitialized=True)
    for b in range(BS):
        grad_out_arr[b * 2] = Scalar[dtype](-1.0 / Float64(BS))
        grad_out_arr[b * 2 + 1] = Scalar[dtype](alpha / Float64(BS))
    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BS, 2), MutAnyOrigin
    ](grad_out_arr.unsafe_ptr())

    var grad_obs_arr = InlineArray[Scalar[dtype], BS * OBS](uninitialized=True)
    var grad_obs_t = LayoutTensor[
        dtype, Layout.row_major(BS, OBS), MutAnyOrigin
    ](grad_obs_arr.unsafe_ptr())

    state.zero_grads()
    SACGraph.backward[BS](grad_out_t, grad_obs_t, params, cache_t, grads)

    # Check gradients are finite and non-zero
    var grad_ok = True
    var any_nonzero = False
    for i in range(SACGraph.PARAM_SIZE):
        var g = Float64(grads.ptr[i])
        if g != g:
            grad_ok = False
            print("  [FAIL] NaN in param grad at index", i)
            break
        if g != 0.0:
            any_nonzero = True

    if grad_ok and any_nonzero:
        print("  [PASS] Backward: param gradients finite and non-zero")
    elif not any_nonzero:
        print("  [FAIL] All param gradients are zero")

    # Verify actor and critic grads are both present
    # Actor params are first, then twin critic params
    comptime ACTOR_PS = ActorModel.PARAM_SIZE
    comptime CRITIC_PS = CriticModel.PARAM_SIZE
    var actor_nonzero = 0
    for i in range(ACTOR_PS):
        if Float64(grads.ptr[i]) != 0.0:
            actor_nonzero += 1
    var critic_nonzero = 0
    for i in range(2 * CRITIC_PS):
        if Float64(grads.ptr[ACTOR_PS + i]) != 0.0:
            critic_nonzero += 1

    print("  Actor grads non-zero:", actor_nonzero, "/", ACTOR_PS)
    print("  Critic grads non-zero:", critic_nonzero, "/", 2 * CRITIC_PS)

    if actor_nonzero > 0 and critic_nonzero > 0:
        print("  [PASS] Both actor and critic receive gradients!")
    else:
        print("  [FAIL] Missing gradients in actor or critic")


fn main() raises:
    print("=== SAC Autodiff Composition Tests ===")
    test_sac_graph_shapes()
    print()
    test_sac_graph_forward_backward()
    print()
    print("=== Done ===")
