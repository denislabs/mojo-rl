"""Test workspace and critic group prototypes."""

from std.testing import assert_equal, assert_true
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Linear, LinearReLU, Sequential
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.training import NetworkPair
from mojo_rl.deep_agents.core.workspace import (
    OffPolicyTrainWS,
    SampleBatch,
    ExplorationWS,
)
from mojo_rl.deep_agents.core.critic_group import CriticGroup


# Test dimensions (small for testing)
comptime OBS = 4
comptime ACT = 2
comptime ACTOR_OUT = 4  # 2*ACT for SAC-style
comptime CI = OBS + ACT  # 6
comptime CO = 1
comptime BS = 8
comptime HIDDEN = 16

# Test models
comptime ActorModel = Sequential[LinearReLU[OBS, HIDDEN], Linear[HIDDEN, ACTOR_OUT]]
comptime CriticModel = Sequential[LinearReLU[CI, HIDDEN], Linear[HIDDEN, CO]]
comptime CriticOpt = Adam[0.001]

comptime CCS = CriticModel.CACHE_SIZE
comptime ACS = ActorModel.CACHE_SIZE

# Workspace type aliases
comptime WS = OffPolicyTrainWS[BS, OBS, ACT, ACTOR_OUT, CI, CO, CCS, ACS, 0, 0, 2]
comptime WS_SINGLE = OffPolicyTrainWS[BS, OBS, ACT, ACTOR_OUT, CI, CO, CCS, ACS, 0, 0, 1]
comptime SB = SampleBatch[BS, OBS, ACT]
comptime EWS = ExplorationWS[64, ACTOR_OUT, 0]


def test_workspace_offsets() raises:
    """Verify workspace offsets are properly chained."""
    print("test_workspace_offsets...")

    # Offsets should be monotonically increasing
    assert_true(WS._O_NEXT_ACT == 0, "next_act starts at 0")
    assert_true(WS._O_NEXT_LP > WS._O_NEXT_ACT, "next_lp after next_act")
    assert_true(WS._O_NEXT_CI > WS._O_NEXT_LP, "next_ci after next_lp")
    assert_true(WS._O_NEXT_Q > WS._O_NEXT_CI, "next_q after next_ci")
    assert_true(WS._O_TARGETS > WS._O_NEXT_Q, "targets after next_q")
    assert_true(WS._O_CI > WS._O_TARGETS, "ci after targets")
    assert_true(WS._O_Q_OUTS > WS._O_CI, "q_outs after ci")
    assert_true(WS._O_Q_CACHES > WS._O_Q_OUTS, "q_caches after q_outs")
    assert_true(WS._O_Q_GRAD > WS._O_Q_CACHES, "q_grad after q_caches")
    assert_true(WS._O_D_CI > WS._O_Q_GRAD, "d_ci after q_grad")
    assert_true(WS.TOTAL_SIZE > 0, "total size positive")

    # Verify specific offset calculations
    assert_equal(WS._O_NEXT_LP, BS * ACT)  # after next_act
    assert_equal(WS._O_NEXT_CI, BS * ACT + BS)  # after next_lp

    print("  TOTAL_SIZE (2 critics):", WS.TOTAL_SIZE)
    print("  TOTAL_SIZE (1 critic):", WS_SINGLE.TOTAL_SIZE)
    assert_true(
        WS.TOTAL_SIZE > WS_SINGLE.TOTAL_SIZE,
        "twin critics need more space",
    )
    print("  PASSED")


def test_workspace_cpu_views() raises:
    """Test CPU workspace allocation and view access."""
    print("test_workspace_cpu_views...")

    var data = WS.alloc_cpu()
    var ws = WS(data.unsafe_ptr())

    # Write through a view using ptr access
    var next_act_t = ws.next_act()
    for b in range(BS):
        for a in range(ACT):
            next_act_t.ptr[b * ACT + a] = Scalar[dtype](Float64(b * ACT + a))

    # Read back through the same view
    var check = ws.next_act()
    for b in range(BS):
        for a in range(ACT):
            assert_equal(
                Float64(check.ptr[b * ACT + a]),
                Float64(b * ACT + a),
            )

    # Verify critic-indexed views don't overlap
    var q0 = ws.q_out(0)
    var q1 = ws.q_out(1)
    q0.ptr[0] = Scalar[dtype](42.0)
    q1.ptr[0] = Scalar[dtype](99.0)
    assert_equal(Float64(ws.q_out(0).ptr[0]), 42.0)
    assert_equal(Float64(ws.q_out(1).ptr[0]), 99.0)

    # Same for caches
    var c0 = ws.q_cache(0)
    var c1 = ws.q_cache(1)
    c0.ptr[0] = Scalar[dtype](1.0)
    c1.ptr[0] = Scalar[dtype](2.0)
    assert_equal(Float64(ws.q_cache(0).ptr[0]), 1.0)
    assert_equal(Float64(ws.q_cache(1).ptr[0]), 2.0)

    print("  PASSED")


def test_sample_batch() raises:
    """Test SampleBatch typed views."""
    print("test_sample_batch...")

    var data = SB.alloc_cpu()
    var sb = SB(data.unsafe_ptr())

    # Write obs
    var obs_t = sb.obs()
    for b in range(BS):
        for o in range(OBS):
            obs_t.ptr[b * OBS + o] = Scalar[dtype](Float64(b + o))

    # Read back
    for b in range(BS):
        for o in range(OBS):
            assert_equal(
                Float64(sb.obs().ptr[b * OBS + o]), Float64(b + o)
            )

    # Write rew (1D)
    var rew_t = sb.rew()
    for b in range(BS):
        rew_t.ptr[b] = Scalar[dtype](Float64(b) * 0.5)
    assert_equal(Float64(sb.rew().ptr[3]), 1.5)

    print("  TOTAL_SIZE:", SB.TOTAL_SIZE)
    assert_equal(SB.TOTAL_SIZE, BS * OBS + BS * ACT + BS + BS * OBS + BS)
    print("  PASSED")


def test_exploration_ws() raises:
    """Test ExplorationWS."""
    print("test_exploration_ws...")

    comptime EWS_SIZE = EWS.TOTAL_SIZE
    var data = List[Scalar[dtype]](capacity=EWS_SIZE)
    for _ in range(EWS_SIZE):
        data.append(Scalar[dtype](0))
    var ews = EWS(data.unsafe_ptr())

    # Write through raw_act view with specific N_ENVS
    var act_t = ews.raw_act[32]()
    act_t.ptr[0] = Scalar[dtype](1.5)
    assert_equal(Float64(ews.raw_act[32]().ptr[0]), 1.5)

    print("  TOTAL_SIZE:", EWS.TOTAL_SIZE)
    print("  PASSED")


def test_critic_group_cpu() raises:
    """Test CriticGroup with twin critics."""
    print("test_critic_group_cpu...")

    # Single critic
    var single = CriticGroup[CriticModel, CriticOpt, 1]()
    single.initialize[]()

    # Verify params are initialized (not all zeros)
    var p0 = single.online_params_view(0)
    var has_nonzero = False
    for i in range(CriticModel.PARAM_SIZE):
        if Float64(p0.ptr[i]) != 0.0:
            has_nonzero = True
            break
    assert_true(has_nonzero, "critic 0 should be initialized")

    # Twin critics
    var twin = CriticGroup[CriticModel, CriticOpt, 2]()
    twin.initialize[]()

    # Verify target == online after initialize (hard copy)
    var p_c0 = twin.online_params_view(0)
    var t_c0 = twin.target_params_view(0)
    for i in range(CriticModel.PARAM_SIZE):
        assert_equal(Float64(p_c0.ptr[i]), Float64(t_c0.ptr[i]))

    # Test batch operations
    twin.zero_all_grads()
    var g0 = twin.online_grads_view(0)
    var g1 = twin.online_grads_view(1)
    for i in range(CriticModel.PARAM_SIZE):
        assert_equal(Float64(g0.ptr[i]), 0.0)
        assert_equal(Float64(g1.ptr[i]), 0.0)

    # Test soft_update_all: tau=1.0 → target = online
    for i in range(CriticModel.PARAM_SIZE):
        twin.pairs[0].online.params[i] = Scalar[dtype](1.0)
        twin.pairs[1].online.params[i] = Scalar[dtype](2.0)

    twin.soft_update_all(1.0)
    assert_equal(Float64(twin.target_params_view(0).ptr[0]), 1.0)
    assert_equal(Float64(twin.target_params_view(1).ptr[0]), 2.0)

    print("  PASSED")


def test_unified_cpu_gpu_api() raises:
    """Demonstrate that workspace API is identical for CPU and GPU backing."""
    print("test_unified_cpu_gpu_api...")

    # CPU path
    var cpu_data = WS.alloc_cpu()
    var cpu_ws = WS(cpu_data.unsafe_ptr())
    cpu_ws.next_act().ptr[0] = Scalar[dtype](1.0)
    assert_equal(Float64(cpu_ws.next_act().ptr[0]), 1.0)

    # The GPU path would be identical:
    #   var gpu_buf = WS.alloc_gpu(ctx)
    #   var gpu_ws = WS(gpu_buf.unsafe_ptr())
    #   gpu_ws.next_act().ptr[0] = ...  # same API!
    #
    # Both paths use the same WS type, same view methods,
    # same compile-time offsets. Only the allocation differs.

    print("  PASSED (CPU only — GPU requires DeviceContext)")


def test_workspace_with_critic_group() raises:
    """Show workspace + critic group working together (the intended pattern)."""
    print("test_workspace_with_critic_group...")

    comptime NUM_CRITICS = 2
    comptime WSType = OffPolicyTrainWS[
        BS, OBS, ACT, ACTOR_OUT, CI, CO, CCS, ACS, 0, 0, NUM_CRITICS,
    ]

    var data = WSType.alloc_cpu()
    var ws = WSType(data.unsafe_ptr())
    var critics = CriticGroup[CriticModel, CriticOpt, NUM_CRITICS]()
    critics.initialize[]()

    # Fill ci with dummy data
    var ci_t = ws.ci()
    for b in range(BS):
        for c in range(CI):
            ci_t.ptr[b * CI + c] = Scalar[dtype](0.1)

    # Forward all critics through workspace views — no comptime if NUM_CRITICS == 2!
    comptime CriticNet = CriticModel
    for i in range(NUM_CRITICS):
        var q_t = ws.q_out(i)
        var p = critics.online_params_view(i)
        CriticNet.forward[BS](ci_t, q_t, p)

    # Verify each critic produced output
    for i in range(NUM_CRITICS):
        var has_output = False
        var q_t = ws.q_out(i)
        for b in range(BS):
            if Float64(q_t.ptr[b]) != 0.0:
                has_output = True
                break
        assert_true(has_output, "critic " + String(i) + " should produce output")

    # Soft update all — one line replaces 4 lines of comptime-if
    critics.soft_update_all(0.005)

    print("  PASSED")


def main() raises:
    test_workspace_offsets()
    test_workspace_cpu_views()
    test_sample_batch()
    test_exploration_ws()
    test_critic_group_cpu()
    test_unified_cpu_gpu_api()
    test_workspace_with_critic_group()
    print("\nAll workspace/critic_group tests passed!")
