"""C.2 — n-step buffer unit tests.

Verifies `NStepBuffer` (CPU) and `GPUNStepBuffer` against hand-computed
n-step return formulas. Both expected behaviours:

  - Ring fills with the first N − 1 transitions (no emit).
  - On the N-th transition (no done): emits `(s_0, a_0, R_n,
    s_N, done=False)` where `R_n = r_0 + γr_1 + … + γ^{N−1}r_{N−1}`.
  - On `done=True` with `k < N` transitions buffered: emits
    `(s_0, a_0, R_k, s_k, done=True)` where `R_k = r_0 + γr_1 + …
    + γ^{k−1}r_{k−1}` and ring resets.
  - GPU `out_*` after `process` matches the equivalent CPU result.

Run:
    pixi run -e apple mojo run -I . tests/nn/test_n_step_replay.mojo
"""

from max.gpu.host import DeviceContext
from std.math import abs as fabs
from std.memory import alloc
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.data.n_step_replay import (
    NStepBuffer, GPUNStepBuffer,
)


comptime N = 3
comptime OBS = 2
comptime ACT = 1
comptime GAMMA: Float64 = 0.99


def _approx(a: Scalar[DT], b: Float64) -> Bool:
    var diff = fabs(Float64(a) - b)
    return diff < 1e-4


# ──────────────────────────────────────────────────────────────────────
# CPU NStepBuffer tests.
# ──────────────────────────────────────────────────────────────────────


def test_cpu_no_emit_until_n() raises:
    var nb = NStepBuffer[N, OBS, ACT].new(gamma=Scalar[DT](GAMMA))
    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var act = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    var nxt = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))

    # Add N − 1 transitions; expect no emit.
    for i in range(N - 1):
        obs[0] = Scalar[DT](Float64(i))
        obs[1] = Scalar[DT](Float64(i) + 0.1)
        act[0] = Scalar[DT](Float64(i) * 10.0)
        nxt[0] = Scalar[DT](Float64(i) + 1.0)
        nxt[1] = Scalar[DT](Float64(i) + 1.1)
        var r = nb.add(obs, act, Scalar[DT](Float64(i) + 1.0), nxt, False)
        assert_true(not r.valid, "Should not emit before N transitions")
    print("  test_cpu_no_emit_until_n PASSED")


def test_cpu_full_ring_emit() raises:
    var nb = NStepBuffer[N, OBS, ACT].new(gamma=Scalar[DT](GAMMA))
    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var act = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    var nxt = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))

    # r_0=1.0, r_1=2.0, r_2=3.0. Expected R_n = 1 + 0.99*2 + 0.99^2 * 3
    #                                       = 1 + 1.98 + 2.9403 = 5.9203.
    var rewards = List[Float64]()
    rewards.append(1.0)
    rewards.append(2.0)
    rewards.append(3.0)

    var result_valid = False
    var result_reward = Scalar[DT](0.0)
    var result_obs0 = Scalar[DT](0.0)
    var result_action0 = Scalar[DT](0.0)
    var result_nxt0 = Scalar[DT](0.0)

    for i in range(N):
        obs[0] = Scalar[DT](Float64(i))
        obs[1] = Scalar[DT](Float64(i) + 0.1)
        act[0] = Scalar[DT](Float64(i) * 10.0)
        nxt[0] = Scalar[DT](Float64(i) + 1.0)
        nxt[1] = Scalar[DT](Float64(i) + 1.1)
        var r = nb.add(
            obs, act, Scalar[DT](rewards[i]), nxt, False,
        )
        if i == N - 1:
            result_valid = r.valid
            result_reward = r.reward
            result_obs0 = r.obs[0]
            result_action0 = r.action[0]
            result_nxt0 = r.next_obs[0]

    assert_true(result_valid, "Ring should emit on N-th transition")
    var expected_r = 1.0 + GAMMA * 2.0 + GAMMA * GAMMA * 3.0
    assert_true(
        _approx(result_reward, expected_r),
        "R_n mismatch: got " + String(Float64(result_reward))
        + " expected " + String(expected_r),
    )
    # s_0 was {0.0, 0.1}, a_0 was {0.0}, last next_obs was {3.0, 3.1}.
    assert_true(_approx(result_obs0, 0.0), "obs[0] mismatch")
    assert_true(_approx(result_action0, 0.0), "action[0] mismatch")
    assert_true(_approx(result_nxt0, 3.0), "next_obs[0] mismatch")
    print("  test_cpu_full_ring_emit PASSED (R_n=", result_reward, ")")


def test_cpu_done_flush_short() raises:
    """Push 2 transitions with done=True on the 2nd; expect a 2-step
    return flush with done=True."""
    var nb = NStepBuffer[N, OBS, ACT].new(gamma=Scalar[DT](GAMMA))
    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var act = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    var nxt = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))

    obs[0] = Scalar[DT](10.0)
    obs[1] = Scalar[DT](11.0)
    act[0] = Scalar[DT](100.0)
    nxt[0] = Scalar[DT](20.0)
    nxt[1] = Scalar[DT](21.0)
    var r1 = nb.add(obs, act, Scalar[DT](5.0), nxt, False)
    assert_true(not r1.valid, "First transition shouldn't emit")

    obs[0] = Scalar[DT](20.0)
    obs[1] = Scalar[DT](21.0)
    act[0] = Scalar[DT](200.0)
    nxt[0] = Scalar[DT](30.0)
    nxt[1] = Scalar[DT](31.0)
    var r2 = nb.add(obs, act, Scalar[DT](6.0), nxt, True)
    assert_true(r2.valid, "Done should flush")
    assert_true(r2.done, "Flush should carry done=True")
    var expected = 5.0 + GAMMA * 6.0
    assert_true(
        _approx(r2.reward, expected),
        "Done-flush R_k mismatch: got " + String(Float64(r2.reward))
        + " expected " + String(expected),
    )
    assert_true(_approx(r2.obs[0], 10.0), "obs[0] mismatch")
    assert_true(_approx(r2.action[0], 100.0), "action[0] mismatch")
    assert_true(_approx(r2.next_obs[0], 30.0), "next_obs[0] mismatch")
    # Buffer should be reset.
    assert_true(nb.count == 0, "Done should reset count")
    print("  test_cpu_done_flush_short PASSED (R_k=", r2.reward, ")")


# ──────────────────────────────────────────────────────────────────────
# GPU GPUNStepBuffer tests (parity vs CPU).
# ──────────────────────────────────────────────────────────────────────


def test_gpu_parity_with_cpu() raises:
    """Feed the same trajectory through CPU and GPU buffers, compare
    the emitted reward/obs/action/next_obs on the N-th step."""
    comptime N_ENVS = 2

    var ctx = DeviceContext()
    var nb_cpu = NStepBuffer[N, OBS, ACT].new(gamma=Scalar[DT](GAMMA))
    var nb_gpu = GPUNStepBuffer[N, OBS, ACT, N_ENVS].new(
        ctx, gamma=Scalar[DT](GAMMA),
    )

    # Per-step device buffers (env 0 and env 1 fed identical data).
    var dev_obs = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var dev_act = ctx.enqueue_create_buffer[DT](N_ENVS * ACT)
    var dev_rew = ctx.enqueue_create_buffer[DT](N_ENVS)
    var dev_nobs = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var dev_done = ctx.enqueue_create_buffer[DT](N_ENVS)

    var host_obs = alloc[Scalar[DT]](N_ENVS * OBS)
    var host_act = alloc[Scalar[DT]](N_ENVS * ACT)
    var host_rew = alloc[Scalar[DT]](N_ENVS)
    var host_nobs = alloc[Scalar[DT]](N_ENVS * OBS)
    var host_done = alloc[Scalar[DT]](N_ENVS)

    # CPU oracle takes Lists (NStepBuffer.add signature).
    var cpu_obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var cpu_act = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    var cpu_nxt = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))

    var cpu_emit_reward = Scalar[DT](0.0)
    for i in range(N):
        # Fill host scratch.
        for e in range(N_ENVS):
            host_obs[e * OBS + 0] = Scalar[DT](Float64(i))
            host_obs[e * OBS + 1] = Scalar[DT](Float64(i) + 0.1)
            host_act[e * ACT + 0] = Scalar[DT](Float64(i) * 10.0)
            host_rew[e] = Scalar[DT](Float64(i) + 1.0)
            host_nobs[e * OBS + 0] = Scalar[DT](Float64(i) + 1.0)
            host_nobs[e * OBS + 1] = Scalar[DT](Float64(i) + 1.1)
            host_done[e] = Scalar[DT](0.0)
        ctx.enqueue_copy(dev_obs, host_obs)
        ctx.enqueue_copy(dev_act, host_act)
        ctx.enqueue_copy(dev_rew, host_rew)
        ctx.enqueue_copy(dev_nobs, host_nobs)
        ctx.enqueue_copy(dev_done, host_done)
        nb_gpu.process(ctx, dev_obs, dev_act, dev_rew, dev_nobs, dev_done)

        cpu_obs[0] = Scalar[DT](Float64(i))
        cpu_obs[1] = Scalar[DT](Float64(i) + 0.1)
        cpu_act[0] = Scalar[DT](Float64(i) * 10.0)
        cpu_nxt[0] = Scalar[DT](Float64(i) + 1.0)
        cpu_nxt[1] = Scalar[DT](Float64(i) + 1.1)
        var r = nb_cpu.add(
            cpu_obs, cpu_act,
            Scalar[DT](Float64(i) + 1.0),
            cpu_nxt, False,
        )
        if r.valid:
            cpu_emit_reward = r.reward

    # D2H the GPU emit + valid flags.
    var h_out_rew = alloc[Scalar[DT]](N_ENVS)
    var h_out_valid = alloc[Scalar[DType.int32]](N_ENVS)
    var h_out_obs = alloc[Scalar[DT]](N_ENVS * OBS)
    var h_out_act = alloc[Scalar[DT]](N_ENVS * ACT)
    var h_out_nobs = alloc[Scalar[DT]](N_ENVS * OBS)
    ctx.enqueue_copy(h_out_rew, nb_gpu.out_rew)
    ctx.enqueue_copy(h_out_valid, nb_gpu.out_valid)
    ctx.enqueue_copy(h_out_obs, nb_gpu.out_obs)
    ctx.enqueue_copy(h_out_act, nb_gpu.out_act)
    ctx.enqueue_copy(h_out_nobs, nb_gpu.out_nobs)
    ctx.synchronize()

    for e in range(N_ENVS):
        assert_true(
            Int(h_out_valid[e]) == 1,
            "GPU env " + String(e) + " should have emitted by step N",
        )
        assert_true(
            _approx(h_out_rew[e], Float64(cpu_emit_reward)),
            "GPU env " + String(e) + " R_n mismatch with CPU: "
            + "gpu=" + String(Float64(h_out_rew[e]))
            + " cpu=" + String(Float64(cpu_emit_reward)),
        )
        assert_true(
            _approx(h_out_obs[e * OBS + 0], 0.0),
            "obs[0] mismatch env " + String(e),
        )
        assert_true(
            _approx(h_out_act[e * ACT + 0], 0.0),
            "action[0] mismatch env " + String(e),
        )
        # Last next_obs[0] was 3.0 (from i = N - 1 = 2 + 1).
        assert_true(
            _approx(h_out_nobs[e * OBS + 0], Float64(N)),
            "next_obs[0] mismatch env " + String(e),
        )

    print(
        "  test_gpu_parity_with_cpu PASSED (R_n=", cpu_emit_reward, ")",
    )


def main() raises:
    print("=" * 60)
    print("C.2 n-step buffer unit tests")
    print("=" * 60)
    test_cpu_no_emit_until_n()
    test_cpu_full_ring_emit()
    test_cpu_done_flush_short()
    test_gpu_parity_with_cpu()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
