"""Regression test for MuZero two-player sign flipping in n-step value targets.

Verifies that `nstep_value_targets_kernel` (kernels.mojo) correctly applies
the muzero-general sign-flipping convention:
  - rewards stored in the step's player-to-move perspective
  - bootstrap value stored in the bootstrap step's player-to-move perspective
  - value target is computed in the *base step's* player-to-move perspective
  - any reward/bootstrap from a different player has its sign negated
Reference: muzero-general/replay_buffer.py:242-259.

Two cases are checked end-to-end against hand-computed expectations:

  Case A — single-player (to_play all zeros): no flip applied; the kernel
  must produce standard n-step bootstrap.

  Case B — two-player turn-taking (to_play alternates [0,1,0,1]): flip
  applied wherever the step's player differs from the base step's player.

Test config: BATCH=2, K=3, N=2, gamma=0.99.
"""

from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.deep_agents.muzero.kernels import nstep_value_targets_kernel


def main() raises:
    print("=== MuZero n-step sign-flip regression ===")

    var ctx = DeviceContext()

    comptime BATCH = 2
    comptime K = 3
    comptime N = 2
    comptime TPB = 32
    comptime BLOCKS = (BATCH * (K + 1) + TPB - 1) // TPB
    comptime gamma = Float64(0.99)

    # ── Hand-built batch ─────────────────────────────────────────────
    # b=0: rewards=[0.5, 1.0, -0.5], dones=[0,0,0], values=[10,20,30,40]
    # b=1: rewards=[2.0,-1.0,  0.0], dones=[0,1,0], values=[ 5,15,25,35]
    # batch_rewards layout: [b * K + k]
    # batch_values  layout: [b * (K+1) + k]

    var rew_buf = ctx.enqueue_create_buffer[dtype](BATCH * K)
    var don_buf = ctx.enqueue_create_buffer[dtype](BATCH * K)
    var val_buf = ctx.enqueue_create_buffer[dtype](BATCH * (K + 1))
    var tp_buf = ctx.enqueue_create_buffer[DType.uint8](BATCH * (K + 1))

    var rew_host = ctx.enqueue_create_host_buffer[dtype](BATCH * K)
    var don_host = ctx.enqueue_create_host_buffer[dtype](BATCH * K)
    var val_host = ctx.enqueue_create_host_buffer[dtype](BATCH * (K + 1))
    var tp_host = ctx.enqueue_create_host_buffer[DType.uint8](
        BATCH * (K + 1)
    )

    # b=0 rewards
    rew_host[0 * K + 0] = Scalar[dtype](0.5)
    rew_host[0 * K + 1] = Scalar[dtype](1.0)
    rew_host[0 * K + 2] = Scalar[dtype](-0.5)
    # b=0 dones
    don_host[0 * K + 0] = Scalar[dtype](0.0)
    don_host[0 * K + 1] = Scalar[dtype](0.0)
    don_host[0 * K + 2] = Scalar[dtype](0.0)
    # b=0 values (K+1=4 entries)
    val_host[0 * (K + 1) + 0] = Scalar[dtype](10.0)
    val_host[0 * (K + 1) + 1] = Scalar[dtype](20.0)
    val_host[0 * (K + 1) + 2] = Scalar[dtype](30.0)
    val_host[0 * (K + 1) + 3] = Scalar[dtype](40.0)

    # b=1 rewards
    rew_host[1 * K + 0] = Scalar[dtype](2.0)
    rew_host[1 * K + 1] = Scalar[dtype](-1.0)
    rew_host[1 * K + 2] = Scalar[dtype](0.0)
    # b=1 dones — terminal at k=1
    don_host[1 * K + 0] = Scalar[dtype](0.0)
    don_host[1 * K + 1] = Scalar[dtype](1.0)
    don_host[1 * K + 2] = Scalar[dtype](0.0)
    # b=1 values
    val_host[1 * (K + 1) + 0] = Scalar[dtype](5.0)
    val_host[1 * (K + 1) + 1] = Scalar[dtype](15.0)
    val_host[1 * (K + 1) + 2] = Scalar[dtype](25.0)
    val_host[1 * (K + 1) + 3] = Scalar[dtype](35.0)

    # ── Output buffers ────────────────────────────────────────────────
    # value_targets layout: [(K+1) * BATCH] indexed [k * BATCH + b]
    # reward_targets layout: [K * BATCH] indexed [k * BATCH + b]
    var vt_buf = ctx.enqueue_create_buffer[dtype]((K + 1) * BATCH)
    var rt_buf = ctx.enqueue_create_buffer[dtype](K * BATCH)
    var vt_host = ctx.enqueue_create_host_buffer[dtype]((K + 1) * BATCH)
    var rt_host = ctx.enqueue_create_host_buffer[dtype](K * BATCH)

    var rew_t = LayoutTensor[
        dtype, Layout.row_major(BATCH * K), MutAnyOrigin
    ](rew_buf.unsafe_ptr())
    var don_t = LayoutTensor[
        dtype, Layout.row_major(BATCH * K), MutAnyOrigin
    ](don_buf.unsafe_ptr())
    var val_t = LayoutTensor[
        dtype, Layout.row_major(BATCH * (K + 1)), MutAnyOrigin
    ](val_buf.unsafe_ptr())
    var tp_t = LayoutTensor[
        DType.uint8, Layout.row_major(BATCH * (K + 1)), MutAnyOrigin
    ](tp_buf.unsafe_ptr())
    var vt_t = LayoutTensor[
        dtype, Layout.row_major((K + 1) * BATCH), MutAnyOrigin
    ](vt_buf.unsafe_ptr())
    var rt_t = LayoutTensor[
        dtype, Layout.row_major(K * BATCH), MutAnyOrigin
    ](rt_buf.unsafe_ptr())

    var num_failures = 0

    # ───────────────────────────────────────────────────────────────────
    # Case A: single-player (to_play all zeros)
    # ───────────────────────────────────────────────────────────────────
    print()
    print("--- Case A: to_play all zeros (single-player, no flip) ---")
    for i in range(BATCH * (K + 1)):
        tp_host[i] = UInt8(0)

    ctx.enqueue_copy(rew_buf, rew_host)
    ctx.enqueue_copy(don_buf, don_host)
    ctx.enqueue_copy(val_buf, val_host)
    ctx.enqueue_copy(tp_buf, tp_host)

    @parameter
    @always_inline
    def run_kernel_a(
        vt: LayoutTensor[
            dtype, Layout.row_major((K + 1) * BATCH), MutAnyOrigin
        ],
        rt: LayoutTensor[dtype, Layout.row_major(K * BATCH), MutAnyOrigin],
        rw: LayoutTensor[dtype, Layout.row_major(BATCH * K), MutAnyOrigin],
        dn: LayoutTensor[dtype, Layout.row_major(BATCH * K), MutAnyOrigin],
        bv: LayoutTensor[
            dtype, Layout.row_major(BATCH * (K + 1)), MutAnyOrigin
        ],
        tp: LayoutTensor[
            DType.uint8, Layout.row_major(BATCH * (K + 1)), MutAnyOrigin
        ],
        g: Scalar[dtype],
    ):
        nstep_value_targets_kernel[BATCH, K, N, dtype](
            vt, rt, rw, dn, bv, tp, g
        )

    ctx.enqueue_function[run_kernel_a, run_kernel_a](
        vt_t,
        rt_t,
        rew_t,
        don_t,
        val_t,
        tp_t,
        Scalar[dtype](gamma),
        grid_dim=(BLOCKS,),
        block_dim=(TPB,),
    )

    ctx.enqueue_copy(vt_host, vt_buf)
    ctx.enqueue_copy(rt_host, rt_buf)
    ctx.synchronize()

    # Hand-computed expected values for Case A:
    # b=0:
    #   k=0: 0.5 + 0.99*1.0 + 0.99^2 * 30 = 0.5 + 0.99 + 29.403 = 30.893
    #   k=1: 1.0 + 0.99*(-0.5) + 0.99^2 * 40 = 1.0 - 0.495 + 39.204 = 39.709
    #   k=2: -0.5 (only 1 step available, no bootstrap)
    #   k=3: 0.0 (no steps available)
    # b=1:
    #   k=0: 2.0 + 0.99*(-1.0) [hit terminal] = 1.01
    #   k=1: -1.0 [hit terminal at i=0]
    #   k=2: 0.0 (only 1 step available, no bootstrap)
    #   k=3: 0.0
    var expected_a = List[Float64]()
    expected_a.append(30.893)  # k=0, b=0
    expected_a.append(1.01)    # k=0, b=1
    expected_a.append(39.709)  # k=1, b=0
    expected_a.append(-1.0)    # k=1, b=1
    expected_a.append(-0.5)    # k=2, b=0
    expected_a.append(0.0)     # k=2, b=1
    expected_a.append(0.0)     # k=3, b=0
    expected_a.append(0.0)     # k=3, b=1

    var labels = List[String]()
    labels.append("k=0 b=0")
    labels.append("k=0 b=1")
    labels.append("k=1 b=0")
    labels.append("k=1 b=1")
    labels.append("k=2 b=0")
    labels.append("k=2 b=1")
    labels.append("k=3 b=0")
    labels.append("k=3 b=1")

    for i in range((K + 1) * BATCH):
        var got = Float64(vt_host[i])
        var exp = expected_a[i]
        var diff = got - exp
        if diff < 0.0:
            diff = -diff
        var ok = diff < 1e-3
        var status = "PASS" if ok else "FAIL"
        print(
            status,
            " ",
            labels[i],
            " : got ",
            got,
            " expected ",
            exp,
            " diff ",
            diff,
        )
        if not ok:
            num_failures += 1

    # Reward targets should equal raw rewards (no transform here).
    var expected_rew = List[Float64]()
    expected_rew.append(0.5)   # k=0, b=0
    expected_rew.append(2.0)   # k=0, b=1
    expected_rew.append(1.0)   # k=1, b=0
    expected_rew.append(-1.0)  # k=1, b=1
    expected_rew.append(-0.5)  # k=2, b=0
    expected_rew.append(0.0)   # k=2, b=1
    print()
    print("Reward targets (Case A):")
    for i in range(K * BATCH):
        var got = Float64(rt_host[i])
        var exp = expected_rew[i]
        var diff = got - exp
        if diff < 0.0:
            diff = -diff
        var ok = diff < 1e-3
        var status = "PASS" if ok else "FAIL"
        print(status, " idx ", i, " got ", got, " expected ", exp)
        if not ok:
            num_failures += 1

    # ───────────────────────────────────────────────────────────────────
    # Case B: two-player turn-taking (to_play alternates 0,1,0,1)
    # Same data; only to_play changes. Hand-computed sign flips:
    #
    # b=0 (perspective alternates by k):
    #   k=0 perspective=0: r[0] (player 0, keep) + γ·r[1] (player 1, FLIP)
    #                      + γ²·v[2] (player 0, keep)
    #     = 0.5 - 0.99*1.0 + 0.99² * 30 = 0.5 - 0.99 + 29.403 = 28.913
    #   k=1 perspective=1: r[1] (player 1, keep) + γ·r[2] (player 0, FLIP)
    #                      + γ²·v[3] (player 1, keep)
    #     = 1.0 + 0.99*0.5 + 0.99² * 40 = 1.0 + 0.495 + 39.204 = 40.699
    #   k=2 perspective=0: r[2] (player 0, keep). 1 step, no bootstrap.
    #     = -0.5
    #   k=3 perspective=1: 0 steps, no bootstrap.
    #     = 0.0
    # b=1 (perspective alternates by k):
    #   k=0 perspective=0: r[0] (player 0) + γ·r[1] (player 1, FLIP) [terminal]
    #     = 2.0 + 0.99*1.0 = 2.99
    #   k=1 perspective=1: r[1] (player 1) [terminal]
    #     = -1.0
    #   k=2 perspective=0: r[2] (player 0). 1 step, no bootstrap.
    #     = 0.0
    #   k=3 perspective=1: 0 steps.
    #     = 0.0
    # ───────────────────────────────────────────────────────────────────
    print()
    print("--- Case B: to_play alternating (two-player, sign flip) ---")
    for b in range(BATCH):
        for k in range(K + 1):
            tp_host[b * (K + 1) + k] = UInt8(k & 1)

    ctx.enqueue_copy(tp_buf, tp_host)

    @parameter
    @always_inline
    def run_kernel_b(
        vt: LayoutTensor[
            dtype, Layout.row_major((K + 1) * BATCH), MutAnyOrigin
        ],
        rt: LayoutTensor[dtype, Layout.row_major(K * BATCH), MutAnyOrigin],
        rw: LayoutTensor[dtype, Layout.row_major(BATCH * K), MutAnyOrigin],
        dn: LayoutTensor[dtype, Layout.row_major(BATCH * K), MutAnyOrigin],
        bv: LayoutTensor[
            dtype, Layout.row_major(BATCH * (K + 1)), MutAnyOrigin
        ],
        tp: LayoutTensor[
            DType.uint8, Layout.row_major(BATCH * (K + 1)), MutAnyOrigin
        ],
        g: Scalar[dtype],
    ):
        nstep_value_targets_kernel[BATCH, K, N, dtype](
            vt, rt, rw, dn, bv, tp, g
        )

    ctx.enqueue_function[run_kernel_b, run_kernel_b](
        vt_t,
        rt_t,
        rew_t,
        don_t,
        val_t,
        tp_t,
        Scalar[dtype](gamma),
        grid_dim=(BLOCKS,),
        block_dim=(TPB,),
    )

    ctx.enqueue_copy(vt_host, vt_buf)
    ctx.enqueue_copy(rt_host, rt_buf)
    ctx.synchronize()

    var expected_b = List[Float64]()
    expected_b.append(28.913)  # k=0, b=0
    expected_b.append(2.99)    # k=0, b=1
    expected_b.append(40.699)  # k=1, b=0
    expected_b.append(-1.0)    # k=1, b=1
    expected_b.append(-0.5)    # k=2, b=0
    expected_b.append(0.0)     # k=2, b=1
    expected_b.append(0.0)     # k=3, b=0
    expected_b.append(0.0)     # k=3, b=1

    for i in range((K + 1) * BATCH):
        var got = Float64(vt_host[i])
        var exp = expected_b[i]
        var diff = got - exp
        if diff < 0.0:
            diff = -diff
        var ok = diff < 1e-3
        var status = "PASS" if ok else "FAIL"
        print(
            status,
            " ",
            labels[i],
            " : got ",
            got,
            " expected ",
            exp,
            " diff ",
            diff,
        )
        if not ok:
            num_failures += 1

    print()
    if num_failures == 0:
        print("=== ALL TESTS PASSED ===")
    else:
        print("=== ", num_failures, " ASSERTIONS FAILED ===")
        raise Error("regression test failed")
