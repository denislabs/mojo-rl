"""Regression test for MuZero n-step value target kernel.

Verifies:
  - TIME-MAJOR layout (post P-LAYOUT): tensors indexed [t * BATCH + b].
  - Full N-step bootstrap when window covers K+N timesteps (post P-WINDOW):
    bootstrap fires for every base k in [0, K] regardless of K vs N.
  - Two-player sign flip (P0): rewards/bootstrap from a different player
    are negated. Reference: muzero-general/replay_buffer.py:242-259.

Two cases hand-computed against the kernel output:

  Case A — to_play all zeros (single-player): no flip.
  Case B — to_play alternates [0,1,0,1,0,1] (two-player turn-taking).

Test config: BATCH=2, K=3, N=2, gamma=0.99, window length K+N+1 = 6.
"""

from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.deep_agents.muzero.kernels import nstep_value_targets_kernel


def main() raises:
    print("=== MuZero n-step sign-flip regression (time-major) ===")

    var ctx = DeviceContext()

    comptime BATCH = 2
    comptime K = 3
    comptime N = 2
    comptime WIN_FULL = K + N + 1  # 6: values/to_play
    comptime WIN_TRN = K + N  # 5: rewards/dones
    comptime TPB = 32
    comptime BLOCKS = (BATCH * (K + 1) + TPB - 1) // TPB
    comptime gamma = Float64(0.99)

    # ── Hand-built batch (TIME-MAJOR: index = t * BATCH + b) ─────────
    # b=0: rewards=[0.5, 1.0, -0.5, 0.0, 0.0]   (5)
    #      dones  =[0,   0,    0,   0,   0]
    #      values =[10, 20, 30, 40, 50, 60]      (6)
    # b=1: rewards=[2.0, -1.0, 0.0, 0.0, 0.0]
    #      dones  =[0,    1,   0,   0,   0]      (terminal at t=1)
    #      values =[5, 15, 25, 35, 45, 55]

    var rew_buf = ctx.enqueue_create_buffer[dtype](WIN_TRN * BATCH)
    var don_buf = ctx.enqueue_create_buffer[dtype](WIN_TRN * BATCH)
    var val_buf = ctx.enqueue_create_buffer[dtype](WIN_FULL * BATCH)
    var tp_buf = ctx.enqueue_create_buffer[DType.uint8](WIN_FULL * BATCH)

    var rew_host = ctx.enqueue_create_host_buffer[dtype](WIN_TRN * BATCH)
    var don_host = ctx.enqueue_create_host_buffer[dtype](WIN_TRN * BATCH)
    var val_host = ctx.enqueue_create_host_buffer[dtype](WIN_FULL * BATCH)
    var tp_host = ctx.enqueue_create_host_buffer[DType.uint8](
        WIN_FULL * BATCH
    )

    var b0_rew = List[Float64]()
    b0_rew.append(0.5)
    b0_rew.append(1.0)
    b0_rew.append(-0.5)
    b0_rew.append(0.0)
    b0_rew.append(0.0)
    var b1_rew = List[Float64]()
    b1_rew.append(2.0)
    b1_rew.append(-1.0)
    b1_rew.append(0.0)
    b1_rew.append(0.0)
    b1_rew.append(0.0)

    var b0_don = List[Float64]()
    for _ in range(WIN_TRN):
        b0_don.append(0.0)
    var b1_don = List[Float64]()
    b1_don.append(0.0)
    b1_don.append(1.0)  # terminal
    b1_don.append(0.0)
    b1_don.append(0.0)
    b1_don.append(0.0)

    var b0_val = List[Float64]()
    b0_val.append(10.0)
    b0_val.append(20.0)
    b0_val.append(30.0)
    b0_val.append(40.0)
    b0_val.append(50.0)
    b0_val.append(60.0)
    var b1_val = List[Float64]()
    b1_val.append(5.0)
    b1_val.append(15.0)
    b1_val.append(25.0)
    b1_val.append(35.0)
    b1_val.append(45.0)
    b1_val.append(55.0)

    # Write time-major
    for t in range(WIN_TRN):
        rew_host[t * BATCH + 0] = Scalar[dtype](b0_rew[t])
        rew_host[t * BATCH + 1] = Scalar[dtype](b1_rew[t])
        don_host[t * BATCH + 0] = Scalar[dtype](b0_don[t])
        don_host[t * BATCH + 1] = Scalar[dtype](b1_don[t])
    for t in range(WIN_FULL):
        val_host[t * BATCH + 0] = Scalar[dtype](b0_val[t])
        val_host[t * BATCH + 1] = Scalar[dtype](b1_val[t])

    # ── Output buffers ────────────────────────────────────────────────
    var vt_buf = ctx.enqueue_create_buffer[dtype]((K + 1) * BATCH)
    var rt_buf = ctx.enqueue_create_buffer[dtype](K * BATCH)
    var vt_host = ctx.enqueue_create_host_buffer[dtype]((K + 1) * BATCH)
    var rt_host = ctx.enqueue_create_host_buffer[dtype](K * BATCH)

    var rew_t = LayoutTensor[
        dtype, Layout.row_major(WIN_TRN * BATCH), MutAnyOrigin
    ](rew_buf.unsafe_ptr())
    var don_t = LayoutTensor[
        dtype, Layout.row_major(WIN_TRN * BATCH), MutAnyOrigin
    ](don_buf.unsafe_ptr())
    var val_t = LayoutTensor[
        dtype, Layout.row_major(WIN_FULL * BATCH), MutAnyOrigin
    ](val_buf.unsafe_ptr())
    var tp_t = LayoutTensor[
        DType.uint8, Layout.row_major(WIN_FULL * BATCH), MutAnyOrigin
    ](tp_buf.unsafe_ptr())
    var vt_t = LayoutTensor[
        dtype, Layout.row_major((K + 1) * BATCH), MutAnyOrigin
    ](vt_buf.unsafe_ptr())
    var rt_t = LayoutTensor[
        dtype, Layout.row_major(K * BATCH), MutAnyOrigin
    ](rt_buf.unsafe_ptr())

    var num_failures = 0

    var labels = List[String]()
    labels.append("k=0 b=0")
    labels.append("k=0 b=1")
    labels.append("k=1 b=0")
    labels.append("k=1 b=1")
    labels.append("k=2 b=0")
    labels.append("k=2 b=1")
    labels.append("k=3 b=0")
    labels.append("k=3 b=1")

    # ───────────────────────────────────────────────────────────────────
    # Case A — single-player (no sign flip)
    # ───────────────────────────────────────────────────────────────────
    print()
    print("--- Case A: to_play all zeros (single-player, no flip) ---")
    for i in range(WIN_FULL * BATCH):
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
        rw: LayoutTensor[
            dtype, Layout.row_major(WIN_TRN * BATCH), MutAnyOrigin
        ],
        dn: LayoutTensor[
            dtype, Layout.row_major(WIN_TRN * BATCH), MutAnyOrigin
        ],
        bv: LayoutTensor[
            dtype, Layout.row_major(WIN_FULL * BATCH), MutAnyOrigin
        ],
        tp: LayoutTensor[
            DType.uint8, Layout.row_major(WIN_FULL * BATCH), MutAnyOrigin
        ],
        g: Scalar[dtype],
    ):
        nstep_value_targets_kernel[BATCH, K, N, dtype](
            vt, rt, rw, dn, bv, tp, g
        )

    ctx.enqueue_function[run_kernel_a, run_kernel_a](
        vt_t, rt_t, rew_t, don_t, val_t, tp_t,
        Scalar[dtype](gamma),
        grid_dim=(BLOCKS,),
        block_dim=(TPB,),
    )

    ctx.enqueue_copy(vt_host, vt_buf)
    ctx.enqueue_copy(rt_host, rt_buf)
    ctx.synchronize()

    # Hand-computed expected values for Case A:
    # b=0:
    #   k=0: r0+γr1 + γ²·v2 = 0.5 + 0.99 + 0.9801·30 = 30.893
    #   k=1: r1+γr2 + γ²·v3 = 1.0 - 0.495 + 0.9801·40 = 39.709
    #   k=2: r2+γr3 + γ²·v4 = -0.5 + 0 + 0.9801·50 = 48.505
    #   k=3: r3+γr4 + γ²·v5 = 0 + 0 + 0.9801·60 = 58.806
    # b=1:
    #   k=0: r0+γr1 [terminal] = 2.0 - 0.99 = 1.01
    #   k=1: r1 [terminal] = -1.0
    #   k=2: r2+γr3 + γ²·v4 = 0 + 0 + 0.9801·45 = 44.1045
    #   k=3: r3+γr4 + γ²·v5 = 0 + 0 + 0.9801·55 = 53.9055
    var expected_a = List[Float64]()
    expected_a.append(30.893)
    expected_a.append(1.01)
    expected_a.append(39.709)
    expected_a.append(-1.0)
    expected_a.append(48.505)
    expected_a.append(44.1045)
    expected_a.append(58.806)
    expected_a.append(53.9055)

    for i in range((K + 1) * BATCH):
        var got = Float64(vt_host[i])
        var exp = expected_a[i]
        var diff = got - exp
        if diff < 0.0:
            diff = -diff
        var ok = diff < 1e-3
        var status = "PASS" if ok else "FAIL"
        print(status, " ", labels[i], " : got ", got, " expected ", exp)
        if not ok:
            num_failures += 1

    var expected_rew = List[Float64]()
    expected_rew.append(0.5)
    expected_rew.append(2.0)
    expected_rew.append(1.0)
    expected_rew.append(-1.0)
    expected_rew.append(-0.5)
    expected_rew.append(0.0)
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
    # Case B — two-player turn-taking (alternating to_play [0,1,0,1,0,1])
    # ───────────────────────────────────────────────────────────────────
    print()
    print("--- Case B: to_play alternating (two-player, sign flip) ---")
    for t in range(WIN_FULL):
        for b in range(BATCH):
            tp_host[t * BATCH + b] = UInt8(t & 1)

    ctx.enqueue_copy(tp_buf, tp_host)

    @parameter
    @always_inline
    def run_kernel_b(
        vt: LayoutTensor[
            dtype, Layout.row_major((K + 1) * BATCH), MutAnyOrigin
        ],
        rt: LayoutTensor[dtype, Layout.row_major(K * BATCH), MutAnyOrigin],
        rw: LayoutTensor[
            dtype, Layout.row_major(WIN_TRN * BATCH), MutAnyOrigin
        ],
        dn: LayoutTensor[
            dtype, Layout.row_major(WIN_TRN * BATCH), MutAnyOrigin
        ],
        bv: LayoutTensor[
            dtype, Layout.row_major(WIN_FULL * BATCH), MutAnyOrigin
        ],
        tp: LayoutTensor[
            DType.uint8, Layout.row_major(WIN_FULL * BATCH), MutAnyOrigin
        ],
        g: Scalar[dtype],
    ):
        nstep_value_targets_kernel[BATCH, K, N, dtype](
            vt, rt, rw, dn, bv, tp, g
        )

    ctx.enqueue_function[run_kernel_b, run_kernel_b](
        vt_t, rt_t, rew_t, don_t, val_t, tp_t,
        Scalar[dtype](gamma),
        grid_dim=(BLOCKS,),
        block_dim=(TPB,),
    )

    ctx.enqueue_copy(vt_host, vt_buf)
    ctx.enqueue_copy(rt_host, rt_buf)
    ctx.synchronize()

    # Hand-computed Case B (sign flips per-step):
    # b=0:
    #   k=0 (P0): r0(P0) + γ·-r1(P1) + γ²·v2(P0) = 0.5 - 0.99 + 0.9801·30 = 28.913
    #   k=1 (P1): r1(P1) + γ·-r2(P0) + γ²·v3(P1) = 1.0 + 0.495 + 0.9801·40 = 40.699
    #   k=2 (P0): r2(P0) + γ·-r3(P1)=0 + γ²·v4(P0) = -0.5 + 0 + 49.005 = 48.505
    #   k=3 (P1): r3(P1)=0 + γ·-r4(P0)=0 + γ²·v5(P1) = 0.9801·60 = 58.806
    # b=1:
    #   k=0 (P0): r0(P0)+γ·-r1(P1) [terminal] = 2.0 + 0.99 = 2.99
    #   k=1 (P1): r1(P1) [terminal] = -1.0
    #   k=2 (P0): 0 + 0 + 0.9801·45 = 44.1045
    #   k=3 (P1): 0 + 0 + 0.9801·55 = 53.9055
    var expected_b = List[Float64]()
    expected_b.append(28.913)
    expected_b.append(2.99)
    expected_b.append(40.699)
    expected_b.append(-1.0)
    expected_b.append(48.505)
    expected_b.append(44.1045)
    expected_b.append(58.806)
    expected_b.append(53.9055)

    for i in range((K + 1) * BATCH):
        var got = Float64(vt_host[i])
        var exp = expected_b[i]
        var diff = got - exp
        if diff < 0.0:
            diff = -diff
        var ok = diff < 1e-3
        var status = "PASS" if ok else "FAIL"
        print(status, " ", labels[i], " : got ", got, " expected ", exp)
        if not ok:
            num_failures += 1

    print()
    if num_failures == 0:
        print("=== ALL TESTS PASSED ===")
    else:
        print("=== ", num_failures, " ASSERTIONS FAILED ===")
        raise Error("regression test failed")
