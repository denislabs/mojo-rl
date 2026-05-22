"""PPOActorLoss[ACT] tests — Phase 6.4.

Covers:
  - forward sanity: simple BATCH=1 case with new_log_prob = old_log_prob
    (ratio = 1, unclipped, surrogate = adv → L = -adv - entropy_coef*H)
  - gradient zero for clipped samples (only entropy grad on log_std)
  - FD gradcheck on grad_actor_output (both mu and log_std columns)
  - GPU parity vs CPU
"""

from std.math import abs as fabs, exp, log
from std.memory import alloc
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.loss.ppo_actor_loss import PPOActorLoss


# ──────────────────────────────────────────────────────────────────────────
# CPU forward + backward sanity
# ──────────────────────────────────────────────────────────────────────────


def test_forward_unclipped_ratio_one() raises:
    """BATCH=1, ACT=1, mu=0, log_std=0 (std=1).
    Action = 0 → new_log_prob = -0.5 * log(2π) = -0.9189.
    Set old_log_prob = -0.9189 → diff = 0, ratio = 1.
    advantage = 2.0 → unclipped = clipped = 2.0 → min = 2.0 → loss = -2.0.
    entropy_coef = 0 → final L = -2.0."""
    comptime BATCH = 1
    comptime ACT = 1
    var loss = PPOActorLoss[ACT].make["cpu"](
        clip_eps=Scalar[DT](0.2), entropy_coef=Scalar[DT](0.0)
    )

    var ao_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var ac_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * ACT)
    var ol_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var ad_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)

    ao_buf[0] = 0.0  # mu
    ao_buf[1] = 0.0  # log_std → std = 1
    ac_buf[0] = 0.0
    ol_buf[0] = -0.9189385332046727  # -0.5 * log(2π) (matches new_log_prob)
    ad_buf[0] = 2.0

    var ao = TileTensor(ao_buf, row_major[BATCH, 2 * ACT]())
    var ac = TileTensor(ac_buf, row_major[BATCH, ACT]())
    var ol = TileTensor(ol_buf, row_major[BATCH]())
    var ad = TileTensor(ad_buf, row_major[BATCH]())

    var L = loss.forward["cpu", BATCH](ao, ac, ol, ad)
    print("L (expect -2.0) = ", L)
    assert_true(fabs(L - (-2.0)) < 1e-4, "wrong loss: " + String(L))

    ao_buf.free()
    ac_buf.free()
    ol_buf.free()
    ad_buf.free()
    print("  test_forward_unclipped_ratio_one PASSED")


def test_entropy_term() raises:
    """With ratio=1 and advantage=0, the loss is purely -entropy_coef*H.
    H = 0.5 * (log(2π) + 1 + 2*log_std). For log_std=0: H=0.5*(log(2π)+1)≈1.4189."""
    comptime BATCH = 1
    comptime ACT = 1
    var loss = PPOActorLoss[ACT].make["cpu"](
        clip_eps=Scalar[DT](0.2), entropy_coef=Scalar[DT](0.5)
    )

    var ao_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var ac_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * ACT)
    var ol_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var ad_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    ao_buf[0] = 0.0
    ao_buf[1] = 0.0
    ac_buf[0] = 0.0
    ol_buf[0] = -0.9189385332046727
    ad_buf[0] = 0.0  # zero advantage isolates the entropy term

    var ao = TileTensor(ao_buf, row_major[BATCH, 2 * ACT]())
    var ac = TileTensor(ac_buf, row_major[BATCH, ACT]())
    var ol = TileTensor(ol_buf, row_major[BATCH]())
    var ad = TileTensor(ad_buf, row_major[BATCH]())

    var L = loss.forward["cpu", BATCH](ao, ac, ol, ad)
    # H = 0.5 * (log(2π) + 1) ≈ 1.4189
    # Expected: L = -0 - 0.5 * 1.4189 = -0.7095
    print("L (entropy isolated) = ", L)
    assert_true(fabs(L - (-0.7094693)) < 1e-4)

    ao_buf.free()
    ac_buf.free()
    ol_buf.free()
    ad_buf.free()
    print("  test_entropy_term PASSED")


# ──────────────────────────────────────────────────────────────────────────
# Clipped sample → grad zero on mu (entropy may still flow on log_std)
# ──────────────────────────────────────────────────────────────────────────


def test_clipped_sample_grad_zero_on_mu() raises:
    """If ratio is clipped on the lower side (ratio < 1-eps) with adv > 0,
    OR upper side (ratio > 1+eps) with adv < 0, the clipped objective is
    smaller and min picks it → grad zero on mu.

    Force ratio > 1+eps with adv > 0: set new_log_prob much larger than
    old_log_prob. Then clipped_obj = (1+eps)*adv < ratio*adv = unclipped.
    So min picks clipped, and gradient flows through clipped (constant
    w.r.t. mu/log_std), so grad_mu = 0."""
    comptime BATCH = 1
    comptime ACT = 1
    var loss = PPOActorLoss[ACT].make["cpu"](
        clip_eps=Scalar[DT](0.2), entropy_coef=Scalar[DT](0.0)
    )

    var ao_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var ac_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * ACT)
    var ol_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var ad_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var go_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)

    ao_buf[0] = 0.0  # mu
    ao_buf[1] = 0.0  # log_std
    ac_buf[0] = 0.0  # action == mu → new_log_prob = -log(2π)/2 ≈ -0.9189
    ol_buf[0] = -10.0  # diff = ~9.08 → ratio = ~8761 >> 1+eps
    ad_buf[0] = 1.0  # positive advantage
    for k in range(BATCH * 2 * ACT):
        go_buf[k] = -999.0

    var ao = TileTensor(ao_buf, row_major[BATCH, 2 * ACT]())
    var ac = TileTensor(ac_buf, row_major[BATCH, ACT]())
    var ol = TileTensor(ol_buf, row_major[BATCH]())
    var ad = TileTensor(ad_buf, row_major[BATCH]())
    var go = TileTensor(go_buf, row_major[BATCH, 2 * ACT]())

    loss.vjp["cpu", BATCH](ao, ac, ol, ad, go)
    # grad_mu = 0; grad_log_std = 0 (entropy_coef=0)
    assert_true(fabs(go[0, 0]) < 1e-6, "grad_mu not zero on clip: " + String(go[0, 0]))
    assert_true(fabs(go[0, 1]) < 1e-6, "grad_log_std not zero on clip: " + String(go[0, 1]))

    ao_buf.free()
    ac_buf.free()
    ol_buf.free()
    ad_buf.free()
    go_buf.free()
    print("  test_clipped_sample_grad_zero_on_mu PASSED")


# ──────────────────────────────────────────────────────────────────────────
# FD gradcheck
# ──────────────────────────────────────────────────────────────────────────


def test_gradcheck_fd() raises:
    """FD vs analytical grad on actor_output for non-clipped samples.
    Use small log_prob_diff so we stay in the unclipped regime."""
    comptime BATCH = 3
    comptime ACT = 2
    comptime EPS: Scalar[DT] = 1e-3
    comptime TOL_REL: Scalar[DT] = 5e-3  # PPO has many nonlinearities

    var loss = PPOActorLoss[ACT].make["cpu"](
        clip_eps=Scalar[DT](0.2), entropy_coef=Scalar[DT](0.01)
    )

    var ao_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var ac_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * ACT)
    var ol_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var ad_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var go_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)

    # Initial actor_output: mu small, log_std around -0.5, all in valid range.
    for b in range(BATCH):
        for j in range(ACT):
            ao_buf[b * 2 * ACT + j] = Scalar[DT](0.1 * Float64(j))  # mu_j
            ao_buf[b * 2 * ACT + ACT + j] = Scalar[DT](-0.5)  # log_std_j
        ad_buf[b] = Scalar[DT](0.5 + 0.1 * Float64(b))  # advantage
        for j in range(ACT):
            ac_buf[b * ACT + j] = Scalar[DT](0.05 + 0.02 * Float64(b * ACT + j))
        # Compute "old_log_prob" so diff = 0 → ratio = 1 → unclipped.
        var nlp: Scalar[DT] = 0.0
        for j in range(ACT):
            var mu = ao_buf[b * 2 * ACT + j]
            var ls = ao_buf[b * 2 * ACT + ACT + j]
            var std = exp(ls)
            var a = ac_buf[b * ACT + j]
            var z = (a - mu) / (std + Scalar[DT](1e-6))
            nlp += Scalar[DT](-0.5) * (
                Scalar[DT](1.8378770664093453)
                + Scalar[DT](2.0) * ls
                + z * z
            )
        ol_buf[b] = nlp
    for k in range(BATCH * 2 * ACT):
        go_buf[k] = 0.0

    var ao = TileTensor(ao_buf, row_major[BATCH, 2 * ACT]())
    var ac = TileTensor(ac_buf, row_major[BATCH, ACT]())
    var ol = TileTensor(ol_buf, row_major[BATCH]())
    var ad = TileTensor(ad_buf, row_major[BATCH]())
    var go = TileTensor(go_buf, row_major[BATCH, 2 * ACT]())

    loss.vjp["cpu", BATCH](ao, ac, ol, ad, go)

    # FD: perturb each entry of actor_output, recompute L.
    var max_rel: Scalar[DT] = 0.0
    for b in range(BATCH):
        for j in range(2 * ACT):
            var saved = ao_buf[b * 2 * ACT + j]
            ao_buf[b * 2 * ACT + j] = saved + EPS
            var Lp = loss.forward["cpu", BATCH](ao, ac, ol, ad)
            ao_buf[b * 2 * ACT + j] = saved - EPS
            var Lm = loss.forward["cpu", BATCH](ao, ac, ol, ad)
            ao_buf[b * 2 * ACT + j] = saved
            var fd = (Lp - Lm) / (Scalar[DT](2.0) * EPS)
            var an = go[b, j]
            var denom = fabs(an) + Scalar[DT](1e-6)
            var rel = fabs(fd - an) / denom
            if rel > max_rel:
                max_rel = rel

    print("  PPO FD gradcheck max_rel = ", max_rel)
    assert_true(max_rel < TOL_REL, "gradcheck failed")

    ao_buf.free()
    ac_buf.free()
    ol_buf.free()
    ad_buf.free()
    go_buf.free()
    print("  test_gradcheck_fd PASSED")


# ──────────────────────────────────────────────────────────────────────────
# GPU parity
# ──────────────────────────────────────────────────────────────────────────


def test_gpu_parity() raises:
    comptime BATCH = 4
    comptime ACT = 2
    comptime TOL_FWD: Scalar[DT] = 1e-5
    comptime TOL_BWD: Scalar[DT] = 1e-5

    var ctx = DeviceContext()
    var loss_cpu = PPOActorLoss[ACT].make["cpu"](
        clip_eps=Scalar[DT](0.2), entropy_coef=Scalar[DT](0.01)
    )
    var loss_gpu = PPOActorLoss[ACT].make["gpu"](
        ctx, clip_eps=Scalar[DT](0.2), entropy_coef=Scalar[DT](0.01)
    )

    var ao_host = ctx.enqueue_create_host_buffer[DT](BATCH * 2 * ACT)
    var ac_host = ctx.enqueue_create_host_buffer[DT](BATCH * ACT)
    var ol_host = ctx.enqueue_create_host_buffer[DT](BATCH)
    var ad_host = ctx.enqueue_create_host_buffer[DT](BATCH)
    ctx.synchronize()

    # Mix of clipped + unclipped samples.
    for b in range(BATCH):
        for j in range(ACT):
            ao_host.unsafe_ptr()[b * 2 * ACT + j] = Scalar[DT](
                0.1 * Float64(b * ACT + j)
            )
            ao_host.unsafe_ptr()[b * 2 * ACT + ACT + j] = Scalar[DT](-0.3)
            ac_host.unsafe_ptr()[b * ACT + j] = Scalar[DT](
                0.05 + 0.03 * Float64(b * ACT + j)
            )
        ol_host.unsafe_ptr()[b] = Scalar[DT](-1.0 - 0.5 * Float64(b))
        ad_host.unsafe_ptr()[b] = Scalar[DT](1.0 - 0.3 * Float64(b))

    var ao_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var ac_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * ACT)
    var ol_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var ad_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    for k in range(BATCH * 2 * ACT):
        ao_cpu[k] = ao_host.unsafe_ptr()[k]
    for k in range(BATCH * ACT):
        ac_cpu[k] = ac_host.unsafe_ptr()[k]
    for k in range(BATCH):
        ol_cpu[k] = ol_host.unsafe_ptr()[k]
        ad_cpu[k] = ad_host.unsafe_ptr()[k]
    var ao_t_cpu = TileTensor(ao_cpu, row_major[BATCH, 2 * ACT]())
    var ac_t_cpu = TileTensor(ac_cpu, row_major[BATCH, ACT]())
    var ol_t_cpu = TileTensor(ol_cpu, row_major[BATCH]())
    var ad_t_cpu = TileTensor(ad_cpu, row_major[BATCH]())

    var ao_dev = ctx.enqueue_create_buffer[DT](BATCH * 2 * ACT)
    var ac_dev = ctx.enqueue_create_buffer[DT](BATCH * ACT)
    var ol_dev = ctx.enqueue_create_buffer[DT](BATCH)
    var ad_dev = ctx.enqueue_create_buffer[DT](BATCH)
    ctx.enqueue_copy(ao_dev, ao_host)
    ctx.enqueue_copy(ac_dev, ac_host)
    ctx.enqueue_copy(ol_dev, ol_host)
    ctx.enqueue_copy(ad_dev, ad_host)
    var ao_t_gpu = TileTensor(ao_dev, row_major[BATCH, 2 * ACT]())
    var ac_t_gpu = TileTensor(ac_dev, row_major[BATCH, ACT]())
    var ol_t_gpu = TileTensor(ol_dev, row_major[BATCH]())
    var ad_t_gpu = TileTensor(ad_dev, row_major[BATCH]())

    var L_cpu = loss_cpu.forward["cpu", BATCH](
        ao_t_cpu, ac_t_cpu, ol_t_cpu, ad_t_cpu
    )
    var L_gpu = loss_gpu.forward["gpu", BATCH](
        ao_t_gpu, ac_t_gpu, ol_t_gpu, ad_t_gpu
    )
    print("L_cpu = ", L_cpu, "  L_gpu = ", L_gpu)
    assert_true(fabs(L_cpu - L_gpu) < TOL_FWD, "forward parity failed")

    var go_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    for k in range(BATCH * 2 * ACT):
        go_cpu[k] = 0.0
    var go_t_cpu = TileTensor(go_cpu, row_major[BATCH, 2 * ACT]())
    loss_cpu.vjp["cpu", BATCH](
        ao_t_cpu, ac_t_cpu, ol_t_cpu, ad_t_cpu, go_t_cpu
    )

    var go_dev = ctx.enqueue_create_buffer[DT](BATCH * 2 * ACT)
    var go_t_gpu = TileTensor(go_dev, row_major[BATCH, 2 * ACT]())
    loss_gpu.vjp["gpu", BATCH](
        ao_t_gpu, ac_t_gpu, ol_t_gpu, ad_t_gpu, go_t_gpu
    )
    var go_host = ctx.enqueue_create_host_buffer[DT](BATCH * 2 * ACT)
    ctx.enqueue_copy(go_host, go_dev)
    ctx.synchronize()

    var max_diff: Scalar[DT] = 0.0
    for k in range(BATCH * 2 * ACT):
        var d = fabs(go_cpu[k] - go_host.unsafe_ptr()[k])
        if d > max_diff:
            max_diff = d
    print("max-diff grad_actor_output = " + String(max_diff))
    assert_true(max_diff < TOL_BWD, "backward parity failed")

    ao_cpu.free()
    ac_cpu.free()
    ol_cpu.free()
    ad_cpu.free()
    go_cpu.free()
    print("  test_gpu_parity PASSED")


def main() raises:
    print("=" * 60)
    print("nn2 PPOActorLoss tests (CPU + GPU, Phase 6.4)")
    print("=" * 60)
    test_forward_unclipped_ratio_one()
    test_entropy_term()
    test_clipped_sample_grad_zero_on_mu()
    test_gradcheck_fd()
    test_gpu_parity()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
