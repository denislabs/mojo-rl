"""RSample[ACT] CPU tests — Phase 8.4.

Three tests:

  1. test_rsample_forward_matches_free_function — RSample.forward with
     a fixed RNG seed produces the same `action` and `log_prob` output as
     the free-function `squashed_gaussian_sample` fed with the z that
     RSample drew internally. Bit-identical (same arithmetic, just
     pack/unpack difference).

  2. test_rsample_backward_matches_free_function — given the same
     forward-pass cache, RSample.backward (consuming
     grad_output=[grad_action | grad_log_prob] with
     grad_log_prob = α/BATCH constant) produces the same
     grad_actor_output as the free-function `sac_actor_backward(..., α)`.
     This is the load-bearing equivalence: the composed-form SAC loss
     graph (where the α factor and the 1/BATCH mean reduction live in
     downstream Modules) must invert to the same actor-output gradient
     as the bespoke free-function form.

  3. test_rsample_fd_gradcheck — finite-difference gradcheck on RSample
     end-to-end. The "virtual loss" reads grad_output directly off the
     packed output, so we probe each actor_output (= input) entry and
     verify analytical backward matches numerical (eps=1e-3, 1% rel-err).
"""

from std.math import abs as fabs, exp as fexp, log as flog, tanh as ftanh
from std.memory import alloc
from std.random import seed
from std.testing import assert_almost_equal, assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.rsample import RSample
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.loss.sac_actor_loss import (
    squashed_gaussian_sample,
    sac_actor_backward,
)


def test_rsample_forward_matches_free_function() raises:
    """RSample.forward draws z internally, then writes packed
    [action | log_prob]. Re-running squashed_gaussian_sample on the
    same z (read out of the cache) must produce the same action +
    log_prob."""
    comptime BATCH = 3
    comptime ACT = 2

    var ao = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var out_rs = alloc[Scalar[DT]](BATCH * (ACT + 1))
    var act_ff = alloc[Scalar[DT]](BATCH * ACT)
    var lp_ff = alloc[Scalar[DT]](BATCH)

    # Hand-picked inputs.
    ao[0]  =  0.3; ao[1]  = -0.6; ao[2]  = -0.5; ao[3]  =  0.1
    ao[4]  =  0.8; ao[5]  =  0.0; ao[6]  = -1.0; ao[7]  = -0.3
    ao[8]  = -0.4; ao[9]  =  0.7; ao[10] = -0.2; ao[11] = -0.8

    var rs = RSample[ACT].make[target="cpu", INIT=Zero]()
    rs.action_scale = 2.0

    seed(1234)
    var ao_t = TileTensor(ao, row_major[BATCH, 2 * ACT]())
    var out_t = TileTensor(out_rs, row_major[BATCH, ACT + 1]())
    rs.forward["cpu", BATCH](ao_t, out_t)

    # Replay free-function with the z RSample just drew.
    var z_ptr = rs.z_cache.unsafe_ptr()
    var z_t = TileTensor(z_ptr, row_major[BATCH, ACT]())
    var act_t = TileTensor(act_ff, row_major[BATCH, ACT]())
    var lp_t = TileTensor(lp_ff, row_major[BATCH]())
    squashed_gaussian_sample[ACT, BATCH](
        ao_t, z_t, rs.action_scale, act_t, lp_t
    )

    # Compare.
    var max_diff: Scalar[DT] = 0.0
    for b in range(BATCH):
        for j in range(ACT):
            var d = fabs(out_rs[b * (ACT + 1) + j] - act_ff[b * ACT + j])
            if d > max_diff:
                max_diff = d
        var d_lp = fabs(out_rs[b * (ACT + 1) + ACT] - lp_ff[b])
        if d_lp > max_diff:
            max_diff = d_lp
    print(
        "  test_rsample_forward_matches_free_function max_diff=", max_diff
    )
    assert_true(
        max_diff < Scalar[DT](1e-7),
        "RSample.forward and squashed_gaussian_sample disagree (max_diff > 1e-7)",
    )

    ao.free(); out_rs.free(); act_ff.free(); lp_ff.free()


def test_rsample_backward_matches_free_function() raises:
    """Construct grad_output = [grad_action | (α/BATCH)·ones] and verify
    RSample.backward produces the same grad_actor_output as
    sac_actor_backward(..., α). This is the equivalence that lets the
    composed-form SAC actor loss (where α and Mean live downstream) drop
    in for the free-function form."""
    comptime BATCH = 3
    comptime ACT = 2

    var ao = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var grad_out = alloc[Scalar[DT]](BATCH * (ACT + 1))
    var grad_in_rs = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var out_rs = alloc[Scalar[DT]](BATCH * (ACT + 1))
    var grad_action_ff = alloc[Scalar[DT]](BATCH * ACT)
    var grad_in_ff = alloc[Scalar[DT]](BATCH * 2 * ACT)

    # Same setup as the free-function gradcheck.
    ao[0]  =  0.3; ao[1]  = -0.6; ao[2]  = -0.5; ao[3]  =  0.1
    ao[4]  =  0.8; ao[5]  =  0.0; ao[6]  = -1.0; ao[7]  = -0.3
    ao[8]  = -0.4; ao[9]  =  0.7; ao[10] = -0.2; ao[11] = -0.8
    # Pretend grad_action arrives from critic backward. Same shape as the
    # free-function test.
    var ga_b0_0:  Scalar[DT] =  0.4
    var ga_b0_1:  Scalar[DT] = -0.15
    var ga_b1_0:  Scalar[DT] = -0.25
    var ga_b1_1:  Scalar[DT] =  0.5
    var ga_b2_0:  Scalar[DT] =  0.1
    var ga_b2_1:  Scalar[DT] = -0.3

    var alpha: Scalar[DT] = 0.18
    var alpha_per_batch: Scalar[DT] = alpha / Scalar[DT](BATCH)

    # Build grad_output for RSample:
    #   [b, 0..ACT)  = grad_action[b, j]
    #   [b, ACT]     = grad_log_prob = alpha/BATCH (uniform — matches the
    #                  Mean+Scale chain downstream)
    grad_out[0 * (ACT + 1) + 0] = ga_b0_0
    grad_out[0 * (ACT + 1) + 1] = ga_b0_1
    grad_out[0 * (ACT + 1) + ACT] = alpha_per_batch
    grad_out[1 * (ACT + 1) + 0] = ga_b1_0
    grad_out[1 * (ACT + 1) + 1] = ga_b1_1
    grad_out[1 * (ACT + 1) + ACT] = alpha_per_batch
    grad_out[2 * (ACT + 1) + 0] = ga_b2_0
    grad_out[2 * (ACT + 1) + 1] = ga_b2_1
    grad_out[2 * (ACT + 1) + ACT] = alpha_per_batch

    grad_action_ff[0] = ga_b0_0; grad_action_ff[1] = ga_b0_1
    grad_action_ff[2] = ga_b1_0; grad_action_ff[3] = ga_b1_1
    grad_action_ff[4] = ga_b2_0; grad_action_ff[5] = ga_b2_1

    var rs = RSample[ACT].make[target="cpu", INIT=Zero]()
    rs.action_scale = 2.0

    seed(99)
    var ao_t = TileTensor(ao, row_major[BATCH, 2 * ACT]())
    var out_t = TileTensor(out_rs, row_major[BATCH, ACT + 1]())
    rs.forward["cpu", BATCH](ao_t, out_t)

    var go_t = TileTensor(grad_out, row_major[BATCH, ACT + 1]())
    var gi_rs_t = TileTensor(grad_in_rs, row_major[BATCH, 2 * ACT]())
    rs.backward["cpu", BATCH](go_t, gi_rs_t)

    # Free-function form using the same z.
    var z_ptr = rs.z_cache.unsafe_ptr()
    var z_t = TileTensor(z_ptr, row_major[BATCH, ACT]())
    var ga_t = TileTensor(grad_action_ff, row_major[BATCH, ACT]())
    var gi_ff_t = TileTensor(grad_in_ff, row_major[BATCH, 2 * ACT]())
    sac_actor_backward[ACT, BATCH](
        ao_t, z_t, ga_t, alpha, rs.action_scale, gi_ff_t
    )

    var max_diff: Scalar[DT] = 0.0
    for i in range(BATCH * 2 * ACT):
        var d = fabs(grad_in_rs[i] - grad_in_ff[i])
        if d > max_diff:
            max_diff = d
    print(
        "  test_rsample_backward_matches_free_function max_diff=", max_diff
    )
    assert_true(
        max_diff < Scalar[DT](1e-6),
        "RSample.backward and sac_actor_backward disagree (max_diff > 1e-6)",
    )

    ao.free(); grad_out.free(); grad_in_rs.free(); out_rs.free()
    grad_action_ff.free(); grad_in_ff.free()


def test_rsample_fd_gradcheck() raises:
    """End-to-end FD gradcheck for RSample. We probe each input entry
    against the virtual loss
        L = Σ_b ( Σ_j grad_out[b, j]·output[b, j]
                  + grad_out[b, ACT]·output[b, ACT] )
    whose analytical backward is exactly RSample.backward(grad_out, .).
    """
    comptime BATCH = 3
    comptime ACT = 2

    var ao = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var grad_out = alloc[Scalar[DT]](BATCH * (ACT + 1))
    var grad_in = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var out_buf = alloc[Scalar[DT]](BATCH * (ACT + 1))

    ao[0]  =  0.3; ao[1]  = -0.6; ao[2]  = -0.5; ao[3]  =  0.1
    ao[4]  =  0.8; ao[5]  =  0.0; ao[6]  = -1.0; ao[7]  = -0.3
    ao[8]  = -0.4; ao[9]  =  0.7; ao[10] = -0.2; ao[11] = -0.8

    grad_out[0] =  0.4; grad_out[1]  = -0.15; grad_out[2]  =  0.05
    grad_out[3] = -0.25; grad_out[4] =  0.5;  grad_out[5]  =  0.07
    grad_out[6] =  0.1;  grad_out[7] = -0.3;  grad_out[8]  = -0.04

    var rs = RSample[ACT].make[target="cpu", INIT=Zero]()
    rs.action_scale = 2.0

    seed(7)
    var ao_t = TileTensor(ao, row_major[BATCH, 2 * ACT]())
    var out_t = TileTensor(out_buf, row_major[BATCH, ACT + 1]())
    rs.forward["cpu", BATCH](ao_t, out_t)

    var go_t = TileTensor(grad_out, row_major[BATCH, ACT + 1]())
    var gi_t = TileTensor(grad_in, row_major[BATCH, 2 * ACT]())
    rs.backward["cpu", BATCH](go_t, gi_t)

    # FD: perturb each ao[idx], rerun forward (with the SAME z each time —
    # we have to freeze RNG via re-seeding to match). Actually the cleanest
    # approach is to capture z once, then for each FD probe call
    # squashed_gaussian_sample with that same z (it doesn't draw new
    # noise). This sidesteps reseeding-races.
    var z_ptr = rs.z_cache.unsafe_ptr()
    var z_t = TileTensor(z_ptr, row_major[BATCH, ACT]())
    var act_scratch = alloc[Scalar[DT]](BATCH * ACT)
    var lp_scratch = alloc[Scalar[DT]](BATCH)
    var act_t = TileTensor(act_scratch, row_major[BATCH, ACT]())
    var lp_t = TileTensor(lp_scratch, row_major[BATCH]())

    var eps: Scalar[DT] = 1e-3
    var max_rel: Scalar[DT] = 0.0
    var max_abs: Scalar[DT] = 0.0
    for idx in range(BATCH * 2 * ACT):
        var orig = ao[idx]
        # +eps
        ao[idx] = orig + eps
        squashed_gaussian_sample[ACT, BATCH](
            ao_t, z_t, rs.action_scale, act_t, lp_t
        )
        var L_plus: Scalar[DT] = 0.0
        for b in range(BATCH):
            for j in range(ACT):
                L_plus += grad_out[b * (ACT + 1) + j] * act_scratch[b * ACT + j]
            L_plus += grad_out[b * (ACT + 1) + ACT] * lp_scratch[b]
        # -eps
        ao[idx] = orig - eps
        squashed_gaussian_sample[ACT, BATCH](
            ao_t, z_t, rs.action_scale, act_t, lp_t
        )
        var L_minus: Scalar[DT] = 0.0
        for b in range(BATCH):
            for j in range(ACT):
                L_minus += grad_out[b * (ACT + 1) + j] * act_scratch[b * ACT + j]
            L_minus += grad_out[b * (ACT + 1) + ACT] * lp_scratch[b]
        ao[idx] = orig

        var num_grad = (L_plus - L_minus) / (Scalar[DT](2.0) * eps)
        var ana_grad = grad_in[idx]
        var abs_err = fabs(num_grad - ana_grad)
        var denom = fabs(num_grad) + fabs(ana_grad) + Scalar[DT](1e-6)
        var rel_err = abs_err / denom
        if rel_err > max_rel:
            max_rel = rel_err
        if abs_err > max_abs:
            max_abs = abs_err

    print("  test_rsample_fd_gradcheck max_abs=", max_abs, " max_rel=", max_rel)
    assert_true(max_rel < Scalar[DT](1e-2), "FD gradcheck rel-err > 1%")

    ao.free(); grad_out.free(); grad_in.free(); out_buf.free()
    act_scratch.free(); lp_scratch.free()


def main() raises:
    print("=" * 70)
    print("nn2 Phase 8.4 — RSample[ACT] CPU tests")
    print("=" * 70)
    test_rsample_forward_matches_free_function()
    print("  test_rsample_forward_matches_free_function PASSED")
    test_rsample_backward_matches_free_function()
    print("  test_rsample_backward_matches_free_function PASSED")
    test_rsample_fd_gradcheck()
    print("  test_rsample_fd_gradcheck PASSED")
    print("=" * 70)
    print("ALL PASSED — RSample[ACT] equivalent to free-function form")
    print("=" * 70)
