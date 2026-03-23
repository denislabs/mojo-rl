"""Test ComputeGraph-based prediction heads for Dreamer V3 RSSM.

Verifies that:
1. predict_all_heads produces correct outputs (matches individual heads)
2. backward_all_heads produces non-zero grad_feat (the fix for the bug)
3. backward_all_heads accumulates param grads for all 3 heads
4. Gradient check: analytical vs finite-difference
"""

from mojo_rl.nn.constants import dtype
from mojo_rl.deep_agents.dreamer_v3.rssm import RSSM
from layout import Layout, LayoutTensor
from std.math import abs


def test_heads_forward() raises:
    """Verify predict_all_heads matches individual head outputs."""
    print("Test 1: Heads forward consistency...")

    # Small RSSM for testing
    comptime OBS = 6
    comptime ACT = 2
    comptime DETER = 32
    comptime HIDDEN = 16
    comptime STOCH = 4
    comptime CLASSES = 4
    comptime UNITS = 16
    comptime BINS = 15
    comptime BATCH = 2

    comptime RSSMType = RSSM[
        OBS, ACT, DETER, HIDDEN, STOCH, CLASSES, UNITS, BINS
    ]
    comptime FEAT = RSSMType.FEAT_DIM  # DETER + STOCH*CLASSES = 32+16 = 48
    comptime HEADS_OUT = RSSMType.HEADS_OUT_DIM  # OBS + BINS + 1 = 22

    var rssm = RSSMType()

    # Create feat input
    var feat_arr = InlineArray[Scalar[dtype], BATCH * FEAT](uninitialized=True)
    for i in range(BATCH * FEAT):
        feat_arr[i] = Scalar[dtype](Float64(i + 1) * 0.01)
    var feat_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, FEAT), MutAnyOrigin
    ](feat_arr.unsafe_ptr())

    # Forward via individual heads
    var dec_out = InlineArray[Scalar[dtype], BATCH * OBS](uninitialized=True)
    var dec_t = LayoutTensor[dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin](
        dec_out.unsafe_ptr()
    )
    rssm.decode[BATCH](feat_t, dec_t)

    var rew_out = InlineArray[Scalar[dtype], BATCH * BINS](uninitialized=True)
    var rew_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, BINS), MutAnyOrigin
    ](rew_out.unsafe_ptr())
    rssm.predict_reward[BATCH](feat_t, rew_t)

    # Note: predict_continue applies sigmoid, but HeadsGraph outputs raw logit
    # So we compare with ContModel forward directly
    var cont_out = InlineArray[Scalar[dtype], BATCH * 1](uninitialized=True)
    var cont_t = LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin](
        cont_out.unsafe_ptr()
    )
    RSSMType.ContNet.forward[BATCH](
        feat_t, cont_t, rssm.continue_head.params_view()
    )

    # Forward via ComputeGraph heads
    var heads_out = InlineArray[Scalar[dtype], BATCH * HEADS_OUT](
        uninitialized=True
    )
    var heads_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, HEADS_OUT), MutAnyOrigin
    ](heads_out.unsafe_ptr())
    var cache_arr = InlineArray[
        Scalar[dtype], BATCH * RSSMType.HEADS_CACHE_SIZE
    ](uninitialized=True)
    var cache_t = LayoutTensor[
        dtype,
        Layout.row_major(BATCH, RSSMType.HEADS_CACHE_SIZE),
        MutAnyOrigin,
    ](cache_arr.unsafe_ptr())

    rssm.predict_all_heads[BATCH](feat_t, heads_t, cache_t)

    # Compare outputs
    # HeadsGraph output: [obs_hat(6), rew_logits(15), cont_logit(1)]
    var max_err: Float64 = 0.0

    # Decoder portion
    for i in range(BATCH * OBS):
        var b = i // OBS
        var d = i % OBS
        var graph_val = Float64(heads_out[b * HEADS_OUT + d])
        var direct_val = Float64(dec_out[i])
        var err = abs(graph_val - direct_val)
        if err > max_err:
            max_err = err

    # Reward portion
    for i in range(BATCH * BINS):
        var b = i // BINS
        var d = i % BINS
        var graph_val = Float64(heads_out[b * HEADS_OUT + OBS + d])
        var direct_val = Float64(rew_out[i])
        var err = abs(graph_val - direct_val)
        if err > max_err:
            max_err = err

    # Continue portion
    for b in range(BATCH):
        var graph_val = Float64(heads_out[b * HEADS_OUT + OBS + BINS])
        var direct_val = Float64(cont_out[b])
        var err = abs(graph_val - direct_val)
        if err > max_err:
            max_err = err

    print("  Max output error:", max_err)
    if max_err > 1e-5:
        raise Error("Heads forward output mismatch")

    print("  PASSED")


def test_heads_backward_produces_grad_feat() raises:
    """Verify backward produces non-zero grad_feat from all 3 heads."""
    print("Test 2: Heads backward produces grad_feat...")

    comptime OBS = 6
    comptime ACT = 2
    comptime DETER = 32
    comptime HIDDEN = 16
    comptime STOCH = 4
    comptime CLASSES = 4
    comptime UNITS = 16
    comptime BINS = 15
    comptime BATCH = 2

    comptime RSSMType = RSSM[
        OBS, ACT, DETER, HIDDEN, STOCH, CLASSES, UNITS, BINS
    ]
    comptime FEAT = RSSMType.FEAT_DIM
    comptime HEADS_OUT = RSSMType.HEADS_OUT_DIM

    var rssm = RSSMType()

    # Feat input
    var feat_arr = InlineArray[Scalar[dtype], BATCH * FEAT](uninitialized=True)
    for i in range(BATCH * FEAT):
        feat_arr[i] = Scalar[dtype](Float64(i + 1) * 0.01)
    var feat_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, FEAT), MutAnyOrigin
    ](feat_arr.unsafe_ptr())

    # Forward
    var heads_out = InlineArray[Scalar[dtype], BATCH * HEADS_OUT](
        uninitialized=True
    )
    var heads_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, HEADS_OUT), MutAnyOrigin
    ](heads_out.unsafe_ptr())
    var cache_arr = InlineArray[
        Scalar[dtype], BATCH * RSSMType.HEADS_CACHE_SIZE
    ](uninitialized=True)
    var cache_t = LayoutTensor[
        dtype,
        Layout.row_major(BATCH, RSSMType.HEADS_CACHE_SIZE),
        MutAnyOrigin,
    ](cache_arr.unsafe_ptr())

    rssm.predict_all_heads[BATCH](feat_t, heads_t, cache_t)

    # Backward with gradient seeds for each head
    var grad_out = InlineArray[Scalar[dtype], BATCH * HEADS_OUT](
        uninitialized=True
    )
    # Decoder gradient: MSE-like
    for b in range(BATCH):
        for i in range(OBS):
            grad_out[b * HEADS_OUT + i] = Scalar[dtype](0.1)
    # Reward gradient
    for b in range(BATCH):
        for i in range(BINS):
            grad_out[b * HEADS_OUT + OBS + i] = Scalar[dtype](0.05)
    # Continue gradient
    for b in range(BATCH):
        grad_out[b * HEADS_OUT + OBS + BINS] = Scalar[dtype](0.2)

    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, HEADS_OUT), MutAnyOrigin
    ](grad_out.unsafe_ptr())

    # Zero grads before backward
    rssm.zero_all_grads()

    var grad_feat = InlineArray[Scalar[dtype], BATCH * FEAT](uninitialized=True)
    var grad_feat_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, FEAT), MutAnyOrigin
    ](grad_feat.unsafe_ptr())

    rssm.backward_all_heads[BATCH](grad_out_t, grad_feat_t, cache_t)

    # Check grad_feat is non-zero
    var any_nonzero = False
    var grad_feat_norm: Float64 = 0.0
    for i in range(BATCH * FEAT):
        var v = Float64(grad_feat[i])
        grad_feat_norm += v * v
        if abs(v) > 1e-10:
            any_nonzero = True

    print("  grad_feat L2 norm:", sqrt(grad_feat_norm))
    if not any_nonzero:
        raise Error("grad_feat is all zeros — backward failed")

    # Check ALL THREE networks have non-zero param grads
    var dec_grad_norm: Float64 = 0.0
    var dec_grads = rssm.decoder.grads_view()
    for i in range(RSSMType.DecModel.PARAM_SIZE):
        var v = Float64(dec_grads.ptr[i])
        dec_grad_norm += v * v

    var rew_grad_norm: Float64 = 0.0
    var rew_grads = rssm.reward_head.grads_view()
    for i in range(RSSMType.RewModel.PARAM_SIZE):
        var v = Float64(rew_grads.ptr[i])
        rew_grad_norm += v * v

    var cont_grad_norm: Float64 = 0.0
    var cont_grads = rssm.continue_head.grads_view()
    for i in range(RSSMType.ContModel.PARAM_SIZE):
        var v = Float64(cont_grads.ptr[i])
        cont_grad_norm += v * v

    print(
        "  Param grad norms — dec:",
        sqrt(dec_grad_norm),
        "rew:",
        sqrt(rew_grad_norm),
        "cont:",
        sqrt(cont_grad_norm),
    )

    if dec_grad_norm < 1e-20:
        raise Error("Decoder has zero param grads")
    if rew_grad_norm < 1e-20:
        raise Error("Reward head has zero param grads")
    if cont_grad_norm < 1e-20:
        raise Error("Continue head has zero param grads")

    print("  PASSED (all 3 heads have non-zero grads)")


def sqrt(x: Float64) -> Float64:
    """Simple sqrt via std.math."""
    from std.math import sqrt as _sqrt

    return _sqrt(x)


def test_heads_grad_check() raises:
    """Finite-difference gradient check for prediction heads."""
    print("Test 3: Heads gradient check...")

    comptime OBS = 4
    comptime ACT = 2
    comptime DETER = 16
    comptime HIDDEN = 8
    comptime STOCH = 2
    comptime CLASSES = 4
    comptime UNITS = 8
    comptime BINS = 5
    comptime BATCH = 1

    comptime RSSMType = RSSM[
        OBS, ACT, DETER, HIDDEN, STOCH, CLASSES, UNITS, BINS
    ]
    comptime FEAT = RSSMType.FEAT_DIM  # 16 + 8 = 24
    comptime HEADS_OUT = RSSMType.HEADS_OUT_DIM  # 4 + 5 + 1 = 10
    comptime M = RSSMType.HeadsGraph

    var rssm = RSSMType()

    # Use the heads graph directly for grad check
    # Assemble params
    comptime CP = RSSMType.HeadsCP
    var combined = InlineArray[Scalar[dtype], CP.TOTAL_SIZE](uninitialized=True)
    CP.assemble(
        combined.unsafe_ptr(),
        rssm.decoder.params_view().ptr,
        rssm.reward_head.params_view().ptr,
        rssm.continue_head.params_view().ptr,
    )

    var feat_arr = InlineArray[Scalar[dtype], BATCH * FEAT](uninitialized=True)
    for i in range(FEAT):
        feat_arr[i] = Scalar[dtype](Float64(i + 1) * 0.05)
    var feat_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, FEAT), MutAnyOrigin
    ](feat_arr.unsafe_ptr())

    var grad_out_arr = InlineArray[Scalar[dtype], BATCH * HEADS_OUT](
        uninitialized=True
    )
    for i in range(HEADS_OUT):
        grad_out_arr[i] = Scalar[dtype](1.0)

    var params_t = LayoutTensor[
        dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin
    ](combined.unsafe_ptr())

    # Analytical forward + backward
    var output_arr = InlineArray[Scalar[dtype], BATCH * HEADS_OUT](
        uninitialized=True
    )
    var output_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, HEADS_OUT), MutAnyOrigin
    ](output_arr.unsafe_ptr())
    var cache_arr = InlineArray[Scalar[dtype], BATCH * M.CACHE_SIZE](
        uninitialized=True
    )
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.CACHE_SIZE), MutAnyOrigin
    ](cache_arr.unsafe_ptr())

    M.forward[BATCH](feat_t, output_t, params_t, cache_t)

    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, HEADS_OUT), MutAnyOrigin
    ](grad_out_arr.unsafe_ptr())
    var grad_feat_arr = InlineArray[Scalar[dtype], BATCH * FEAT](
        uninitialized=True
    )
    var grad_feat_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, FEAT), MutAnyOrigin
    ](grad_feat_arr.unsafe_ptr())
    var grads_arr = InlineArray[Scalar[dtype], M.PARAM_SIZE](uninitialized=True)
    for i in range(M.PARAM_SIZE):
        grads_arr[i] = Scalar[dtype](0.0)
    var grads_t = LayoutTensor[
        dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin
    ](grads_arr.unsafe_ptr())

    M.backward[BATCH](grad_out_t, grad_feat_t, params_t, cache_t, grads_t)

    # Finite difference — input gradients (most important for BPTT)
    var eps = Float64(1e-4)
    var max_rel_i: Float64 = 0.0

    for in_idx in range(FEAT):
        var orig = feat_arr[in_idx]

        feat_arr[in_idx] = orig + Scalar[dtype](eps)
        var op = InlineArray[Scalar[dtype], BATCH * HEADS_OUT](
            uninitialized=True
        )
        var op_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, HEADS_OUT), MutAnyOrigin
        ](op.unsafe_ptr())
        M.forward[BATCH](feat_t, op_t, params_t)

        feat_arr[in_idx] = orig - Scalar[dtype](eps)
        var om = InlineArray[Scalar[dtype], BATCH * HEADS_OUT](
            uninitialized=True
        )
        var om_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, HEADS_OUT), MutAnyOrigin
        ](om.unsafe_ptr())
        M.forward[BATCH](feat_t, om_t, params_t)

        feat_arr[in_idx] = orig

        var fd: Float64 = 0.0
        for o in range(HEADS_OUT):
            fd += Float64(op[o] - om[o]) / (2.0 * eps)

        var ag = Float64(grad_feat_arr[in_idx])
        var ae = abs(fd - ag)
        var dn = max(abs(fd), abs(ag))
        if dn > 1e-4 and ae > 1e-5:
            var rel = ae / dn
            if rel > max_rel_i:
                max_rel_i = rel

    print("  Input grad max relative error:", max_rel_i)
    if max_rel_i > 0.05:
        raise Error("Input gradient check failed: " + String(max_rel_i))

    print("  PASSED")


def test_backward_feat_to_encoder() raises:
    """Verify backward_feat_to_encoder produces encoder + posterior grads."""
    print("Test 4: Backward feat → encoder...")

    comptime OBS = 6
    comptime ACT = 2
    comptime DETER = 32
    comptime HIDDEN = 16
    comptime STOCH = 4
    comptime CLASSES = 4
    comptime UNITS = 16
    comptime BINS = 15
    comptime BATCH = 2

    comptime RSSMType = RSSM[
        OBS, ACT, DETER, HIDDEN, STOCH, CLASSES, UNITS, BINS
    ]
    comptime FEAT = RSSMType.FEAT_DIM
    comptime HEADS_OUT = RSSMType.HEADS_OUT_DIM
    comptime STOCH_FLAT = STOCH * CLASSES

    var rssm = RSSMType()

    # Create inputs: obs, deter, post_probs
    var obs_arr = InlineArray[Scalar[dtype], BATCH * OBS](uninitialized=True)
    for i in range(BATCH * OBS):
        obs_arr[i] = Scalar[dtype](Float64(i + 1) * 0.1)
    var obs_t = LayoutTensor[dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin](
        obs_arr.unsafe_ptr()
    )

    var deter_arr = InlineArray[Scalar[dtype], BATCH * DETER](
        uninitialized=True
    )
    for i in range(BATCH * DETER):
        deter_arr[i] = Scalar[dtype](Float64(i + 1) * 0.02)
    var deter_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DETER), MutAnyOrigin
    ](deter_arr.unsafe_ptr())

    # Run observe_step to get realistic post_probs
    var prev_stoch = InlineArray[Scalar[dtype], BATCH * STOCH_FLAT](
        uninitialized=True
    )
    for i in range(BATCH * STOCH_FLAT):
        prev_stoch[i] = Scalar[dtype](0.0)
    var prev_stoch_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, STOCH_FLAT), MutAnyOrigin
    ](prev_stoch.unsafe_ptr())

    var prev_action = InlineArray[Scalar[dtype], BATCH * ACT](
        uninitialized=True
    )
    for i in range(BATCH * ACT):
        prev_action[i] = Scalar[dtype](0.0)
    var prev_action_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, ACT), MutAnyOrigin
    ](prev_action.unsafe_ptr())

    var new_deter = InlineArray[Scalar[dtype], BATCH * DETER](
        uninitialized=True
    )
    var new_deter_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DETER), MutAnyOrigin
    ](new_deter.unsafe_ptr())
    var new_stoch = InlineArray[Scalar[dtype], BATCH * STOCH_FLAT](
        uninitialized=True
    )
    var new_stoch_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, STOCH_FLAT), MutAnyOrigin
    ](new_stoch.unsafe_ptr())
    var post_probs = InlineArray[Scalar[dtype], BATCH * STOCH_FLAT](
        uninitialized=True
    )
    var post_probs_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, STOCH_FLAT), MutAnyOrigin
    ](post_probs.unsafe_ptr())
    var prior_probs = InlineArray[Scalar[dtype], BATCH * STOCH_FLAT](
        uninitialized=True
    )
    var prior_probs_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, STOCH_FLAT), MutAnyOrigin
    ](prior_probs.unsafe_ptr())
    var feat = InlineArray[Scalar[dtype], BATCH * FEAT](uninitialized=True)
    var feat_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, FEAT), MutAnyOrigin
    ](feat.unsafe_ptr())

    rssm.observe_step[BATCH](
        obs_t,
        deter_t,
        prev_stoch_t,
        prev_action_t,
        new_deter_t,
        new_stoch_t,
        post_probs_t,
        prior_probs_t,
        feat_t,
        training=False,  # deterministic for reproducibility
    )

    # Forward prediction heads
    var heads_out = InlineArray[Scalar[dtype], BATCH * HEADS_OUT](
        uninitialized=True
    )
    var heads_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, HEADS_OUT), MutAnyOrigin
    ](heads_out.unsafe_ptr())
    var cache_arr = InlineArray[
        Scalar[dtype], BATCH * RSSMType.HEADS_CACHE_SIZE
    ](uninitialized=True)
    var cache_t = LayoutTensor[
        dtype,
        Layout.row_major(BATCH, RSSMType.HEADS_CACHE_SIZE),
        MutAnyOrigin,
    ](cache_arr.unsafe_ptr())

    rssm.predict_all_heads[BATCH](feat_t, heads_t, cache_t)

    # Backward heads → grad_feat
    rssm.zero_all_grads()

    var grad_out = InlineArray[Scalar[dtype], BATCH * HEADS_OUT](
        uninitialized=True
    )
    for i in range(BATCH * HEADS_OUT):
        grad_out[i] = Scalar[dtype](1.0)
    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, HEADS_OUT), MutAnyOrigin
    ](grad_out.unsafe_ptr())

    var grad_feat = InlineArray[Scalar[dtype], BATCH * FEAT](uninitialized=True)
    var grad_feat_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, FEAT), MutAnyOrigin
    ](grad_feat.unsafe_ptr())

    rssm.backward_all_heads[BATCH](grad_out_t, grad_feat_t, cache_t)

    # Now backward feat → encoder
    var grad_deter = InlineArray[Scalar[dtype], BATCH * DETER](
        uninitialized=True
    )
    var grad_deter_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DETER), MutAnyOrigin
    ](grad_deter.unsafe_ptr())

    rssm.backward_feat_to_encoder[BATCH](
        grad_feat_t, obs_t, new_deter_t, post_probs_t, grad_deter_t
    )

    # Check grad_deter is non-zero
    var gd_norm: Float64 = 0.0
    for i in range(BATCH * DETER):
        var v = Float64(grad_deter[i])
        gd_norm += v * v
    print("  grad_deter L2 norm:", sqrt(gd_norm))

    # Check encoder has non-zero param grads
    var enc_grad_norm: Float64 = 0.0
    var enc_grads = rssm.encoder.grads_view()
    for i in range(RSSMType.EncModel.PARAM_SIZE):
        var v = Float64(enc_grads.ptr[i])
        enc_grad_norm += v * v

    # Check posterior has non-zero param grads
    var post_grad_norm: Float64 = 0.0
    var post_grads = rssm.posterior.grads_view()
    for i in range(RSSMType.PostModel.PARAM_SIZE):
        var v = Float64(post_grads.ptr[i])
        post_grad_norm += v * v

    print(
        "  Param grad norms — encoder:",
        sqrt(enc_grad_norm),
        "posterior:",
        sqrt(post_grad_norm),
    )

    if enc_grad_norm < 1e-20:
        raise Error("Encoder has zero param grads")
    if post_grad_norm < 1e-20:
        raise Error("Posterior has zero param grads")
    if gd_norm < 1e-20:
        raise Error("grad_deter is zero")

    print("  PASSED (encoder + posterior have grads, grad_deter non-zero)")


def test_backward_kl_loss() raises:
    """Verify backward_kl_loss produces prior + posterior + encoder grads."""
    print("Test 5: Backward KL loss...")

    comptime OBS = 6
    comptime ACT = 2
    comptime DETER = 32
    comptime HIDDEN = 16
    comptime STOCH = 4
    comptime CLASSES = 4
    comptime UNITS = 16
    comptime BINS = 15
    comptime BATCH = 2

    comptime RSSMType = RSSM[
        OBS, ACT, DETER, HIDDEN, STOCH, CLASSES, UNITS, BINS
    ]
    comptime FEAT = RSSMType.FEAT_DIM
    comptime STOCH_FLAT = STOCH * CLASSES

    var rssm = RSSMType()

    # Run observe_step to get realistic probs
    var obs_arr = InlineArray[Scalar[dtype], BATCH * OBS](uninitialized=True)
    for i in range(BATCH * OBS):
        obs_arr[i] = Scalar[dtype](Float64(i + 1) * 0.1)
    var obs_t = LayoutTensor[dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin](
        obs_arr.unsafe_ptr()
    )

    var prev_deter = InlineArray[Scalar[dtype], BATCH * DETER](
        uninitialized=True
    )
    for i in range(BATCH * DETER):
        prev_deter[i] = Scalar[dtype](Float64(i + 1) * 0.02)
    var prev_deter_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DETER), MutAnyOrigin
    ](prev_deter.unsafe_ptr())

    var prev_stoch = InlineArray[Scalar[dtype], BATCH * STOCH_FLAT](
        uninitialized=True
    )
    for i in range(BATCH * STOCH_FLAT):
        prev_stoch[i] = Scalar[dtype](0.0)
    var prev_stoch_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, STOCH_FLAT), MutAnyOrigin
    ](prev_stoch.unsafe_ptr())

    var prev_action = InlineArray[Scalar[dtype], BATCH * ACT](
        uninitialized=True
    )
    for i in range(BATCH * ACT):
        prev_action[i] = Scalar[dtype](0.0)
    var prev_action_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, ACT), MutAnyOrigin
    ](prev_action.unsafe_ptr())

    var new_deter = InlineArray[Scalar[dtype], BATCH * DETER](
        uninitialized=True
    )
    var new_deter_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DETER), MutAnyOrigin
    ](new_deter.unsafe_ptr())
    var new_stoch = InlineArray[Scalar[dtype], BATCH * STOCH_FLAT](
        uninitialized=True
    )
    var new_stoch_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, STOCH_FLAT), MutAnyOrigin
    ](new_stoch.unsafe_ptr())
    var post_probs = InlineArray[Scalar[dtype], BATCH * STOCH_FLAT](
        uninitialized=True
    )
    var post_probs_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, STOCH_FLAT), MutAnyOrigin
    ](post_probs.unsafe_ptr())
    var prior_probs = InlineArray[Scalar[dtype], BATCH * STOCH_FLAT](
        uninitialized=True
    )
    var prior_probs_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, STOCH_FLAT), MutAnyOrigin
    ](prior_probs.unsafe_ptr())
    var feat = InlineArray[Scalar[dtype], BATCH * FEAT](uninitialized=True)
    var feat_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, FEAT), MutAnyOrigin
    ](feat.unsafe_ptr())

    rssm.observe_step[BATCH](
        obs_t,
        prev_deter_t,
        prev_stoch_t,
        prev_action_t,
        new_deter_t,
        new_stoch_t,
        post_probs_t,
        prior_probs_t,
        feat_t,
        training=False,
    )

    # Compute KL to verify it's above free nats
    from mojo_rl.deep_agents.dreamer_v3.rssm import kl_divergence

    var kl_val = kl_divergence[BATCH, STOCH, CLASSES](
        post_probs_t, prior_probs_t
    )
    print("  KL divergence:", kl_val, "(free_nats: 1.0)")

    # Zero all grads
    rssm.zero_all_grads()

    # Run KL backward
    var grad_deter = InlineArray[Scalar[dtype], BATCH * DETER](
        uninitialized=True
    )
    for i in range(BATCH * DETER):
        grad_deter[i] = Scalar[dtype](0.0)
    var grad_deter_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DETER), MutAnyOrigin
    ](grad_deter.unsafe_ptr())

    rssm.backward_kl_loss[BATCH](
        obs_t,
        new_deter_t,
        post_probs_t,
        prior_probs_t,
        0.5,  # dyn_scale
        0.1,  # rep_scale
        grad_deter_t,
    )

    # Check grad_deter
    var gd_norm: Float64 = 0.0
    for i in range(BATCH * DETER):
        var v = Float64(grad_deter[i])
        gd_norm += v * v

    # Check prior has grads (from dyn_kl)
    var prior_grad_norm: Float64 = 0.0
    var prior_grads = rssm.prior.grads_view()
    for i in range(RSSMType.PriorModel.PARAM_SIZE):
        var v = Float64(prior_grads.ptr[i])
        prior_grad_norm += v * v

    # Check posterior has grads (from rep_kl)
    var post_grad_norm: Float64 = 0.0
    var post_grads = rssm.posterior.grads_view()
    for i in range(RSSMType.PostModel.PARAM_SIZE):
        var v = Float64(post_grads.ptr[i])
        post_grad_norm += v * v

    # Check encoder has grads (from rep_kl → posterior → encoder)
    var enc_grad_norm: Float64 = 0.0
    var enc_grads = rssm.encoder.grads_view()
    for i in range(RSSMType.EncModel.PARAM_SIZE):
        var v = Float64(enc_grads.ptr[i])
        enc_grad_norm += v * v

    print("  grad_deter L2 norm:", sqrt(gd_norm))
    print(
        "  Param grad norms — prior:",
        sqrt(prior_grad_norm),
        "posterior:",
        sqrt(post_grad_norm),
        "encoder:",
        sqrt(enc_grad_norm),
    )

    if kl_val > 1.0:
        # KL above free nats — should have gradients
        if prior_grad_norm < 1e-20:
            raise Error("Prior has zero grads (dyn_kl should train it)")
        if post_grad_norm < 1e-20:
            raise Error("Posterior has zero grads (rep_kl should train it)")
        print("  PASSED (KL > free_nats: all networks have grads)")
    else:
        # KL below free nats — grads should be zero (clamped)
        print("  OK (KL < free_nats: no gradient applied — correct)")
        print("  Running again with FREE_NATS=0 to verify gradient flow...")

        # Test with a zero free_nats RSSM to confirm gradient flow works
        comptime RSSMType0 = RSSM[
            OBS,
            ACT,
            DETER,
            HIDDEN,
            STOCH,
            CLASSES,
            UNITS,
            BINS,
            FREE_NATS=0.0,
        ]
        var rssm0 = RSSMType0()

        # Copy params from original rssm
        for i in range(RSSMType0.EncModel.PARAM_SIZE):
            rssm0.encoder.params_view().ptr[i] = rssm.encoder.params_view().ptr[
                i
            ]
        for i in range(RSSMType0.PostModel.PARAM_SIZE):
            rssm0.posterior.params_view().ptr[
                i
            ] = rssm.posterior.params_view().ptr[i]
        for i in range(RSSMType0.PriorModel.PARAM_SIZE):
            rssm0.prior.params_view().ptr[i] = rssm.prior.params_view().ptr[i]
        # Copy GRU params too
        for i in range(RSSMType0.DeterProj.PARAM_SIZE):
            rssm0.deter_proj.params_view().ptr[
                i
            ] = rssm.deter_proj.params_view().ptr[i]
        for i in range(RSSMType0.StochProj.PARAM_SIZE):
            rssm0.stoch_proj.params_view().ptr[
                i
            ] = rssm.stoch_proj.params_view().ptr[i]
        for i in range(RSSMType0.ActionProj.PARAM_SIZE):
            rssm0.action_proj.params_view().ptr[
                i
            ] = rssm.action_proj.params_view().ptr[i]
        for i in range(RSSMType0.GRUHiddenModel.PARAM_SIZE):
            rssm0.gru_hidden.params_view().ptr[
                i
            ] = rssm.gru_hidden.params_view().ptr[i]
        for i in range(RSSMType0.GRUGateModel.PARAM_SIZE):
            rssm0.gru_gates.params_view().ptr[
                i
            ] = rssm.gru_gates.params_view().ptr[i]

        # Re-run observe_step with new rssm
        var new_deter0 = InlineArray[Scalar[dtype], BATCH * DETER](
            uninitialized=True
        )
        var new_deter0_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, DETER), MutAnyOrigin
        ](new_deter0.unsafe_ptr())
        var new_stoch0 = InlineArray[Scalar[dtype], BATCH * STOCH_FLAT](
            uninitialized=True
        )
        var new_stoch0_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, STOCH_FLAT), MutAnyOrigin
        ](new_stoch0.unsafe_ptr())
        var post_probs0 = InlineArray[Scalar[dtype], BATCH * STOCH_FLAT](
            uninitialized=True
        )
        var post_probs0_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, STOCH_FLAT), MutAnyOrigin
        ](post_probs0.unsafe_ptr())
        var prior_probs0 = InlineArray[Scalar[dtype], BATCH * STOCH_FLAT](
            uninitialized=True
        )
        var prior_probs0_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, STOCH_FLAT), MutAnyOrigin
        ](prior_probs0.unsafe_ptr())
        var feat0 = InlineArray[Scalar[dtype], BATCH * FEAT](uninitialized=True)
        var feat0_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, FEAT), MutAnyOrigin
        ](feat0.unsafe_ptr())

        rssm0.observe_step[BATCH](
            obs_t,
            prev_deter_t,
            prev_stoch_t,
            prev_action_t,
            new_deter0_t,
            new_stoch0_t,
            post_probs0_t,
            prior_probs0_t,
            feat0_t,
            training=False,
        )

        rssm0.zero_all_grads()
        var gd0 = InlineArray[Scalar[dtype], BATCH * DETER](uninitialized=True)
        for i in range(BATCH * DETER):
            gd0[i] = Scalar[dtype](0.0)
        var gd0_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, DETER), MutAnyOrigin
        ](gd0.unsafe_ptr())

        rssm0.backward_kl_loss[BATCH](
            obs_t,
            new_deter0_t,
            post_probs0_t,
            prior_probs0_t,
            0.5,
            0.1,
            gd0_t,
        )

        var prior_gn0: Float64 = 0.0
        var pri_g0 = rssm0.prior.grads_view()
        for i in range(RSSMType0.PriorModel.PARAM_SIZE):
            var v = Float64(pri_g0.ptr[i])
            prior_gn0 += v * v

        var post_gn0: Float64 = 0.0
        var pos_g0 = rssm0.posterior.grads_view()
        for i in range(RSSMType0.PostModel.PARAM_SIZE):
            var v = Float64(pos_g0.ptr[i])
            post_gn0 += v * v

        var gd0_norm: Float64 = 0.0
        for i in range(BATCH * DETER):
            var v = Float64(gd0[i])
            gd0_norm += v * v

        print(
            "  FREE_NATS=0 — prior:",
            sqrt(prior_gn0),
            "posterior:",
            sqrt(post_gn0),
            "grad_deter:",
            sqrt(gd0_norm),
        )

        if prior_gn0 < 1e-20:
            raise Error("Prior has zero grads with FREE_NATS=0")
        if post_gn0 < 1e-20:
            raise Error("Posterior has zero grads with FREE_NATS=0")

        print("  PASSED (KL gradient flow verified with FREE_NATS=0)")


def test_backward_gru_core() raises:
    """Verify backward_gru_core produces grads for all GRU sub-networks."""
    print("Test 6: Backward GRU core...")

    comptime OBS = 6
    comptime ACT = 2
    comptime DETER = 32
    comptime HIDDEN = 16
    comptime STOCH = 4
    comptime CLASSES = 4
    comptime UNITS = 16
    comptime BINS = 15
    comptime BATCH = 2

    comptime RSSMType = RSSM[
        OBS, ACT, DETER, HIDDEN, STOCH, CLASSES, UNITS, BINS
    ]
    comptime STOCH_FLAT = STOCH * CLASSES

    var rssm = RSSMType()

    # Create inputs
    var prev_deter = InlineArray[Scalar[dtype], BATCH * DETER](
        uninitialized=True
    )
    for i in range(BATCH * DETER):
        prev_deter[i] = Scalar[dtype](Float64(i + 1) * 0.01)
    var prev_deter_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DETER), MutAnyOrigin
    ](prev_deter.unsafe_ptr())

    var prev_stoch = InlineArray[Scalar[dtype], BATCH * STOCH_FLAT](
        uninitialized=True
    )
    for i in range(BATCH * STOCH_FLAT):
        prev_stoch[i] = Scalar[dtype](Float64(i + 1) * 0.05)
    var prev_stoch_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, STOCH_FLAT), MutAnyOrigin
    ](prev_stoch.unsafe_ptr())

    var prev_action = InlineArray[Scalar[dtype], BATCH * ACT](
        uninitialized=True
    )
    for i in range(BATCH * ACT):
        prev_action[i] = Scalar[dtype](Float64(i + 1) * 0.3)
    var prev_action_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, ACT), MutAnyOrigin
    ](prev_action.unsafe_ptr())

    # Simulated grad_new_deter (as if from prediction heads + KL)
    var grad_new_deter = InlineArray[Scalar[dtype], BATCH * DETER](
        uninitialized=True
    )
    for i in range(BATCH * DETER):
        grad_new_deter[i] = Scalar[dtype](Float64(i % 7 + 1) * 0.01)
    var grad_nd_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DETER), MutAnyOrigin
    ](grad_new_deter.unsafe_ptr())

    # Zero all grads
    rssm.zero_all_grads()

    # Run GRU backward
    var grad_prev_deter = InlineArray[Scalar[dtype], BATCH * DETER](
        uninitialized=True
    )
    var grad_pd_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DETER), MutAnyOrigin
    ](grad_prev_deter.unsafe_ptr())

    var grad_prev_stoch = InlineArray[Scalar[dtype], BATCH * STOCH_FLAT](
        uninitialized=True
    )
    var grad_ps_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, STOCH_FLAT), MutAnyOrigin
    ](grad_prev_stoch.unsafe_ptr())

    rssm.backward_gru_core[BATCH](
        grad_nd_t,
        prev_deter_t,
        prev_stoch_t,
        prev_action_t,
        grad_pd_t,
        grad_ps_t,
    )

    # Check grad_prev_deter non-zero
    var gpd_norm: Float64 = 0.0
    for i in range(BATCH * DETER):
        var v = Float64(grad_prev_deter[i])
        gpd_norm += v * v

    # Check grad_prev_stoch non-zero
    var gps_norm: Float64 = 0.0
    for i in range(BATCH * STOCH_FLAT):
        var v = Float64(grad_prev_stoch[i])
        gps_norm += v * v

    print(
        "  grad_prev_deter L2:",
        sqrt(gpd_norm),
        "grad_prev_stoch L2:",
        sqrt(gps_norm),
    )

    # Check all 5 GRU sub-network param grads
    var dp_norm: Float64 = 0.0
    var dp_g = rssm.deter_proj.grads_view()
    for i in range(RSSMType.DeterProj.PARAM_SIZE):
        var v = Float64(dp_g.ptr[i])
        dp_norm += v * v

    var sp_norm: Float64 = 0.0
    var sp_g = rssm.stoch_proj.grads_view()
    for i in range(RSSMType.StochProj.PARAM_SIZE):
        var v = Float64(sp_g.ptr[i])
        sp_norm += v * v

    var ap_norm: Float64 = 0.0
    var ap_g = rssm.action_proj.grads_view()
    for i in range(RSSMType.ActionProj.PARAM_SIZE):
        var v = Float64(ap_g.ptr[i])
        ap_norm += v * v

    var gh_norm: Float64 = 0.0
    var gh_g = rssm.gru_hidden.grads_view()
    for i in range(RSSMType.GRUHiddenModel.PARAM_SIZE):
        var v = Float64(gh_g.ptr[i])
        gh_norm += v * v

    var gg_norm: Float64 = 0.0
    var gg_g = rssm.gru_gates.grads_view()
    for i in range(RSSMType.GRUGateModel.PARAM_SIZE):
        var v = Float64(gg_g.ptr[i])
        gg_norm += v * v

    print(
        "  Param grads — deter_proj:",
        sqrt(dp_norm),
        "stoch_proj:",
        sqrt(sp_norm),
        "action_proj:",
        sqrt(ap_norm),
    )
    print(
        "              gru_hidden:",
        sqrt(gh_norm),
        "gru_gates:",
        sqrt(gg_norm),
    )

    if gpd_norm < 1e-20:
        raise Error("grad_prev_deter is zero")
    if gps_norm < 1e-20:
        raise Error("grad_prev_stoch is zero")
    if dp_norm < 1e-20:
        raise Error("DeterProj has zero grads")
    if sp_norm < 1e-20:
        raise Error("StochProj has zero grads")
    if ap_norm < 1e-20:
        raise Error("ActionProj has zero grads")
    if gh_norm < 1e-20:
        raise Error("GRUHidden has zero grads")
    if gg_norm < 1e-20:
        raise Error("GRUGates has zero grads")

    print("  PASSED (all 5 GRU networks + prev_deter/stoch have grads)")


def main() raises:
    print("=" * 60)
    print("Dreamer V3 Autodiff Backward Tests")
    print("=" * 60)

    test_heads_forward()
    test_heads_backward_produces_grad_feat()
    test_heads_grad_check()
    test_backward_feat_to_encoder()
    test_backward_kl_loss()
    test_backward_gru_core()

    print("=" * 60)
    print("All Dreamer backward tests PASSED!")
    print("=" * 60)
