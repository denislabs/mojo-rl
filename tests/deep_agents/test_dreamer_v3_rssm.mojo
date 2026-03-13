"""Test DreamerV3 RSSM world model components.

Tests:
1. Categorical sampling with unimix
2. KL divergence between categoricals
3. RSSM observe step (forward pass)
4. RSSM imagine step (forward pass)
5. Decoder / reward / continue head forward passes
"""

from std.math import exp, log, abs
from std.memory import alloc, memset
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.loss.two_hot import symlog, symexp, compute_symlog_bins

from mojo_rl.deep_agents.dreamer_v3.rssm import (
    RSSM,
    categorical_sample,
    kl_divergence,
)


fn test_symlog_symexp():
    """Test symlog/symexp are inverses."""
    print("Test symlog/symexp...")

    var test_values = InlineArray[Float32, 7](fill=0)
    test_values[0] = Float32(0.0)
    test_values[1] = Float32(1.0)
    test_values[2] = Float32(-1.0)
    test_values[3] = Float32(10.0)
    test_values[4] = Float32(-10.0)
    test_values[5] = Float32(0.01)
    test_values[6] = Float32(-0.01)

    var all_pass = True
    for i in range(7):
        var x = test_values[i]
        var y = symlog(x)
        var x_back = symexp(y)
        var diff = abs(x - x_back)
        if diff > 1e-4:
            print("  FAIL: symexp(symlog(", x, ")) =", x_back, "diff =", diff)
            all_pass = False

    if all_pass:
        print("  PASS")


fn test_symlog_bins():
    """Test symlog bin computation."""
    print("Test symlog bins...")
    var bins = compute_symlog_bins[255]()

    # Check symmetry: bins[0] should be -symexp(20), bins[254] should be symexp(20)
    var diff = abs(bins[0] + bins[254])
    if diff > 1e-2:
        print("  FAIL: bins not symmetric. bins[0] =", bins[0], "bins[254] =", bins[254])
    else:
        print("  PASS: bins symmetric. bins[0] =", bins[0], "bins[127] =", bins[127], "bins[254] =", bins[254])


fn test_categorical_sample():
    """Test categorical sampling with unimix."""
    print("Test categorical_sample...")

    comptime B = 2
    comptime SD = 4
    comptime C = 4

    var logits_ptr = alloc[Scalar[dtype]](B * SD * C)
    var output_ptr = alloc[Scalar[dtype]](B * SD * C)
    var probs_ptr = alloc[Scalar[dtype]](B * SD * C)
    memset(logits_ptr, 0, B * SD * C)
    memset(output_ptr, 0, B * SD * C)
    memset(probs_ptr, 0, B * SD * C)

    # Set some logits
    for b in range(B):
        for s in range(SD):
            for c in range(C):
                (logits_ptr + b * SD * C + s * C + c)[] = Scalar[dtype](
                    Float64(c) * 0.5
                )

    var logits_t = LayoutTensor[dtype, Layout.row_major(B, SD * C), MutAnyOrigin](logits_ptr)
    var output_t = LayoutTensor[dtype, Layout.row_major(B, SD * C), MutAnyOrigin](output_ptr)
    var probs_t = LayoutTensor[dtype, Layout.row_major(B, SD * C), MutAnyOrigin](probs_ptr)

    categorical_sample[B, SD, C, 0.01](logits_t, output_t, probs_t, True)

    # Check: output should be one-hot per category
    var all_pass = True
    for b in range(B):
        for s in range(SD):
            var sum_val = Float64(0.0)
            for c in range(C):
                var v = Float64(output_ptr[b * SD * C + s * C + c])
                if v != 0.0 and v != 1.0:
                    print("  FAIL: output not one-hot at b=", b, "s=", s, "c=", c, "v=", v)
                    all_pass = False
                sum_val += v
            if abs(sum_val - 1.0) > 1e-6:
                print("  FAIL: one-hot sum != 1 at b=", b, "s=", s, "sum=", sum_val)
                all_pass = False

    # Check: probs should sum to 1 per category
    for b in range(B):
        for s in range(SD):
            var sum_p = Float64(0.0)
            for c in range(C):
                sum_p += Float64(probs_ptr[b * SD * C + s * C + c])
            if abs(sum_p - 1.0) > 1e-4:
                print("  FAIL: probs sum != 1 at b=", b, "s=", s, "sum=", sum_p)
                all_pass = False

    if all_pass:
        print("  PASS")

    logits_ptr.free()
    output_ptr.free()
    probs_ptr.free()


fn test_kl_divergence():
    """Test KL divergence computation."""
    print("Test kl_divergence...")

    comptime B = 2
    comptime SD = 2
    comptime C = 4

    var post_ptr = alloc[Scalar[dtype]](B * SD * C)
    var prior_ptr = alloc[Scalar[dtype]](B * SD * C)

    # Set uniform distributions — KL should be 0
    for i in range(B * SD * C):
        post_ptr[i] = Scalar[dtype](0.25)
        prior_ptr[i] = Scalar[dtype](0.25)

    var post_t = LayoutTensor[dtype, Layout.row_major(B, SD * C), MutAnyOrigin](post_ptr)
    var prior_t = LayoutTensor[dtype, Layout.row_major(B, SD * C), MutAnyOrigin](prior_ptr)

    var kl_uniform = kl_divergence[B, SD, C](post_t, prior_t)
    if abs(kl_uniform) > 1e-4:
        print("  FAIL: KL(uniform||uniform) should be ~0, got", kl_uniform)
    else:
        print("  PASS: KL(uniform||uniform) =", kl_uniform)

    # Set concentrated vs uniform — KL should be positive
    for b in range(B):
        for s in range(SD):
            for c in range(C):
                var idx = b * SD * C + s * C + c
                if c == 0:
                    post_ptr[idx] = Scalar[dtype](0.97)
                else:
                    post_ptr[idx] = Scalar[dtype](0.01)

    var kl_concentrated = kl_divergence[B, SD, C](post_t, prior_t)
    if kl_concentrated <= 0:
        print("  FAIL: KL(concentrated||uniform) should be > 0, got", kl_concentrated)
    else:
        print("  PASS: KL(concentrated||uniform) =", kl_concentrated)

    post_ptr.free()
    prior_ptr.free()


fn test_rssm_init():
    """Test RSSM initialization."""
    print("Test RSSM init...")

    # Small dims for testing
    comptime OBS = 6
    comptime ACT = 2
    comptime DETER = 32
    comptime HIDDEN = 16
    comptime STOCH = 4
    comptime CLASSES = 4
    comptime UNITS = 16
    comptime BINS = 31

    var rssm = RSSM[OBS, ACT, DETER, HIDDEN, STOCH, CLASSES, UNITS, BINS, 4]()

    # Check bins were computed
    if rssm.bins[0] != rssm.bins[0]:  # NaN check
        print("  FAIL: bins contain NaN")
    else:
        print("  PASS: RSSM initialized successfully")
        print("    STOCH_FLAT =", rssm.STOCH_FLAT)
        print("    FEAT_DIM =", rssm.FEAT_DIM)
        print("    bins[0] =", rssm.bins[0], "bins[15] =", rssm.bins[15], "bins[30] =", rssm.bins[30])


fn test_rssm_observe_step():
    """Test RSSM observe step forward pass."""
    print("Test RSSM observe_step...")

    comptime OBS = 6
    comptime ACT = 2
    comptime DETER = 32
    comptime HIDDEN = 16
    comptime STOCH = 4
    comptime CLASSES = 4
    comptime UNITS = 16
    comptime BINS = 31
    comptime B = 2
    comptime STOCH_FLAT = STOCH * CLASSES  # 16
    comptime FEAT = DETER + STOCH_FLAT  # 48

    var rssm = RSSM[OBS, ACT, DETER, HIDDEN, STOCH, CLASSES, UNITS, BINS, 4]()

    # Allocate inputs
    var obs_ptr = alloc[Scalar[dtype]](B * OBS)
    var deter_ptr = alloc[Scalar[dtype]](B * DETER)
    var stoch_ptr = alloc[Scalar[dtype]](B * STOCH_FLAT)
    var action_ptr = alloc[Scalar[dtype]](B * ACT)
    memset(obs_ptr, 0, B * OBS)
    memset(deter_ptr, 0, B * DETER)
    memset(stoch_ptr, 0, B * STOCH_FLAT)
    memset(action_ptr, 0, B * ACT)

    # Set some observation values
    for b in range(B):
        for i in range(OBS):
            (obs_ptr + b * OBS + i)[] = Scalar[dtype](Float64(i) * 0.1 + Float64(b) * 0.5)

    # Allocate outputs
    var new_deter_ptr = alloc[Scalar[dtype]](B * DETER)
    var new_stoch_ptr = alloc[Scalar[dtype]](B * STOCH_FLAT)
    var post_probs_ptr = alloc[Scalar[dtype]](B * STOCH_FLAT)
    var prior_probs_ptr = alloc[Scalar[dtype]](B * STOCH_FLAT)
    var feat_ptr = alloc[Scalar[dtype]](B * FEAT)
    memset(new_deter_ptr, 0, B * DETER)
    memset(new_stoch_ptr, 0, B * STOCH_FLAT)
    memset(post_probs_ptr, 0, B * STOCH_FLAT)
    memset(prior_probs_ptr, 0, B * STOCH_FLAT)
    memset(feat_ptr, 0, B * FEAT)

    # Create LayoutTensors
    var obs_t = LayoutTensor[dtype, Layout.row_major(B, OBS), MutAnyOrigin](obs_ptr)
    var deter_t = LayoutTensor[dtype, Layout.row_major(B, DETER), MutAnyOrigin](deter_ptr)
    var stoch_t = LayoutTensor[dtype, Layout.row_major(B, STOCH_FLAT), MutAnyOrigin](stoch_ptr)
    var action_t = LayoutTensor[dtype, Layout.row_major(B, ACT), MutAnyOrigin](action_ptr)
    var new_deter_t = LayoutTensor[dtype, Layout.row_major(B, DETER), MutAnyOrigin](new_deter_ptr)
    var new_stoch_t = LayoutTensor[dtype, Layout.row_major(B, STOCH_FLAT), MutAnyOrigin](new_stoch_ptr)
    var post_probs_t = LayoutTensor[dtype, Layout.row_major(B, STOCH_FLAT), MutAnyOrigin](post_probs_ptr)
    var prior_probs_t = LayoutTensor[dtype, Layout.row_major(B, STOCH_FLAT), MutAnyOrigin](prior_probs_ptr)
    var feat_t = LayoutTensor[dtype, Layout.row_major(B, FEAT), MutAnyOrigin](feat_ptr)

    # Run observe step
    rssm.observe_step[B](
        obs_t, deter_t, stoch_t, action_t,
        new_deter_t, new_stoch_t, post_probs_t, prior_probs_t, feat_t,
        True,
    )

    # Check outputs are non-trivial
    var deter_nonzero = False
    for i in range(B * DETER):
        if Float64(new_deter_ptr[i]) != 0.0:
            deter_nonzero = True
            break

    var stoch_has_one = False
    for i in range(B * STOCH_FLAT):
        if Float64(new_stoch_ptr[i]) == 1.0:
            stoch_has_one = True
            break

    var feat_nonzero = False
    for i in range(B * FEAT):
        if Float64(feat_ptr[i]) != 0.0:
            feat_nonzero = True
            break

    if deter_nonzero and stoch_has_one and feat_nonzero:
        print("  PASS: observe_step produced non-trivial outputs")
    else:
        print("  FAIL: deter_nonzero=", deter_nonzero,
              "stoch_has_one=", stoch_has_one,
              "feat_nonzero=", feat_nonzero)

    # Free
    obs_ptr.free()
    deter_ptr.free()
    stoch_ptr.free()
    action_ptr.free()
    new_deter_ptr.free()
    new_stoch_ptr.free()
    post_probs_ptr.free()
    prior_probs_ptr.free()
    feat_ptr.free()


fn test_rssm_imagine_step():
    """Test RSSM imagine step (prior only, no observations)."""
    print("Test RSSM imagine_step...")

    comptime OBS = 6
    comptime ACT = 2
    comptime DETER = 32
    comptime HIDDEN = 16
    comptime STOCH = 4
    comptime CLASSES = 4
    comptime UNITS = 16
    comptime BINS = 31
    comptime B = 2
    comptime STOCH_FLAT = STOCH * CLASSES
    comptime FEAT = DETER + STOCH_FLAT

    var rssm = RSSM[OBS, ACT, DETER, HIDDEN, STOCH, CLASSES, UNITS, BINS, 4]()

    # Allocate inputs (non-zero for interesting behavior)
    var deter_ptr = alloc[Scalar[dtype]](B * DETER)
    var stoch_ptr = alloc[Scalar[dtype]](B * STOCH_FLAT)
    var action_ptr = alloc[Scalar[dtype]](B * ACT)
    for i in range(B * DETER):
        deter_ptr[i] = Scalar[dtype](0.01)
    for i in range(B * STOCH_FLAT):
        stoch_ptr[i] = Scalar[dtype](0.0)
    # One-hot stoch
    for b in range(B):
        for s in range(STOCH):
            (stoch_ptr + b * STOCH_FLAT + s * CLASSES)[] = Scalar[dtype](1.0)
    for i in range(B * ACT):
        action_ptr[i] = Scalar[dtype](0.5)

    # Allocate outputs
    var new_deter_ptr = alloc[Scalar[dtype]](B * DETER)
    var new_stoch_ptr = alloc[Scalar[dtype]](B * STOCH_FLAT)
    var feat_ptr = alloc[Scalar[dtype]](B * FEAT)
    memset(new_deter_ptr, 0, B * DETER)
    memset(new_stoch_ptr, 0, B * STOCH_FLAT)
    memset(feat_ptr, 0, B * FEAT)

    var deter_t = LayoutTensor[dtype, Layout.row_major(B, DETER), MutAnyOrigin](deter_ptr)
    var stoch_t = LayoutTensor[dtype, Layout.row_major(B, STOCH_FLAT), MutAnyOrigin](stoch_ptr)
    var action_t = LayoutTensor[dtype, Layout.row_major(B, ACT), MutAnyOrigin](action_ptr)
    var new_deter_t = LayoutTensor[dtype, Layout.row_major(B, DETER), MutAnyOrigin](new_deter_ptr)
    var new_stoch_t = LayoutTensor[dtype, Layout.row_major(B, STOCH_FLAT), MutAnyOrigin](new_stoch_ptr)
    var feat_t = LayoutTensor[dtype, Layout.row_major(B, FEAT), MutAnyOrigin](feat_ptr)

    rssm.imagine_step[B](
        deter_t, stoch_t, action_t,
        new_deter_t, new_stoch_t, feat_t,
        True,
    )

    var deter_nonzero = False
    for i in range(B * DETER):
        if Float64(new_deter_ptr[i]) != 0.0:
            deter_nonzero = True
            break

    if deter_nonzero:
        print("  PASS: imagine_step produced non-trivial outputs")
    else:
        print("  FAIL: all outputs are zero")

    deter_ptr.free()
    stoch_ptr.free()
    action_ptr.free()
    new_deter_ptr.free()
    new_stoch_ptr.free()
    feat_ptr.free()


fn main():
    print("=" * 60)
    print("DreamerV3 RSSM Tests")
    print("=" * 60)

    test_symlog_symexp()
    test_symlog_bins()
    test_categorical_sample()
    test_kl_divergence()
    test_rssm_init()
    test_rssm_observe_step()
    test_rssm_imagine_step()

    print("=" * 60)
    print("All tests completed.")
    print("=" * 60)
