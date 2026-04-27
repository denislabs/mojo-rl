"""Level 4: Optimizer step correctness tests.

Verifies that each optimizer's update rule matches its mathematical formula
by computing expected values in Float64 and comparing against the actual
float32 step output.

Tests SGD, Adam, and AdamW over multiple steps with varied gradients.

Usage:
    pixi run mojo run -I . tests/nn/test_optimizer_step.mojo
"""

from std.math import abs, sqrt
from std.memory import alloc, memset, UnsafePointer
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.optimizer import SGD, Adam, AdamW


def test_sgd() raises:
    """SGD: p -= lr * g."""
    print("SGD step correctness:")
    comptime PS = 4
    comptime LR = 0.01

    var params = alloc[Scalar[dtype]](PS)
    var grads = alloc[Scalar[dtype]](PS)
    var state = alloc[Scalar[dtype]](PS)
    memset(state, 0, PS)

    # Initial params and gradients
    var p0 = alloc[Float64](PS)
    var g = alloc[Float64](PS)
    (p0 + 0)[] = 1.0;   (g + 0)[] = 2.0
    (p0 + 1)[] = -0.5;  (g + 1)[] = -1.0
    (p0 + 2)[] = 0.0;   (g + 2)[] = 0.5
    (p0 + 3)[] = 3.14;  (g + 3)[] = 0.0

    for i in range(PS):
        (params + i)[] = Scalar[dtype]((p0 + i)[])
        (grads + i)[] = Scalar[dtype]((g + i)[])

    var params_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](params)
    var grads_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](grads)
    var state_t = LayoutTensor[dtype, Layout.row_major(PS, 1), MutAnyOrigin](state)
    # Zero-length opt_global_state (SGD doesn't use it).
    var og_t = LayoutTensor[dtype, Layout.row_major(0), MutAnyOrigin](
        UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=0)
    )

    # Step 1
    SGD[LR].step[PS](params_t, grads_t, state_t, og_t, step_num=1)

    var max_err: Float64 = 0.0
    for i in range(PS):
        var expected = (p0 + i)[] - LR * (g + i)[]
        var err = abs(Float64((params + i)[]) - expected)
        if err > max_err:
            max_err = err
        (p0 + i)[] = expected  # track for step 2

    if max_err < 1e-6:
        print("  [PASS] step 1: max_err=", max_err)
    else:
        print("  [FAIL] step 1: max_err=", max_err)

    # Step 2: uniform gradient, lr_scale=0.5
    for i in range(PS):
        (grads + i)[] = Scalar[dtype](0.1)

    comptime SCALE = 0.5
    SGD[LR].step[PS](params_t, grads_t, state_t, og_t, step_num=2, lr_scale=SCALE)

    max_err = 0.0
    for i in range(PS):
        var expected = (p0 + i)[] - LR * SCALE * 0.1
        var err = abs(Float64((params + i)[]) - expected)
        if err > max_err:
            max_err = err

    if max_err < 1e-6:
        print("  [PASS] step 2 (lr_scale=0.5): max_err=", max_err)
    else:
        print("  [FAIL] step 2 (lr_scale=0.5): max_err=", max_err)

    params.free()
    grads.free()
    state.free()
    p0.free()
    g.free()
    print()


def test_adam() raises:
    """Adam: verify moments, bias correction, and param update over 2 steps."""
    print("Adam step correctness:")
    comptime PS = 4
    comptime LR = 0.001
    comptime B1 = 0.9
    comptime B2 = 0.999
    comptime EPS = 1e-8

    var params = alloc[Scalar[dtype]](PS)
    var grads = alloc[Scalar[dtype]](PS)
    var state = alloc[Scalar[dtype]](PS * 2)
    memset(state, 0, PS * 2)

    # Track expected values in Float64
    var p = alloc[Float64](PS)
    var m = alloc[Float64](PS)
    var v = alloc[Float64](PS)
    var g1 = alloc[Float64](PS)
    var g2 = alloc[Float64](PS)

    (p + 0)[] = 1.0;  (p + 1)[] = -0.5; (p + 2)[] = 0.0;  (p + 3)[] = 2.0
    (g1 + 0)[] = 0.5; (g1 + 1)[] = -0.3; (g1 + 2)[] = 1.0; (g1 + 3)[] = 0.0
    (g2 + 0)[] = 0.2; (g2 + 1)[] = 0.1; (g2 + 2)[] = -0.5; (g2 + 3)[] = 0.8
    for i in range(PS):
        (m + i)[] = 0.0
        (v + i)[] = 0.0
        (params + i)[] = Scalar[dtype]((p + i)[])
        (grads + i)[] = Scalar[dtype]((g1 + i)[])

    var params_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](params)
    var grads_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](grads)
    var state_t = LayoutTensor[dtype, Layout.row_major(PS, 2), MutAnyOrigin](state)
    # Phase 4: Adam.GLOBAL_STATE_SIZE = 1 (Float32 slot bit-patterning a UInt32
    # device step counter). The CPU `step()` path doesn't consult it, but the
    # signature still requires a 1-element tensor.
    var og_buf = alloc[Scalar[dtype]](1)
    (og_buf + 0)[] = Scalar[dtype](0.0)
    var og_t = LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin](og_buf)

    # Step 1
    Adam[LR, B1, B2, EPS].step[PS](params_t, grads_t, state_t, og_t, step_num=1)

    var bc1 = 1.0 - B1   # 0.1
    var bc2 = 1.0 - B2   # 0.001
    var max_err: Float64 = 0.0
    for i in range(PS):
        (m + i)[] = B1 * 0.0 + (1.0 - B1) * (g1 + i)[]
        (v + i)[] = B2 * 0.0 + (1.0 - B2) * (g1 + i)[] * (g1 + i)[]
        var m_hat = (m + i)[] / bc1
        var v_hat = (v + i)[] / bc2
        (p + i)[] = (p + i)[] - LR * m_hat / (sqrt(v_hat) + EPS)
        var err = abs(Float64((params + i)[]) - (p + i)[])
        if err > max_err:
            max_err = err

    if max_err < 1e-6:
        print("  [PASS] step 1: max_err=", max_err)
    else:
        print("  [FAIL] step 1: max_err=", max_err)

    # Verify state (moments)
    var state_err: Float64 = 0.0
    for i in range(PS):
        var m_actual = Float64((state + i * 2 + 0)[])
        var v_actual = Float64((state + i * 2 + 1)[])
        var e1 = abs(m_actual - (m + i)[])
        var e2 = abs(v_actual - (v + i)[])
        if e1 > state_err:
            state_err = e1
        if e2 > state_err:
            state_err = e2

    if state_err < 1e-6:
        print("  [PASS] moments after step 1: max_err=", state_err)
    else:
        print("  [FAIL] moments after step 1: max_err=", state_err)

    # Step 2 with different gradients
    for i in range(PS):
        (grads + i)[] = Scalar[dtype]((g2 + i)[])

    Adam[LR, B1, B2, EPS].step[PS](params_t, grads_t, state_t, og_t, step_num=2)

    var bc1_2 = 1.0 - B1 * B1    # 1 - 0.9^2 = 0.19
    var bc2_2 = 1.0 - B2 * B2    # 1 - 0.999^2 = 0.001999
    max_err = 0.0
    for i in range(PS):
        (m + i)[] = B1 * (m + i)[] + (1.0 - B1) * (g2 + i)[]
        (v + i)[] = B2 * (v + i)[] + (1.0 - B2) * (g2 + i)[] * (g2 + i)[]
        var m_hat = (m + i)[] / bc1_2
        var v_hat = (v + i)[] / bc2_2
        (p + i)[] = (p + i)[] - LR * m_hat / (sqrt(v_hat) + EPS)
        var err = abs(Float64((params + i)[]) - (p + i)[])
        if err > max_err:
            max_err = err

    if max_err < 1e-5:
        print("  [PASS] step 2: max_err=", max_err)
    else:
        print("  [FAIL] step 2: max_err=", max_err)

    params.free()
    grads.free()
    state.free()
    og_buf.free()
    p.free()
    m.free()
    v.free()
    g1.free()
    g2.free()
    print()


def test_adamw() raises:
    """AdamW: verify decoupled weight decay."""
    print("AdamW step correctness:")
    comptime PS = 4
    comptime LR = 0.001
    comptime B1 = 0.9
    comptime B2 = 0.999
    comptime EPS = 1e-8
    comptime WD = 0.01

    var params = alloc[Scalar[dtype]](PS)
    var grads = alloc[Scalar[dtype]](PS)
    var state = alloc[Scalar[dtype]](PS * 2)
    memset(state, 0, PS * 2)

    var p = alloc[Float64](PS)
    var g = alloc[Float64](PS)
    (p + 0)[] = 1.0;  (p + 1)[] = -0.5; (p + 2)[] = 0.0;  (p + 3)[] = 2.0
    (g + 0)[] = 0.5;  (g + 1)[] = -0.3; (g + 2)[] = 1.0;  (g + 3)[] = 0.0

    for i in range(PS):
        (params + i)[] = Scalar[dtype]((p + i)[])
        (grads + i)[] = Scalar[dtype]((g + i)[])

    var params_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](params)
    var grads_t = LayoutTensor[dtype, Layout.row_major(PS), MutAnyOrigin](grads)
    var state_t = LayoutTensor[dtype, Layout.row_major(PS, 2), MutAnyOrigin](state)
    # Phase 4: AdamW.GLOBAL_STATE_SIZE = 1 — see test_adam.
    var og_buf = alloc[Scalar[dtype]](1)
    (og_buf + 0)[] = Scalar[dtype](0.0)
    var og_t = LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin](og_buf)

    # Step 1
    AdamW[LR, B1, B2, EPS, WD].step[PS](params_t, grads_t, state_t, og_t, step_num=1)

    var bc1 = 1.0 - B1
    var bc2 = 1.0 - B2
    var wd_factor = 1.0 - LR * WD

    var max_err: Float64 = 0.0
    for i in range(PS):
        var m_new = (1.0 - B1) * (g + i)[]
        var v_new = (1.0 - B2) * (g + i)[] * (g + i)[]
        var m_hat = m_new / bc1
        var v_hat = v_new / bc2
        var expected = (p + i)[] * wd_factor - LR * m_hat / (sqrt(v_hat) + EPS)
        var err = abs(Float64((params + i)[]) - expected)
        if err > max_err:
            max_err = err
        (p + i)[] = expected

    if max_err < 1e-6:
        print("  [PASS] step 1: max_err=", max_err)
    else:
        print("  [FAIL] step 1: max_err=", max_err)

    # Verify weight decay: param[3] had g=0, so Adam wouldn't move it,
    # but AdamW shrinks it by wd_factor
    var p3_actual = Float64((params + 3)[])
    var p3_expected = 2.0 * wd_factor  # Only WD, no adam update (g=0 -> m=v=0)
    if abs(p3_actual - p3_expected) < 1e-6:
        print("  [PASS] weight decay shrinks param with zero gradient")
    else:
        print(
            "  [FAIL] weight decay: actual=",
            p3_actual,
            "expected=",
            p3_expected,
        )

    params.free()
    grads.free()
    state.free()
    og_buf.free()
    p.free()
    g.free()
    print()


def main() raises:
    print("=== Optimizer Step Correctness ===")
    print()
    test_sgd()
    test_adam()
    test_adamw()
    print("=== Done ===")
