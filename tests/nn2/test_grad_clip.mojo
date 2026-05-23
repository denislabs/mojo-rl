"""B.3 — grad-clip walker unit tests.

Three checks:
  1. Disabled (max_grad_norm=0): walker no-ops, grads unchanged.
  2. Small grads (norm < max_grad_norm): grads unchanged, returned norm
     is the actual pre-clip norm.
  3. Large grads (norm > max_grad_norm): grads scaled so post-clip norm
     equals max_grad_norm exactly (to FP32 tol).
"""

from std.math import sqrt
from std.random import seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.core.grad_clip import clip_grads_auto
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU


comptime IN = 4
comptime HID = 8
comptime OUT = 2
comptime MLP = Sequential[
    Linear[IN, HID], ReLU[HID], Linear[HID, OUT],
]


def _set_all_grads(mut model: MLP, value: Scalar[DT]) raises:
    """Brute-force set every Param.grad to `value` across the MLP."""
    # children: [Linear[IN,HID], ReLU, Linear[HID,OUT]]
    # Linear.weight (IN*HID), Linear.bias (HID)
    var n0_w = IN * HID
    var n0_b = HID
    var n2_w = HID * OUT
    var n2_b = OUT
    var l0_w_ptr = model.children[0].weight.grad_unsafe_ptr_cpu()
    var l0_b_ptr = model.children[0].bias.grad_unsafe_ptr_cpu()
    var l2_w_ptr = model.children[2].weight.grad_unsafe_ptr_cpu()
    var l2_b_ptr = model.children[2].bias.grad_unsafe_ptr_cpu()
    for i in range(n0_w):
        l0_w_ptr[i] = value
    for i in range(n0_b):
        l0_b_ptr[i] = value
    for i in range(n2_w):
        l2_w_ptr[i] = value
    for i in range(n2_b):
        l2_b_ptr[i] = value


def _grad_norm(model: MLP) -> Scalar[DT]:
    var n0_w = IN * HID
    var n0_b = HID
    var n2_w = HID * OUT
    var n2_b = OUT
    var l0_w_ptr = model.children[0].weight.grad_unsafe_ptr_cpu()
    var l0_b_ptr = model.children[0].bias.grad_unsafe_ptr_cpu()
    var l2_w_ptr = model.children[2].weight.grad_unsafe_ptr_cpu()
    var l2_b_ptr = model.children[2].bias.grad_unsafe_ptr_cpu()
    var s = Scalar[DT](0.0)
    for i in range(n0_w):
        s += l0_w_ptr[i] * l0_w_ptr[i]
    for i in range(n0_b):
        s += l0_b_ptr[i] * l0_b_ptr[i]
    for i in range(n2_w):
        s += l2_w_ptr[i] * l2_w_ptr[i]
    for i in range(n2_b):
        s += l2_b_ptr[i] * l2_b_ptr[i]
    return sqrt(s)


def test_disabled_is_noop() raises:
    print("test_disabled_is_noop ...")
    seed(42)
    var net = MLP.make[target="cpu", INIT=Kaiming]()
    _set_all_grads(net, Scalar[DT](7.0))
    var pre_norm = _grad_norm(net)
    var returned = clip_grads_auto[MLP, "cpu"](net, Scalar[DT](0.0))
    var post_norm = _grad_norm(net)
    assert_true(
        returned == Scalar[DT](0.0),
        "Disabled walker should return 0, got " + String(returned),
    )
    assert_true(
        (pre_norm - post_norm).__abs__() < Scalar[DT](1e-5),
        "Grads should be unchanged when disabled: pre="
        + String(pre_norm) + " post=" + String(post_norm),
    )
    print("  PASSED (pre=post=", pre_norm, ")")


def test_small_grads_unchanged() raises:
    print("test_small_grads_unchanged ...")
    seed(42)
    var net = MLP.make[target="cpu", INIT=Kaiming]()
    _set_all_grads(net, Scalar[DT](0.01))
    var pre_norm = _grad_norm(net)
    # Threshold 100x the actual norm — clip should never trigger.
    var max_norm = pre_norm * Scalar[DT](100.0)
    var returned = clip_grads_auto[MLP, "cpu"](net, max_norm)
    var post_norm = _grad_norm(net)
    assert_true(
        (returned - pre_norm).__abs__() < Scalar[DT](1e-4),
        "Returned norm should match pre-clip norm: "
        + String(returned) + " vs " + String(pre_norm),
    )
    assert_true(
        (post_norm - pre_norm).__abs__() < Scalar[DT](1e-5),
        "Grads must not change when norm < max_norm: pre="
        + String(pre_norm) + " post=" + String(post_norm),
    )
    print("  PASSED (norm=", pre_norm, " < max=", max_norm, ")")


def test_huge_grads_clipped_to_max() raises:
    print("test_huge_grads_clipped_to_max ...")
    seed(42)
    var net = MLP.make[target="cpu", INIT=Kaiming]()
    _set_all_grads(net, Scalar[DT](100.0))
    var pre_norm = _grad_norm(net)
    var max_norm = Scalar[DT](1.0)
    var returned = clip_grads_auto[MLP, "cpu"](net, max_norm)
    var post_norm = _grad_norm(net)
    assert_true(
        (returned - pre_norm).__abs__() < Scalar[DT](1.0),
        "Returned norm should equal pre-clip norm: "
        + String(returned) + " vs " + String(pre_norm),
    )
    assert_true(
        (post_norm - max_norm).__abs__() < Scalar[DT](1e-4),
        "Post-clip norm should equal max_norm: post="
        + String(post_norm) + " max=" + String(max_norm),
    )
    print("  PASSED (pre=", pre_norm, " -> post=", post_norm, ")")


def main() raises:
    print("=" * 60)
    print("B.3 grad-clip walker")
    print("=" * 60)
    test_disabled_is_noop()
    test_small_grads_unchanged()
    test_huge_grads_clipped_to_max()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
