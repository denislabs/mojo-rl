"""InferenceMode propagation — Phase 5.3.

Verifies that `set_inference(value)` propagates through nested
Sequentials to every leaf Module. Current leaves (Linear, ReLU, Tanh,
StopGrad) don't yet change behavior on the flag — this test pins down
the propagation mechanism so that future inference-sensitive layers
(Dropout, BN, NoisyLinear) inherit working machinery from day one.

Also verifies that `set_inference(False)` correctly reverts the flag.
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.tanh import Tanh
from mojo_rl.nn2.primitives.stop_grad import StopGrad
from mojo_rl.nn2.combinators import Sequential


def test_default_is_train_mode_cpu() raises:
    var net = Sequential[
        Linear[4, 8], ReLU[8], Linear[8, 4], Tanh[4]
    ].make["cpu", INIT=Zero]()
    # All freshly-made modules should be in train mode (_inference=False).
    assert_true(not net._inference, "Sequential default mode")
    assert_true(not net.children[0]._inference, "child[0] Linear default mode")
    assert_true(not net.children[1]._inference, "child[1] ReLU default mode")
    assert_true(not net.children[2]._inference, "child[2] Linear default mode")
    assert_true(not net.children[3]._inference, "child[3] Tanh default mode")
    print("  test_default_is_train_mode_cpu PASSED")


def test_set_inference_propagates_cpu() raises:
    """set_inference(True) reaches every leaf in a nested Sequential."""
    # Nested topology: outer Sequential containing an inner Sequential
    # mixed with a leaf StopGrad — exercises recursion through
    # combinator-of-combinator.
    var inner = Sequential[Linear[3, 5], ReLU[5], Linear[5, 2]].make[
        "cpu", INIT=Zero,
    ]()
    var outer = Sequential[
        Linear[2, 3],
        StopGrad[3],
        Linear[3, 4],
    ].make["cpu", INIT=Zero]()

    # Test on the simpler 4-layer chain first.
    outer.set_inference(True)
    assert_true(outer._inference, "outer flag set")
    assert_true(outer.children[0]._inference, "child[0] flag set")
    assert_true(outer.children[1]._inference, "child[1] StopGrad flag set")
    assert_true(outer.children[2]._inference, "child[2] flag set")

    # Toggle back.
    outer.set_inference(False)
    assert_true(not outer._inference, "outer flag cleared")
    assert_true(not outer.children[0]._inference, "child[0] flag cleared")
    assert_true(not outer.children[1]._inference, "child[1] flag cleared")
    assert_true(not outer.children[2]._inference, "child[2] flag cleared")

    # Nested case.
    inner.set_inference(True)
    assert_true(inner._inference, "inner flag set")
    assert_true(inner.children[0]._inference, "inner.child[0] flag set")
    assert_true(inner.children[1]._inference, "inner.child[1] flag set")
    assert_true(inner.children[2]._inference, "inner.child[2] flag set")

    print("  test_set_inference_propagates_cpu PASSED")


def test_set_inference_propagates_gpu() raises:
    """GPU mirror — set_inference works on a GPU-built Sequential."""
    var ctx = DeviceContext()
    var net = Sequential[
        Linear[4, 8], ReLU[8], Linear[8, 4], Tanh[4], StopGrad[4]
    ].make["gpu", INIT=Zero](ctx)

    assert_true(not net._inference, "default train mode")

    net.set_inference(True)
    assert_true(net._inference)
    assert_true(net.children[0]._inference)
    assert_true(net.children[1]._inference)
    assert_true(net.children[2]._inference)
    assert_true(net.children[3]._inference)
    assert_true(net.children[4]._inference)

    net.set_inference(False)
    assert_true(not net._inference)
    assert_true(not net.children[0]._inference)
    assert_true(not net.children[1]._inference)
    assert_true(not net.children[2]._inference)
    assert_true(not net.children[3]._inference)
    assert_true(not net.children[4]._inference)

    print("  test_set_inference_propagates_gpu PASSED")


def main() raises:
    print("=" * 60)
    print("nn2 InferenceMode propagation (Phase 5.3)")
    print("=" * 60)
    test_default_is_train_mode_cpu()
    test_set_inference_propagates_cpu()
    test_set_inference_propagates_gpu()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
