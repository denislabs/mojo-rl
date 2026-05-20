"""Polyak/hard-copy parameter walker CPU tests — Phase 7.2.

Covers:
  - polyak_update on a single Linear: hand-checked expected output.
  - hard_copy_params (tau=1.0) makes target identical to online.
  - polyak_update on a deeper Sequential[Linear, Tanh, Linear]:
    every leaf moves by the right amount.
  - polyak_update with tau=0 is a no-op on target.
"""

from std.math import abs as fabs
from std.testing import assert_almost_equal, assert_equal
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.tanh import Tanh
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.core.map_params import polyak_update, hard_copy_params


def test_polyak_single_linear() raises:
    """Linear[2, 3]: weight 6 elements, bias 3 elements.

    online.weight[i] = i+1                    (1..6)
    target.weight[i] = 10*(i+1)               (10..60)
    online.bias[i]   = 100*(i+1)              (100..300)
    target.bias[i]   = 1000*(i+1)             (1000..3000)
    tau = 0.1
    Expected target.weight[i] = 0.9 * 10*(i+1) + 0.1 * (i+1) = 9.1 * (i+1)
    Expected target.bias[i]   = 0.9 * 1000*(i+1) + 0.1 * 100*(i+1) = 910 * (i+1)
    """
    var online = Linear[2, 3].make["cpu", INIT=Zero]()
    var target_net = Linear[2, 3].make["cpu", INIT=Zero]()

    var ow = TileTensor(online.weight, row_major[3, 2]())
    var tw = TileTensor(target_net.weight, row_major[3, 2]())
    var ob = TileTensor(online.bias, row_major[3]())
    var tb = TileTensor(target_net.bias, row_major[3]())
    for i in range(6):
        var ri = i // 2
        var ci = i % 2
        ow[ri, ci] = Scalar[DT](i + 1)
        tw[ri, ci] = Scalar[DT](10 * (i + 1))
    for i in range(3):
        ob[i] = Scalar[DT](100 * (i + 1))
        tb[i] = Scalar[DT](1000 * (i + 1))

    polyak_update["cpu", M=Linear[2, 3]](online, target_net, Scalar[DT](0.1))

    var tw_after = TileTensor(target_net.weight, row_major[3, 2]())
    var tb_after = TileTensor(target_net.bias, row_major[3]())
    for i in range(6):
        var ri = i // 2
        var ci = i % 2
        assert_almost_equal(
            tw_after[ri, ci], Scalar[DT](9.1 * (i + 1)), atol=1e-5
        )
    for i in range(3):
        assert_almost_equal(
            tb_after[i], Scalar[DT](910 * (i + 1)), atol=1e-4
        )
    print("  test_polyak_single_linear PASSED")


def test_polyak_tau_zero_is_noop() raises:
    """Tau=0 is a no-op — target_net is untouched."""
    var online = Linear[2, 3].make["cpu", INIT=Zero]()
    var target_net = Linear[2, 3].make["cpu", INIT=Zero]()

    var ow = TileTensor(online.weight, row_major[3, 2]())
    var tw = TileTensor(target_net.weight, row_major[3, 2]())
    var ob = TileTensor(online.bias, row_major[3]())
    var tb = TileTensor(target_net.bias, row_major[3]())
    for i in range(6):
        var ri = i // 2
        var ci = i % 2
        ow[ri, ci] = Scalar[DT](999.0)
        tw[ri, ci] = Scalar[DT](i + 1)
    for i in range(3):
        ob[i] = Scalar[DT](999.0)
        tb[i] = Scalar[DT](i + 1)

    polyak_update["cpu", M=Linear[2, 3]](online, target_net, Scalar[DT](0.0))

    var tw_after = TileTensor(target_net.weight, row_major[3, 2]())
    var tb_after = TileTensor(target_net.bias, row_major[3]())
    for i in range(6):
        var ri = i // 2
        var ci = i % 2
        assert_almost_equal(
            tw_after[ri, ci], Scalar[DT](i + 1), atol=1e-8
        )
    for i in range(3):
        assert_almost_equal(tb_after[i], Scalar[DT](i + 1), atol=1e-8)
    print("  test_polyak_tau_zero_is_noop PASSED")


def test_hard_copy_params() raises:
    """Hard copy (tau=1) leaves target equal to online elementwise."""
    var online = Linear[2, 3].make["cpu", INIT=Zero]()
    var target_net = Linear[2, 3].make["cpu", INIT=Zero]()

    var ow = TileTensor(online.weight, row_major[3, 2]())
    var ob = TileTensor(online.bias, row_major[3]())
    for i in range(6):
        var ri = i // 2
        var ci = i % 2
        ow[ri, ci] = Scalar[DT](i * 0.3 - 0.5)
    for i in range(3):
        ob[i] = Scalar[DT](i * 0.7 + 0.1)

    hard_copy_params["cpu", M=Linear[2, 3]](online, target_net)

    var ow_after = TileTensor(online.weight, row_major[3, 2]())
    var tw_after = TileTensor(target_net.weight, row_major[3, 2]())
    var ob_after = TileTensor(online.bias, row_major[3]())
    var tb_after = TileTensor(target_net.bias, row_major[3]())
    for i in range(6):
        var ri = i // 2
        var ci = i % 2
        assert_almost_equal(
            tw_after[ri, ci], ow_after[ri, ci], atol=1e-7
        )
    for i in range(3):
        assert_almost_equal(tb_after[i], ob_after[i], atol=1e-7)
    print("  test_hard_copy_params PASSED")


def test_polyak_sequential_deep() raises:
    """Walk a Sequential[Linear[3,4], Tanh[4], Linear[4,2]] with tau=0.5.

    Initialize everything to zero except a few hand-picked entries.
    After update, each touched cell should be 0.5 * (online + target_before).
    """
    comptime Net = Sequential[Linear[3, 4], Tanh[4], Linear[4, 2]]
    var online = Net.make["cpu", INIT=Zero]()
    var target_net = Net.make["cpu", INIT=Zero]()

    # Fill online layer 0 weight + layer 2 bias.
    var l0_w_online = TileTensor(online.children[0].weight, row_major[4, 3]())
    var l0_w_target = TileTensor(
        target_net.children[0].weight, row_major[4, 3]()
    )
    var l2_b_online = TileTensor(online.children[2].bias, row_major[2]())
    var l2_b_target = TileTensor(target_net.children[2].bias, row_major[2]())

    l0_w_online[0, 0] = 1.0
    l0_w_online[3, 2] = -0.5
    l0_w_target[0, 0] = 5.0
    l0_w_target[3, 2] = 10.0

    l2_b_online[0] = 2.0
    l2_b_target[1] = 4.0

    polyak_update["cpu", M=Net](online, target_net, Scalar[DT](0.5))

    var l0_w_t_after = TileTensor(
        target_net.children[0].weight, row_major[4, 3]()
    )
    var l2_b_t_after = TileTensor(
        target_net.children[2].bias, row_major[2]()
    )
    # target[0,0] = 0.5 * 5.0 + 0.5 * 1.0 = 3.0
    assert_almost_equal(l0_w_t_after[0, 0], Scalar[DT](3.0), atol=1e-6)
    # target[3,2] = 0.5 * 10 + 0.5 * (-0.5) = 4.75
    assert_almost_equal(l0_w_t_after[3, 2], Scalar[DT](4.75), atol=1e-6)
    # l2.bias[0] = 0.5 * 0 + 0.5 * 2.0 = 1.0
    assert_almost_equal(l2_b_t_after[0], Scalar[DT](1.0), atol=1e-6)
    # l2.bias[1] = 0.5 * 4.0 + 0.5 * 0 = 2.0
    assert_almost_equal(l2_b_t_after[1], Scalar[DT](2.0), atol=1e-6)

    # Untouched cells should still be 0.
    assert_almost_equal(l0_w_t_after[1, 1], 0.0, atol=1e-8)
    assert_almost_equal(l0_w_t_after[2, 0], 0.0, atol=1e-8)

    print("  test_polyak_sequential_deep PASSED")


def main() raises:
    print("=" * 70)
    print("polyak_update / hard_copy_params CPU tests (Phase 7.2)")
    print("=" * 70)
    test_polyak_single_linear()
    test_polyak_tau_zero_is_noop()
    test_hard_copy_params()
    test_polyak_sequential_deep()
    print("All polyak/hard_copy tests PASSED.")
