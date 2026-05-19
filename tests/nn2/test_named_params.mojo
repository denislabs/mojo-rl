"""named_params — verify dotted names + apply_decay flags + per-leaf sizes
on a multi-layer MLP, CPU and GPU.

Layer-local apply_decay rule:
  Linear weight -> True
  Linear bias   -> False
  ReLU          -> no params, no visits
"""

from std.testing import assert_equal, assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators import Sequential
from mojo_rl.nn2.core import named_params


def test_named_params_cpu_mlp() raises:
    """3-layer MLP CPU: Linear(4,3)→ReLU→Linear(3,2). Expect 4 leaf params:
      "0.weight" (12 elems, decay=True)
      "0.bias"   (3 elems,  decay=False)
      "2.weight" (6 elems,  decay=True)
      "2.bias"   (2 elems,  decay=False)
    Index 1 (ReLU) contributes nothing.
    """
    var net = Sequential[Linear[4, 3], ReLU[3], Linear[3, 2]].make[
        "cpu", INIT=Zero,
    ]()
    var params = named_params["cpu"](net)

    assert_equal(len(params), 4, "expected 4 leaf params")

    assert_equal(params[0].name, String("0.weight"))
    assert_equal(params[0].n_elems, 12)
    assert_true(params[0].apply_decay)

    assert_equal(params[1].name, String("0.bias"))
    assert_equal(params[1].n_elems, 3)
    assert_true(not params[1].apply_decay)

    assert_equal(params[2].name, String("2.weight"))
    assert_equal(params[2].n_elems, 6)
    assert_true(params[2].apply_decay)

    assert_equal(params[3].name, String("2.bias"))
    assert_equal(params[3].n_elems, 2)
    assert_true(not params[3].apply_decay)

    print("  test_named_params_cpu_mlp PASSED (4 leaves, names + decay correct)")


def test_named_params_gpu_mlp() raises:
    """Same MLP on GPU. Names + sizes + decay flags must match CPU."""
    var ctx = DeviceContext()
    var net = Sequential[Linear[4, 3], ReLU[3], Linear[3, 2]].make[
        "gpu", INIT=Zero,
    ](ctx)
    var params = named_params["gpu"](net)

    assert_equal(len(params), 4)
    assert_equal(params[0].name, String("0.weight"))
    assert_equal(params[0].n_elems, 12)
    assert_true(params[0].apply_decay)
    assert_equal(params[1].name, String("0.bias"))
    assert_equal(params[1].n_elems, 3)
    assert_true(not params[1].apply_decay)
    assert_equal(params[2].name, String("2.weight"))
    assert_equal(params[2].n_elems, 6)
    assert_true(params[2].apply_decay)
    assert_equal(params[3].name, String("2.bias"))
    assert_equal(params[3].n_elems, 2)
    assert_true(not params[3].apply_decay)

    # GPU pointers must be non-null (DeviceBuffer-backed).
    for i in range(4):
        assert_true(
            Int(params[i].param_ptr) != 0,
            "GPU param_ptr["+String(i)+"] is null",
        )

    print("  test_named_params_gpu_mlp PASSED")


def main() raises:
    print("=" * 60)
    print("nn2 named_params tests (CPU + GPU, Phase 4)")
    print("=" * 60)
    test_named_params_cpu_mlp()
    test_named_params_gpu_mlp()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
