"""ExternalUnaryNode + set_external smoke (Phase 3).

Builds a tiny ComputeGraph that contains an ExternalUnaryNode: the
trainer (this test) owns the inner Scale instance; the graph references
it per call via `g.set_external["scaler"](mut my_scale)`. Verifies that
1. forward dispatches through the external module
2. backward dispatches through the external module
3. set_external rebinding mid-test routes to the new instance
4. ExternalBinaryNode dispatches through its module too

If this passes, the SACActorLossCG migration in §8.6.1 can declare
ACTOR / CRITIC as ExternalUnaryNode without owning copies.
"""

from std.memory import alloc
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators import (
    ComputeGraph,
    InputSlot,
    UnaryNode,
    BinaryNode,
    ExternalUnaryNode,
    ExternalBinaryNode,
)
from mojo_rl.nn2.primitives.scale import Scale
from mojo_rl.nn2.primitives.binary_sub import BinarySub
from mojo_rl.nn2.initializer import Kaiming

from layout import TileTensor, row_major


def test_external_unary_node_basic() raises:
    """Graph: InputSlot["s"] → ExternalUnary["scaler", Scale[1]] →
    output. Confirms set_external dispatch through the trainer-owned
    Scale instance."""
    print("test_external_unary_node_basic ...")
    comptime BATCH = 4

    comptime G = ComputeGraph[
        1,
        InputSlot["s", 1],
        ExternalUnaryNode["scaler", Scale[1], "s"],
    ]
    var g = G.make[target="cpu", INIT=Kaiming]()

    # Trainer-owned Scale instance, multiplier=3.
    var ext = Scale[1].make[target="cpu", INIT=Kaiming]()
    ext.multiplier = Scalar[DT](3.0)
    g.set_external["scaler", Scale[1]](ext)

    var in_buf = alloc[Scalar[DT]](BATCH)
    var out_buf = alloc[Scalar[DT]](BATCH)
    var go_buf = alloc[Scalar[DT]](BATCH)
    for b in range(BATCH):
        in_buf[b] = Scalar[DT](b + 1)
        go_buf[b] = Scalar[DT](0.1 * Float64(b + 1))

    var in_t = TileTensor(in_buf, row_major[BATCH, 1]())
    var out_t = TileTensor(out_buf, row_major[BATCH, 1]())
    g.set_input["s", BATCH](in_t)
    g.forward["cpu", BATCH](out_t)
    for b in range(BATCH):
        var expected = Scalar[DT](Float64(b + 1) * 3.0)
        assert_true(
            (out_buf[b] - expected).__abs__() < Scalar[DT](1e-5),
            "ExternalUnaryNode forward must equal 3·input",
        )

    # Confirm pointer is to the same instance: mutate `ext.multiplier`
    # to 5 and rerun forward; expect outputs scale to 5·input.
    ext.multiplier = Scalar[DT](5.0)
    g.forward["cpu", BATCH](out_t)
    for b in range(BATCH):
        var expected = Scalar[DT](Float64(b + 1) * 5.0)
        assert_true(
            (out_buf[b] - expected).__abs__() < Scalar[DT](1e-5),
            "set_external must bind by pointer (rebind to current value)",
        )

    var go_t = TileTensor(go_buf, row_major[BATCH, 1]())
    g.vjp["cpu", BATCH](go_t)
    var gi_p = g.grad_input_ptr["s"]()
    for b in range(BATCH):
        # Scale backward: grad_in = 5 * grad_out (multiplier=5 from above).
        var expected = Scalar[DT](go_buf[b] * Scalar[DT](5.0))
        assert_true(
            (gi_p[b] - expected).__abs__() < Scalar[DT](1e-5),
            "ExternalUnaryNode backward must equal 5·grad_output",
        )

    in_buf.free()
    out_buf.free()
    go_buf.free()
    print("  ok")


def test_external_unary_rebind() raises:
    """Two distinct external Scale instances bound sequentially. Confirms
    set_external rebinding routes forward to the new instance."""
    print("test_external_unary_rebind ...")
    comptime BATCH = 2

    comptime G = ComputeGraph[
        1,
        InputSlot["s", 1],
        ExternalUnaryNode["scaler", Scale[1], "s"],
    ]
    var g = G.make[target="cpu", INIT=Kaiming]()

    var a = Scale[1].make[target="cpu", INIT=Kaiming]()
    a.multiplier = Scalar[DT](2.0)
    var b = Scale[1].make[target="cpu", INIT=Kaiming]()
    b.multiplier = Scalar[DT](-1.0)

    var in_buf = alloc[Scalar[DT]](BATCH)
    var out_buf = alloc[Scalar[DT]](BATCH)
    in_buf[0] = Scalar[DT](1.0)
    in_buf[1] = Scalar[DT](10.0)
    var in_t = TileTensor(in_buf, row_major[BATCH, 1]())
    var out_t = TileTensor(out_buf, row_major[BATCH, 1]())
    g.set_input["s", BATCH](in_t)

    g.set_external["scaler", Scale[1]](a)
    g.forward["cpu", BATCH](out_t)
    assert_true(out_buf[0] == Scalar[DT](2.0), "First bind (a) failed")
    assert_true(out_buf[1] == Scalar[DT](20.0), "First bind (a) failed")

    g.set_external["scaler", Scale[1]](b)
    g.forward["cpu", BATCH](out_t)
    assert_true(out_buf[0] == Scalar[DT](-1.0), "Rebind to (b) failed")
    assert_true(out_buf[1] == Scalar[DT](-10.0), "Rebind to (b) failed")

    in_buf.free()
    out_buf.free()
    print("  ok")


def test_external_binary_node() raises:
    """Graph: InputSlot["a"] + InputSlot["b"] →
        ExternalBinary["sub_ext", BinarySub[1]] → output.
    Verifies ExternalBinaryNode dispatch through a trainer-owned BinarySub."""
    print("test_external_binary_node ...")
    comptime BATCH = 3

    comptime G = ComputeGraph[
        1,
        InputSlot["a", 1],
        InputSlot["b", 1],
        ExternalBinaryNode["sub_ext", BinarySub[1], "a", "b"],
    ]
    var g = G.make[target="cpu", INIT=Kaiming]()

    var sub = BinarySub[1].make[target="cpu", INIT=Kaiming]()
    g.set_external_binary["sub_ext", BinarySub[1]](sub)

    var a_buf = alloc[Scalar[DT]](BATCH)
    var b_buf = alloc[Scalar[DT]](BATCH)
    var out_buf = alloc[Scalar[DT]](BATCH)
    a_buf[0] = Scalar[DT](5.0); a_buf[1] = Scalar[DT](7.0); a_buf[2] = Scalar[DT](10.0)
    b_buf[0] = Scalar[DT](2.0); b_buf[1] = Scalar[DT](3.0); b_buf[2] = Scalar[DT](4.0)

    var a_t = TileTensor(a_buf, row_major[BATCH, 1]())
    var b_t = TileTensor(b_buf, row_major[BATCH, 1]())
    var out_t = TileTensor(out_buf, row_major[BATCH, 1]())
    g.set_input["a", BATCH](a_t)
    g.set_input["b", BATCH](b_t)
    g.forward["cpu", BATCH](out_t)

    for i in range(BATCH):
        var expected = a_buf[i] - b_buf[i]
        assert_true(
            (out_buf[i] - expected).__abs__() < Scalar[DT](1e-5),
            "ExternalBinaryNode forward must equal a − b",
        )

    a_buf.free()
    b_buf.free()
    out_buf.free()
    # Mojo nightly: explicitly extend `sub`'s lifetime past `g.forward` so the
    # pointer stored in the graph by `set_external_binary` isn't dangling.
    # Without this, the compiler can end `sub`'s lifetime at the
    # `set_external_binary` call site and zero-fill its storage.
    _ = sub^
    print("  ok")


def main() raises:
    print("=" * 70)
    print("ExternalUnaryNode + ExternalBinaryNode smoke (Phase 3)")
    print("=" * 70)
    test_external_unary_node_basic()
    test_external_unary_rebind()
    test_external_binary_node()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
