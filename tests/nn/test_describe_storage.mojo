"""CPU test for the additive `describe()` module-introspection helper.

Builds Sequential[Linear[D, H], Linear[H, O]] on CPU, calls describe["cpu"],
prints the table, and asserts the total param count + row count match the
expected Linear layout (weight = IN*OUT, bias = OUT).

Run: pixi run mojo run -I . tests/nn/test_describe_storage.mojo
"""

from std.testing import assert_equal, assert_true

from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.core.describe import describe


comptime D = 4
comptime H = 6
comptime O = 3
comptime NET = Sequential[Linear[D, H], Linear[H, O]]


def main() raises:
    var model = NET.make["cpu", Deterministic](None)

    var table = describe["cpu"](model)
    print(table)

    # Linear has weight (IN*OUT) + bias (OUT). Two linears chained.
    comptime EXPECTED = D * H + H + H * O + O
    # 4 tensors: l0.weight, l0.bias, l1.weight, l1.bias
    comptime EXPECTED_ROWS = 4

    assert_true(table.byte_length() > 0, "describe table must be non-empty")
    assert_true(
        String("total params: ") + String(EXPECTED) in table,
        "footer must report total params == " + String(EXPECTED),
    )
    assert_true(
        String("across ") + String(EXPECTED_ROWS) + String(" tensors") in table,
        "footer must report " + String(EXPECTED_ROWS) + " tensors",
    )
    assert_true("weight: " + String(D * H) in table, "l0 weight row")
    assert_true("bias: " + String(H) in table, "l0 bias row")
    assert_true("weight: " + String(H * O) in table, "l1 weight row")
    assert_true("bias: " + String(O) in table, "l1 bias row")

    print("DESCRIBE OK")
