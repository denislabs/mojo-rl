"""named_params / named_states dotted-name reification (storage surface).

Sequential[Linear[D,H], BatchNorm1D[H], Linear[H,O]] must reify to:
  params: 0.weight(D*H) 0.bias(H) 1.gamma(H) 1.beta(H) 2.weight(H*O) 2.bias(O)
  states: 1.running_mean(H) 1.running_var(H)
proving combinator child-index + field-name dotted paths, sizes, and the
param/state split (gamma/beta are Params, running stats are States).

Run: pixi run mojo run -I . tests/nn/test_named_params_storage.mojo
"""

from std.testing import assert_true, assert_equal

from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.core.named_params import named_params, named_states
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.primitives.batch_norm_1d import BatchNorm1D
from mojo_rl.nn.storage.combinators.sequential import Sequential


comptime D = 4
comptime H = 5
comptime O = 3
comptime NET = Sequential[Linear[D, H], BatchNorm1D[H], Linear[H, O]]


def main() raises:
    print("named_params / named_states reification")
    var net = NET.make["cpu", Deterministic](None)

    var ps = named_params["cpu"](net)
    var exp_names = List[String]()
    exp_names.append("0.weight"); exp_names.append("0.bias")
    exp_names.append("1.gamma"); exp_names.append("1.beta")
    exp_names.append("2.weight"); exp_names.append("2.bias")
    var exp_sizes = List[Int]()
    exp_sizes.append(D * H); exp_sizes.append(H); exp_sizes.append(H)
    exp_sizes.append(H); exp_sizes.append(H * O); exp_sizes.append(O)
    var exp_decay = List[Bool]()
    exp_decay.append(True); exp_decay.append(False); exp_decay.append(False)
    exp_decay.append(False); exp_decay.append(True); exp_decay.append(False)
    assert_equal(len(ps), 6, "param count")
    for i in range(6):
        print("  ", ps[i].name, ps[i].size, ps[i].decay)
        assert_true(ps[i].name == exp_names[i], "param name " + String(i))
        assert_equal(ps[i].size, exp_sizes[i], "param size " + String(i))
        assert_true(ps[i].decay == exp_decay[i], "param decay " + String(i))

    var ss = named_states["cpu"](net)
    var exp_snames = List[String]()
    exp_snames.append("1.running_mean"); exp_snames.append("1.running_var")
    assert_equal(len(ss), 2, "state count")
    for i in range(2):
        print("  ", ss[i].name, ss[i].size)
        assert_true(ss[i].name == exp_snames[i], "state name " + String(i))
        assert_equal(ss[i].size, H, "state size " + String(i))

    print("NAMED PARAMS OK")
