"""EZv2-Atari Stage-2 polish — init_zero on head/reward output layers (CPU).

Pins the EZv2 `init_zero=True` behavior (docs/EZV2_ATARI_PARITY.md §B): the
LAST Linear of the value, policy and reward heads is zeroed at init (neutral
value/reward + uniform policy prior, stable MCTS targets before the heads
learn). `ez_atari_init_zero_{pred,dyn}` reuse the DreamerV3 `scale_output_*`
visitors (scale=0.0 == exact zero).

Checks, after applying init_zero to a freshly Kaiming-init pred + dyn:
  1. Targeted OUTPUT layers are all-zero: pred `1.a.5.*` (policy), `1.b.5.*`
     (value); dyn `rew.5.*` (reward).
  2. The hidden layers right below them are NOT all-zero (we only touch the
     output layer, so the hidden gradient is not choked): pred `1.a.3.weight`,
     dyn `rew.3.weight`.

Run:
    pixi run -e apple mojo run -I . tests/deep_agents2/test_ezv2_atari_init_zero.mojo
"""

from std.math import abs
from std.gpu.memory import AddressSpace
from std.testing import assert_true
from layout import TileTensor

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.core import ParamVisitor
from mojo_rl.deep_agents2.efficient_zero_v2.nets_atari import (
    EZPredNetAtari, EZDynNetAtari,
    ez_atari_init_zero_pred, ez_atari_init_zero_dyn,
)


comptime ACT = 6
comptime BINS = 51  # smaller than Atari's 601 — only affects head Linear size


# Probe a single named param: did we see it, and what is its max |value|?
struct _Probe(ParamVisitor):
    var key: String
    var found: Bool
    var max_abs: Scalar[DT]

    def __init__(out self, key: String):
        self.key = key
        self.found = False
        self.max_abs = Scalar[DT](0.0)

    def visit(
        mut self, name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int, apply_decay: Bool,
    ) raises:
        if name == self.key:
            self.found = True
            var p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](param.ptr)
            for i in range(n_elems):
                var a = abs(p[i])
                if a > self.max_abs:
                    self.max_abs = a


def _pred_maxabs(mut m: EZPredNetAtari[ACT, BINS], key: String) raises -> Scalar[DT]:
    var v = _Probe(key)
    m.for_each_param["cpu", _Probe](String(""), v)
    assert_true(v.found, "pred param exists: " + key)
    return v.max_abs


def _dyn_maxabs(mut d: EZDynNetAtari[ACT, BINS], key: String) raises -> Scalar[DT]:
    # init_zero scales d.graph directly (prefix ""), so probe the graph too.
    var v = _Probe(key)
    d.graph.for_each_param["cpu", _Probe](String(""), v)
    assert_true(v.found, "dyn graph param exists: " + key)
    return v.max_abs


def main() raises:
    print("=" * 70)
    print("EZv2-Atari init_zero on head/reward output layers (CPU)")
    print("=" * 70)

    var pred = EZPredNetAtari[ACT, BINS].make[target="cpu", INIT=Kaiming]()
    var dyn = EZDynNetAtari[ACT, BINS].make[target="cpu", INIT=Kaiming]()

    # Before: output layers are Kaiming-init → non-zero.
    var pol_before = _pred_maxabs(pred, "1.a.5.weight")
    var val_before = _pred_maxabs(pred, "1.b.5.weight")
    var rew_before = _dyn_maxabs(dyn, "rew.5.weight")
    print("  before: |policy_out|=", pol_before, " |value_out|=", val_before,
          " |reward_out|=", rew_before)
    assert_true(pol_before > Scalar[DT](0.0), "policy out non-zero pre-init_zero")
    assert_true(val_before > Scalar[DT](0.0), "value out non-zero pre-init_zero")
    assert_true(rew_before > Scalar[DT](0.0), "reward out non-zero pre-init_zero")

    ez_atari_init_zero_pred["cpu", ACT, BINS](pred)
    ez_atari_init_zero_dyn["cpu", ACT, BINS](dyn)

    # After: targeted output layers exactly zero (weight AND bias).
    var pol_w = _pred_maxabs(pred, "1.a.5.weight")
    var pol_b = _pred_maxabs(pred, "1.a.5.bias")
    var val_w = _pred_maxabs(pred, "1.b.5.weight")
    var val_b = _pred_maxabs(pred, "1.b.5.bias")
    var rew_w = _dyn_maxabs(dyn, "rew.5.weight")
    var rew_b = _dyn_maxabs(dyn, "rew.5.bias")
    print("  after : policy_out w/b=", pol_w, "/", pol_b,
          " value_out w/b=", val_w, "/", val_b,
          " reward_out w/b=", rew_w, "/", rew_b)
    assert_true(pol_w == Scalar[DT](0.0) and pol_b == Scalar[DT](0.0),
                "policy output layer zeroed")
    assert_true(val_w == Scalar[DT](0.0) and val_b == Scalar[DT](0.0),
                "value output layer zeroed")
    assert_true(rew_w == Scalar[DT](0.0) and rew_b == Scalar[DT](0.0),
                "reward output layer zeroed")

    # Hidden layers below the output are untouched (gradient not choked).
    var pol_hid = _pred_maxabs(pred, "1.a.3.weight")
    var rew_hid = _dyn_maxabs(dyn, "rew.3.weight")
    print("  hidden untouched: |policy_hid|=", pol_hid, " |reward_hid|=", rew_hid)
    assert_true(pol_hid > Scalar[DT](0.0), "policy hidden layer untouched")
    assert_true(rew_hid > Scalar[DT](0.0), "reward hidden layer untouched")

    _ = pred^
    _ = dyn^
    print("=" * 70)
    print("PASSED")
    print("=" * 70)
