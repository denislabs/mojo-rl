"""`FBTrainer` smoke gate — does a step run, and does `B` avoid collapsing?

A trainer with five networks and three accumulating vjps into one of them has
many ways to be wrong that produce no error. This gate checks the two that a
loss curve cannot show:

  [2] `B` does not collapse. `L_ortho` exists to prevent every row of `B` from
      shrinking to the same direction, and §11's warning is that `L_FB` keeps
      descending while it happens. So the assertion is on the ROW SPREAD of
      `B` after training, not on the loss.

  [3] the three `B` vjps really do accumulate. Checked directly: a step with
      `ortho_weight = 0` and a step with `ortho_weight = 1` must move `B`'s
      parameters DIFFERENTLY. If the third vjp were overwriting rather than
      accumulating — or if `zero_grad` ran mid-step — the ortho contribution
      would be dropped and the two would coincide.

⚠ This is a smoke gate on random data, not a convergence test. It runs a few
dozen steps on a synthetic dataset and asserts structural properties. Whether
FB learns anything USEFUL is milestone 1's `point_mass` evaluation, which needs
a real dataset — see `docs/BFM_ZERO_SHOT_RL.md` §13 step 4a.

Run:
    pixi run mojo run -I . tests/deep_agents/test_fb_trainer_smoke.mojo
"""

from std.math import abs, sqrt
from std.random import random_float64, seed
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import Tanh, ReLU
from mojo_rl.deep_agents.fb.trainer import FBTrainer
from mojo_rl.deep_agents.fb import sample_z_uniform


comptime OBS: Int = 4
comptime ACT: Int = 2
comptime D: Int = 6
comptime BATCH: Int = 8
comptime HID: Int = 32
comptime STEPS: Int = 40
comptime SEED: Int = 20260805

comptime F_IN = OBS + ACT + D
comptime A_IN = OBS + D

comptime FNet = Sequential[
    Linear[F_IN, HID], ReLU[HID], Linear[HID, D]
]
comptime BNet = Sequential[
    Linear[OBS, HID], ReLU[HID], Linear[HID, D]
]
comptime ANet = Sequential[
    Linear[A_IN, HID], ReLU[HID], Linear[HID, ACT], Tanh[ACT]
]

comptime Trainer = FBTrainer[FNet, BNet, ANet, OBS, ACT, D, BATCH]


def _rand_tensor(n: Int, scale: Float64) raises -> Tensor:
    var t = Tensor.alloc(n)
    for i in range(n):
        t.data[i] = Scalar[DT]((random_float64() * 2.0 - 1.0) * scale)
    return t^


def _z_tensor(batch: Int) raises -> Tensor:
    var z = sample_z_uniform[D](batch)
    var t = Tensor.alloc(batch * D)
    for i in range(batch * D):
        t.data[i] = z[i]
    return t^


def _row_spread(ref b: Tensor, rows: Int) -> Float64:
    """Mean pairwise distance between normalised rows of `B`.

    Normalising first is what makes this a COLLAPSE metric rather than a scale
    metric: a `B` whose rows all shrink together keeps its directions and
    should not be flagged, while a `B` whose rows converge in DIRECTION should
    be, at any magnitude.
    """
    var acc = Float64(0)
    var pairs = 0
    for i in range(rows):
        for j in range(i + 1, rows):
            var ni = Float64(0)
            var nj = Float64(0)
            for k in range(D):
                ni += Float64(b.data[i * D + k]) * Float64(b.data[i * D + k])
                nj += Float64(b.data[j * D + k]) * Float64(b.data[j * D + k])
            ni = sqrt(ni)
            nj = sqrt(nj)
            if ni < 1e-9 or nj < 1e-9:
                continue
            var d2 = Float64(0)
            for k in range(D):
                var u = Float64(b.data[i * D + k]) / ni
                var v = Float64(b.data[j * D + k]) / nj
                d2 += (u - v) * (u - v)
            acc += sqrt(d2)
            pairs += 1
    return acc / Float64(pairs) if pairs > 0 else 0.0


def test_step_runs_and_reports() raises:
    print("[1] a train step runs end to end ...")
    seed(SEED)
    var t = Trainer.make(lr=1e-3)

    var s = _rand_tensor(BATCH * OBS, 1.0)
    var a = _rand_tensor(BATCH * ACT, 1.0)
    var sn = _rand_tensor(BATCH * OBS, 1.0)
    var sp = _rand_tensor(BATCH * OBS, 1.0)
    var z = _z_tensor(BATCH)

    t.load_batch(s, a, sn, sp, z)
    var l = t.train_step()
    print("      measure", l.measure, " ortho", l.ortho, " actor", l.actor)
    print("      |F|", l.f_norm, " |B|", l.b_norm)
    assert_true(l.measure == l.measure, "measure loss is NaN")
    assert_true(l.ortho == l.ortho, "ortho loss is NaN")
    assert_true(l.actor == l.actor, "actor loss is NaN")
    assert_true(l.b_norm > 1e-9, "B output is identically zero after one step")


def test_b_does_not_collapse() raises:
    print("[2] B keeps distinct row directions over", STEPS, "steps ...")
    seed(SEED)
    var t = Trainer.make(lr=1e-3)

    var probe = _rand_tensor(BATCH * OBS, 1.0)
    var b0 = Tensor()
    t.backward_embed[BATCH](probe, b0)
    var spread0 = _row_spread(b0, BATCH)

    for _ in range(STEPS):
        var s = _rand_tensor(BATCH * OBS, 1.0)
        var a = _rand_tensor(BATCH * ACT, 1.0)
        var sn = _rand_tensor(BATCH * OBS, 1.0)
        var sp = _rand_tensor(BATCH * OBS, 1.0)
        var z = _z_tensor(BATCH)
        t.load_batch(s, a, sn, sp, z)
        _ = t.train_step()

    var b1 = Tensor()
    t.backward_embed[BATCH](probe, b1)
    var spread1 = _row_spread(b1, BATCH)
    print("      mean pairwise direction distance:", spread0, "->", spread1)
    assert_true(
        spread1 > 0.1,
        "B's rows collapsed onto one direction (spread " + String(spread1)
        + "). L_ortho is not doing its job, and the measure loss would have"
        " kept descending regardless — see this file's docstring.",
    )


def test_ortho_weight_changes_the_update() raises:
    """The third `B` vjp must actually reach `B`'s parameters.

    Two trainers, identical seed and identical data, differing only in
    `ortho_weight`. If the ortho gradient were dropped — overwritten by a later
    vjp, or zeroed mid-step — the two would produce the SAME `B`.
    """
    print("[3] ortho_weight changes B's update (the vjps accumulate) ...")

    var probe = Tensor.alloc(BATCH * OBS)
    for i in range(BATCH * OBS):
        probe.data[i] = Scalar[DT](0.21 * Float64(i % 9) - 0.7)

    var outs = List[Float64]()
    for variant in range(2):
        var w = 0.0 if variant == 0 else 1.0
        seed(SEED)
        var t = Trainer.make(lr=1e-3, ortho_weight=w)
        seed(SEED + 1)
        for _ in range(5):
            var s = _rand_tensor(BATCH * OBS, 1.0)
            var a = _rand_tensor(BATCH * ACT, 1.0)
            var sn = _rand_tensor(BATCH * OBS, 1.0)
            var sp = _rand_tensor(BATCH * OBS, 1.0)
            var z = _z_tensor(BATCH)
            t.load_batch(s, a, sn, sp, z)
            _ = t.train_step()
        var b = Tensor()
        t.backward_embed[BATCH](probe, b)
        var acc = Float64(0)
        for i in range(BATCH * D):
            acc += Float64(b.data[i]) * Float64(b.data[i])
        outs.append(sqrt(acc))

    print("      ||B(probe)||: ortho_weight=0 ->", outs[0],
          "  ortho_weight=1 ->", outs[1])
    assert_true(
        abs(outs[0] - outs[1]) > 1e-4,
        "ortho_weight had no effect on B (" + String(outs[0]) + " vs "
        + String(outs[1]) + "). The L_ortho gradient is not reaching B's"
        " parameters — most likely a vjp overwrote it or zero_grad ran"
        " mid-step.",
    )


def main() raises:
    test_step_runs_and_reports()
    test_b_does_not_collapse()
    test_ortho_weight_changes_the_update()
    print("\n[PASS] FB trainer smoke gate")
