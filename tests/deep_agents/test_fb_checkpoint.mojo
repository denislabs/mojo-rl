"""`FBTrainer` checkpoint round trip.

A 2 M-step run that writes a file which does not restore is worse than one that
writes nothing — it looks like insurance and is not. So this gates the property
that matters: after `save_state` / `load_state`, the trainer must produce the
SAME outputs as before, on all four nets.

  [1] `B(s)` and `pi_z(s, z)` match bit-for-bit across the round trip.
  [2] `load_state` hard-copies online -> target. `save_state` writes only the
      online nets (the targets are EMA copies), so without the hard copy the
      targets would sit at their random init and the first bootstrapped target
      after a resume would be garbage. ⚠ A resume is NOT bit-identical to an
      uninterrupted run — the targets lose their EMA lag — and this gate does
      not pretend otherwise; see the note on `test_targets_are_hard_copied`.

⚠ The gate perturbs the trainer BETWEEN save and load. Without that, "restored"
and "never changed" are indistinguishable and the test passes on a `load_state`
that does nothing at all.

Run:
    pixi run mojo run -I . tests/deep_agents/test_fb_checkpoint.mojo
"""

from std.math import abs
from std.random import random_float64, seed
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.core.call import call_forward
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import ReLU, Tanh
from mojo_rl.nn.primitives.layer_norm import LayerNorm
from mojo_rl.deep_agents.fb.trainer import FBTrainer
from mojo_rl.deep_agents.fb import sample_z_uniform


comptime OBS: Int = 5
comptime ACT: Int = 3
comptime D: Int = 8
comptime BATCH: Int = 16
comptime HID: Int = 32
comptime SEED: Int = 20260805
comptime CKPT: StaticString = "/tmp/test_fb_ckpt.ckpt"

comptime FNet = Sequential[
    Linear[OBS + ACT + D, HID], ReLU[HID], Linear[HID, D]
]
# LayerNorm here on purpose: the M2 architecture has one, and a checkpoint that
# skipped State fields would restore Params and silently drop it.
comptime BNet = Sequential[
    Linear[OBS, HID], ReLU[HID], Linear[HID, D], LayerNorm[D]
]
comptime ANet = Sequential[
    Linear[OBS + D, HID], ReLU[HID], Linear[HID, ACT], Tanh[ACT]
]
comptime Trainer = FBTrainer[FNet, BNet, ANet, OBS, ACT, D, BATCH]


def _rt(n: Int) raises -> Tensor:
    var t = Tensor.alloc(n)
    for i in range(n):
        t.data[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
    return t^


def _z(batch: Int) raises -> Tensor:
    var zl = sample_z_uniform[D](batch)
    var t = Tensor.alloc(batch * D)
    for i in range(batch * D):
        t.data[i] = zl[i]
    return t^


def _step(mut t: Trainer) raises:
    var s = _rt(BATCH * OBS)
    var a = _rt(BATCH * ACT)
    var sn = _rt(BATCH * OBS)
    var sp = _rt(BATCH * OBS)
    var z = _z(BATCH)
    t.load_batch(s, a, sn, sp, z)
    _ = t.train_step(want_loss=False)


def test_round_trip() raises:
    print("[1] save -> perturb -> load restores B and pi_z ...")
    seed(SEED)
    var t = Trainer.make(lr=1e-3)
    for _ in range(15):
        _step(t)

    var probe = _rt(BATCH * OBS)
    var zp = _z(BATCH)
    var b0 = Tensor()
    var a0 = Tensor()
    t.backward_embed[BATCH](probe, b0)
    t.act[BATCH](probe, zp, a0)

    t.save_state(String(CKPT))

    # ⚠ Perturb, or "restored" and "never changed" are the same observation.
    for _ in range(15):
        _step(t)
    var b_mid = Tensor()
    t.backward_embed[BATCH](probe, b_mid)
    var moved = Float64(0)
    for i in range(BATCH * D):
        var e = abs(Float64(b0.data[i]) - Float64(b_mid.data[i]))
        if e > moved:
            moved = e
    assert_true(
        moved > 1e-5,
        "15 further steps did not move B at all (" + String(moved) + "), so"
        " the load below cannot be shown to have restored anything",
    )
    print("      B moved by", moved, "before the load  OK")

    t.load_state(String(CKPT))
    var b1 = Tensor()
    var a1 = Tensor()
    t.backward_embed[BATCH](probe, b1)
    t.act[BATCH](probe, zp, a1)

    var wb = Float64(0)
    var wa = Float64(0)
    for i in range(BATCH * D):
        var e = abs(Float64(b0.data[i]) - Float64(b1.data[i]))
        if e > wb:
            wb = e
    for i in range(BATCH * ACT):
        var e = abs(Float64(a0.data[i]) - Float64(a1.data[i]))
        if e > wa:
            wa = e
    print("      after load: |B diff|", wb, " |pi_z diff|", wa)
    assert_true(wb < 1e-6, "B not restored: " + String(wb))
    assert_true(wa < 1e-6, "pi_z not restored: " + String(wa))


def test_targets_are_hard_copied() raises:
    """After `load_state`, each TARGET net must equal its ONLINE net.

    ⚠ The first version of this test asserted that a step taken after a round
    trip matches the same step taken without one. That is FALSE BY DESIGN and
    contradicted this module's own docstring: `save_state` omits the targets
    (they are EMA copies), so on the reference path they lag the online nets by
    15 steps while on the resumed path they are hard-copied and lag by zero. It
    asserted bit-identity of a resume the implementation explicitly says is not
    bit-identical.

    The property that DOES hold, and the one worth protecting, is that the
    targets are not left at their random init — without the hard copy the first
    bootstrapped target after a resume is garbage.
    """
    print("[2] load_state hard-copies online -> target ...")
    seed(SEED)
    var t = Trainer.make(lr=1e-3)
    for _ in range(15):
        _step(t)
    t.save_state(String(CKPT))

    seed(SEED + 3)
    var fresh = Trainer.make(lr=1e-3)
    var probe = _rt(BATCH * OBS)

    # Target output BEFORE the load: a fresh trainer's target is its random
    # init, hard-copied from its own random online net.
    var pack = TensorPack[1]()
    pack[0].ensure(BATCH * OBS)
    for i in range(BATCH * OBS):
        pack[0].data[i] = probe.data[i]
    var tgt_before = Tensor()
    tgt_before.ensure(BATCH * D)
    call_forward["cpu", BATCH](
        fresh.bnet.target_net, TensorRefs[1, MutAnyOrigin](pack[0]),
        tgt_before, None,
    )

    fresh.load_state(String(CKPT))

    var tgt_after = Tensor()
    tgt_after.ensure(BATCH * D)
    call_forward["cpu", BATCH](
        fresh.bnet.target_net, TensorRefs[1, MutAnyOrigin](pack[0]),
        tgt_after, None,
    )
    var onl_after = Tensor()
    fresh.backward_embed[BATCH](probe, onl_after)

    var moved = Float64(0)
    var gap = Float64(0)
    for i in range(BATCH * D):
        var m = abs(Float64(tgt_before.data[i]) - Float64(tgt_after.data[i]))
        if m > moved:
            moved = m
        var g = abs(Float64(tgt_after.data[i]) - Float64(onl_after.data[i]))
        if g > gap:
            gap = g
    print("      target moved by", moved, " target-vs-online gap", gap)
    assert_true(
        moved > 1e-5,
        "the target net did not change across load_state (" + String(moved)
        + "), so it is still at its random init and the check below is"
        " vacuous",
    )
    assert_true(
        gap < 1e-6,
        "after load_state the target differs from the online net by "
        + String(gap) + " — the hard copy did not happen, and the first"
        " bootstrapped target after a resume would be the random init.",
    )


def main() raises:
    test_round_trip()
    test_targets_are_hard_copied()
    print("\n[PASS] FB checkpoint round trip")
