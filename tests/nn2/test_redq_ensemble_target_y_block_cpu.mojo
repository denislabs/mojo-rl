"""Phase R.1 integration smoke for `EnsembleTargetYBlock` (CPU).

Checks the block-level wiring (actor + rsample + N target critics +
combine kernel + terminal mask) end-to-end at N=4, N_MIN=2, MODE=MIN.

Strategy — snapshot the intermediate scratches the block writes
(`_mb_stacked_q [N, BATCH]` and `_mb_lp [BATCH]`) after `step` and
recompute the expected `y[b]` formula on the host using THE SAME
stacked-Q and log-prob values the block consumed. Byte-equality is
then a clean gate on the combine + mask path; we do NOT try to
reproduce the actor + rsample draws (those are validated by RSample's
own tests).

Checks:
  (a) All `mb_y[b]` finite.
  (b) For `term[b] = 1`: `mb_y[b] == r[b]` exactly (bootstrap dropped).
  (c) For `term[b] = 0`: `mb_y[b] == r[b] + γ · (min(q[0,b], q[1,b])
      − α · log_prob[b])` reconstructed from the block's own
      `_mb_stacked_q` + `_mb_lp` (subset pinned to [0, 1] via
      `set_subset_idxs`).
"""

from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators import Sequential
from mojo_rl.nn2.initializer import Xavier

from mojo_rl.deep_agents2.training.trainer_block import TrainerState
from mojo_rl.deep_agents2.redq import (
    CriticEnsemble,
    EnsembleTargetYBlock,
    REDQ_TARGET_MIN,
)


comptime OBS = 3
comptime ACT = 2
comptime BATCH = 8
comptime N = 4
comptime N_MIN = 2
comptime SA_DIM = OBS + ACT

comptime ActorNet = Sequential[
    Linear[OBS, 16],
    ReLU[16],
    Linear[16, 2 * ACT],
]
comptime CriticNet = Sequential[
    Linear[SA_DIM, 16],
    ReLU[16],
    Linear[16, 1],
]


def _list_int_2(a: Int, b: Int) raises -> List[Int]:
    var out = List[Int](length=2, fill=0)
    out[0] = a
    out[1] = b
    return out^


def _fill_target_y_inputs(
    mut state: TrainerState[OBS, ACT, BATCH],
) raises:
    """Hand-fill mb_sp (next states), mb_r (rewards), mb_d (terminal
    flags). Half the batch is terminated to exercise both branches of
    the terminal mask."""
    var sp = state.mb_sp.cpu_ptr()
    var r  = state.mb_r.cpu_ptr()
    var d  = state.mb_d.cpu_ptr()
    for b in range(BATCH):
        for k in range(OBS):
            sp[b * OBS + k] = Scalar[DT](
                0.1 * Float64(b) + 0.07 * Float64(k) - 0.3
            )
        r[b] = Scalar[DT](-0.5 + 0.2 * Float64(b))
        # b=0..3 truncation (term=0, bootstrap kept), b=4..7 termination.
        d[b] = Scalar[DT](0.0) if b < 4 else Scalar[DT](1.0)


def test_ensemble_target_y_block_cpu() raises:
    print(
        "--- EnsembleTargetYBlock[Actor, Critic, N=4, N_MIN=2, MIN] CPU ---"
    )

    var actor = ActorNet.make["cpu", Xavier]()
    var ensemble = CriticEnsemble[CriticNet, N].make["cpu", Xavier]()
    var block = EnsembleTargetYBlock[
        ActorNet, CriticNet, N, BATCH, OBS, ACT, N_MIN, REDQ_TARGET_MIN,
    ].make["cpu"](
        action_scale=Scalar[DT](1.0),
        gamma=Scalar[DT](0.97),
    )
    var state = TrainerState[OBS, ACT, BATCH].make["cpu"]()

    block.set_subset_idxs(_list_int_2(0, 1))
    _fill_target_y_inputs(state)

    var alpha = Scalar[DT](0.15)
    block.step["cpu"](
        actor,
        ensemble,
        state.mb_sp.cpu_ptr(),
        state.mb_r.cpu_ptr(),
        state.mb_d.cpu_ptr(),
        alpha,
        state.mb_y.cpu_ptr(),
    )

    # (a) All y finite.
    var y_p = state.mb_y.cpu_ptr()
    for b in range(BATCH):
        var v = Float64(y_p[b])
        assert_true(v == v, "y[b] must be finite (no NaN)")

    # (b) Terminated samples: y == r exactly.
    var r_p = state.mb_r.cpu_ptr()
    var d_p = state.mb_d.cpu_ptr()
    for b in range(BATCH):
        if d_p[b] == Scalar[DT](1.0):
            print("  b=", b, " (term=1) y =", y_p[b], " r =", r_p[b])
            assert_true(
                y_p[b] == r_p[b],
                "term=1 ⇒ y == r exactly (CleanRL semantics)",
            )

    # (c) Non-terminated: reconstruct the formula from the block's own
    # intermediates and verify byte-equality. This isolates the
    # combine + mask path (kernel + bootstrap-add) from the actor + rsample
    # draws (already validated by RSample's tests).
    var stacked = block._mb_stacked_q.cpu_ptr()  # [N, BATCH]
    var lp_p = block._mb_lp.cpu_ptr()             # [BATCH]
    var gamma = block.gamma
    var max_dev: Float64 = 0.0
    for b in range(BATCH):
        if d_p[b] == Scalar[DT](0.0):
            # Subset pinned to [0, 1] → combined = min(stacked[0,b], stacked[1,b]).
            var q0 = stacked[0 * BATCH + b]
            var q1 = stacked[1 * BATCH + b]
            var combined = q0 if q0 < q1 else q1
            var expected = r_p[b] + gamma * (combined - alpha * lp_p[b])
            var dev = Float64(y_p[b]) - Float64(expected)
            if dev < 0.0:
                dev = -dev
            if dev > max_dev:
                max_dev = dev
            print(
                "  b=", b, " (term=0) y =", y_p[b],
                " expected =", expected,
            )
    print("  max |y - expected| over term=0 samples =", max_dev)
    assert_true(
        max_dev == 0.0,
        "REDQ y formula must be bit-identical to the hand-reconstructed expression",
    )

    print("PASS — EnsembleTargetYBlock end-to-end smoke green.")


def main() raises:
    test_ensemble_target_y_block_cpu()
