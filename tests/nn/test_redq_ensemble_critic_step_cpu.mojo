"""Phase R.1 integration smoke for `EnsembleCriticStep` (CPU).

Checks the block wires the N-critic loop correctly:
  (a) `state.critic_loss` is finite and positive after step 1 (per-critic
      MSE losses summed).
  (b) All N=4 online critics' params CHANGED (each member got its own
      `Adam.step` against its own gradient).
  (c) All N=4 TARGET nets' params UNCHANGED (this block does not run
      polyak — that's PolyakStep's job).
  (d) After K=20 SGD-style steps against a FIXED y, mean per-critic
      loss at step K-1 is strictly smaller than at step 0 — the
      critics are actually converging onto the target.

Hand-fills `state.mb_s` / `state.mb_a` / `state.mb_y` with
deterministic patterns so the test is reproducible. No actor, no
target_y_block — that's R.1's other smoke.
"""

from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.named_params import named_params
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.nn.combinators import Sequential
from mojo_rl.nn.initializer import Xavier

from mojo_rl.deep_agents.training.trainer_block import TrainerState
from mojo_rl.deep_agents.redq import CriticEnsemble, EnsembleCriticStep


comptime OBS = 3
comptime ACT = 2
comptime BATCH = 8
comptime N = 4
comptime SA_DIM = OBS + ACT

comptime CriticNet = Sequential[
    Linear[SA_DIM, 16],
    ReLU[16],
    Linear[16, 1],
]


def _fill_mb_state(mut state: TrainerState[OBS, ACT, BATCH]) raises:
    """Deterministic hand-fill of mb_s, mb_a, mb_y (used as fixed
    targets for K steps). Values are bounded so the network outputs
    don't explode."""
    var s = state.mb_s.cpu_ptr()
    var a = state.mb_a.cpu_ptr()
    var y = state.mb_y.cpu_ptr()
    for b in range(BATCH):
        for d in range(OBS):
            s[b * OBS + d] = Scalar[DT](0.1 * Float64(b) + 0.07 * Float64(d) - 0.5)
        for j in range(ACT):
            a[b * ACT + j] = Scalar[DT](0.05 * Float64(b) - 0.03 * Float64(j) + 0.2)
        # Targets in a moderate range so MSE is finite and meaningful.
        y[b] = Scalar[DT](-0.5 + 0.1 * Float64(b))


def _snapshot_param_sum[M: Module](mut model: M) raises -> Float64:
    """Sum of absolute leaf values. Used as a coarse "has this model
    changed" signature — robust to FP noise and quicker than walking
    every element. Different param values → different sum (with very
    high probability for non-pathological weight tensors)."""
    var ps = named_params["cpu", M](model)
    var acc: Float64 = 0.0
    for i in range(len(ps)):
        ref p = ps[i]
        for k in range(p.n_elems):
            var v = Float64(p.param_ptr[k])
            if v < 0.0:
                v = -v
            acc += v
    return acc


def test_ensemble_critic_step_cpu() raises:
    print("--- EnsembleCriticStep[CriticNet, N=4] CPU smoke ---")

    var ensemble = CriticEnsemble[CriticNet, N].make["cpu", Xavier]()
    var block = EnsembleCriticStep[CriticNet, N, OBS, ACT, BATCH].make["cpu"]()
    var state = TrainerState[OBS, ACT, BATCH].make["cpu"]()

    # Per-critic LR knob — at N critics × BATCH the default is fine but
    # explicit is clearer.
    for i in range(N):
        ensemble.opts[i].lr = Scalar[DT](1e-3)

    _fill_mb_state(state)

    # Snapshot before training: per-member online and target signatures.
    var online_before = List[Float64](length=N, fill=0.0)
    var target_before = List[Float64](length=N, fill=0.0)
    for i in range(N):
        online_before[i] = _snapshot_param_sum[CriticNet](
            ensemble.pairs[i].online
        )
        target_before[i] = _snapshot_param_sum[CriticNet](
            ensemble.pairs[i].target_net
        )

    # Step 0 — capture initial sum-of-losses.
    block.step["cpu"](state, ensemble)
    var loss_first = Float64(state.critic_loss)
    print("  step 0 critic_loss (Σᵢ MSEᵢ) =", loss_first)
    assert_true(loss_first == loss_first, "step 0 loss finite")  # NaN check
    assert_true(loss_first > 0.0, "step 0 sum-of-losses > 0")

    # Steps 1..K-1 — track final loss.
    comptime K = 20
    var loss_last: Float64 = loss_first
    for _ in range(K - 1):
        block.step["cpu"](state, ensemble)
        loss_last = Float64(state.critic_loss)
    print("  step", K - 1, "critic_loss =", loss_last)
    assert_true(loss_last == loss_last, "final loss finite")

    # (d) Convergence: loss MUST strictly decrease over K steps when
    # training against a fixed y.
    assert_true(
        loss_last < loss_first,
        "ensemble must converge on fixed y over K=20 steps",
    )
    print(
        "  convergence: loss[0] =", loss_first,
        "→ loss[K-1] =", loss_last,
    )

    # (b)+(c) Every online critic changed; every target critic unchanged.
    for i in range(N):
        var on_after = _snapshot_param_sum[CriticNet](
            ensemble.pairs[i].online
        )
        var tg_after = _snapshot_param_sum[CriticNet](
            ensemble.pairs[i].target_net
        )
        var on_d = on_after - online_before[i]
        if on_d < 0.0:
            on_d = -on_d
        var tg_d = tg_after - target_before[i]
        if tg_d < 0.0:
            tg_d = -tg_d
        print(
            "  member", i,
            " online |Δ|sum =", on_d,
            " target |Δ|sum =", tg_d,
        )
        assert_true(
            on_d > 0.0,
            "online critic must have moved (Adam.step ran for this member)",
        )
        assert_true(
            tg_d == 0.0,
            "target net must be byte-identical (this block does not polyak)",
        )

    print("PASS — EnsembleCriticStep N=4 CPU smoke green.")


def main() raises:
    test_ensemble_critic_step_cpu()
