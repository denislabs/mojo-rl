"""Phase R.3 smoke for `EnsemblePolyakStep` (CPU). Same idea as the
R.0 polyak round-trip but routed through the block (which reads
`state.ctx` and τ from itself). At N=4 verify all 4 targets shift
toward their online twins; onlines stay byte-identical."""

from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.core.named_params import named_params
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators import Sequential
from mojo_rl.nn2.initializer import Xavier

from mojo_rl.deep_agents2.training.trainer_block import TrainerState
from mojo_rl.deep_agents2.redq import (
    CriticEnsemble,
    EnsemblePolyakStep,
)


comptime OBS = 3
comptime ACT = 2
comptime BATCH = 4
comptime N = 4

comptime CriticNet = Sequential[
    Linear[OBS + ACT, 16],
    ReLU[16],
    Linear[16, 1],
]


def _snapshot_param_sum[M: Module](mut model: M) raises -> Float64:
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


def _fill_model[M: Module](
    mut model: M, start: Float64, step: Float64,
) raises:
    var ps = named_params["cpu", M](model)
    var v = start
    for i in range(len(ps)):
        ref p = ps[i]
        for k in range(p.n_elems):
            p.param_ptr[k] = Scalar[DT](v)
            v += step


def test_ensemble_polyak_step_cpu() raises:
    print("--- EnsemblePolyakStep[CriticNet, N=4] CPU ---")
    var ensemble = CriticEnsemble[CriticNet, N].make["cpu", Xavier]()
    var block = EnsemblePolyakStep[CriticNet, N, OBS, ACT, BATCH].make(
        tau=Scalar[DT](0.3),
    )
    var state = TrainerState[OBS, ACT, BATCH].make["cpu"]()

    # Force online ≠ target per pair so polyak isn't a no-op.
    for i in range(N):
        _fill_model[CriticNet](
            ensemble.pairs[i].online,
            0.0 + 0.01 * Float64(i),
            0.001,
        )
        _fill_model[CriticNet](
            ensemble.pairs[i].target_net,
            1.0 - 0.01 * Float64(i),
            -0.0007,
        )

    var online_before = List[Float64](length=N, fill=0.0)
    for i in range(N):
        online_before[i] = _snapshot_param_sum[CriticNet](
            ensemble.pairs[i].online
        )

    # K polyak rounds.
    comptime K = 5
    for _ in range(K):
        block.step["cpu"](state, ensemble)

    # Onlines untouched; targets moved toward online (gap reduced).
    for i in range(N):
        var on_after = _snapshot_param_sum[CriticNet](
            ensemble.pairs[i].online
        )
        var on_d = on_after - online_before[i]
        if on_d < 0.0:
            on_d = -on_d
        # Distance online ↔ target after K rounds, should be strictly
        # less than at start (gap was ~|1.0 - 0.0| = 1.0 per leaf).
        var ps_on = named_params["cpu", CriticNet](
            ensemble.pairs[i].online
        )
        var ps_tg = named_params["cpu", CriticNet](
            ensemble.pairs[i].target_net
        )
        var max_gap: Float64 = 0.0
        for k in range(len(ps_on)):
            ref pon = ps_on[k]
            ref ptg = ps_tg[k]
            for j in range(pon.n_elems):
                var d = Float64(pon.param_ptr[j]) - Float64(ptg.param_ptr[j])
                if d < 0.0:
                    d = -d
                if d > max_gap:
                    max_gap = d
        print(
            "  pair", i,
            " online |Δ|sum =", on_d,
            " max|online - target| after K =", max_gap,
        )
        assert_true(
            on_d == 0.0,
            "polyak must not touch online (block writes targets only)",
        )
        assert_true(
            max_gap < 0.5,
            "K=5 τ=0.3 polyak rounds should shrink the 1.0 init gap",
        )

    print("PASS — EnsemblePolyakStep N=4 CPU smoke green.")


def main() raises:
    test_ensemble_polyak_step_cpu()
