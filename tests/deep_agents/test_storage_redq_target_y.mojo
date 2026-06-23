"""EnsembleTargetYBlock isolation gate — target-y vs hand oracle (CPU).

Builds a tiny actor (StochasticActor) + N tiny critics and the storage
`EnsembleTargetYBlock`, runs `step` with a PINNED MIN subset, then recomputes
`state.mb_y` from an INDEPENDENT host oracle.

RSample draws fresh reparam noise each forward, so rather than re-running the
stochastic policy the oracle READS BACK the action + log_prob the block used
(`blk._mb_alp`, the packed [B, ACT+1] rsample output) and re-forwards the SAME
critic nets on host into a fresh stacked buffer — isolating the concat / N-critic
stacking / MIN-subset combine / α·logp / γ / terminal-mask wiring from the RNG.

    sa = concat(s', action_used)
    q[i, b] = critic_i(sa)
    combined[b] = min over subset of q[·, b]
    y_ref[b] = r[b] + (1-term[b])·γ·(combined[b] − α·logp_used[b])

Assert match < 1e-4.

Run: pixi run mojo run -I . tests/deep_agents/test_storage_redq_target_y.mojo
"""

from std.math import isnan, isinf
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.linear_relu import LinearReLU
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.core.initializer import Xavier
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs

from mojo_rl.deep_agents.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents.core.online_target_pair import OnlineTargetPair
from mojo_rl.deep_agents.training.trainer_block import TrainerState
from mojo_rl.deep_agents.redq.ensemble import CriticEnsemble
from mojo_rl.deep_agents.redq.ensemble_target_y_block import EnsembleTargetYBlock
from mojo_rl.deep_agents.redq.kernels import REDQ_TARGET_MIN


comptime OBS = 3
comptime ACT = 1
comptime SA = OBS + ACT
comptime H = 16
comptime BATCH = 16
comptime N = 4
comptime N_MIN = 2
comptime GAMMA = Scalar[DT](0.99)
comptime ASCALE = Scalar[DT](2.0)
comptime ALPHA = Scalar[DT](0.3)
comptime ACTOR = StochasticActor[OBS, ACT, LinearReLU[OBS, H], LinearReLU[H, H]]
comptime CRITIC = Sequential[LinearReLU[SA, H], LinearReLU[H, H], Linear[H, 1]]
comptime BLK = EnsembleTargetYBlock[
    ACTOR, CRITIC, N, BATCH, OBS, ACT, N_MIN, REDQ_TARGET_MIN
]


def main() raises:
    print("=" * 60)
    print("EnsembleTargetYBlock isolation gate (pinned subset, CPU)")
    print("=" * 60)

    var blk = BLK.make["cpu"](action_scale=ASCALE, gamma=GAMMA)
    var actor = ACTOR.make["cpu", Xavier]()
    var ensemble = CriticEnsemble[CRITIC, N].make["cpu", Xavier]()
    var st = TrainerState[OBS, ACT, BATCH].make["cpu"]()
    for i in range(BATCH * OBS):
        st.mb_sp.data[i] = Scalar[DT]((i % 7) - 3) * 0.2
    for b in range(BATCH):
        st.mb_r.data[b] = Scalar[DT]((b % 4) - 2) * 0.3
        st.mb_d.data[b] = Scalar[DT](1.0) if (b % 8 == 7) else Scalar[DT](0.0)

    # Pin the MIN subset deterministically.
    var subset = List[Int](length=N_MIN, fill=0)
    subset[0] = 2
    subset[1] = 0
    blk.set_subset_idxs(subset)

    blk.step["cpu"](st, actor, ensemble, ALPHA)

    # ── Independent oracle. Read back the action + logp the block used. ──
    var sp = Tensor.alloc(BATCH * OBS)
    for i in range(BATCH * OBS):
        sp.data[i] = st.mb_sp.data[i]
    # action_used[b, j] = blk._mb_alp[b, ACT+1][j]; logp_used[b] = [..][ACT]
    var sa = Tensor.alloc(BATCH * SA)
    var logp_used = List[Scalar[DT]](length=BATCH, fill=Scalar[DT](0))
    for b in range(BATCH):
        for d in range(OBS):
            sa.data[b * SA + d] = sp.data[b * OBS + d]
        for j in range(ACT):
            sa.data[b * SA + OBS + j] = blk._mb_alp.data[b * (ACT + 1) + j]
        logp_used[b] = blk._mb_alp.data[b * (ACT + 1) + ACT]

    # Re-forward each critic on host into a fresh stacked [N, BATCH].
    var q_stack = Tensor.alloc(N * BATCH)
    var q_i = Tensor.alloc(BATCH)
    for i in range(N):
        ensemble.pairs[i].target_net.forward["cpu", BATCH](
            TensorRefs[CRITIC.ARITY](sa), q_i
        )
        for b in range(BATCH):
            q_stack.data[i * BATCH + b] = q_i.data[b]

    var ok = True
    var max_err = Scalar[DT](0.0)
    for b in range(BATCH):
        var a = q_stack.data[2 * BATCH + b]
        var c = q_stack.data[0 * BATCH + b]
        var combined = a if a < c else c
        var nonterm = Scalar[DT](1.0) - st.mb_d.data[b]
        var y_ref = (
            st.mb_r.data[b]
            + nonterm * GAMMA * (combined - ALPHA * logp_used[b])
        )
        if isnan(st.mb_y.data[b]) or isinf(st.mb_y.data[b]):
            ok = False
        var err = abs(st.mb_y.data[b] - y_ref)
        if err > max_err:
            max_err = err
        if err > Scalar[DT](1e-4):
            ok = False

    print("  max |mb_y - oracle| :", max_err)
    assert_true(
        ok,
        "mb_y == r + (1-term)·γ·(min-subset Q − α·logp) over the used action",
    )
    print("REDQ TARGET-Y ISOLATION OK")
