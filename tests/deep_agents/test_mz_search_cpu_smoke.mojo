"""MuZero CPU learned-dynamics MCTS smoke — GenericCPUMCTS + MZ CPU adapters.

The CPU counterpart of `test_mz_search_smoke` (GPU). Validates the single-device
CPU search path the CartPole lighthouse driver will use: rep/dyn/pred CPU
adapters wrapping the nn h/g/f nets, threaded through `GenericCPUMCTS.search`
(SinglePlayer, learned dynamics). Asserts the returned visit policy is a valid
distribution over a few searches from arbitrary observations.

Run (no GPU):
    pixi run mojo run -I . tests/deep_agents/test_mz_search_cpu_smoke.mojo
"""

from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.deep_agents.muzero.nets import MZRepNet, MZDynNet, MZPredNet
from mojo_rl.deep_agents.zero.mcts_adapters_mz_cpu import (
    MZRepCPU, MZDynCPU, MZPredCPU,
)
from mojo_rl.planners.tree_search import (
    GenericCPUMCTS,
    MuZeroPUCT,
    DirichletNoise,
    SinglePlayer,
)


def main() raises:
    comptime OBS = 4        # CartPole
    comptime ACT = 2
    comptime LATENT = 16
    comptime BINS = 51
    comptime H = 32
    comptime NUM_SIMS = 24
    comptime MAX_NODES = 64
    var v_min = Scalar[DT](-10.0)
    var v_max = Scalar[DT](10.0)

    comptime Rep = MZRepNet[OBS, LATENT, H]
    comptime Dyn = MZDynNet[LATENT, ACT, BINS, H]
    comptime Pred = MZPredNet[LATENT, ACT, BINS, H]

    var rep_net = Rep.make["cpu", INIT=Kaiming]()
    var dyn_net = Dyn.make["cpu", INIT=Kaiming]()
    var pred_net = Pred.make["cpu", INIT=Kaiming]()
    var rep = MZRepCPU[OBS, LATENT, Rep](net=UnsafePointer(to=rep_net))
    var dyn = MZDynCPU[LATENT, ACT, BINS, Dyn](
        net=UnsafePointer(to=dyn_net), v_min=v_min, v_max=v_max
    )
    var pred = MZPredCPU[LATENT, ACT, BINS, Pred](
        net=UnsafePointer(to=pred_net), v_min=v_min, v_max=v_max
    )

    var mcts = GenericCPUMCTS[
        ACT, LATENT, NUM_SIMS, MAX_NODES,
        MuZeroPUCT[1.25], DirichletNoise[0.25, 0.25], SinglePlayer,
    ](gamma=0.997)

    for t in range(3):
        var obs = List[Float64]()
        for j in range(OBS):
            obs.append(0.05 * Float64(t) - 0.1 * Float64(j))
        var policy = mcts.search[type_of(rep), type_of(dyn), type_of(pred)](
            rep, dyn, pred, obs, add_noise=True
        )
        var s = 0.0
        var best = 0
        for a in range(ACT):
            assert_true(policy[a] == policy[a], "policy NaN")
            assert_true(policy[a] >= 0.0, "policy negative")
            s += policy[a]
            if policy[a] > policy[best]:
                best = a
        assert_true(s > 0.99 and s < 1.01, "visit policy does not sum to 1")
        print("search", t, "best action", best, "p", policy[best], "sum", s)

    _ = rep_net^   # keepalive for the adapters' non-owning pointers
    _ = dyn_net^
    _ = pred_net^
    print("MuZero CPU learned-dynamics search smoke: OK")
