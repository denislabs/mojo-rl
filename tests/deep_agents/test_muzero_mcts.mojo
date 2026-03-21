"""Test MuZero MCTS with actual networks."""

from mojo_rl.deep_agents.muzero.state import MuZeroCPUState
from mojo_rl.deep_agents.muzero.mcts import MCTS, MCTSNode
from mojo_rl.nn.constants import dtype


fn main():
    print("=== MuZero MCTS Tests ===")

    comptime OBS = 4
    comptime ACT = 2
    comptime LATENT = 32
    comptime BINS = 21
    comptime SIMS = 10

    comptime StateType = MuZeroCPUState[OBS, ACT, LATENT_DIM=LATENT, HIDDEN_DIM=32, NUM_BINS=BINS]

    # Create networks
    var state = StateType()

    # Create MCTS
    var mcts = MCTS[ACT, LATENT, BINS, SIMS](gamma=0.99)
    print("MCTS created")

    # Create a dummy observation
    var obs = List[Scalar[dtype]](capacity=OBS)
    for i in range(OBS):
        obs.append(Scalar[dtype](0.1 * (i + 1)))

    # Run MCTS search
    print("Running MCTS search with", SIMS, "simulations...")
    var policy = mcts.search[
        StateType.RepModel,
        StateType.DynModel,
        StateType.PredModel,
        StateType.OptType,
        StateType.OptType,
        StateType.OptType,
    ](
        obs,
        state.representation,
        state.dynamics,
        state.prediction,
        -10.0,
        10.0,
        add_noise=True,
    )

    print("MCTS policy:")
    var sum_policy = Float64(0.0)
    for a in range(ACT):
        print("  action", a, ":", policy[a])
        sum_policy += policy[a]
    print("  sum:", sum_policy)

    if sum_policy > 0.99 and sum_policy < 1.01:
        print("PASS: policy sums to 1")
    else:
        print("FAIL: policy sums to", sum_policy)

    # Check that nodes were created
    print("Nodes in tree:", len(mcts.nodes))
    if len(mcts.nodes) > 1:
        print("PASS: tree expanded")
    else:
        print("FAIL: tree not expanded")

    # Check root visit counts
    var root = mcts.nodes[0]
    var total_visits = 0
    for a in range(ACT):
        total_visits += root.visit_count[a]
    print("Root total visits:", total_visits, "(expected", SIMS, ")")
    if total_visits == SIMS:
        print("PASS: correct visit count")
    else:
        print("FAIL: visit count mismatch")

    print("=== Done ===")
