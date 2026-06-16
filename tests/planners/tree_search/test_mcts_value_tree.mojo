"""Phase 3 planners: CPU MCTS on a fixed-shape ``KnownValueTree``.

Tests the expansion/backup loop end-to-end on a synthetic tree with
known leaf values. The adapter encodes the path-so-far in the hidden
state, so:

  * ``Representation`` writes ``hidden = [depth=0, path_index=0]`` at the
    root.
  * ``Dynamics`` follows action ``a``: new depth = depth+1, new
    path_index = path_index * BRANCHING + a. Reward is 0 except at leaf
    depth where it equals the leaf's value.
  * ``Prediction`` returns a uniform prior. Value is the leaf's stored
    value at leaf depth, 0 otherwise.

With enough simulations to fully expand a small tree, the MCTS root
visit-count policy should concentrate on the action that leads to the
sub-tree containing the maximum leaf value (SinglePlayer / no noise /
no discount → just argmax over leaves).

This test exercises:
  * Non-trivial tree expansion past depth 1.
  * Q-value propagation up the search path with gamma=1.
  * MinMax Q-normalization across nodes (without it, identical priors
    on a balanced tree would never break ties).

Usage:
    pixi run mojo run -I . tests/planners/tree_search/test_mcts_value_tree.mojo
"""

from std.math import abs as math_abs
from std.random import seed as _set_seed
from std.testing import assert_true

from mojo_rl.planners.tree_search import (
    GenericCPUMCTS,
    AlphaGoPUCT,
    NoNoise,
    SinglePlayer,
    Representation,
    Dynamics,
    Prediction,
)


# ─── Tree shape comptime params ───────────────────────────────────────────


comptime BRANCH: Int = 2
comptime DEPTH: Int = 3
comptime ACT: Int = BRANCH
comptime LATENT: Int = 2  # [depth, path_index]
comptime NUM_LEAVES: Int = 8  # = BRANCH ** DEPTH


# ─── Adapters ─────────────────────────────────────────────────────────────


@fieldwise_init
struct TreeRepresentation(
    Movable, ImplicitlyDeletable, Representation,
):
    """Root state: depth=0, path_index=0."""

    comptime OBS_DIM: Int = 1
    comptime LATENT_DIM: Int = LATENT

    def encode_cpu(
        mut self,
        obs: List[Float64],
        mut hidden_out: List[Float64],
    ) raises:
        hidden_out[0] = Float64(0.0)
        hidden_out[1] = Float64(0.0)


@fieldwise_init
struct TreeDynamics(Movable, ImplicitlyDeletable, Dynamics):
    """Advance one level: depth += 1, path_index = path_index * BRANCH + a.

    Reward is the leaf value at depth=DEPTH and 0 otherwise. This makes
    the optimal action the one whose sub-tree contains the maximum leaf.
    """

    comptime LATENT_DIM: Int = LATENT
    comptime ACTION_DIM: Int = ACT

    var leaf_values: List[Float64]

    def step_cpu(
        mut self,
        hidden_in: List[Float64],
        action: Int,
        mut hidden_out: List[Float64],
    ) raises -> Float64:
        var depth = Int(hidden_in[0])
        var path_idx = Int(hidden_in[1])

        var new_depth = depth + 1
        var new_path = path_idx * BRANCH + action

        hidden_out[0] = Float64(new_depth)
        hidden_out[1] = Float64(new_path)

        if new_depth == DEPTH:
            # Reached a leaf — reward = leaf value.
            return self.leaf_values[new_path]
        return Float64(0.0)


@fieldwise_init
struct TreePrediction(Movable, ImplicitlyDeletable, Prediction):
    """Uniform prior. Value is the leaf value at leaf depth, 0 elsewhere.

    Putting the value at the leaf gives MCTS a non-zero bootstrap when
    its expansion is shallower than DEPTH; for the configured
    NUM_SIMULATIONS we always fully expand so the bootstrap is
    redundant — but the trait still has to return something.
    """

    comptime LATENT_DIM: Int = LATENT
    comptime ACTION_DIM: Int = ACT

    var leaf_values: List[Float64]

    def predict_cpu(
        mut self,
        hidden: List[Float64],
        mut policy_out: List[Float64],
    ) raises -> Float64:
        for a in range(ACT):
            policy_out[a] = 1.0 / Float64(ACT)

        var depth = Int(hidden[0])
        var path_idx = Int(hidden[1])
        if depth == DEPTH:
            return self.leaf_values[path_idx]
        return Float64(0.0)


def _approx(a: Float64, b: Float64, tol: Float64 = 1e-9) -> Bool:
    return math_abs(a - b) <= tol


def _argmax_2(p0: Float64, p1: Float64) -> Int:
    return 0 if p0 >= p1 else 1


# ─── Test: argmax leaf ────────────────────────────────────────────────────


def test_mcts_picks_subtree_with_max_leaf() raises:
    """A small tree where the best leaf is reachable through action 0:

        leaves indexed 0..7 (row-major over a binary depth-3 tree):
        [10, 0, 0, 0, 0, 0, 0, 1]

        action 0 at the root descends into leaves [10, 0, 0, 0]
        action 1 at the root descends into leaves [0, 0, 0, 1]

    With enough sims to expand both subtrees, MCTS should prefer
    action 0 (max leaf = 10).
    """
    _set_seed(0x73E1)

    var leaves: List[Float64] = [
        10.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 1.0,
    ]

    # Enough simulations to expand every node twice over.
    var planner = GenericCPUMCTS[
        ACT, LATENT,
        128,      # NUM_SIMULATIONS
        256,      # MAX_NODES
        AlphaGoPUCT[c_puct=2.5],
        NoNoise,
        SinglePlayer,
    ](gamma=1.0)  # gamma=1: leaf reward propagates undiscounted.

    var rep = TreeRepresentation()
    var dyn = TreeDynamics(leaf_values=leaves.copy())
    var pred = TreePrediction(leaf_values=leaves.copy())

    var root_obs: List[Float64] = [0.0]
    var policy = planner.search[
        TreeRepresentation, TreeDynamics, TreePrediction
    ](rep, dyn, pred, root_obs, add_noise=False)

    assert_true(
        _approx(policy[0] + policy[1], 1.0, tol=1e-9),
        "visit-count policy should sum to 1",
    )

    var best = _argmax_2(policy[0], policy[1])
    assert_true(
        best == 0,
        "action 0 leads to the leaf with value 10 (max); got argmax="
        + String(best)
        + " policy=("
        + String(policy[0]) + ", " + String(policy[1]) + ")",
    )

    # Sanity: with 128 sims and a clear value gap (10 vs 1) action 0
    # should claim a sizeable majority of visits (> 70%).
    assert_true(
        policy[0] >= 0.7,
        "action 0 should claim a clear majority of visits, got "
        + String(policy[0]),
    )


def test_mcts_swap_winner_swaps_argmax() raises:
    """Mirror the previous test: put the max leaf in the right
    sub-tree and assert MCTS now prefers action 1. Falsifies any
    hard-coded action-0 bias in the selection/backup path.
    """
    _set_seed(0x73E2)

    var leaves: List[Float64] = [
         0.0, 0.0, 0.0, 1.0,
        10.0, 0.0, 0.0, 0.0,
    ]

    var planner = GenericCPUMCTS[
        ACT, LATENT,
        128, 256, AlphaGoPUCT[c_puct=2.5], NoNoise, SinglePlayer,
    ](gamma=1.0)

    var rep = TreeRepresentation()
    var dyn = TreeDynamics(leaf_values=leaves.copy())
    var pred = TreePrediction(leaf_values=leaves.copy())

    var root_obs: List[Float64] = [0.0]
    var policy = planner.search[
        TreeRepresentation, TreeDynamics, TreePrediction
    ](rep, dyn, pred, root_obs, add_noise=False)

    var best = _argmax_2(policy[0], policy[1])
    assert_true(
        best == 1,
        "swapping the best leaf to the right sub-tree should flip the"
        + " argmax; got policy=("
        + String(policy[0]) + ", " + String(policy[1]) + ")",
    )
    assert_true(
        policy[1] >= 0.7,
        "action 1 should claim a clear majority of visits after swap, got "
        + String(policy[1]),
    )


def test_mcts_root_value_within_leaf_range() raises:
    """``root_value`` should land inside the [min_leaf, max_leaf] range.
    A sanity that backup isn't producing out-of-bounds Q estimates.
    """
    _set_seed(0x73E3)

    var leaves: List[Float64] = [
        5.0, 3.0, 4.0, 2.0,
        1.0, 0.0, 7.0, 6.0,
    ]

    var planner = GenericCPUMCTS[
        ACT, LATENT,
        64, 256, AlphaGoPUCT[c_puct=2.5], NoNoise, SinglePlayer,
    ](gamma=1.0)

    var rep = TreeRepresentation()
    var dyn = TreeDynamics(leaf_values=leaves.copy())
    var pred = TreePrediction(leaf_values=leaves.copy())

    var root_obs: List[Float64] = [0.0]
    var _policy = planner.search[
        TreeRepresentation, TreeDynamics, TreePrediction
    ](rep, dyn, pred, root_obs, add_noise=False)

    var v = planner.root_value()
    # min leaf = 0, max leaf = 7. Allow 1e-9 slack.
    assert_true(
        v >= 0.0 - 1e-9 and v <= 7.0 + 1e-9,
        "root_value must lie inside [min_leaf, max_leaf]=[0, 7], got "
        + String(v),
    )


def main() raises:
    print("=== Phase 3 planners: CPU MCTS on KnownValueTree ===")
    test_mcts_picks_subtree_with_max_leaf()
    print("  PASS argmax matches the max-leaf sub-tree (best=action 0)")
    test_mcts_swap_winner_swaps_argmax()
    print("  PASS swapping best leaf flips argmax (best=action 1)")
    test_mcts_root_value_within_leaf_range()
    print("  PASS root_value stays inside [min_leaf, max_leaf]")
    print("OK")
