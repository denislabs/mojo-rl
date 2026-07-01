"""Phase-0 end-to-end smoke: RNG → maze-gen → engine → render → obs → solve.

Builds the maze game, BFS-solves the grid to the goal, executes the moves
through the real `step()` (grid movement + collision + goal logic), and asserts
the level completes with reward 10. Also checks the 64×64×3 observation is
well-formed and non-degenerate. Proves the full spike pipeline.

Requires the reference asset dir; run from repo root:
    pixi run mojo run -I . tests/envs/procgen/test_maze_spike.mojo
"""

from std.testing import assert_equal, assert_true, TestSuite

from mojo_rl.envs.procgen.games import MazeSpikeGame
from mojo_rl.envs.procgen.core.object_ids import WALL_OBJ

comptime ASSET_ROOT = String(
    "references/procgen-master/procgen/data/assets/"
)
comptime GOAL = 2
comptime WORLD_DIM = 25


def _bfs_actions(game: MazeSpikeGame) -> List[Int]:
    """BFS from the agent's cell to the GOAL cell; return the action sequence.
    Action codes: right=7, left=1, up(+y)=5, down(-y)=3."""
    var w = game.w
    var start_x = Int(game.agent.x)
    var start_y = Int(game.agent.y)

    var goal_x = -1
    var goal_y = -1
    for y in range(game.h):
        for x in range(w):
            if game._get_obj(x, y) == GOAL:
                goal_x = x
                goal_y = y

    var prev = List[Int]()
    prev.resize(w * game.h, -2)  # -2 = unvisited
    var queue = List[Int]()
    var start_idx = start_y * w + start_x
    queue.append(start_idx)
    prev[start_idx] = -1  # root

    var dxs: List[Int] = [1, -1, 0, 0]
    var dys: List[Int] = [0, 0, 1, -1]

    var head = 0
    while head < len(queue):
        var cur = queue[head]
        head += 1
        var cx = cur % w
        var cy = cur // w
        if cx == goal_x and cy == goal_y:
            break
        for d in range(4):
            var nx = cx + dxs[d]
            var ny = cy + dys[d]
            if nx < 0 or nx >= w or ny < 0 or ny >= game.h:
                continue
            if game._get_obj(nx, ny) == WALL_OBJ:
                continue
            var nidx = ny * w + nx
            if prev[nidx] != -2:
                continue
            prev[nidx] = cur
            queue.append(nidx)

    # Reconstruct path goal→start, then reverse into actions.
    var cells = List[Int]()
    var node = goal_y * w + goal_x
    while node != -1:
        cells.append(node)
        node = prev[node]

    var actions = List[Int]()
    for k in range(len(cells) - 1, 0, -1):
        var frm = cells[k]
        var to = cells[k - 1]
        var fx = frm % w
        var fy = frm // w
        var tx = to % w
        var ty = to // w
        if tx == fx + 1:
            actions.append(7)  # right
        elif tx == fx - 1:
            actions.append(1)  # left
        elif ty == fy + 1:
            actions.append(5)  # up (+y)
        elif ty == fy - 1:
            actions.append(3)  # down (-y)
    return actions^


def test_maze_reset_and_obs_wellformed() raises:
    var game = MazeSpikeGame(ASSET_ROOT)
    game.reset(7)
    var obs = game.render()
    assert_equal(len(obs), 64 * 64 * 3)
    # Non-degenerate: at least one lit pixel (walls/agent over black bg).
    var lit = 0
    for i in range(len(obs)):
        if obs[i] > 0:
            lit += 1
    assert_true(lit > 500, "render produced a near-black frame")


def test_maze_bfs_solves_level() raises:
    # A few seeds: BFS-solve → reward 10 + level_complete.
    var seeds: List[Int] = [0, 1, 7, 42]
    for si in range(len(seeds)):
        var game = MazeSpikeGame(ASSET_ROOT)
        game.reset(seeds[si])
        var actions = _bfs_actions(game)
        assert_true(len(actions) > 0, "no path found for seed")
        var total: Float32 = 0.0
        for k in range(len(actions)):
            total += game.step(actions[k])
        assert_true(game.level_complete, "level not completed")
        assert_equal(Int(total), 10)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
