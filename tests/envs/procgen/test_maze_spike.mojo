"""Phase-0 end-to-end smoke: RNG → maze-gen → engine → render → obs → solve.

Builds the maze game, BFS-solves the grid to the goal, executes the moves
through the real `step()` (grid movement + collision + goal logic), and asserts
the level completes with reward 10. Also checks the 64×64×3 observation is
well-formed and non-degenerate. Proves the full spike pipeline.

Requires the reference asset dir; run from repo root:
    pixi run mojo run -I . tests/envs/procgen/test_maze_spike.mojo
"""

from std.testing import assert_equal, assert_true, TestSuite

from mojo_rl.envs.procgen.games import MazeGame, MazeEnv
from mojo_rl.envs.procgen.core.object_ids import WALL_OBJ

comptime ASSET_ROOT = String("assets/procgen/")
comptime GOAL = 2
comptime WORLD_DIM = 25


def _bfs_actions(game: MazeGame) -> List[Int]:
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
    var game = MazeGame(ASSET_ROOT)
    game.reset(7)
    var obs = game.render()
    assert_equal(len(obs), 64 * 64 * 3)
    # Non-degenerate: background + walls fill the frame with lit pixels, and the
    # frame is not a single flat color (walls vs background vs sprites differ).
    var lit = 0
    var mn = 255
    var mx = 0
    for i in range(len(obs)):
        var v = Int(obs[i])
        if v > 0:
            lit += 1
        if v < mn:
            mn = v
        if v > mx:
            mx = v
    assert_true(lit > 500, "render produced a near-black frame")
    assert_true(mx - mn > 40, "render produced a flat frame")


def test_maze_bfs_solves_level() raises:
    # A few seeds: BFS-solve → reward 10 + level_complete.
    var seeds: List[Int] = [0, 1, 7, 42]
    for si in range(len(seeds)):
        var game = MazeGame(ASSET_ROOT)
        game.reset(seeds[si])
        var actions = _bfs_actions(game)
        assert_true(len(actions) > 0, "no path found for seed")
        var total: Float32 = 0.0
        for k in range(len(actions)):
            total += game.step(actions[k])
        assert_true(game.level_complete, "level not completed")
        assert_equal(Int(total), 10)


def test_maze_env_episode() raises:
    # Benchmark env: reset samples a level from [0,200); BFS-solve the episode
    # through the MazeEnv API and confirm terminal reward + done via step().
    var env = MazeEnv(ASSET_ROOT, rand_seed=0, num_levels=200)
    var obs = env.reset()
    assert_equal(len(obs), MazeEnv.OBS_DIM)
    assert_true(
        env.current_level_seed >= 0 and env.current_level_seed < 200,
        "level seed out of configured range",
    )
    var actions = _bfs_actions(env.game)
    assert_true(len(actions) > 0, "no path found in env level")

    var last_reward: Float32 = 0.0
    var got_done = False
    var got_complete = False
    for k in range(len(actions)):
        var res = env.step(actions[k])
        assert_equal(len(res.obs), MazeEnv.OBS_DIM)
        last_reward = res.reward
        if res.done:
            got_done = True
        if res.level_complete:
            got_complete = True
    assert_true(got_done, "episode never terminated")
    assert_true(got_complete, "level never completed")
    assert_equal(Int(last_reward), 10)

    # A second reset picks another (in-range) level seed.
    _ = env.reset()
    assert_true(
        env.current_level_seed >= 0 and env.current_level_seed < 200,
        "second level seed out of range",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
