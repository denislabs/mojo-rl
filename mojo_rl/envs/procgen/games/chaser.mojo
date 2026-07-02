"""Chaser game — Pac-Man-like maze pursuit (port of `games/chaser.cpp`).

Chaser is the first game on the `BasicAbstractGame` **entity substrate**: the
agent and enemies are continuous-position entities that move by velocity + grid
collision (not grid-locked like maze). `game_reset(level_seed)` replays the exact
BasicAbstractGame base-reset + ChaserGame::game_reset RNG order so a level seed
reproduces reference Procgen's layout (gated by `test_chaser_reset_parity.mojo`).

Fidelity = level-exact (bit-exact draw order) + visual-approx. See
`docs/PROCGEN_CHASER_SCOPE.md`.

Phase 0 = reset parity (this file's `game_reset` + grid/entity helpers). The step
physics (enemy AI, orb pickup, completion) + rendering land in P1/P2.
"""

from ..core.entity import Entity
from ..core.randgen import RandGen
from ..core.mazegen import MazeGen, MAZE_OFFSET
from ..core.object_ids import SPACE, WALL_OBJ, PLAYER

# Chaser object ids (chaser.cpp).
comptime LARGE_ORB = 2
comptime ENEMY_WEAK = 3
comptime ENEMY_EGG = 4
comptime MAZE_WALL = 5
comptime ENEMY = 6
comptime ENEMY2 = 7
comptime ENEMY3 = 8
comptime MARKER = 1001
comptime ORB = 1002

comptime ORB_REWARD: Float32 = 0.04
comptime COMPLETION_BONUS: Float32 = 10.0
comptime A_R: Float32 = 0.4  # base agent radius used for the base-reset spawn draws
comptime BG_COUNT = 1  # topdown_simple_backgrounds (resources.cpp) → randn(1) == 0
comptime EAT_TIMEOUT = 75
comptime EGG_TIMEOUT = 50
comptime MAXSPEED: Float32 = 0.5

# DistributionMode (game.h): chaser supports Easy / Hard / Extreme.
comptime DIST_EASY = 0
comptime DIST_HARD = 1
comptime DIST_EXTREME = 2


struct ChaserGame(Copyable, Movable):
    var rand_gen: RandGen
    var grid: List[Int]
    var w: Int
    var h: Int
    var maze_dim: Int
    var dist_mode: Int
    var total_enemies: Int
    var extra_orb_sign: Int
    var agent: Entity
    var entities: List[Entity]  # non-agent entities (large orbs, eggs, enemies)
    var total_orbs: Int
    var orbs_collected: Int
    var eat_time: Int
    var free_cells: List[Int]
    var is_space_vec: List[Bool]
    var bg_pct_x: Float32
    var background_index: Int
    var episode_reward: Float32
    var done: Bool
    var level_complete: Bool
    var cur_time: Int

    def __init__(out self, dist_mode: Int = DIST_EASY):
        self.rand_gen = RandGen()
        self.dist_mode = dist_mode
        # (maze_dim, total_enemies, extra_orb_sign) per distribution mode.
        if dist_mode == DIST_EXTREME:
            self.maze_dim = 19
            self.total_enemies = 5
            self.extra_orb_sign = 1
        elif dist_mode == DIST_HARD:
            self.maze_dim = 13
            self.total_enemies = 3
            self.extra_orb_sign = -1
        else:  # EasyMode
            self.maze_dim = 11
            self.total_enemies = 3
            self.extra_orb_sign = 0
        self.w = self.maze_dim
        self.h = self.maze_dim
        self.grid = List[Int]()
        self.agent = Entity.make(0.5, 0.5, A_R, PLAYER)
        self.entities = List[Entity]()
        self.total_orbs = 0
        self.orbs_collected = 0
        self.eat_time = -EAT_TIMEOUT
        self.free_cells = List[Int]()
        self.is_space_vec = List[Bool]()
        self.bg_pct_x = 0.0
        self.background_index = 0
        self.episode_reward = 0.0
        self.done = False
        self.level_complete = False
        self.cur_time = 0

    # --- grid helpers (BasicAbstractGame subset; set_obj(i,j) → idx j*w+i) ---
    def _get_obj(self, x: Int, y: Int) -> Int:
        return self.grid[y * self.w + x]

    def _get_obj_idx(self, idx: Int) -> Int:
        return self.grid[idx]

    def _set_obj(mut self, x: Int, y: Int, v: Int):
        self.grid[y * self.w + x] = v

    def _set_obj_idx(mut self, idx: Int, v: Int):
        self.grid[idx] = v

    def _cells_with_type(self, type: Int) -> List[Int]:
        # get_cells_with_type: scan grid indices in ascending order.
        var cells = List[Int]()
        for i in range(self.w * self.h):
            if self.grid[i] == type:
                cells.append(i)
        return cells^

    def _add_entity(
        mut self, x: Float32, y: Float32, r: Float32, type: Int
    ) -> Int:
        # add_entity(x,y,0,0,r,type); returns its index in self.entities.
        self.entities.append(Entity(x, y, 0.0, 0.0, r, r, type))
        return len(self.entities) - 1

    def _spawn_egg(mut self, cell: Int):
        var idx = self._add_entity(
            Float32(cell % self.maze_dim) + 0.5,
            Float32(cell // self.maze_dim) + 0.5,
            0.5,
            ENEMY_EGG,
        )
        self.entities[idx].health = Float32(EGG_TIMEOUT)

    def reset(mut self, level_seed: Int):
        self.rand_gen.seed(level_seed)
        self.episode_reward = 0.0
        self.done = False
        self.level_complete = False
        self.cur_time = 0
        self.orbs_collected = 0
        self.eat_time = -EAT_TIMEOUT
        self.entities = List[Entity]()

        # --- BasicAbstractGame::game_reset base draws, in exact order ---
        self.bg_pct_x = self.rand_gen.rand01()
        self.background_index = self.rand_gen.randn(BG_COUNT)  # == 0
        # random_agent_start=true → two rand01 draws (base agent x/y). Chaser
        # overwrites these below, but the draws MUST be consumed for parity.
        _ = self.rand_gen.rand01() * (Float32(self.w) - 2 * A_R) + A_R
        _ = self.rand_gen.rand01() * (Float32(self.h) - 2 * A_R) + A_R

        # --- ChaserGame::game_reset ---
        # Whole grid → MAZE_WALL, then overlay the no-dead-ends maze.
        self.grid = List[Int]()
        self.grid.resize(self.w * self.h, MAZE_WALL)

        var mg = MazeGen(self.maze_dim)
        mg.generate_maze_no_dead_ends(self.rand_gen)

        var num_quadrants = 4
        var extra_quad = self.rand_gen.randn(num_quadrants)

        var orbs_for_quadrant = List[Int]()
        for i in range(num_quadrants):
            orbs_for_quadrant.append(
                1 + (self.extra_orb_sign if i == extra_quad else 0)
            )

        # Overlay maze + collect per-quadrant free (SPACE) cells (i-major, j-inner).
        var quadrants = List[List[Int]]()
        for _ in range(num_quadrants):
            quadrants.append(List[Int]())
        var half = Float32(self.maze_dim) / 2.0
        for i in range(self.maze_dim):
            for j in range(self.maze_dim):
                var obj = mg.grid.get(i + MAZE_OFFSET, j + MAZE_OFFSET)
                self._set_obj(i, j, MAZE_WALL if obj == WALL_OBJ else obj)
                if obj == SPACE:
                    var idx = j * self.maze_dim + i
                    var qx = 1 if Float32(i) >= half else 0
                    var qy = 1 if Float32(j) >= half else 0
                    quadrants[qx * 2 + qy].append(idx)

        # Place large orbs per quadrant (marks their cells; entities spawned).
        for i in range(num_quadrants):
            var num_orbs = orbs_for_quadrant[i]
            var sel = self.rand_gen.simple_choose(len(quadrants[i]), num_orbs)
            for s in range(len(sel)):
                var cell = quadrants[i][sel[s]]
                _ = self._add_entity(
                    Float32(cell % self.maze_dim) + 0.5,
                    Float32(cell // self.maze_dim) + 0.5,
                    0.4,
                    LARGE_ORB,
                )
                self._set_obj_idx(cell, MARKER)

        # Agent start + enemy eggs from remaining SPACE cells.
        self.free_cells = self._cells_with_type(SPACE)
        var sel = self.rand_gen.simple_choose(
            len(self.free_cells), 1 + self.total_enemies
        )
        var start = self.free_cells[sel[0]]
        self.agent = Entity.make(
            Float32(start % self.maze_dim) + 0.5,
            Float32(start // self.maze_dim) + 0.5,
            0.5,
            PLAYER,
        )
        for i in range(self.total_enemies):
            var cell = self.free_cells[sel[i + 1]]
            self._set_obj_idx(cell, MARKER)
            self._spawn_egg(cell)

        # All (SPACE) free cells become orbs; large-orb marker cells revert to SPACE.
        for k in range(len(self.free_cells)):
            self._set_obj_idx(self.free_cells[k], ORB)
        self.total_orbs = len(self.free_cells)

        for i in range(self.w * self.h):
            if self.grid[i] == MARKER:
                self.grid[i] = SPACE

        # Rebuild free_cells / is_space_vec = non-wall cells.
        self.free_cells = List[Int]()
        self.is_space_vec = List[Bool]()
        for i in range(self.w * self.h):
            var is_space = self.grid[i] != MAZE_WALL
            if is_space:
                self.free_cells.append(i)
            self.is_space_vec.append(is_space)
