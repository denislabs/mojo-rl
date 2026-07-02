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

from std.math import floor, ceil, sqrt

from ..core.entity import Entity
from ..core.randgen import RandGen
from ..core.mazegen import MazeGen, MAZE_OFFSET
from ..core.object_ids import SPACE, WALL_OBJ, PLAYER, INVALID_OBJ, INVALID_IDX


def _fsign(x: Float32) -> Float32:
    # cpp-utils sign(): -1 / 0 / +1 (note sign(0) == 0).
    if x > 0:
        return 1.0
    if x == 0:
        return 0.0
    return -1.0

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
    var step_rand_int: Int
    var action_vx: Float32
    var action_vy: Float32
    var reward: Float32

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
        self.step_rand_int = 0
        self.action_vx = 0.0
        self.action_vy = 0.0
        self.reward = 0.0

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

    # --- physics / collision helpers (BasicAbstractGame subset) ---
    def _obj_from_floats(self, fi: Float32, fj: Float32) -> Int:
        if fi < 0.0 or fj < 0.0:
            return INVALID_OBJ
        var xi = Int(floor(fi))
        var yj = Int(floor(fj))
        if xi < 0 or xi >= self.w or yj < 0 or yj >= self.h:
            return INVALID_OBJ
        return self.grid[yj * self.w + xi]

    def _is_blocked(self, target: Int) -> Bool:
        # chaser is_blocked: MAZE_WALL blocks; base adds WALL_OBJ + out-of-bounds.
        return target == MAZE_WALL or target == WALL_OBJ or target == INVALID_OBJ

    def _to_grid_idx(self, x: Int, y: Int) -> Int:
        if x < 0 or x >= self.w or y < 0 or y >= self.h:
            return INVALID_IDX
        return y * self.w + x

    def get_agent_index(self) -> Int:
        return Int(self.agent.y) * self.w + Int(self.agent.x)

    def _manhattan_dist(self, a: Int, b: Int) -> Int:
        return abs((a % self.w) - (b % self.w)) + abs(
            (a // self.w) - (b // self.w)
        )

    def _get_adjacent(self, idx: Int) -> List[Int]:
        var x = idx % self.w
        var y = idx // self.w
        var neighbors = List[Int]()
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                if i != 0 and j != 0:
                    continue
                var n = self._to_grid_idx(x + i, y + j)
                if n != INVALID_IDX:
                    neighbors.append(n)
        return neighbors^

    def can_eat_enemies(self) -> Bool:
        return self.cur_time - self.eat_time < EAT_TIMEOUT

    def _sub_step(self, mut obj: Entity, vx: Float32, vy: Float32) -> Bool:
        # grid-only sub_step (chaser entities never block/reflect each other, so
        # the reference entity-push loop is a no-op → dropped). Returns whether
        # the move was blocked on its axis.
        if obj.will_erase:
            return False
        var nx = obj.x + vx
        var ny = obj.y + vy
        var margin: Float32 = 0.98
        var is_horizontal = vx != 0.0
        var block = False
        for i in range(2):
            for j in range(2):
                var t = self._obj_from_floats(
                    nx + obj.rx * margin * Float32(2 * i - 1),
                    ny + obj.ry * margin * Float32(2 * j - 1),
                )
                if self._is_blocked(t):
                    block = True
        if block:
            if is_horizontal:
                if vx > 0.0:
                    nx = floor(nx + obj.rx) - obj.rx
                else:
                    nx = ceil(nx - obj.rx) + obj.rx
            else:
                if vy > 0.0:
                    ny = floor(ny + obj.ry) - obj.ry
                else:
                    ny = ceil(ny - obj.ry) + obj.ry
        obj.x = nx
        obj.y = ny
        return block

    def _basic_step_object(self, mut obj: Entity):
        if obj.will_erase:
            return
        var speed = sqrt(obj.vx * obj.vx + obj.vy * obj.vy)
        var nsub = Int(4.0 * speed)
        if nsub < 4:
            nsub = 4
        var pct = 1.0 / Float32(nsub)
        var cmp = abs(obj.vx) - abs(obj.vy)
        var step_x_first: Bool
        if cmp == 0.0:
            step_x_first = self.step_rand_int % 2 == 0
        else:
            step_x_first = cmp > 0.0
        if obj.type == PLAYER:
            if self.action_vx != 0.0:
                step_x_first = True
            if self.action_vy != 0.0:
                step_x_first = False
        var vx_pct: Float32 = 0.0
        var vy_pct: Float32 = 0.0
        for _ in range(nsub):
            var bx: Bool
            var by: Bool
            if step_x_first:
                bx = self._sub_step(obj, obj.vx * pct, 0.0)
                by = self._sub_step(obj, 0.0, obj.vy * pct)
            else:
                by = self._sub_step(obj, 0.0, obj.vy * pct)
                bx = self._sub_step(obj, obj.vx * pct, 0.0)
            if not bx:
                vx_pct += 1.0
            if not by:
                vy_pct += 1.0
            if bx and by:
                break
        obj.vx *= vx_pct / Float32(nsub)
        obj.vy *= vy_pct / Float32(nsub)

    def step(mut self, action: Int) -> Float32:
        self.cur_time += 1
        self.reward = 0.0
        self.done = False
        self.level_complete = False

        # --- BasicAbstractGame::game_step base ---
        self.step_rand_int = self.rand_gen.randint(0, 1000000)
        var move_action = action % 9
        if action >= 9:
            move_action = 4  # special action → stand still
        self.action_vx = Float32(move_action // 3 - 1)
        self.action_vy = Float32(move_action % 3 - 1)

        # update_agent_velocity (chaser override): constant-speed Pac-Man movement.
        if self.action_vx != 0.0:
            self.agent.vx = MAXSPEED * self.action_vx
        if self.action_vy != 0.0:
            self.agent.vy = MAXSPEED * self.action_vy
        self.agent.vx = _fsign(self.agent.vx) * MAXSPEED
        self.agent.vy = _fsign(self.agent.vy) * MAXSPEED

        # step_entities: move smart entities (enemies) then the agent. Chaser
        # entities don't interact during movement → step order is irrelevant.
        for i in range(len(self.entities) - 1, -1, -1):
            if self.entities[i].smart_step:
                var e = self.entities[i].copy()
                self._basic_step_object(e)
                self.entities[i] = e^
        var a = self.agent.copy()
        self._basic_step_object(a)
        self.agent = a^

        # agent-collision pass + erase.
        for i in range(len(self.entities) - 1, -1, -1):
            var etype = self.entities[i].type
            var tx = self.entities[i].rx + self.agent.rx
            var ty = self.entities[i].ry + self.agent.ry
            if (
                abs(self.entities[i].x - self.agent.x) < tx
                and abs(self.entities[i].y - self.agent.y) < ty
            ):
                if etype == LARGE_ORB:
                    self.eat_time = self.cur_time
                    self.reward += ORB_REWARD
                    self.entities[i].will_erase = True
                elif etype == ENEMY:
                    if self.can_eat_enemies():
                        self.entities[i].will_erase = True
                    else:
                        self.done = True
        for i in range(len(self.entities) - 1, -1, -1):
            if self.entities[i].will_erase:
                _ = self.entities.pop(i)

        # --- ChaserGame::game_step: enemy AI + spawns + orb pickup + completion ---
        var num_enemies = 0
        var default_enemy_speed: Float32 = 0.5
        var vscale = (
            default_enemy_speed * 0.5
        ) if self.can_eat_enemies() else default_enemy_speed
        for j in range(len(self.entities) - 1, -1, -1):
            var etype = self.entities[j].type
            if etype == ENEMY_EGG:
                num_enemies += 1
                var hp = self.entities[j].health - 1.0
                self.entities[j].health = hp
                if hp == 0.0:
                    self.entities[j].will_erase = True
                    var ex = self.entities[j].x
                    var ey = self.entities[j].y
                    var enemy = Entity(ex, ey, 0.0, 0.0, 0.5, 0.5, ENEMY)
                    enemy.smart_step = True
                    self.entities.append(enemy^)
            elif etype == ENEMY:
                num_enemies += 1
                var x = self.entities[j].x - 0.5
                var y = self.entities[j].y - 0.5
                var evx = self.entities[j].vx
                var evy = self.entities[j].vy
                var dist_scale = -1 if self.can_eat_enemies() else 1
                var enemy_idx = self._to_grid_idx(Int(x), Int(y))
                var agent_idx = self._to_grid_idx(
                    Int(self.agent.x), Int(self.agent.y)
                )
                var rx_ = floor(x + 0.5)
                var ry_ = floor(y + 0.5)
                var is_at_junction = abs(x - rx_) + abs(y - ry_) < 0.01
                var be_agressive = self.step_rand_int % 2 == 0
                if (evx == 0.0 and evy == 0.0) or is_at_junction:
                    var prev_idx = self._to_grid_idx(
                        Int(x - _fsign(evx)), Int(y - _fsign(evy))
                    )
                    var adj = self._get_adjacent(enemy_idx)
                    var min_dist = 2 * self.w
                    var space_neighbors = List[Int]()
                    for ai in range(len(adj)):
                        var nb = adj[ai]
                        if self.is_space_vec[nb] and nb != prev_idx:
                            var md = self._manhattan_dist(nb, agent_idx) * dist_scale
                            if be_agressive:
                                if md < min_dist:
                                    min_dist = md
                                    space_neighbors = List[Int]()
                                    space_neighbors.append(nb)
                                elif md == min_dist:
                                    space_neighbors.append(nb)
                            else:
                                space_neighbors.append(nb)
                    var ni = self.step_rand_int % len(space_neighbors)
                    var neighbor = space_neighbors[ni]
                    var nx = neighbor % self.w
                    var ny = neighbor // self.w
                    self.entities[j].vx = (Float32(nx) - x) * vscale
                    self.entities[j].vy = (Float32(ny) - y) * vscale

        if num_enemies < self.total_enemies:
            var si = self.step_rand_int % len(self.free_cells)
            var cell = self.free_cells[si]
            var egg = Entity(
                Float32(cell % self.maze_dim) + 0.5,
                Float32(cell // self.maze_dim) + 0.5,
                0.0,
                0.0,
                0.5,
                0.5,
                ENEMY_EGG,
            )
            egg.health = Float32(EGG_TIMEOUT)
            self.entities.append(egg^)

        var agent_idx = self.get_agent_index()
        if self.grid[agent_idx] == ORB:
            self.grid[agent_idx] = SPACE
            self.reward += ORB_REWARD
            self.orbs_collected += 1
        if self.orbs_collected == self.total_orbs:
            self.reward += COMPLETION_BONUS
            self.level_complete = True
            self.done = True

        self.episode_reward += self.reward
        return self.reward
