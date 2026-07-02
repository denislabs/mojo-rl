"""Heist game — maze with keys, locked doors, and a gem exit (port of `heist.cpp`).

Completes the maze-family (maze / chaser / heist share `MazeGen`). The player
navigates a `generate_maze_with_doors` maze, collects colored keys, opens the
matching locked doors, and reaches the gem (exit). No enemies, gravity, or
shooting — it reuses the chaser entity substrate almost entirely, adding only the
door/key logic and (in P1) door collision-blocking.

`game_reset(level_seed)` replays the exact BasicAbstractGame base-reset +
HeistGame::game_reset RNG order (bg draws → agent-spawn draws → difficulty →
num_keys → generate_maze_with_doors → off_x/off_y → collision-checked KEY/EXIT
spawns). Level-exact + visual-approx. See `docs/PROCGEN_HEIST_SCOPE.md`.

P0 = reset parity (this file's game_reset). Step/door-blocking + render + env
follow in P1/P2.
"""

from std.math import floor
from std.memory import ArcPointer

from ..core.entity import Entity
from ..core.randgen import RandGen
from ..core.mazegen import MazeGen, MAZE_OFFSET
from ..core.object_ids import (
    SPACE,
    WALL_OBJ,
    PLAYER,
    DOOR_OBJ,
    KEY_OBJ,
    EXIT_OBJ,
    AGENT_OBJ,
)

# Heist entity type ids (heist.cpp).
comptime LOCKED_DOOR = 1
comptime KEY = 2
comptime EXIT = 9
comptime KEY_ON_RING = 11

comptime COMPLETION_BONUS: Float32 = 10.0
comptime A_R: Float32 = 0.4  # base agent radius (base-reset spawn draws)
comptime BG_COUNT = 9  # topdown_backgrounds (resources.cpp)
comptime MAXSPEED: Float32 = 0.75

# DistributionMode (game.h): heist supports Easy / Hard / Memory.
comptime DIST_EASY = 0
comptime DIST_HARD = 1
comptime DIST_MEMORY = 10


def heist_world_dim(dist_mode: Int) -> Int:
    if dist_mode == DIST_MEMORY:
        return 23
    if dist_mode == DIST_HARD:
        return 13
    return 9  # EasyMode


struct HeistGame(Copyable, Movable):
    var rand_gen: RandGen
    var grid: List[Int]
    var w: Int
    var h: Int
    var world_dim: Int
    var dist_mode: Int
    var agent: Entity
    var entities: List[Entity]  # keys, locked doors, exit (+ HUD ring in P2)
    var has_keys: List[Bool]
    var num_keys: Int
    var maze_dim: Int
    var off_x: Int
    var off_y: Int
    var bg_pct_x: Float32
    var background_index: Int
    var episode_reward: Float32
    var done: Bool
    var level_complete: Bool
    var cur_time: Int

    def __init__(out self, dist_mode: Int = DIST_EASY):
        self.rand_gen = RandGen()
        self.dist_mode = dist_mode
        self.world_dim = heist_world_dim(dist_mode)
        self.w = self.world_dim
        self.h = self.world_dim
        self.grid = List[Int]()
        self.agent = Entity.make(0.5, 0.5, A_R, PLAYER)
        self.entities = List[Entity]()
        self.has_keys = List[Bool]()
        self.num_keys = 0
        self.maze_dim = 0
        self.off_x = 0
        self.off_y = 0
        self.bg_pct_x = 0.0
        self.background_index = 0
        self.episode_reward = 0.0
        self.done = False
        self.level_complete = False
        self.cur_time = 0

    # --- collision-checked spawns (BasicAbstractGame subset heist needs) ---
    @staticmethod
    def _rand_pos(mut rg: RandGen, r: Float32, mn: Float32, mx: Float32) -> Float32:
        if mx - mn <= 2 * r:
            return (mx + mn) / 2
        return (mx - mn - 2 * r) * rg.rand01() + r + mn

    def _has_collision(
        self, ex: Float32, ey: Float32, erx: Float32, ery: Float32,
        ox: Float32, oy: Float32, orx: Float32, ory: Float32,
    ) -> Bool:
        return abs(ex - ox) < (erx + orx) and abs(ey - oy) < (ery + ory)

    def _spawns_collide(
        self, ex: Float32, ey: Float32, erx: Float32, ery: Float32
    ) -> Bool:
        # has_agent_collision (skipped for PLAYER — spawns are never PLAYER) +
        # has_any_collision against already-placed entities.
        if self._has_collision(
            ex, ey, erx, ery, self.agent.x, self.agent.y, self.agent.rx, self.agent.ry
        ):
            return True
        for i in range(len(self.entities)):
            ref o = self.entities[i]
            if self._has_collision(ex, ey, erx, ery, o.x, o.y, o.rx, o.ry):
                return True
        return False

    def _spawn_entity(
        mut self, r: Float32, type: Int, x: Float32, y: Float32, theme: Int
    ):
        # spawn_entity(r, type, x, y, w=1, h=1, check=true) with rx=ry=r.
        var ex = HeistGame._rand_pos(self.rand_gen, r, x, x + 1.0)
        var ey = HeistGame._rand_pos(self.rand_gen, r, y, y + 1.0)
        var count = 0
        while self._spawns_collide(ex, ey, r, r) and count < 100:
            ex = HeistGame._rand_pos(self.rand_gen, r, x, x + 1.0)
            ey = HeistGame._rand_pos(self.rand_gen, r, y, y + 1.0)
            count += 1
        var e = Entity(ex, ey, 0.0, 0.0, r, r, type)
        e.image_theme = theme
        self.entities.append(e^)

    def _add_entity(
        mut self, x: Float32, y: Float32, r: Float32, type: Int, theme: Int
    ):
        var e = Entity(x, y, 0.0, 0.0, r, r, type)
        e.image_theme = theme
        self.entities.append(e^)

    def _set_obj(mut self, x: Int, y: Int, v: Int):
        self.grid[y * self.w + x] = v

    def reset(mut self, level_seed: Int):
        self.rand_gen.seed(level_seed)
        self.episode_reward = 0.0
        self.done = False
        self.level_complete = False
        self.cur_time = 0
        self.entities = List[Entity]()

        # --- BasicAbstractGame::game_reset base draws (heist bg = topdown_backgrounds) ---
        self.bg_pct_x = self.rand_gen.rand01()
        self.background_index = self.rand_gen.randn(BG_COUNT)
        # random_agent_start=true → two rand01 draws (overwritten below).
        _ = self.rand_gen.rand01() * (Float32(self.w) - 2 * A_R) + A_R
        _ = self.rand_gen.rand01() * (Float32(self.h) - 2 * A_R) + A_R

        # --- HeistGame::game_reset ---
        var max_diff = (self.world_dim - 5) // 2
        var difficulty = self.rand_gen.randn(max_diff + 1)
        if self.dist_mode == DIST_MEMORY:
            self.num_keys = self.rand_gen.randn(4)
        else:
            self.num_keys = difficulty + self.rand_gen.randn(2)
        if self.num_keys > 3:
            self.num_keys = 3

        self.has_keys = List[Bool]()
        for _ in range(self.num_keys):
            self.has_keys.append(False)

        self.maze_dim = difficulty * 2 + 5
        # maze_scale = main_height / world_dim = 1.0 → cell coords are integers.
        self.agent = Entity.make(-1.0, -1.0, 0.375, PLAYER)

        var mg = MazeGen(self.maze_dim)
        mg.generate_maze_with_doors(self.rand_gen, self.num_keys)

        self.off_x = self.rand_gen.randn(self.world_dim - self.maze_dim + 1)
        self.off_y = self.rand_gen.randn(self.world_dim - self.maze_dim + 1)

        self.grid = List[Int]()
        self.grid.resize(self.w * self.h, WALL_OBJ)

        var r_ent: Float32 = 0.5  # scale / 2
        for i in range(self.maze_dim):
            for j in range(self.maze_dim):
                var x = self.off_x + i
                var y = self.off_y + j
                var obj = mg.grid.get(i + MAZE_OFFSET, j + MAZE_OFFSET)
                var obj_x = Float32(x) + 0.5
                var obj_y = Float32(y) + 0.5
                if obj != WALL_OBJ:
                    self._set_obj(x, y, SPACE)
                if obj >= KEY_OBJ:
                    self._spawn_entity(
                        0.375, KEY, Float32(x), Float32(y), obj - KEY_OBJ - 1
                    )
                elif obj >= DOOR_OBJ:
                    self._add_entity(
                        obj_x, obj_y, r_ent, LOCKED_DOOR, obj - DOOR_OBJ - 1
                    )
                elif obj == EXIT_OBJ:
                    self._spawn_entity(0.375, EXIT, Float32(x), Float32(y), 0)
                elif obj == AGENT_OBJ:
                    self.agent.x = obj_x
                    self.agent.y = obj_y
