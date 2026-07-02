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

from std.math import floor, ceil, sqrt, atan2
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
comptime MIXRATE: Float32 = 0.5  # base agent-velocity blend (heist doesn't override)
comptime POS_EPS: Float32 = -0.001

# DistributionMode (game.h): heist supports Easy / Hard / Memory.
comptime DIST_EASY = 0
comptime DIST_HARD = 1
comptime DIST_MEMORY = 10


def _fsign(x: Float32) -> Float32:
    if x > 0:
        return 1.0
    if x == 0:
        return 0.0
    return -1.0


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
    var step_rand_int: Int
    var action_vx: Float32
    var action_vy: Float32
    var reward: Float32

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
        self.step_rand_int = 0
        self.action_vx = 0.0
        self.action_vy = 0.0
        self.reward = 0.0

    # --- collision-checked spawns (BasicAbstractGame subset heist needs) ---
    @staticmethod
    def _rand_pos(mut rg: RandGen, r: Float32, mn: Float32, mx: Float32) -> Float32:
        if mx - mn <= 2 * r:
            return (mx + mn) / 2
        return (mx - mn - 2 * r) * rg.rand01() + r + mn

    def _has_collision(
        self, ex: Float32, ey: Float32, erx: Float32, ery: Float32,
        ox: Float32, oy: Float32, orx: Float32, ory: Float32,
        margin: Float32 = 0.0,
    ) -> Bool:
        return (
            abs(ex - ox) < (erx + orx + margin)
            and abs(ey - oy) < (ery + ory + margin)
        )

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

    # --- step physics (BasicAbstractGame continuous + door blocking) ---
    def _obj_from_floats(self, fi: Float32, fj: Float32) -> Int:
        # out_of_bounds_object == WALL_OBJ for heist.
        if fi < 0.0 or fj < 0.0:
            return WALL_OBJ
        var xi = Int(floor(fi))
        var yj = Int(floor(fj))
        if xi < 0 or xi >= self.w or yj < 0 or yj >= self.h:
            return WALL_OBJ
        return self.grid[yj * self.w + xi]

    def _push_obj(
        self,
        src_x: Float32,
        src_y: Float32,
        src_rx: Float32,
        src_ry: Float32,
        mut obj: Entity,
        is_h: Bool,
        depth: Int,
    ) -> Bool:
        # Reposition `obj` just outside the blocking `src` and zero its velocity
        # on that axis (chaser could skip this; heist's locked doors need it).
        var t_vx: Float32 = 0.0
        var t_vy: Float32 = 0.0
        if is_h:
            var rsum = src_rx + obj.rx
            t_vx = src_x + _fsign(obj.x - src_x) * rsum - obj.x
        else:
            var rsum = src_ry + obj.ry
            t_vy = src_y + _fsign(obj.y - src_y) * rsum - obj.y
        var block = False
        if depth < 5:
            block = self._sub_step(obj, t_vx, t_vy, depth + 1)
        if is_h:
            obj.vx = 0.0
        else:
            obj.vy = 0.0
        return block

    def _sub_step(
        self, mut obj: Entity, vx: Float32, vy: Float32, depth: Int
    ) -> Bool:
        if obj.will_erase:
            return False
        var nx = obj.x + vx
        var ny = obj.y + vy
        var margin: Float32 = 0.98
        var is_h = vx != 0.0
        var block = False
        for i in range(2):
            for j in range(2):
                var t = self._obj_from_floats(
                    nx + obj.rx * margin * Float32(2 * i - 1),
                    ny + obj.ry * margin * Float32(2 * j - 1),
                )
                if t == WALL_OBJ:
                    block = True
        if block:
            if is_h:
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

        # Entity-collision pass: locked doors (no key) block the agent.
        var block2 = False
        for i in range(len(self.entities) - 1, -1, -1):
            if self.entities[i].will_erase:
                continue
            var mt = self.entities[i].type
            var mtheme = self.entities[i].image_theme
            var mx = self.entities[i].x
            var my = self.entities[i].y
            var mrx = self.entities[i].rx
            var mry = self.entities[i].ry
            if self._has_collision(
                obj.x, obj.y, obj.rx, obj.ry, mx, my, mrx, mry, POS_EPS
            ):
                if mt == LOCKED_DOOR and not self.has_keys[mtheme]:
                    _ = self._push_obj(mx, my, mrx, mry, obj, is_h, depth)
                    block2 = True
        return block or block2

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
                bx = self._sub_step(obj, obj.vx * pct, 0.0, 0)
                by = self._sub_step(obj, 0.0, obj.vy * pct, 0)
            else:
                by = self._sub_step(obj, 0.0, obj.vy * pct, 0)
                bx = self._sub_step(obj, obj.vx * pct, 0.0, 0)
            if not bx:
                vx_pct += 1.0
            if not by:
                vy_pct += 1.0
            if bx and by:
                break
        obj.vx *= vx_pct / Float32(nsub)
        obj.vy *= vy_pct / Float32(nsub)

    def _out_of_bounds(self, e: Entity) -> Bool:
        return (
            e.x + e.rx < 0.0
            or e.y + e.ry < 0.0
            or e.x - e.rx > Float32(self.world_dim)
            or e.y - e.ry > Float32(self.world_dim)
        )

    def step(mut self, action: Int) -> Float32:
        self.cur_time += 1
        self.reward = 0.0
        self.done = False
        self.level_complete = False

        # --- BasicAbstractGame::game_step base ---
        self.step_rand_int = self.rand_gen.randint(0, 1000000)
        var move = action % 9
        if action >= 9:
            move = 4
        self.action_vx = Float32(move // 3 - 1)
        self.action_vy = Float32(move % 3 - 1)

        # update_agent_velocity (base: momentum blend + 0.9 decay).
        self.agent.vx = (
            (1.0 - MIXRATE) * self.agent.vx + MIXRATE * MAXSPEED * self.action_vx
        )
        self.agent.vy = (
            (1.0 - MIXRATE) * self.agent.vy + MIXRATE * MAXSPEED * self.action_vy
        )
        self.agent.vx *= 0.9
        self.agent.vy *= 0.9

        # Only the agent is smart_step (keys/doors/exit are static).
        var a = self.agent.copy()
        self._basic_step_object(a)
        self.agent = a^

        # face_direction (cosmetic; used by the renderer).
        if self.action_vx != 0.0 or self.action_vy != 0.0:
            self.agent.rotation = -atan2(self.action_vy, self.action_vx)

        # Agent-collision pass: exit / key pickup / door open.
        for i in range(len(self.entities) - 1, -1, -1):
            var etype = self.entities[i].type
            var tx = self.entities[i].rx + self.agent.rx
            var ty = self.entities[i].ry + self.agent.ry
            if (
                abs(self.entities[i].x - self.agent.x) < tx
                and abs(self.entities[i].y - self.agent.y) < ty
            ):
                if etype == EXIT:
                    self.done = True
                    self.reward += COMPLETION_BONUS
                    self.level_complete = True
                elif etype == KEY:
                    self.entities[i].will_erase = True
                    self.has_keys[self.entities[i].image_theme] = True
                elif etype == LOCKED_DOOR:
                    if self.has_keys[self.entities[i].image_theme]:
                        self.entities[i].will_erase = True
        for i in range(len(self.entities) - 1, -1, -1):
            if self.entities[i].will_erase:
                _ = self.entities.pop(i)

        if self._out_of_bounds(self.agent):
            self.done = True

        self.episode_reward += self.reward
        return self.reward
