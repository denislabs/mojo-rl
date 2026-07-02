"""Caveflyer game — cave-flying shooter (port of `games/caveflyer.cpp`).

Rotate-and-thrust ship (Asteroids-style) navigating a procedurally carved cave to
a green UFO goal (+10); shoot red target ships (+3 each) with the projectile
substrate, dodge meteors + enemy ships + cave walls (any touch = death). ESTABLISHES
the roomgen substrate (cellular-automata cave gen → largest room → BFS path →
dilation), shared with jumper. Adds rotational dynamics (vrot/MIXRATEROT/MAXVTHETA)
on top of the projectile + entity engine.

`game_reset`/`game_step` replay the exact RNG order. Level-exact + visual-approx.
See `docs/PROCGEN_CAVEFLYER_SCOPE.md`. P0+P1 = reset+step parity; render/env in P2.
"""

from std.math import floor, ceil, sqrt, cos, sin, atan2

from ..core.entity import Entity
from ..core.randgen import RandGen
from ..core.grid import Grid
from ..core.roomgen import (
    roomgen_update, roomgen_find_best_room, roomgen_expand_room, roomgen_find_path
)
from ..core.assets import Sprite, load_sprite, load_sprites
from ..core.rasterizer import Canvas, RES, downscale
from ..core.object_ids import PLAYER, SPACE, WALL_OBJ, EXPLOSION, EXPLOSION5

comptime GOAL = 1
comptime OBSTACLE = 2
comptime TARGET = 3
comptime PLAYER_BULLET = 4
comptime ENEMY = 5
comptime CAVEWALL = 8
comptime EXHAUST = 9
comptime MARKER = 1003

comptime GOAL_REWARD: Float32 = 10.0
comptime TARGET_REWARD: Float32 = 3.0
comptime PI: Float32 = 3.14159265358979
comptime MAXVTHETA: Float32 = 15.0 * PI / 180.0
comptime MIXRATEROT: Float32 = 0.5
comptime A_R: Float32 = 0.4
comptime BG_COUNT = 13  # space_backgrounds (value only; not in RNG-stream parity)
comptime OBS_SS = 4
comptime RENDER_EPS: Float32 = 0.0

comptime DIST_EASY = 0
comptime DIST_HARD = 1


def _fsign(x: Float32) -> Float32:
    if x > 0.0:
        return 1.0
    elif x == 0.0:
        return 0.0
    return -1.0


struct CaveflyerAssets(Movable):
    """Read-only sprite set (shared via ArcPointer; passed into render())."""

    var ship: Sprite
    var goal: Sprite  # ufoGreen2
    var obstacle: Sprite  # meteorBrown_big1
    var target: Sprite  # ufoRed2
    var bullet: Sprite  # laserBlue02
    var enemy: Sprite  # enemyShipBlue4
    var cavewall: Sprite  # groundA
    var exhaust: Sprite  # towerDefense_tile295
    var explosion: Sprite
    var backgrounds: List[Sprite]  # space_backgrounds (13)

    def __init__(out self, asset_root: String) raises:
        self.ship = load_sprite(asset_root, "misc_assets/playerShip1_red.png")
        self.goal = load_sprite(asset_root, "misc_assets/ufoGreen2.png")
        self.obstacle = load_sprite(asset_root, "misc_assets/meteorBrown_big1.png")
        self.target = load_sprite(asset_root, "misc_assets/ufoRed2.png")
        self.bullet = load_sprite(asset_root, "misc_assets/laserBlue02.png")
        self.enemy = load_sprite(asset_root, "misc_assets/enemyShipBlue4.png")
        self.cavewall = load_sprite(asset_root, "misc_assets/groundA.png")
        self.exhaust = load_sprite(asset_root, "misc_assets/towerDefense_tile295.png")
        self.explosion = load_sprite(asset_root, "misc_assets/explosion1.png")
        var names: List[String] = [
            "deep_space_01", "spacegen_01", "milky_way_01", "ez_space_lite_01",
            "meyespace_v1_01", "eye_nebula_01", "deep_sky_01", "space_nebula_01",
            "Background-1", "Background-2", "Background-3", "Background-4",
            "parallax-space-backgound",
        ]
        var sbp = List[String]()
        for i in range(len(names)):
            sbp.append("space_backgrounds/" + names[i] + ".png")
        self.backgrounds = load_sprites(asset_root, sbp)


struct CaveflyerGame(Copyable, Movable):
    var rand_gen: RandGen
    var dist_mode: Int
    var grid: Grid
    var w: Int
    var h: Int
    var oob: Int
    var agent: Entity
    var entities: List[Entity]
    var mixrate: Float32
    var maxspeed: Float32
    var action_vx: Float32
    var action_vy: Float32
    var action_vrot: Float32
    var special_action: Int
    var step_rand_int: Int
    var bg_pct_x: Float32
    var background_index: Int
    var reward: Float32
    var episode_reward: Float32
    var done: Bool
    var level_complete: Bool
    var cur_time: Int
    var agent_cell: Int
    var goal_cell: Int

    def __init__(out self, dist_mode: Int = DIST_EASY):
        self.rand_gen = RandGen()
        self.dist_mode = dist_mode
        self.grid = Grid()
        var wd = 30 if dist_mode == DIST_EASY else 40
        self.grid.resize(wd, wd)
        self.w = wd
        self.h = wd
        self.oob = CAVEWALL
        self.agent = Entity(0.5, 0.5, 0.0, 0.0, 0.5, 0.5, PLAYER)
        self.entities = List[Entity]()
        self.mixrate = 0.9
        self.maxspeed = 0.5
        self.action_vx = 0.0
        self.action_vy = 0.0
        self.action_vrot = 0.0
        self.special_action = 0
        self.step_rand_int = 0
        self.bg_pct_x = 0.0
        self.background_index = 0
        self.reward = 0.0
        self.episode_reward = 0.0
        self.done = False
        self.level_complete = False
        self.cur_time = 0
        self.agent_cell = 0
        self.goal_cell = 0

    # --- grid helpers ---
    def _obj_ff(self, fi: Float32, fj: Float32) -> Int:
        if fi < 0.0 or fj < 0.0:
            return self.oob
        var x = Int(floor(fi))
        var y = Int(floor(fj))
        if x < 0 or x >= self.w or y < 0 or y >= self.h:
            return self.oob
        return self.grid.data[y * self.w + x]

    @staticmethod
    def _is_blocked(src_type: Int, target: Int, oob: Int) -> Bool:
        if target == WALL_OBJ or target == oob:
            return True
        if src_type == PLAYER and target == CAVEWALL:
            return True
        return False

    @staticmethod
    def _will_reflect(src_type: Int, target: Int, oob: Int) -> Bool:
        return src_type == ENEMY and (target == CAVEWALL or target == oob)

    @staticmethod
    def _has_collision(
        ax: Float32, ay: Float32, arx: Float32, ary: Float32,
        bx: Float32, by: Float32, brx: Float32, bry: Float32, m: Float32,
    ) -> Bool:
        return abs(ax - bx) < (arx + brx + m) and abs(ay - by) < (ary + bry + m)

    def _out_of_bounds(self, e: Entity) -> Bool:
        return (
            e.x + e.rx < 0.0 or e.y + e.ry < 0.0
            or e.x - e.rx > Float32(self.w) or e.y - e.ry > Float32(self.h)
        )

    # --- physics (grid-only sub_step; no entity blocks/reflects the agent/enemies) ---
    def _sub_step(self, mut obj: Entity, vx: Float32, vy: Float32) -> Bool:
        if obj.will_erase:
            return False
        var nx = obj.x + vx
        var ny = obj.y + vy
        var margin: Float32 = 0.98
        var is_h = vx != 0.0
        var block = False
        var reflect = False
        for i in range(2):
            for j in range(2):
                var t = self._obj_ff(
                    nx + obj.rx * margin * Float32(2 * i - 1),
                    ny + obj.ry * margin * Float32(2 * j - 1),
                )
                if self._is_blocked(obj.type, t, self.oob):
                    block = True
                if self._will_reflect(obj.type, t, self.oob):
                    reflect = True
        if reflect:
            if is_h:
                var d: Float32
                if vx < 0.0:
                    d = ceil(nx - obj.rx) - (nx - obj.rx)
                else:
                    d = floor(nx + obj.rx) - (nx + obj.rx)
                obj.vx = -obj.vx
                nx = nx + 2 * d
            else:
                var d: Float32
                if vy < 0.0:
                    d = ceil(ny - obj.ry) - (ny - obj.ry)
                else:
                    d = floor(ny + obj.ry) - (ny + obj.ry)
                obj.vy = -obj.vy
                ny = ny + 2 * d
        elif block:
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
        var sxf: Bool
        if cmp == 0.0:
            sxf = self.step_rand_int % 2 == 0
        else:
            sxf = cmp > 0.0
        if obj.type == PLAYER:
            if self.action_vx != 0.0:
                sxf = True
            if self.action_vy != 0.0:
                sxf = False
        var vx_pct: Float32 = 0.0
        var vy_pct: Float32 = 0.0
        for _ in range(nsub):
            var bx: Bool
            var by: Bool
            if sxf:
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

    def _set_action_xy(mut self, mv: Int):
        var accel = Float32(mv % 3 - 1)
        if accel < 0.0:
            accel *= 0.33
        var theta = -1.0 * self.agent.rotation + PI / 2
        if accel > 0.0:
            var ex = Entity(
                self.agent.x - self.agent.rx * cos(theta),
                self.agent.y - self.agent.ry * sin(theta),
                0.0, 0.0, 0.5 * self.agent.rx, 0.5 * self.agent.rx, EXHAUST,
            )
            ex.expire_time = 4
            ex.rotation = -1.0 * theta - PI / 2
            ex.grow_rate = 1.25
            ex.alpha_decay = 0.8
            self.entities.append(ex^)
        self.action_vy = accel * sin(theta)
        self.action_vx = accel * cos(theta)
        self.action_vrot = Float32(mv // 3 - 1)

    def _update_agent_velocity(mut self):
        var a = self.agent.copy()
        a.vx += self.mixrate * self.maxspeed * self.action_vx * 0.2
        a.vy += self.mixrate * self.maxspeed * self.action_vy * 0.2
        a.vx *= 0.9
        a.vy *= 0.9
        self.agent = a^

    def _handle_agent_collision(mut self, t: Int):
        if t == GOAL:
            self.reward += GOAL_REWARD
            self.level_complete = True
            self.done = True
        elif t == OBSTACLE or t == ENEMY or t == TARGET:
            self.done = True

    # --- level gen ---
    def _spawn_at(mut self, idx: Int, r: Float32, type: Int):
        var x = Float32(idx % self.w) + 0.5
        var y = Float32(idx // self.w) + 0.5
        self.entities.append(Entity(x, y, 0.0, 0.0, r, r, type))

    def reset(mut self, level_seed: Int):
        self.rand_gen.seed(level_seed)
        self.entities = List[Entity]()
        self.reward = 0.0
        self.episode_reward = 0.0
        self.done = False
        self.level_complete = False
        self.cur_time = 0
        var gsz = self.w * self.h

        self.bg_pct_x = self.rand_gen.rand01()
        self.background_index = self.rand_gen.randn(BG_COUNT)
        for i in range(gsz):
            self.grid.data[i] = SPACE  # base game_reset

        self.oob = WALL_OBJ
        for i in range(gsz):
            self.grid.data[i] = WALL_OBJ if self.rand_gen.rand01() < 0.5 else SPACE
        for _ in range(4):
            roomgen_update(self.grid, self.oob)

        var best = roomgen_find_best_room(self.grid)
        for i in range(gsz):
            self.grid.data[i] = WALL_OBJ
        var free_cells = List[Int]()
        for i in range(gsz):
            if best[i]:
                self.grid.data[i] = SPACE
                free_cells.append(i)

        var sel = self.rand_gen.simple_choose(len(free_cells), 2)
        self.agent_cell = free_cells[sel[0]]
        self.goal_cell = free_cells[sel[1]]
        self.agent = Entity(
            Float32(self.agent_cell % self.w) + 0.5,
            Float32(self.agent_cell // self.w) + 0.5,
            0.0, 0.0, 0.5, 0.5, PLAYER,
        )
        self.agent.smart_step = True
        self.agent.render_z = 1

        var goal = Entity(
            Float32(self.goal_cell % self.w) + 0.5,
            Float32(self.goal_cell // self.w) + 0.5,
            0.0, 0.0, 0.5, 0.5, GOAL,
        )
        goal.collides_with_entities = True
        self.entities.append(goal^)

        var goal_path = roomgen_find_path(self.grid, self.agent_cell, self.goal_cell)

        var should_prune = self.dist_mode != 2  # not MemoryMode
        if should_prune:
            var wide = List[Bool]()
            wide.resize(gsz, False)
            for i in range(len(goal_path)):
                wide[goal_path[i]] = True
            roomgen_expand_room(self.grid, wide, 4)
            for i in range(gsz):
                self.grid.data[i] = WALL_OBJ
            for i in range(gsz):
                if wide[i]:
                    self.grid.data[i] = SPACE

        for _ in range(4):
            roomgen_update(self.grid, self.oob)
            for i in range(len(goal_path)):
                self.grid.data[goal_path[i]] = SPACE

        for i in range(len(goal_path)):
            self.grid.data[goal_path[i]] = MARKER

        free_cells = List[Int]()
        for i in range(gsz):
            if self.grid.data[i] == SPACE:
                free_cells.append(i)
            elif self.grid.data[i] == WALL_OBJ:
                self.grid.data[i] = CAVEWALL

        var chunk_size = len(free_cells) // 80
        var num_objs = 3 * chunk_size
        var oi = self.rand_gen.simple_choose(len(free_cells), num_objs)
        for i in range(num_objs):
            var val = free_cells[oi[i]]
            var x = Float32(val % self.w) + 0.5
            var y = Float32(val // self.w) + 0.5
            if i < chunk_size:
                var e = Entity(x, y, 0.0, 0.0, 0.5, 0.5, OBSTACLE)
                e.collides_with_entities = True
                self.entities.append(e^)
            elif i < 2 * chunk_size:
                var e = Entity(x, y, 0.0, 0.0, 0.5, 0.5, TARGET)
                e.health = 5.0
                e.collides_with_entities = True
                self.entities.append(e^)
            else:
                var vel = (0.1 * self.rand_gen.rand01() + 0.1) * Float32(
                    self.rand_gen.randn(2) * 2 - 1
                )
                var e = Entity(x, y, 0.0, 0.0, 0.5, 0.5, ENEMY)
                if self.rand_gen.rand01() < 0.5:
                    e.vx = vel
                else:
                    e.vy = vel
                e.smart_step = True
                e.collides_with_entities = True
                self.entities.append(e^)

        for i in range(gsz):
            if self.grid.data[i] == MARKER:
                self.grid.data[i] = SPACE
        self.oob = CAVEWALL

    def step(mut self, action: Int) -> Float32:
        self.cur_time += 1
        self.step_rand_int = self.rand_gen.randint(0, 1000000)
        var mv = action % 9
        self.special_action = 0
        if action >= 9:
            self.special_action = action - 8
            mv = 4
        self.action_vrot = 0.0
        self.action_vx = 0.0
        self.action_vy = 0.0
        self.reward = 0.0
        self.done = False
        self.level_complete = False

        self._set_action_xy(mv)
        self._update_agent_velocity()
        var a = self.agent.copy()
        a.vrot = MIXRATEROT * a.vrot
        a.vrot += MIXRATEROT * MAXVTHETA * self.action_vrot
        self.agent = a^

        # step_entities: entities (reverse) then the agent (idx 0, stepped last).
        for i in range(len(self.entities) - 1, -1, -1):
            if self.entities[i].smart_step:
                var e = self.entities[i].copy()
                self._basic_step_object(e)
                self.entities[i] = e^
            var e2 = self.entities[i].copy()
            e2.step()
            self.entities[i] = e2^
        var ag = self.agent.copy()
        self._basic_step_object(ag)
        ag.step()
        self.agent = ag^

        # collisions: agent-death/win + collides_with_entities → handle_collision.
        var n = len(self.entities)
        for i in range(n - 1, -1, -1):
            if self.entities[i].type != PLAYER and self._has_collision(
                self.entities[i].x, self.entities[i].y,
                self.entities[i].rx, self.entities[i].ry,
                self.agent.x, self.agent.y, self.agent.rx, self.agent.ry,
                self.entities[i].collision_margin,
            ):
                self._handle_agent_collision(self.entities[i].type)
            if self.entities[i].collides_with_entities:
                for j in range(n - 1, -1, -1):
                    if i == j:
                        continue
                    if (
                        not self.entities[i].will_erase
                        and not self.entities[j].will_erase
                        and self._has_collision(
                            self.entities[i].x, self.entities[i].y,
                            self.entities[i].rx, self.entities[i].ry,
                            self.entities[j].x, self.entities[j].y,
                            self.entities[j].rx, self.entities[j].ry,
                            self.entities[i].collision_margin,
                        )
                    ):
                        self._handle_collision(i, j)

        # erase (bullets/explosions/exhaust/enemies flagged or off-screen).
        for i in range(len(self.entities) - 1, -1, -1):
            if self.entities[i].will_erase or (
                self.entities[i].auto_erase and self._out_of_bounds(self.entities[i])
            ):
                _ = self.entities.pop(i)
        if self._out_of_bounds(self.agent):
            self.done = True

        # caveflyer tail: fire bullet + enemy facing + bullet-vs-cavewall.
        if self.special_action == 1:
            var theta = -1.0 * self.agent.rotation + PI / 2
            var b = Entity(
                self.agent.x, self.agent.y, cos(theta), sin(theta), 0.1, 0.25, PLAYER_BULLET
            )
            b.expire_time = 10
            b.rotation = self.agent.rotation
            self.entities.append(b^)
        for i in range(len(self.entities) - 1, -1, -1):
            if self.entities[i].type == ENEMY:
                var evx = self.entities[i].vx
                var evy = self.entities[i].vy
                if evx != 0.0 or evy != 0.0:
                    self.entities[i].rotation = -1.0 * atan2(evy, evx) + (-1.0 * PI / 2)
            if self.entities[i].type != PLAYER_BULLET:
                continue
            var found_wall = False
            for a2 in range(2):
                for b2 in range(2):
                    var t2 = self._obj_ff(
                        self.entities[i].x + self.entities[i].rx * Float32(2 * a2 - 1),
                        self.entities[i].y + self.entities[i].ry * Float32(2 * b2 - 1),
                    )
                    if t2 == CAVEWALL:
                        found_wall = True
            if found_wall:
                self.entities[i].will_erase = True
                var ex = Entity(
                    self.entities[i].x, self.entities[i].y, 0.0, 0.0,
                    0.5 * self.entities[i].rx, 0.5 * self.entities[i].rx, EXPLOSION,
                )
                self.entities.append(ex^)
        for i in range(len(self.entities) - 1, -1, -1):
            if self.entities[i].will_erase or (
                self.entities[i].auto_erase and self._out_of_bounds(self.entities[i])
            ):
                _ = self.entities.pop(i)

        if self.done:
            pass
        self.episode_reward += self.reward
        return self.reward

    def _handle_collision(mut self, i: Int, j: Int):
        # entities[i] is a collides_with_entities src; act only if entities[j] is a bullet.
        if self.entities[j].type != PLAYER_BULLET:
            return
        var st = self.entities[i].type
        var erase = False
        if st == TARGET:
            self.entities[i].health -= 1.0
            erase = True
            if self.entities[i].health <= 0.0 and not self.entities[i].will_erase:
                var sx = self.entities[i].x
                var sy = self.entities[i].y
                var sr = self.entities[i].rx
                self.entities[i].will_erase = True
                self.reward += TARGET_REWARD
                self.entities.append(Entity(sx, sy, 0.0, 0.0, 0.5 * sr, 0.5 * sr, EXPLOSION))
        elif st == OBSTACLE or st == ENEMY or st == GOAL:
            erase = True
        if erase and not self.entities[j].will_erase:
            var tx = self.entities[j].x
            var ty = self.entities[j].y
            var tr = self.entities[j].rx
            var svx = self.entities[i].vx
            var svy = self.entities[i].vy
            self.entities[j].will_erase = True
            var ex = Entity(tx, ty, svx, svy, 0.5 * tr, 0.5 * tr, EXPLOSION)
            self.entities.append(ex^)
