"""Fruitbot game — vertical scrolling collect-and-shoot (port of `fruitbot.cpp`).

The robot auto-drifts UP a tall field, dodging wall gaps; collect fruit (GOOD +1),
avoid food (BAD −4), shoot locks to open doors, reach the presents row at the top
(+10 win). Crashing into a wall/closed door ends the run. Reuses the projectile
substrate. `game_reset`/`game_step` replay the exact RNG order (partition, walls,
object spawns with collision retries, theme+fit). Level-exact + visual-approx.
See `docs/PROCGEN_FRUITBOT_SCOPE.md`. P0+P1 = reset+step parity; render/env in P2.
"""

from std.math import floor, ceil, sqrt
from std.memory import ArcPointer

from ..core.entity import Entity
from ..core.randgen import RandGen
from ..core.assets import Sprite, load_sprite, load_sprites, load_topdown_backgrounds
from ..core.rasterizer import Canvas, RES, downscale
from ..core.object_ids import PLAYER, WALL_OBJ

comptime BARRIER = 1
comptime OOB_WALL = 2
comptime PLAYER_BULLET = 3
comptime BAD_OBJ = 4
comptime GOOD_OBJ = 7
comptime LOCKED_DOOR = 10
comptime LOCK = 11
comptime PRESENT = 12

comptime KEY_DURATION = 8
comptime DOOR_ASPECT: Float32 = 3.25
comptime COMPLETION_BONUS: Float32 = 10.0
comptime POSITIVE_REWARD: Float32 = 1.0
comptime PENALTY: Float32 = -4.0
comptime A_R: Float32 = 0.4
comptime BG_COUNT = 9  # topdown_backgrounds
comptime MIXRATE: Float32 = 0.5
comptime MAXSPEED: Float32 = 0.85
comptime BULLET_VSCALE: Float32 = 0.5
comptime PI: Float32 = 3.14159265358979
comptime HEIGHT = 60
comptime RENDER_EPS: Float32 = 0.02
comptime OBS_SS = 4

comptime DIST_EASY = 0
comptime DIST_HARD = 1


def _good_aspect(t: Int) -> Float32:
    var w: List[Float32] = [27.0, 24.0, 30.0, 20.0, 23.0, 22.0]
    var h: List[Float32] = [28.0, 30.0, 22.0, 31.0, 23.0, 26.0]
    return w[t] / h[t]


def _bad_aspect(t: Int) -> Float32:
    var w: List[Float32] = [14.0, 28.0, 19.0, 21.0, 22.0, 22.0]
    var h: List[Float32] = [25.0, 13.0, 19.0, 20.0, 24.0, 22.0]
    return w[t] / h[t]


struct FruitbotGame(Copyable, Movable):
    var rand_gen: RandGen
    var w: Int
    var h: Int
    var dist_mode: Int
    var agent: Entity
    var entities: List[Entity]
    var last_fire: Int
    var action_vx: Float32
    var action_vy: Float32
    var special_action: Int
    var step_rand_int: Int
    var bg_pct_x: Float32
    var background_index: Int
    var reward: Float32
    var episode_reward: Float32
    var done: Bool
    var level_complete: Bool
    var cur_time: Int

    def __init__(out self, dist_mode: Int = DIST_EASY):
        self.rand_gen = RandGen()
        self.dist_mode = dist_mode
        self.w = 10 if dist_mode == DIST_EASY else 20
        self.h = HEIGHT
        self.agent = Entity.make(0.5, 0.5, A_R, PLAYER)
        self.entities = List[Entity]()
        self.last_fire = 0
        self.action_vx = 0.0
        self.action_vy = 0.0
        self.special_action = 0
        self.step_rand_int = 0
        self.bg_pct_x = 0.0
        self.background_index = 0
        self.reward = 0.0
        self.episode_reward = 0.0
        self.done = False
        self.level_complete = False
        self.cur_time = 0

    @staticmethod
    def _rand_pos(mut rg: RandGen, r: Float32, mn: Float32, mx: Float32) -> Float32:
        if mx - mn <= 2 * r:
            return (mx + mn) / 2
        return (mx - mn - 2 * r) * rg.rand01() + r + mn

    def _coll(
        self, ax: Float32, ay: Float32, arx: Float32, ary: Float32,
        bx: Float32, by: Float32, brx: Float32, bry: Float32,
    ) -> Bool:
        return abs(ax - bx) < (arx + brx) and abs(ay - by) < (ary + bry)

    def _has_any_coll(self, ex: Float32, ey: Float32, erx: Float32, ery: Float32) -> Bool:
        if self._coll(ex, ey, erx, ery, self.agent.x, self.agent.y, self.agent.rx, self.agent.ry):
            return True
        for i in range(len(self.entities)):
            ref o = self.entities[i]
            if not o.avoids_collisions and self._coll(ex, ey, erx, ery, o.x, o.y, o.rx, o.ry):
                return True
        return False

    def _out_of_bounds(self, e: Entity) -> Bool:
        return (
            e.x + e.rx < 0.0 or e.y + e.ry < 0.0
            or e.x - e.rx > Float32(self.w) or e.y - e.ry > Float32(self.h)
        )

    def _blocked_at(self, fi: Float32, fj: Float32) -> Bool:
        var x = Int(floor(fi))
        var y = Int(floor(fj))
        return x < 0 or x >= self.w or y < 0 or y >= self.h

    def _sub_step(self, mut o: Entity, vx: Float32, vy: Float32) -> Bool:
        var nx = o.x + vx
        var ny = o.y + vy
        var margin: Float32 = 0.98
        var is_h = vx != 0.0
        var block = False
        for i in range(2):
            for j in range(2):
                if self._blocked_at(
                    nx + o.rx * margin * Float32(2 * i - 1),
                    ny + o.ry * margin * Float32(2 * j - 1),
                ):
                    block = True
        if block:
            if is_h:
                if vx > 0.0:
                    nx = floor(nx + o.rx) - o.rx
                else:
                    nx = ceil(nx - o.rx) + o.rx
            else:
                if vy > 0.0:
                    ny = floor(ny + o.ry) - o.ry
                else:
                    ny = ceil(ny - o.ry) + o.ry
        o.x = nx
        o.y = ny
        return block

    def _basic_step_object(self, mut o: Entity):
        var speed = sqrt(o.vx * o.vx + o.vy * o.vy)
        var nsub = Int(4.0 * speed)
        if nsub < 4:
            nsub = 4
        var pct = 1.0 / Float32(nsub)
        var cmp = abs(o.vx) - abs(o.vy)
        var sxf: Bool
        if cmp == 0.0:
            sxf = self.step_rand_int % 2 == 0
        else:
            sxf = cmp > 0.0
        if o.type == PLAYER:
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
                bx = self._sub_step(o, o.vx * pct, 0.0)
                by = self._sub_step(o, 0.0, o.vy * pct)
            else:
                by = self._sub_step(o, 0.0, o.vy * pct)
                bx = self._sub_step(o, o.vx * pct, 0.0)
            if not bx:
                vx_pct += 1.0
            if not by:
                vy_pct += 1.0
            if bx and by:
                break
        o.vx *= vx_pct / Float32(nsub)
        o.vy *= vy_pct / Float32(nsub)

    def _drift(mut self, i: Int):
        self.entities[i].x += self.entities[i].vx
        self.entities[i].y += self.entities[i].vy
        self.entities[i].life_time += 1
        if (
            self.entities[i].expire_time > 0
            and self.entities[i].life_time > self.entities[i].expire_time
        ):
            self.entities[i].will_erase = True

    def _add_walls(mut self, ry: Float32, use_door: Bool, min_pct: Float32):
        var rw = Float32(self.w)
        var wall_ry: Float32 = 0.3
        var lock_rx: Float32 = 0.25
        var lock_ry: Float32 = 0.45
        var pct = min_pct + 0.2 * self.rand_gen.rand01()
        if use_door:
            pct += 0.1
            var lpw = 2 * lock_rx / Float32(self.w)
            var dpw = (wall_ry * 2 * DOOR_ASPECT) / Float32(self.w)
            var nd = Int(ceil((pct - 2 * lpw) / dpw))
            pct = 2 * lpw + dpw * Float32(nd)
        var gapw = pct * rw
        var w1 = self.rand_gen.rand01() * (rw - gapw)
        var w2 = rw - w1 - gapw
        self.entities.append(Entity(w1 / 2, ry, 0.0, 0.0, w1 / 2, wall_ry, BARRIER))
        self.entities.append(Entity(rw - w2 / 2, ry, 0.0, 0.0, w2 / 2, wall_ry, BARRIER))
        if use_door:
            var ior = self.rand_gen.randn(2)
            var lock_x = w1 + lock_rx + Float32(ior) * (gapw - 2 * lock_rx)
            var door_x = w1 + gapw / 2 - Float32(ior * 2 - 1) * lock_rx
            self.entities.append(
                Entity(door_x, ry, 0.0, 0.0, gapw / 2 - lock_rx, wall_ry, LOCKED_DOOR)
            )
            self.entities.append(
                Entity(lock_x, ry - lock_ry + wall_ry, 0.0, 0.0, lock_rx, lock_ry, LOCK)
            )

    def _spawn_entities(mut self, num: Int, r: Float32, type: Int):
        for _ in range(num):
            var ex = FruitbotGame._rand_pos(self.rand_gen, r, 0.0, Float32(self.w))
            var ey = FruitbotGame._rand_pos(self.rand_gen, r, 0.0, Float32(self.h))
            var c = 0
            while self._has_any_coll(ex, ey, r, r) and c < 100:
                ex = FruitbotGame._rand_pos(self.rand_gen, r, 0.0, Float32(self.w))
                ey = FruitbotGame._rand_pos(self.rand_gen, r, 0.0, Float32(self.h))
                c += 1
            self.entities.append(Entity(ex, ey, 0.0, 0.0, r, r, type))

    def reset(mut self, level_seed: Int):
        self.rand_gen.seed(level_seed)
        self.reward = 0.0
        self.episode_reward = 0.0
        self.done = False
        self.level_complete = False
        self.cur_time = 0
        self.entities = List[Entity]()

        self.bg_pct_x = self.rand_gen.rand01()
        self.background_index = self.rand_gen.randn(BG_COUNT)
        var ax = self.rand_gen.rand01() * (Float32(self.w) - 2 * A_R) + A_R
        _ = self.rand_gen.rand01() * (Float32(self.h) - 2 * A_R) + A_R  # ay
        self.agent = Entity.make(ax, A_R, A_R, PLAYER)
        self.last_fire = 0

        var min_sep = 4
        var num_walls = 10
        var ogs = 6
        var buf_h = 4
        var door_prob: Float32 = 0.125
        var min_pct: Float32 = 0.1
        if self.dist_mode == DIST_EASY:
            num_walls = 5
            ogs = 2
            door_prob = 0.0
            min_pct = 0.2

        var partition = self.rand_gen.partition(
            self.h - min_sep * num_walls - buf_h, num_walls
        )
        var curr_h = 0
        for pi in range(len(partition)):
            var dy = min_sep + partition[pi]
            curr_h += dy
            # Short-circuit: rand01 only drawn when dy > 5.
            var use_door = False
            if dy > 5:
                use_door = self.rand_gen.rand01() < door_prob
            self._add_walls(Float32(curr_h), use_door, min_pct)

        self.agent.y = self.agent.ry

        var num_good = self.rand_gen.randn(10) + 10
        var num_bad = self.rand_gen.randn(10) + 10

        for i in range(self.w):
            var p = Entity(Float32(i) + 0.5, Float32(self.h) - 0.5, 0.0, 0.0, 0.5, 0.5, PRESENT)
            p.image_theme = self.rand_gen.randn(3)
            self.entities.append(p^)

        self._spawn_entities(num_good, 0.5, GOOD_OBJ)
        self._spawn_entities(num_bad, 0.5, BAD_OBJ)

        for i in range(len(self.entities)):
            var t = self.entities[i].type
            if t == GOOD_OBJ or t == BAD_OBJ:
                var theme = self.rand_gen.randn(ogs)
                self.entities[i].image_theme = theme
                var ar = _good_aspect(theme) if t == GOOD_OBJ else _bad_aspect(theme)
                if ar > 1.0:
                    self.entities[i].ry = self.entities[i].rx / ar
                else:
                    self.entities[i].rx = self.entities[i].ry * ar

        self.agent.rotation = -1 * PI / 2

    def step(mut self, action: Int) -> Float32:
        self.cur_time += 1
        self.reward = 0.0
        self.done = False
        self.level_complete = False
        self.step_rand_int = self.rand_gen.randint(0, 1000000)

        var move = action % 9
        self.special_action = 0
        if action >= 9:
            self.special_action = action - 8
            move = 4
        self.action_vx = Float32(move // 3 - 1)
        self.action_vy = 0.2  # set_action_xy override: always drift up

        self.agent.vx = (
            (1.0 - MIXRATE) * self.agent.vx + MIXRATE * MAXSPEED * self.action_vx
        )
        self.agent.vy = (
            (1.0 - MIXRATE) * self.agent.vy + MIXRATE * MAXSPEED * self.action_vy
        )
        self.agent.vx *= 0.9
        self.agent.vy *= 0.9
        var a = self.agent.copy()
        self._basic_step_object(a)
        self.agent = a^

        for i in range(len(self.entities)):
            self._drift(i)

        var n = len(self.entities)
        # agent-collision: collect / crash / present.
        for i in range(n - 1, -1, -1):
            var et = self.entities[i].type
            if self._coll(
                self.entities[i].x, self.entities[i].y, self.entities[i].rx,
                self.entities[i].ry, self.agent.x, self.agent.y, self.agent.rx, self.agent.ry
            ):
                if et == BARRIER or et == LOCKED_DOOR:
                    self.done = True
                elif et == BAD_OBJ:
                    self.reward += PENALTY
                    self.entities[i].will_erase = True
                elif et == GOOD_OBJ:
                    self.reward += POSITIVE_REWARD
                    self.entities[i].will_erase = True
                elif et == PRESENT:
                    self.reward += COMPLETION_BONUS
                    self.done = True
                    self.level_complete = True
        # collides_with_entities: bullet vs barrier/lock.
        for i in range(n - 1, -1, -1):
            if not self.entities[i].collides_with_entities:
                continue
            for j in range(n - 1, -1, -1):
                if i == j:
                    continue
                if self.entities[i].will_erase or self.entities[j].will_erase:
                    continue
                if self._coll(
                    self.entities[i].x, self.entities[i].y, self.entities[i].rx,
                    self.entities[i].ry, self.entities[j].x, self.entities[j].y,
                    self.entities[j].rx, self.entities[j].ry,
                ):
                    if self.entities[i].type == PLAYER_BULLET:
                        var tt = self.entities[j].type
                        if tt == BARRIER:
                            self.entities[i].will_erase = True
                        elif tt == LOCK:
                            self.entities[i].will_erase = True
                            self.entities[j].will_erase = True
                            var ly = self.entities[j].y
                            for k in range(len(self.entities)):
                                if (
                                    self.entities[k].type == LOCKED_DOOR
                                    and abs(self.entities[k].y - ly) < 1.0
                                ):
                                    self.entities[k].will_erase = True
                                    break

        for i in range(len(self.entities) - 1, -1, -1):
            if self.entities[i].will_erase or (
                self.entities[i].auto_erase and self._out_of_bounds(self.entities[i])
            ):
                _ = self.entities.pop(i)
        if self._out_of_bounds(self.agent):
            self.done = True

        # fruitbot tail: fire.
        if self.special_action == 1 and (self.cur_time - self.last_fire) >= KEY_DURATION:
            var b = Entity(
                self.agent.x, self.agent.y, 0.0, 1.0 * BULLET_VSCALE, 0.25, 0.25, PLAYER_BULLET
            )
            b.expire_time = KEY_DURATION
            b.collides_with_entities = True
            self.entities.append(b^)
            self.last_fire = self.cur_time

        self.episode_reward += self.reward
        return self.reward
