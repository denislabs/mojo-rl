"""Starpilot game — space shooter (port of `games/starpilot.cpp`).

The ship flies on the left of a 16×16 field; enemy waves fly in from the right (and
sometimes left); you shoot them (+1 each) and survive to `SHOOTER_WIN_TIME=500`, when
a finish line spawns → touch it to win (+10). Any lethal enemy/bullet ends the run.

This is the entry point for the **projectile substrate** (fire → bullet entities →
collide/destroy) shared by the shooter family. `game_reset` pre-schedules the whole
episode's enemy waves (`add_spawners`) with exact RNG order. Level-exact + visual-approx.
See `docs/PROCGEN_STARPILOT_SCOPE.md`. P0 = reset (add_spawners) parity; step/render in P1/P2.
"""

from std.math import floor, ceil, cos, sin, sqrt
from std.memory import ArcPointer

from ..core.entity import Entity
from ..core.randgen import RandGen
from ..core.object_ids import PLAYER, EXPLOSION

# Starpilot object ids (starpilot.cpp).
comptime BULLET_PLAYER = 1
comptime BULLET2 = 2
comptime BULLET3 = 3
comptime FLYER = 4
comptime METEOR = 5
comptime CLOUD = 6
comptime TURRET = 7
comptime FAST_FLYER = 8
comptime FINISH_LINE = 9

comptime V_SCALE: Float32 = 0.4  # 2/5
comptime MIXRATE: Float32 = 0.5
comptime ENEMY_REWARD: Float32 = 1.0
comptime COMPLETION_BONUS: Float32 = 10.0
comptime A_R: Float32 = 0.4
comptime BG_COUNT = 13  # space_backgrounds
comptime NUM_BASIC_OBJECTS = 9
comptime NUM_SHIP_THEMES = 7
comptime SHOOTER_WIN_TIME = 500
comptime PI: Float32 = 3.14159265358979
comptime WORLD = 16

comptime DIST_EASY = 0
comptime DIST_HARD = 1
comptime DIST_EXTREME = 2


def _theme_count(type: Int) -> Int:
    if type == FLYER or type == FAST_FLYER:
        return 7
    if type == METEOR:
        return 8
    if type == CLOUD:
        return 9
    if type == TURRET:
        return 2
    return 1


struct StarpilotGame(Copyable, Movable):
    var rand_gen: RandGen
    var w: Int
    var h: Int
    var dist_mode: Int
    var agent: Entity
    var entities: List[Entity]
    var spawners: List[Entity]  # pre-scheduled enemy waves (by spawn_time)
    # hp tables (indexed by type 0..8)
    var hp_vs: List[Float32]
    var hp_healths: List[Float32]
    var hp_bullet_r: List[Float32]
    var hp_object_r: List[Float32]
    var hp_prob: List[Float32]
    var total_prob: Float32
    var slow_v: Float32
    var spawn_right_threshold: Float32
    var max_group: Int
    var min_dt: Int
    var max_dt: Int
    var maxspeed: Float32
    var bg_pct_x: Float32
    var background_index: Int
    var special_action: Int
    var action_vx: Float32
    var action_vy: Float32
    var step_rand_int: Int
    var reward: Float32
    var episode_reward: Float32
    var done: Bool
    var level_complete: Bool
    var cur_time: Int

    def __init__(out self, dist_mode: Int = DIST_EASY):
        self.rand_gen = RandGen()
        self.dist_mode = dist_mode
        self.w = WORLD
        self.h = WORLD
        self.agent = Entity.make(0.5, 0.5, A_R, PLAYER)
        self.entities = List[Entity]()
        self.spawners = List[Entity]()
        self.hp_vs = List[Float32]()
        self.hp_healths = List[Float32]()
        self.hp_bullet_r = List[Float32]()
        self.hp_object_r = List[Float32]()
        self.hp_prob = List[Float32]()
        self.total_prob = 0.0
        self.slow_v = 0.0
        self.spawn_right_threshold = 0.0
        self.max_group = 0
        self.min_dt = 0
        self.max_dt = 0
        self.maxspeed = 0.0
        self.bg_pct_x = 0.0
        self.background_index = 0
        self.special_action = 0
        self.action_vx = 0.0
        self.action_vy = 0.0
        self.step_rand_int = 0
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

    def _init_hps(mut self):
        var scale: Float32 = 1.0
        self.hp_vs = List[Float32]()
        self.hp_healths = List[Float32]()
        self.hp_bullet_r = List[Float32]()
        self.hp_object_r = List[Float32]()
        self.hp_prob = List[Float32]()
        for _ in range(9):
            self.hp_vs.append(1.0)
            self.hp_healths.append(0.0)
            self.hp_bullet_r.append(0.0)
            self.hp_object_r.append(scale / 2)
            self.hp_prob.append(1.0)
        var default_bullet_r = scale / 2.5
        if self.dist_mode == DIST_EASY:
            self.hp_prob[METEOR] = 0.0
            self.hp_prob[CLOUD] = 0.0
            self.hp_prob[TURRET] = 0.0
            self.hp_prob[FAST_FLYER] = 0.0
            self.hp_vs[FLYER] = 0.75
            self.hp_vs[BULLET2] = 1.25
            self.hp_healths[TURRET] = 5.0
            self.hp_healths[FLYER] = 2.0
            self.hp_healths[FAST_FLYER] = 1.0
            self.maxspeed = 0.75
        elif self.dist_mode == DIST_HARD:
            self.hp_vs[BULLET2] = 2.0
            self.hp_healths[TURRET] = 5.0
            self.hp_healths[FLYER] = 2.0
            self.hp_healths[FAST_FLYER] = 1.0
            self.maxspeed = 0.75
        else:  # Extreme
            self.hp_vs[BULLET2] = 2.0
            self.hp_healths[TURRET] = 10.0
            self.hp_healths[FLYER] = 5.0
            self.hp_healths[FAST_FLYER] = 2.0
            self.maxspeed = 0.5
            default_bullet_r = scale / 5
        for i in range(9):
            self.hp_bullet_r[i] = default_bullet_r
        self.hp_healths[METEOR] = 500.0
        self.hp_vs[FAST_FLYER] = 1.5
        self.hp_vs[BULLET_PLAYER] = 2.0
        self.hp_vs[BULLET3] = 2.0
        self.hp_object_r[TURRET] = scale * 2
        self.hp_object_r[METEOR] = scale * 2
        self.hp_object_r[CLOUD] = scale * 2
        self.hp_prob[FLYER] = 3.0
        self.slow_v = 0.5
        self.max_group = 5
        self.min_dt = 10
        self.max_dt = 30
        self.spawn_right_threshold = 0.9
        self.hp_prob[BULLET_PLAYER] = 0.0
        self.hp_prob[BULLET2] = 0.0
        self.hp_prob[BULLET3] = 0.0
        self.total_prob = 0.0
        for i in range(2, 9):
            self.total_prob += self.hp_prob[i]

    def _add_spawners(mut self):
        var t = 1 + self.rand_gen.randint(self.min_dt, self.max_dt)
        var can_spawn_left = self.dist_mode != DIST_EASY
        while t <= SHOOTER_WIN_TIME:
            var group_size = 1
            var start_weight = self.rand_gen.rand01() * self.total_prob
            var curr = start_weight
            var type = 2
            while type < 9:
                curr -= self.hp_prob[type]
                if curr <= 0:
                    break
                type += 1
            if type >= 9:
                type = 8
            var r = self.hp_object_r[type]
            var flyer_theme = 0
            if type == FLYER or type == FAST_FLYER:
                group_size = self.rand_gen.randint(0, self.max_group) + 1
                flyer_theme = self.rand_gen.randn(NUM_SHIP_THEMES)
            var y_pos = StarpilotGame._rand_pos(
                self.rand_gen, r, 0.0, Float32(self.h)
            )
            for j in range(group_size):
                var spawn_time = t + j * 5
                var fire_time = self.rand_gen.randint(10, 100)
                var k = 2 * PI / 4
                var theta = (self.rand_gen.rand01() - 0.5) * k
                var v_scale = self.hp_vs[type]
                if self.rand_gen.randint(0, 2) == 1:
                    theta = 0.0
                var health = self.hp_healths[type]
                if type == METEOR or type == CLOUD:
                    theta = 0.0
                    v_scale = self.slow_v
                    fire_time = -1
                elif type == TURRET:
                    theta = 0.0
                    v_scale = self.slow_v
                    fire_time = self.rand_gen.randint(20, 30)
                v_scale *= V_SCALE
                var vx = -1 * cos(theta) * v_scale
                var vy = sin(theta) * v_scale
                var spawn_right = True
                if type == FLYER or type == FAST_FLYER:
                    if (
                        self.rand_gen.rand01() > self.spawn_right_threshold
                        and can_spawn_left
                    ):
                        spawn_right = False
                var x_pos: Float32
                if spawn_right:
                    x_pos = Float32(self.w) + r
                else:
                    x_pos = -r
                    vx *= -1
                var theme = 0
                var rotation: Float32 = 0.0
                if type == CLOUD:
                    theme = self.rand_gen.randn(_theme_count(CLOUD))
                elif type == METEOR:
                    theme = self.rand_gen.randn(_theme_count(METEOR))
                elif type == FLYER or type == FAST_FLYER:
                    theme = flyer_theme
                    var rdir: Float32 = -1.0 if vx > 0 else 1.0
                    rotation = rdir * PI / 2
                elif type == TURRET:
                    theme = self.rand_gen.randn(_theme_count(TURRET))
                var sp = Entity(x_pos, y_pos, vx, vy, r, r, type)
                sp.fire_time = fire_time
                sp.spawn_time = spawn_time
                sp.health = health
                sp.image_theme = theme
                sp.rotation = rotation
                self.spawners.append(sp^)
            t += self.rand_gen.randint(self.min_dt, self.max_dt)

    def reset(mut self, level_seed: Int):
        self.rand_gen.seed(level_seed)
        self.reward = 0.0
        self.episode_reward = 0.0
        self.done = False
        self.level_complete = False
        self.cur_time = 0
        self.entities = List[Entity]()
        self.spawners = List[Entity]()

        # BasicAbstractGame::game_reset base draws (bg = space_backgrounds).
        self.bg_pct_x = self.rand_gen.rand01()
        self.background_index = self.rand_gen.randn(BG_COUNT)
        var ax = self.rand_gen.rand01() * (Float32(self.w) - 2 * A_R) + A_R
        var ay = self.rand_gen.rand01() * (Float32(self.h) - 2 * A_R) + A_R
        self.agent = Entity.make(ax, ay, A_R, PLAYER)

        self._init_hps()
        self._add_spawners()
        self._sort_spawners_desc()  # by spawn_time DESC (stable) → pop back to activate

        self.agent.rotation = PI / 2
        self.agent.image_theme = self.rand_gen.randn(NUM_SHIP_THEMES)

    def _sort_spawners_desc(mut self):
        # Stable insertion sort by spawn_time descending (ties keep generation
        # order), matching the probe's std::stable_sort. n ≲ 100 → O(n²) is fine.
        for i in range(1, len(self.spawners)):
            var key = self.spawners[i].copy()
            var j = i - 1
            while j >= 0 and self.spawners[j].spawn_time < key.spawn_time:
                self.spawners[j + 1] = self.spawners[j].copy()
                j -= 1
            self.spawners[j + 1] = key^

    # --- step physics + projectile substrate ---
    def _is_lethal(self, t: Int) -> Bool:
        return (
            t == FLYER or t == FAST_FLYER or t == BULLET2 or t == BULLET3
            or t == TURRET or t == METEOR
        )

    def _is_destructible(self, t: Int) -> Bool:
        return t == FLYER or t == FAST_FLYER or t == TURRET or t == METEOR

    def _should_fire(self, mtype: Int, mfire: Int, mspawn: Int) -> Bool:
        if mfire <= 0:
            return False
        if mtype == TURRET:
            return (self.cur_time - mspawn) % mfire == 0
        return self.cur_time - mspawn == mfire

    def _collides(
        self, ax: Float32, ay: Float32, arx: Float32, ary: Float32,
        bx: Float32, by: Float32, brx: Float32, bry: Float32,
    ) -> Bool:
        return abs(ax - bx) < (arx + brx) and abs(ay - by) < (ary + bry)

    def _out_of_bounds(self, e: Entity) -> Bool:
        return (
            e.x + e.rx < 0.0
            or e.y + e.ry < 0.0
            or e.x - e.rx > Float32(self.w)
            or e.y - e.ry > Float32(self.h)
        )

    def _blocked_at(self, fi: Float32, fj: Float32) -> Bool:
        # world is all-SPACE; only the edges (oob) block.
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
        # non-smart Entity.step: advance by velocity, tick life, grow, expire.
        self.entities[i].x += self.entities[i].vx
        self.entities[i].y += self.entities[i].vy
        self.entities[i].life_time += 1
        if (
            self.entities[i].expire_time > 0
            and self.entities[i].life_time > self.entities[i].expire_time
        ):
            self.entities[i].will_erase = True
        self.entities[i].rx *= self.entities[i].grow_rate
        self.entities[i].ry *= self.entities[i].grow_rate

    def step(mut self, action: Int) -> Float32:
        self.cur_time += 1  # Game::step increments before game_step
        self.reward = 0.0
        self.done = False
        self.level_complete = False
        self.step_rand_int = self.rand_gen.randint(0, 1000000)  # unused

        var move = action % 9
        self.special_action = 0
        if action >= 9:
            self.special_action = action - 8
            move = 4
        self.action_vx = Float32(move // 3 - 1)
        self.action_vy = Float32(move % 3 - 1)

        # update_agent_velocity (base momentum + 0.9 decay).
        self.agent.vx = (
            (1.0 - MIXRATE) * self.agent.vx
            + MIXRATE * self.maxspeed * self.action_vx
        )
        self.agent.vy = (
            (1.0 - MIXRATE) * self.agent.vy
            + MIXRATE * self.maxspeed * self.action_vy
        )
        self.agent.vx *= 0.9
        self.agent.vy *= 0.9
        var a = self.agent.copy()
        self._basic_step_object(a)
        self.agent = a^

        for i in range(len(self.entities)):
            self._drift(i)

        # --- base collision phase (fixed count; appends go past it) ---
        var n = len(self.entities)
        # agent-collision: finish → win, lethal → done.
        for i in range(n - 1, -1, -1):
            var et = self.entities[i].type
            if self._collides(
                self.entities[i].x, self.entities[i].y, self.entities[i].rx,
                self.entities[i].ry, self.agent.x, self.agent.y, self.agent.rx,
                self.agent.ry,
            ):
                if et == FINISH_LINE:
                    self.done = True
                    self.reward += COMPLETION_BONUS
                    self.level_complete = True
                elif self._is_lethal(et):
                    self.done = True
        # collides_with_entities: player bullet damages destructible enemy.
        for i in range(n - 1, -1, -1):
            if not self.entities[i].collides_with_entities:
                continue
            for j in range(n - 1, -1, -1):
                if i == j:
                    continue
                if self.entities[i].will_erase or self.entities[j].will_erase:
                    continue
                if self._collides(
                    self.entities[i].x, self.entities[i].y, self.entities[i].rx,
                    self.entities[i].ry, self.entities[j].x, self.entities[j].y,
                    self.entities[j].rx, self.entities[j].ry,
                ):
                    var st = self.entities[i].type
                    var tt = self.entities[j].type
                    if st == BULLET_PLAYER and tt != CLOUD and self._is_destructible(tt):
                        var sx = self.entities[i].x
                        var sy = self.entities[i].y
                        var sr = self.entities[i].rx
                        var tvx = self.entities[j].vx
                        var tvy = self.entities[j].vy
                        self.entities[i].will_erase = True
                        self.entities[j].health -= 1.0
                        var ex = Entity(sx, sy, tvx, tvy, 0.5 * sr, 0.5 * sr, EXPLOSION)
                        ex.grow_rate = 1.4
                        ex.expire_time = 4
                        self.entities.append(ex^)
        # erase will_erase / out-of-bounds.
        for i in range(len(self.entities) - 1, -1, -1):
            if self.entities[i].will_erase or self._out_of_bounds(self.entities[i]):
                _ = self.entities.pop(i)
        if self._out_of_bounds(self.agent):
            self.done = True

        # --- StarPilotGame::game_step tail ---
        var is_firing = self.special_action != 0
        var m = len(self.entities)
        for i in range(m - 1, -1, -1):
            var mtype = self.entities[i].type
            var mfire = self.entities[i].fire_time
            var mspawn = self.entities[i].spawn_time
            var mhealth = self.entities[i].health
            var mx = self.entities[i].x
            var my = self.entities[i].y
            var mvx = self.entities[i].vx
            var mvy = self.entities[i].vy
            var mrx = self.entities[i].rx
            var mwill = self.entities[i].will_erase
            if self._should_fire(mtype, mfire, mspawn):
                var bt = BULLET3 if mtype == TURRET else BULLET2
                var br = self.hp_bullet_r[mtype]
                var bvx = self.agent.x - mx
                var bvy = self.agent.y - my
                var bs = self.hp_vs[bt] * V_SCALE / sqrt(bvx * bvx + bvy * bvy)
                bvx *= bs
                bvy *= bs
                self.entities.append(Entity(mx, my, bvx, bvy, br, br, bt))
            if mhealth <= 0.0 and self._is_destructible(mtype) and not mwill:
                var ex = Entity(mx, my, mvx, mvy, 0.5 * mrx, 0.5 * mrx, EXPLOSION)
                ex.grow_rate = 1.4
                ex.expire_time = 4
                self.entities.append(ex^)
                self.reward += ENEMY_REWARD
                self.entities[i].will_erase = True

        # activate spawners whose spawn_time == cur_time (pop from back).
        while (
            len(self.spawners) > 0
            and self.cur_time == self.spawners[len(self.spawners) - 1].spawn_time
        ):
            var last = self.spawners.pop()
            self.entities.append(last^)

        # player firing.
        if is_firing:
            var theta: Float32 = PI if self.special_action == 2 else 0.0
            var vs = self.hp_vs[BULLET_PLAYER] * V_SCALE
            var pbr = self.hp_bullet_r[PLAYER]
            var b = Entity(
                self.agent.x + self.agent.rx * cos(theta), self.agent.y,
                cos(theta) * vs, sin(theta) * vs, pbr, pbr, BULLET_PLAYER,
            )
            b.collides_with_entities = True
            self.entities.append(b^)

        # finish line at the win time (choose_random_theme draws randn(4)).
        if self.cur_time == SHOOTER_WIN_TIME:
            var ftheme = self.rand_gen.randn(4)
            var fin = Entity(
                Float32(self.w), Float32(self.h) / 2, -self.slow_v * V_SCALE, 0.0,
                2.0, Float32(self.h) / 2, FINISH_LINE,
            )
            fin.image_theme = ftheme
            fin.x = Float32(self.w) + fin.rx
            self.entities.append(fin^)

        self.episode_reward += self.reward
        return self.reward
