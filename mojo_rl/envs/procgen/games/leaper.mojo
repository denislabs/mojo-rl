"""Leaper game — frogger (port of `games/leaper.cpp`).

The frog hops up a grid through road lanes (moving cars kill on contact) and water
lanes (must ride drifting logs or it drowns) to reach the finish line at the top.
Reuses the substrate; the new pieces are the frog-hop movement model and a heavy
reset that simulates the lanes to steady-state.

`game_reset(level_seed)`/`game_step` replay the exact RNG order (lane layout +
per-step spawn gates + car theme draws). Level-exact + visual-approx.
See `docs/PROCGEN_LEAPER_SCOPE.md`. P0+P1 = reset+step parity; render/env in P2.
"""

from std.math import floor, ceil, sqrt, pi
from std.memory import ArcPointer

from ..core.entity import Entity
from ..core.randgen import RandGen
from ..core.assets import Sprite, load_sprite, load_sprites, load_topdown_backgrounds
from ..core.rasterizer import Canvas, RES, downscale
from ..core.object_ids import SPACE, WALL_OBJ, PLAYER, INVALID_OBJ

comptime LOG = 1
comptime ROAD = 2
comptime WATER = 3
comptime CAR = 4
comptime FINISH_LINE = 5
comptime MONSTER_RADIUS: Float32 = 0.25
comptime LOG_RADIUS: Float32 = 0.45
comptime NSTEP = 5
comptime MAX_SPEED: Float32 = 0.5  # 2/(NSTEP-1)
comptime VEL_DECAY: Float32 = 0.1  # MAX_SPEED/NSTEP
comptime GOAL_REWARD: Float32 = 10.0
comptime A_R: Float32 = 0.4
comptime BG_COUNT = 9  # topdown_backgrounds
comptime RENDER_EPS: Float32 = 0.02
comptime OBS_SS = 4
comptime PI: Float32 = 3.14159265358979

comptime DIST_EASY = 0
comptime DIST_HARD = 1
comptime DIST_EXTREME = 2


def leaper_world_dim(dist_mode: Int) -> Int:
    if dist_mode == DIST_EASY:
        return 9
    if dist_mode == DIST_HARD:
        return 15
    return 20  # ExtremeMode


def _fsign(x: Float32) -> Float32:
    if x > 0:
        return 1.0
    if x == 0:
        return 0.0
    return -1.0


def _decay(v: Float32) -> Float32:
    var s = _fsign(v)
    var r = abs(v) - VEL_DECAY
    if r < 0:
        r = 0.0
    return r * s


struct LeaperAssets(Movable):
    """Read-only sprite set (shared via ArcPointer; passed into render())."""

    var road: Sprite
    var water: Sprite
    var log: Sprite
    var finish: Sprite
    var cars: List[Sprite]  # 5 themes
    var frogs: List[Sprite]  # frog1/2/4/6/7 (animation frames)
    var backgrounds: List[Sprite]

    def __init__(out self, asset_root: String) raises:
        self.road = load_sprite(asset_root, "misc_assets/roadTile6b.png")
        self.water = load_sprite(asset_root, "misc_assets/terrainTile6.png")
        self.log = load_sprite(asset_root, "misc_assets/elementWood044.png")
        self.finish = load_sprite(asset_root, "misc_assets/finish2.png")
        var cp = List[String]()
        cp.append("misc_assets/car_yellow_5.png")
        cp.append("misc_assets/car_black_1.png")
        cp.append("misc_assets/car_blue_2.png")
        cp.append("misc_assets/car_green_3.png")
        cp.append("misc_assets/car_red_4.png")
        self.cars = load_sprites(asset_root, cp)
        var frp = List[String]()
        frp.append("misc_assets/frog1.png")
        frp.append("misc_assets/frog2.png")
        frp.append("misc_assets/frog4.png")
        frp.append("misc_assets/frog6.png")
        frp.append("misc_assets/frog7.png")
        self.frogs = load_sprites(asset_root, frp)
        self.backgrounds = load_topdown_backgrounds(asset_root)


struct LeaperGame(Copyable, Movable):
    var rand_gen: RandGen
    var w: Int
    var h: Int
    var world_dim: Int
    var dist_mode: Int
    var grid: List[Int]
    var agent: Entity
    var entities: List[Entity]  # cars, logs, finish
    var road_speeds: List[Float32]
    var water_speeds: List[Float32]
    var bottom_road_y: Int
    var bottom_water_y: Int
    var goal_y: Int
    var action_vx: Float32
    var action_vy: Float32
    var step_rand_int: Int
    var reward: Float32
    var bg_pct_x: Float32
    var background_index: Int
    var episode_reward: Float32
    var done: Bool
    var level_complete: Bool
    var cur_time: Int

    def __init__(out self, dist_mode: Int = DIST_EASY):
        self.rand_gen = RandGen()
        self.dist_mode = dist_mode
        self.world_dim = leaper_world_dim(dist_mode)
        self.w = self.world_dim
        self.h = self.world_dim
        self.grid = List[Int]()
        self.agent = Entity.make(0.5, 0.4, A_R, PLAYER)
        self.entities = List[Entity]()
        self.road_speeds = List[Float32]()
        self.water_speeds = List[Float32]()
        self.bottom_road_y = 0
        self.bottom_water_y = 0
        self.goal_y = 0
        self.action_vx = 0.0
        self.action_vy = 0.0
        self.step_rand_int = 0
        self.reward = 0.0
        self.bg_pct_x = 0.0
        self.background_index = 0
        self.episode_reward = 0.0
        self.done = False
        self.level_complete = False
        self.cur_time = 0

    # --- physics helpers (grid-only sub_step; oob=INVALID_OBJ blocks edges) ---
    def _obj(self, x: Int, y: Int) -> Int:
        if x < 0 or x >= self.w or y < 0 or y >= self.h:
            return INVALID_OBJ
        return self.grid[y * self.w + x]

    def _blocked_at(self, fi: Float32, fj: Float32) -> Bool:
        if fi < 0.0 or fj < 0.0:
            return True  # INVALID_OBJ blocks
        var t = self._obj(Int(floor(fi)), Int(floor(fj)))
        return t == WALL_OBJ or t == INVALID_OBJ

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

    def _out_of_bounds(self, e: Entity) -> Bool:
        return (
            e.x + e.rx < 0.0
            or e.y + e.ry < 0.0
            or e.x - e.rx > Float32(self.w)
            or e.y - e.ry > Float32(self.h)
        )

    def _collides(
        self, ax: Float32, ay: Float32, arx: Float32, ary: Float32,
        bx: Float32, by: Float32, brx: Float32, bry: Float32, margin: Float32,
    ) -> Bool:
        return (
            abs(ax - bx) < (arx + brx + margin)
            and abs(ay - by) < (ary + bry + margin)
        )

    def _any_collision(
        self, mx: Float32, my: Float32, mrx: Float32, mry: Float32
    ) -> Bool:
        if self._collides(
            mx, my, mrx, mry, self.agent.x, self.agent.y, self.agent.rx, self.agent.ry, 0.0
        ):
            return True
        for i in range(len(self.entities)):
            ref e = self.entities[i]
            if self._collides(mx, my, mrx, mry, e.x, e.y, e.rx, e.ry, 0.0):
                return True
        return False

    def _rand_sign(mut self) -> Float32:
        return 1.0 if self.rand_gen.rand01() < 0.5 else -1.0

    def _randrange(mut self, lo: Float32, hi: Float32) -> Float32:
        return self.rand_gen.rand01() * (hi - lo) + lo

    def _spawn_entities(mut self):
        # Cars.
        for lane in range(len(self.road_speeds)):
            var sp = self.road_speeds[lane]
            if self.rand_gen.rand01() < abs(sp) / 6.0:
                var x = -MONSTER_RADIUS if sp > 0 else Float32(self.w) + MONSTER_RADIUS
                var theme = self.rand_gen.randn(5)  # drawn even if not added
                var cy = Float32(self.bottom_road_y + lane) + 0.5
                if not self._any_collision(x, cy, 2 * MONSTER_RADIUS, MONSTER_RADIUS):
                    var m = Entity(x, cy, sp, 0.0, 2 * MONSTER_RADIUS, MONSTER_RADIUS, CAR)
                    m.image_theme = theme
                    if sp < 0:
                        m.rotation = PI
                    self.entities.append(m^)
        # Logs.
        for lane in range(len(self.water_speeds)):
            var sp = self.water_speeds[lane]
            if self.rand_gen.rand01() < abs(sp) / 2.0:
                var x = -LOG_RADIUS if sp > 0 else Float32(self.w) + LOG_RADIUS
                var ly = Float32(self.bottom_water_y + lane) + 0.5
                if not self._any_collision(x, ly, LOG_RADIUS, LOG_RADIUS):
                    self.entities.append(
                        Entity(x, ly, sp, 0.0, LOG_RADIUS, LOG_RADIUS, LOG)
                    )

    def _drift_entities(mut self):
        for i in range(len(self.entities)):
            self.entities[i].x += self.entities[i].vx

    def _fill_row(mut self, y: Int, type: Int):
        for x in range(self.w):
            self.grid[y * self.w + x] = type

    def reset(mut self, level_seed: Int):
        self.rand_gen.seed(level_seed)
        self.reward = 0.0
        self.episode_reward = 0.0
        self.done = False
        self.level_complete = False
        self.cur_time = 0
        self.entities = List[Entity]()
        self.road_speeds = List[Float32]()
        self.water_speeds = List[Float32]()
        self.grid = List[Int]()
        self.grid.resize(self.w * self.h, SPACE)

        var easy = self.dist_mode == DIST_EASY
        self.bg_pct_x = self.rand_gen.rand01()
        self.background_index = self.rand_gen.randn(BG_COUNT)
        var ax = self.rand_gen.rand01() * (Float32(self.w) - 2 * A_R) + A_R
        _ = self.rand_gen.rand01() * (Float32(self.h) - 2 * A_R) + A_R  # ay
        self.agent = Entity.make(ax, A_R, A_R, PLAYER)  # y = ry (bottom)

        var min_car: Float32 = 0.05
        var max_car: Float32 = 0.2
        var min_log: Float32 = 0.05
        var max_log: Float32 = 0.1
        if easy:
            min_car = 0.03
            max_car = 0.12
            min_log = 0.025
            max_log = 0.075
        elif self.dist_mode == DIST_EXTREME:
            min_car = 0.1
            max_car = 0.3
            min_log = 0.1
            max_log = 0.2

        self.bottom_road_y = (0 if easy else self.rand_gen.randn(2)) + 1
        var max_diff = 3 if easy else 4
        var difficulty = self.rand_gen.randn(max_diff + 1)
        var extra_lane = 0 if easy else self.rand_gen.randn(4)

        var num_road = difficulty + (1 if extra_lane == 2 else 0)
        for lane in range(num_road):
            self.road_speeds.append(self._rand_sign() * self._randrange(min_car, max_car))
            self._fill_row(self.bottom_road_y + lane, ROAD)

        self.bottom_water_y = (
            self.bottom_road_y + num_road + (0 if easy else self.rand_gen.randn(2)) + 1
        )
        var num_water = difficulty + (1 if extra_lane == 3 else 0)
        var curr_sign = self._rand_sign()
        for lane in range(num_water):
            self.water_speeds.append(curr_sign * self._randrange(min_log, max_log))
            curr_sign *= -1
            self._fill_row(self.bottom_water_y + lane, WATER)

        self.goal_y = self.bottom_water_y + num_water + 1

        # Simulate the lanes to steady-state (no erase during this loop).
        var minspeed = min_car if min_car < min_log else min_log
        var i = 0
        while Float32(i) < Float32(self.w) / minspeed:
            self._spawn_entities()
            self._drift_entities()
            i += 1

        # Finish line (one wide entity at the goal row).
        var fin = Entity(
            Float32(self.w) / 2.0, Float32(self.goal_y) - 0.5, 0.0, 0.0,
            Float32(self.w) / 2.0, 0.5, FINISH_LINE,
        )
        self.entities.append(fin^)

    def step(mut self, action: Int) -> Float32:
        self.cur_time += 1
        self.reward = 0.0
        self.done = False
        self.level_complete = False

        # Frog animation frame cycle.
        if self.agent.image_theme >= 1:
            self.agent.image_theme = (self.agent.image_theme + 1) % NSTEP

        # --- base game_step ---
        self.step_rand_int = self.rand_gen.randint(0, 1000000)
        var move = action % 9
        if action >= 9:
            move = 4
        self.action_vx = Float32(move // 3 - 1)
        self.action_vy = Float32(move % 3 - 1)

        # update_agent_velocity (leaper frog hop): start a hop only when settled.
        if self.agent.vx == 0.0 and self.agent.vy == 0.0:
            if self.action_vx != 0.0:
                self.agent.vx = MAX_SPEED * self.action_vx
                self.agent.image_theme = 1
                var rdir: Float32 = 1.0 if self.agent.vx > 0.0 else -1.0
                self.agent.rotation = rdir * PI / 2
            elif self.action_vy != 0.0:
                self.agent.vy = MAX_SPEED * self.action_vy
                self.agent.image_theme = 1
                self.agent.rotation = 0.0 if self.agent.vy > 0.0 else PI
        self.agent.vx = _decay(self.agent.vx)
        self.agent.vy = _decay(self.agent.vy)

        var a = self.agent.copy()
        self._basic_step_object(a)
        self.agent = a^

        self._drift_entities()  # cars/logs

        # collision pass: car kills, finish (if settled) wins.
        for i in range(len(self.entities) - 1, -1, -1):
            var et = self.entities[i].type
            if self._collides(
                self.entities[i].x, self.entities[i].y, self.entities[i].rx,
                self.entities[i].ry, self.agent.x, self.agent.y, self.agent.rx,
                self.agent.ry, 0.0,
            ):
                if et == CAR:
                    self.done = True
                elif et == FINISH_LINE and self.agent.vx == 0.0 and self.agent.vy == 0.0:
                    self.reward += GOAL_REWARD
                    self.done = True
                    self.level_complete = True

        # erase off-screen (auto_erase) or flagged.
        for i in range(len(self.entities) - 1, -1, -1):
            if self.entities[i].will_erase or (
                self.entities[i].auto_erase and self._out_of_bounds(self.entities[i])
            ):
                _ = self.entities.pop(i)

        if self._out_of_bounds(self.agent):
            self.done = True

        # --- leaper game_step tail ---
        self._spawn_entities()

        var on_log = False
        var log_vx: Float32 = 0.0
        var margin = -self.agent.rx
        for i in range(len(self.entities)):
            if self.entities[i].type == LOG and self._collides(
                self.agent.x, self.agent.y, self.agent.rx, self.agent.ry,
                self.entities[i].x, self.entities[i].y, self.entities[i].rx,
                self.entities[i].ry, margin,
            ):
                on_log = True
                log_vx = self.entities[i].vx

        if (
            self._obj(Int(self.agent.x), Int(self.agent.y)) == WATER
            and not on_log
            and self.agent.vx == 0.0
            and self.agent.vy == 0.0
        ):
            self.done = True

        if on_log:
            self.agent.x += log_vx

        if self._out_of_bounds(self.agent):
            self.done = True

        self.episode_reward += self.reward
        return self.reward

    # --- rendering (visual-approx; assets passed in) ---
    def render_obs(
        self, assets: LeaperAssets, res: Int = RES, ss: Int = OBS_SS
    ) -> List[UInt8]:
        return downscale(self.render(assets, res * ss), res * ss, res)

    def render(self, assets: LeaperAssets, out_res: Int = RES) -> List[UInt8]:
        var canvas = Canvas(out_res)
        canvas.fill(0, 0, 0)

        # Camera: center_agent=false → whole world (x_off == y_off == 0).
        var view_dim = Float32(self.w if self.w > self.h else self.h)
        var unit = Float32(out_res) / view_dim

        # Background (topdown, panned by bg_pct_x).
        ref bg = assets.backgrounds[self.background_index]
        var main_w = Float32(self.w) * unit
        var main_h = Float32(self.h) * unit
        var main_y = (view_dim - Float32(self.h)) * unit
        var bg_ar = Float32(bg.w) / Float32(bg.h)
        var world_ar = Float32(self.w) / Float32(self.h)
        var offset_x = self.bg_pct_x * (bg_ar - world_ar)
        canvas.blit(
            bg, main_w * (-offset_x), main_y, main_w * (bg_ar / world_ar), main_h
        )

        # Grid: road / water lane tiles.
        for x in range(self.w):
            for y in range(self.h):
                var t = self._obj(x, y)
                if t != ROAD and t != WATER:
                    continue
                var sx = (Float32(x) - RENDER_EPS) * unit
                var sy = (view_dim - Float32(y + 1) - RENDER_EPS) * unit
                var sz = (1.0 + 2 * RENDER_EPS) * unit
                if t == ROAD:
                    canvas.blit(assets.road, sx, sy, sz, sz)
                else:
                    canvas.blit(assets.water, sx, sy, sz, sz)

        # Entities: finish line, logs, cars.
        for k in range(len(self.entities)):
            ref e = self.entities[k]
            var ex = (e.x - e.rx) * unit
            var ey = (view_dim - (e.y + e.ry)) * unit
            var ew = 2 * e.rx * unit
            var eh = 2 * e.ry * unit
            if e.type == FINISH_LINE:
                canvas.blit(assets.finish, ex, ey, ew, eh)
            elif e.type == LOG:
                canvas.blit(assets.log, ex, ey, ew, eh)
            elif e.type == CAR:
                canvas.blit(assets.cars[e.image_theme], ex, ey, ew, eh, e.vx < 0.0)

        # Frog (animation frame by image_theme; reflected when hopping left).
        var ax = (self.agent.x - self.agent.rx) * unit
        var ay = (view_dim - (self.agent.y + self.agent.ry)) * unit
        canvas.blit(
            assets.frogs[self.agent.image_theme],
            ax,
            ay,
            2 * self.agent.rx * unit,
            2 * self.agent.ry * unit,
            self.agent.rotation < 0.0,
        )

        return canvas.px.copy()
