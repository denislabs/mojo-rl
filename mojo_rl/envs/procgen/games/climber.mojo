"""Climber game — vertical platformer (port of `games/climber.cpp`).

Climb up generated platforms collecting every yellow crystal (+1 each; grab them
all → +10 and level done), dodging patrolling enemies (touch = death). Reuses the
platformer substrate (gravity + jump + has_support + wall collision) minus crates;
adds coins + a coin_quota completion + patrol enemies. `game_reset`/`game_step`
replay the exact RNG order. Level-exact + visual-approx.
See `docs/PROCGEN_CLIMBER_SCOPE.md`. P0+P1 = reset+step parity; render in P2.
"""

from std.math import floor, ceil, sqrt

from ..core.entity import Entity
from ..core.randgen import RandGen
from ..core.assets import Sprite, load_sprite, load_sprites
from ..core.rasterizer import Canvas, RES, downscale
from ..core.object_ids import PLAYER, SPACE, WALL_OBJ

comptime COIN = 1
comptime ENEMY = 5
comptime ENEMY1 = 6
comptime ENEMY2 = 7
comptime PLAYER_JUMP = 9
comptime PLAYER_RIGHT1 = 12
comptime PLAYER_RIGHT2 = 13
comptime WALL_MID = 15
comptime WALL_TOP = 16
comptime ENEMY_BARRIER = 19

comptime OOB = WALL_MID
comptime COIN_REWARD: Float32 = 1.0
comptime COMPLETION_BONUS: Float32 = 10.0
comptime PATROL_RANGE: Float32 = 4.0
comptime H = 64
comptime NUM_WALL_THEMES = 4
comptime NUM_PLAYER_THEMES = 4
comptime BG_COUNT = 37  # platform_backgrounds (value only; not in RNG-stream parity)
comptime ENEMY_ASPECT: Float32 = 44.0 / 32.0  # enemySwimming_1 (match_aspect_ratio)
comptime OBS_SS = 4
comptime RENDER_EPS: Float32 = 0.0

comptime DIST_EASY = 0
comptime DIST_HARD = 1


struct ClimberAssets(Movable):
    """Read-only sprite set (shared via ArcPointer; passed into render())."""

    var players: List[Sprite]  # {color}_stand, 4 themes
    var players_jump: List[Sprite]  # {color}_walk4, 4 themes
    var players_walk: List[Sprite]  # {color}_walk1, 4 themes
    var wall_top: List[Sprite]  # 4 themes
    var wall_mid: List[Sprite]  # 4 themes
    var enemy1: Sprite
    var enemy2: Sprite
    var coin: Sprite
    var backgrounds: List[Sprite]  # platform_backgrounds (37)

    def __init__(out self, asset_root: String) raises:
        var colors: List[String] = ["Blue", "Green", "Grey", "Red"]
        var st = List[String]()
        var jp = List[String]()
        var wk = List[String]()
        for i in range(len(colors)):
            var c = colors[i]
            st.append("platformer/player" + c + "_stand.png")
            jp.append("platformer/player" + c + "_walk4.png")
            wk.append("platformer/player" + c + "_walk1.png")
        self.players = load_sprites(asset_root, st)
        self.players_jump = load_sprites(asset_root, jp)
        self.players_walk = load_sprites(asset_root, wk)
        var wt: List[String] = [
            "platformer/tileBlue_05.png", "platformer/tileGreen_05.png",
            "platformer/tileYellow_06.png", "platformer/tileBrown_06.png",
        ]
        var wm: List[String] = [
            "platformer/tileBlue_08.png", "platformer/tileGreen_08.png",
            "platformer/tileYellow_09.png", "platformer/tileBrown_09.png",
        ]
        self.wall_top = load_sprites(asset_root, wt)
        self.wall_mid = load_sprites(asset_root, wm)
        self.enemy1 = load_sprite(asset_root, "platformer/enemySwimming_1.png")
        self.enemy2 = load_sprite(asset_root, "platformer/enemySwimming_2.png")
        self.coin = load_sprite(asset_root, "platformer/yellowCrystal.png")
        var bnames: List[String] = [
            "alien_bg", "another_world_bg", "back_cave", "caverns", "cyberpunk_bg",
            "parallax_forest", "scifi_bg", "scifi2_bg", "living_tissue_bg",
            "airadventurelevel1", "airadventurelevel2", "airadventurelevel3",
            "airadventurelevel4", "cave_background", "blue_desert", "blue_grass",
            "blue_land", "blue_shroom", "colored_desert", "colored_grass",
            "colored_land", "colored_shroom", "landscape1", "landscape2",
            "landscape3", "landscape4", "battleback1", "battleback2",
            "battleback3", "battleback4", "battleback5", "battleback6",
            "battleback7", "battleback8", "battleback9", "battleback10", "sunrise",
        ]
        var bp = List[String]()
        for i in range(len(bnames)):
            bp.append("platform_backgrounds/" + bnames[i] + ".png")
        self.backgrounds = load_sprites(asset_root, bp)


struct ClimberGame(Copyable, Movable):
    var rand_gen: RandGen
    var dist_mode: Int
    var grid: List[Int]
    var w: Int
    var h: Int
    var agent: Entity
    var entities: List[Entity]
    var gravity: Float32
    var max_jump: Float32
    var air_control: Float32
    var maxspeed: Float32
    var mixrate: Float32
    var has_support: Bool
    var facing_right: Bool
    var coin_quota: Int
    var coins_collected: Int
    var difficulty: Int
    var num_platforms: Int
    var wall_theme: Int
    var background_index: Int
    var bg_pct_x: Float32
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
        self.w = 16 if dist_mode == DIST_EASY else 20
        self.h = H
        self.grid = List[Int]()
        self.grid.resize(self.w * H, SPACE)
        self.agent = Entity(1.5, 1.5, 0.0, 0.0, 0.5, 0.5, PLAYER)
        self.entities = List[Entity]()
        self.gravity = 0.2
        self.max_jump = 1.5
        self.air_control = 0.15
        self.maxspeed = 0.5
        self.mixrate = 0.5
        self.has_support = False
        self.facing_right = True
        self.coin_quota = 0
        self.coins_collected = 0
        self.difficulty = 0
        self.num_platforms = 0
        self.wall_theme = 0
        self.background_index = 0
        self.bg_pct_x = 0.0
        self.action_vx = 0.0
        self.action_vy = 0.0
        self.step_rand_int = 0
        self.reward = 0.0
        self.episode_reward = 0.0
        self.done = False
        self.level_complete = False
        self.cur_time = 0

    # --- grid helpers ---
    def _gset(mut self, x: Int, y: Int, v: Int):
        if x >= 0 and x < self.w and y >= 0 and y < self.h:
            self.grid[y * self.w + x] = v

    def _obj_ff(self, fi: Float32, fj: Float32) -> Int:
        if fi < 0.0 or fj < 0.0:
            return OOB
        var x = Int(floor(fi))
        var y = Int(floor(fj))
        if x < 0 or x >= self.w or y < 0 or y >= self.h:
            return OOB
        return self.grid[y * self.w + x]

    @staticmethod
    def _is_wall(t: Int) -> Bool:
        return t == WALL_MID or t == WALL_TOP

    @staticmethod
    def _can_support(o: Int) -> Bool:
        return ClimberGame._is_wall(o) or o == OOB

    @staticmethod
    def _is_blocked(src_type: Int, target: Int) -> Bool:
        if target == WALL_OBJ or target == WALL_MID:  # WALL_MID == OOB
            return True
        if src_type == PLAYER and target == WALL_TOP:
            return True
        return False

    @staticmethod
    def _will_reflect(src_type: Int, target: Int) -> Bool:
        return src_type == ENEMY and (
            target == WALL_MID or target == WALL_TOP or target == ENEMY_BARRIER
        )

    @staticmethod
    def _has_collision(
        ax: Float32, ay: Float32, arx: Float32, ary: Float32,
        bx: Float32, by: Float32, brx: Float32, bry: Float32,
    ) -> Bool:
        return abs(ax - bx) < (arx + brx) and abs(ay - by) < (ary + bry)

    def _out_of_bounds(self, e: Entity) -> Bool:
        return (
            e.x + e.rx < 0.0 or e.y + e.ry < 0.0
            or e.x - e.rx > Float32(self.w) or e.y - e.ry > Float32(self.h)
        )

    # --- physics (grid-only sub_step; no crates → no entity blocking) ---
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
                if self._is_blocked(obj.type, t):
                    block = True
                if self._will_reflect(obj.type, t):
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
        self.action_vx = Float32(mv // 3 - 1)
        self.action_vy = Float32(mv % 3 - 1)
        if self.action_vy < 0.0:
            self.action_vy = 0.0
        if self.action_vx > 0.0:
            self.facing_right = True
        if self.action_vx < 0.0:
            self.facing_right = False
        var b1 = self._obj_ff(
            self.agent.x - (self.agent.rx - 0.01), self.agent.y - (self.agent.ry + 0.01)
        )
        var b2 = self._obj_ff(
            self.agent.x + (self.agent.rx - 0.01), self.agent.y - (self.agent.ry + 0.01)
        )
        self.has_support = self._can_support(b1) or self._can_support(b2)
        if self.has_support and self.action_vy == 1.0:
            self.action_vy = 1.0
        else:
            self.action_vy = 0.0

    def _update_agent_velocity(mut self):
        var a = self.agent.copy()
        var mx = self.mixrate if self.has_support else (self.mixrate * self.air_control)
        a.vx = (1.0 - mx) * a.vx + mx * self.maxspeed * self.action_vx
        if self.action_vy > 0.0:
            a.vy = self.max_jump
        if not self.has_support:
            if a.vy > -2.0:
                a.vy -= self.gravity
        self.agent = a^

    # --- level gen ---
    def _choose_delta_y(mut self) -> Int:
        var max_dy = Int(self.max_jump * self.max_jump / (2 * self.gravity))
        var min_dy = 3
        return self.rand_gen.randn(max_dy - min_dy + 1) + min_dy

    def _generate_platforms(mut self):
        self.difficulty = self.rand_gen.randn(3)
        var min_p = self.difficulty * self.difficulty + 1
        var max_p = (self.difficulty + 1) * (self.difficulty + 1) + 1
        self.num_platforms = self.rand_gen.randn(max_p - min_p + 1) + min_p
        self.coin_quota = 0
        self.coins_collected = 0
        var curr_x = self.rand_gen.randn(self.w - 4) + 2
        var curr_y = 0
        var margin_x = 3
        var enemy_prob: Float32 = 0.2 if self.dist_mode == DIST_EASY else 0.5

        for i in range(self.num_platforms):
            var delta_y = self._choose_delta_y()
            var can_spawn = curr_x >= margin_x and curr_x <= self.w - margin_x
            if can_spawn and self.rand_gen.rand01() < enemy_prob:
                # add_entity args (right-to-left eval): vx's randn(2) then y's randn(2).
                var vx_draw = self.rand_gen.randn(2)
                var y_draw = self.rand_gen.randn(2)
                var e = Entity(
                    Float32(curr_x) + 0.5, Float32(curr_y + y_draw + 2) + 0.5,
                    0.15 * Float32(vx_draw * 2 - 1), 0.0, 0.5, 0.5, ENEMY,
                )
                e.image_type = ENEMY1
                e.smart_step = True
                e.climber_spawn_x = Float32(curr_x) + 0.5
                e.ry = e.rx / ENEMY_ASPECT  # match_aspect_ratio
                self.entities.append(e^)
            curr_y += delta_y
            var plat_len = 2 + self.rand_gen.randn(10)
            var vx = self.rand_gen.randn(2) * 2 - 1
            if curr_x < margin_x:
                vx = 1
            if curr_x > self.w - margin_x:
                vx = -1
            var cand = List[Int]()
            for j in range(plat_len):
                var nx = curr_x + (j + 1) * vx
                if nx <= 0 or nx >= self.w - 1:
                    break
                cand.append(nx)
                self._gset(nx, curr_y, WALL_TOP)
            if self.rand_gen.rand01() < 0.5 or i == self.num_platforms - 1:
                var cx = self.rand_gen.choose_one(cand)
                self.entities.append(
                    Entity(Float32(cx) + 0.5, Float32(curr_y) + 1.5, 0.0, 0.0, 0.3, 0.3, COIN)
                )
                self.coin_quota += 1
            curr_x = self.rand_gen.choose_one(cand)

    def _init_walls(mut self):
        for x in range(self.w):
            self._gset(x, 0, WALL_TOP)
        for y in range(self.h):
            self._gset(0, y, WALL_MID)
            self._gset(self.w - 1, y, WALL_MID)
        for x in range(self.w):
            self._gset(x, self.h - 1, WALL_MID)

    def reset(mut self, level_seed: Int):
        self.rand_gen.seed(level_seed)
        for i in range(self.w * self.h):
            self.grid[i] = SPACE
        self.entities = List[Entity]()
        self.reward = 0.0
        self.episode_reward = 0.0
        self.done = False
        self.level_complete = False
        self.cur_time = 0

        self.bg_pct_x = self.rand_gen.rand01()
        self.background_index = self.rand_gen.randn(BG_COUNT)

        self.gravity = 0.2
        self.max_jump = 1.5
        self.air_control = 0.15
        self.maxspeed = 0.5
        self.mixrate = 0.5
        self.has_support = False
        self.facing_right = True

        self.agent = Entity(1.5, 1.5, 0.0, 0.0, 0.5, 0.5, PLAYER)
        self.agent.smart_step = True
        self.agent.render_z = 1
        self.agent.image_theme = self.rand_gen.randn(NUM_PLAYER_THEMES)
        self.wall_theme = self.rand_gen.randn(NUM_WALL_THEMES)

        self._init_walls()
        self._generate_platforms()

    def step(mut self, action: Int) -> Float32:
        self.cur_time += 1
        self.step_rand_int = self.rand_gen.randint(0, 1000000)
        var mv = action % 9
        if action >= 9:
            mv = 4
        self.action_vx = 0.0
        self.action_vy = 0.0
        self.reward = 0.0
        self.done = False
        self.level_complete = False

        self._set_action_xy(mv)
        self._update_agent_velocity()

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

        # agent-collision: enemy → death, coin → +1/collect/erase.
        for i in range(len(self.entities) - 1, -1, -1):
            var t = self.entities[i].type
            if t == PLAYER:
                continue
            if self._has_collision(
                self.entities[i].x, self.entities[i].y,
                self.entities[i].rx, self.entities[i].ry,
                self.agent.x, self.agent.y, self.agent.rx, self.agent.ry,
            ):
                if t == ENEMY:
                    self.done = True
                elif t == COIN:
                    self.reward += COIN_REWARD
                    self.coins_collected += 1
                    self.entities[i].will_erase = True

        for i in range(len(self.entities) - 1, -1, -1):
            if self.entities[i].will_erase or (
                self.entities[i].auto_erase and self._out_of_bounds(self.entities[i])
            ):
                _ = self.entities.pop(i)
        if self._out_of_bounds(self.agent):
            self.done = True

        # climber tail: facing + enemy patrol clamp + coin-quota completion.
        if self.action_vx > 0.0:
            self.agent.is_reflected = False
        if self.action_vx < 0.0:
            self.agent.is_reflected = True
        for i in range(len(self.entities) - 1, -1, -1):
            if self.entities[i].type == ENEMY:
                var ex = self.entities[i].x
                var sx = self.entities[i].climber_spawn_x
                if ex > sx + PATROL_RANGE:
                    self.entities[i].vx = -abs(self.entities[i].vx)
                elif ex < sx - PATROL_RANGE:
                    self.entities[i].vx = abs(self.entities[i].vx)
        if self.coin_quota == self.coins_collected:
            self.done = True
            self.reward += COMPLETION_BONUS
            self.level_complete = True

        self.episode_reward += self.reward
        return self.reward

    # --- rendering (visual-approx; assets passed in) ---
    def render_obs(
        self, assets: ClimberAssets, res: Int = RES, ss: Int = OBS_SS
    ) -> List[UInt8]:
        return downscale(self.render(assets, res * ss), res * ss, res)

    def render(self, assets: ClimberAssets, out_res: Int = RES) -> List[UInt8]:
        var canvas = Canvas(out_res)
        canvas.fill(0, 0, 0)

        # Camera: choose_center cx=W/2, cy=agent.y+W/2-5*ry; visibility=W (follows up).
        var visibility = Float32(self.w)
        var center_x = Float32(self.w) / 2.0
        var center_y = self.agent.y + Float32(self.w) / 2.0 - 5.0 * self.agent.ry
        var view_dim = visibility
        var unit = Float32(out_res) / view_dim
        var x_off = unit * (center_x - view_dim / 2)
        var y_off = unit * (center_y - view_dim / 2)

        # Background (platform, panned by bg_pct_x; anchored to world, follows camera).
        ref bg = assets.backgrounds[self.background_index]
        var main_w = Float32(self.w) * unit
        var main_h = Float32(self.h) * unit
        var main_y = (view_dim - Float32(self.h)) * unit + y_off
        var bg_ar = Float32(bg.w) / Float32(bg.h)
        var world_ar = Float32(self.w) / Float32(self.h)
        var offset_x = self.bg_pct_x * (bg_ar - world_ar)
        canvas.blit(
            bg, -x_off + main_w * (-offset_x), main_y, main_w * (bg_ar / world_ar), main_h
        )

        var wt = self.wall_theme
        var vy0 = Int(floor(center_y - view_dim / 2)) - 1
        var vy1 = Int(floor(center_y + view_dim / 2)) + 1
        for x in range(self.w):
            for y in range(vy0, vy1 + 1):
                if y < 0 or y >= self.h:
                    continue
                var t = self.grid[y * self.w + x]
                if t != WALL_MID and t != WALL_TOP:
                    continue
                var sx = (Float32(x) - RENDER_EPS) * unit - x_off
                var sy = (view_dim - Float32(y + 1) - RENDER_EPS) * unit + y_off
                var sz = (1.0 + 2 * RENDER_EPS) * unit
                if t == WALL_TOP:
                    canvas.blit(assets.wall_top[wt], sx, sy, sz, sz)
                else:
                    canvas.blit(assets.wall_mid[wt], sx, sy, sz, sz)

        # Entities (coins, enemies).
        for k in range(len(self.entities)):
            ref e = self.entities[k]
            var ex = (e.x - e.rx) * unit - x_off
            var ey = (view_dim - (e.y + e.ry)) * unit + y_off
            var ew = 2 * e.rx * unit
            var eh = 2 * e.ry * unit
            if e.type == COIN:
                canvas.blit(assets.coin, ex, ey, ew, eh)
            elif e.type == ENEMY:
                canvas.blit(assets.enemy1, ex, ey, ew, eh, e.is_reflected)

        # Player (tall sprite; preserve aspect, feet-anchored).
        var idx = self.agent.image_theme
        var pw = 2 * self.agent.rx * unit
        var feet = (view_dim - (self.agent.y - self.agent.ry)) * unit + y_off
        var px = (self.agent.x - self.agent.rx) * unit - x_off
        if not self.has_support:
            ref ps = assets.players_jump[idx]
            var ph = pw * (Float32(ps.h) / Float32(ps.w))
            canvas.blit(ps, px, feet - ph, pw, ph, self.agent.is_reflected)
        elif abs(self.agent.vx) < 0.01 and self.action_vx == 0.0:
            ref ps = assets.players[idx]
            var ph = pw * (Float32(ps.h) / Float32(ps.w))
            canvas.blit(ps, px, feet - ph, pw, ph, self.agent.is_reflected)
        else:
            ref ps = assets.players_walk[idx]
            var ph = pw * (Float32(ps.h) / Float32(ps.w))
            canvas.blit(ps, px, feet - ph, pw, ph, self.agent.is_reflected)

        return canvas.px.copy()
