"""Ninja game — platformer with charged jump + throwing stars (port of `games/ninja.cpp`).

Charge a jump (hold up to build power, release to leap) across platforms to the
mushroom goal (+10), throwing stars (6 special-action directions) that stick to
walls and detonate bombs into explosions. Death on touching fire/bombs/explosions.
Reuses the platformer substrate (gravity + has_support + wall collision) + the
projectile substrate. `game_reset`/`game_step` replay the exact RNG order.
Level-exact + visual-approx. See `docs/PROCGEN_NINJA_SCOPE.md`. P0+P1 parity; render P2.
"""

from std.math import floor, ceil, sqrt, cos, sin

from ..core.entity import Entity
from ..core.randgen import RandGen
from ..core.assets import Sprite, load_sprite, load_sprites
from ..core.rasterizer import Canvas, RES, downscale
from ..core.object_ids import PLAYER, SPACE, WALL_OBJ, EXPLOSION

comptime GOAL = 1
comptime BOMB = 6
comptime THROWING_STAR = 7
comptime PLAYER_JUMP = 9
comptime PLAYER_RIGHT1 = 12
comptime PLAYER_RIGHT2 = 13
comptime FIRE = 14
comptime WALL_MID = 20

comptime OOB = WALL_MID
comptime GOAL_REWARD: Float32 = 10.0
comptime POS_EPS: Float32 = -0.001
comptime PI: Float32 = 3.14159265358979
comptime H = 64
comptime W = 64
comptime NUM_WALL_THEMES = 3
comptime NUM_GOAL_THEMES = 6
comptime BG_COUNT = 37  # platform_backgrounds (value only; not in RNG-stream parity)
comptime OBS_SS = 4
comptime RENDER_EPS: Float32 = 0.0

comptime DIST_EASY = 0
comptime DIST_HARD = 1


struct NinjaAssets(Movable):
    """Read-only sprite set (shared via ArcPointer; passed into render())."""

    var player: Sprite
    var player_jump: Sprite
    var player_walk1: Sprite
    var player_walk2: Sprite
    var walls: List[Sprite]  # 3 brick themes
    var goals: List[Sprite]  # 6 shroom themes
    var bomb: Sprite
    var star: Sprite
    var explosion: Sprite
    var backgrounds: List[Sprite]  # platform_backgrounds (37)

    def __init__(out self, asset_root: String) raises:
        self.player = load_sprite(asset_root, "platformer/zombie_idle.png")
        self.player_jump = load_sprite(asset_root, "platformer/zombie_jump.png")
        self.player_walk1 = load_sprite(asset_root, "platformer/zombie_walk1.png")
        self.player_walk2 = load_sprite(asset_root, "platformer/zombie_walk2.png")
        var wl: List[String] = [
            "misc_assets/tile_bricksGrey.png", "misc_assets/tile_bricksGrown.png",
            "misc_assets/tile_bricksRed.png",
        ]
        self.walls = load_sprites(asset_root, wl)
        var gl = List[String]()
        for i in range(1, 7):
            gl.append("platformer/shroom" + String(i) + ".png")
        self.goals = load_sprites(asset_root, gl)
        self.bomb = load_sprite(asset_root, "misc_assets/bomb.png")
        self.star = load_sprite(asset_root, "misc_assets/saw.png")
        self.explosion = load_sprite(asset_root, "misc_assets/explosion1.png")
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


struct NinjaGame(Copyable, Movable):
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
    var jump_charge: Float32
    var jump_charge_inc: Float32
    var has_support: Bool
    var facing_right: Bool
    var last_fire_time: Int
    var difficulty: Int
    var wall_theme: Int
    var background_index: Int
    var bg_pct_x: Float32
    var action_vx: Float32
    var action_vy: Float32
    var special_action: Int
    var step_rand_int: Int
    var reward: Float32
    var episode_reward: Float32
    var done: Bool
    var level_complete: Bool
    var cur_time: Int
    var goal_x: Int
    var goal_y: Int

    def __init__(out self, dist_mode: Int = DIST_EASY):
        self.rand_gen = RandGen()
        self.dist_mode = dist_mode
        self.w = W
        self.h = H
        self.grid = List[Int]()
        self.grid.resize(W * H, SPACE)
        self.agent = Entity(1.5, 32.5, 0.0, 0.0, 0.5, 0.5, PLAYER)
        self.entities = List[Entity]()
        self.gravity = 0.2
        self.max_jump = 1.5
        self.air_control = 0.15
        self.maxspeed = 0.5
        self.mixrate = 0.5
        self.jump_charge = 0.0
        self.jump_charge_inc = 0.25
        self.has_support = False
        self.facing_right = True
        self.last_fire_time = 0
        self.difficulty = 0
        self.wall_theme = 0
        self.background_index = 0
        self.bg_pct_x = 0.0
        self.action_vx = 0.0
        self.action_vy = 0.0
        self.special_action = 0
        self.step_rand_int = 0
        self.reward = 0.0
        self.episode_reward = 0.0
        self.done = False
        self.level_complete = False
        self.cur_time = 0
        self.goal_x = 0
        self.goal_y = 0

    # --- grid helpers ---
    def _gset(mut self, x: Int, y: Int, v: Int):
        if x >= 0 and x < W and y >= 0 and y < H:
            self.grid[y * W + x] = v

    def _obj_ff(self, fi: Float32, fj: Float32) -> Int:
        if fi < 0.0 or fj < 0.0:
            return OOB
        var x = Int(floor(fi))
        var y = Int(floor(fj))
        if x < 0 or x >= W or y < 0 or y >= H:
            return OOB
        return self.grid[y * W + x]

    def _fill_elem(mut self, x: Int, y: Int, dx: Int, dy: Int, e: Int):
        for j in range(dx):
            for k in range(dy):
                self._gset(x + j, y + k, e)

    def _fill_block_top(mut self, x: Int, y: Int, dx: Int, dy: Int, fill: Int, top: Int):
        if dy <= 0:
            return
        self._fill_elem(x, y, dx, dy - 1, fill)
        self._fill_elem(x, y + dy - 1, dx, 1, top)

    def _fill_ground(mut self, x: Int, y: Int, dx: Int, dy: Int):
        self._fill_block_top(x, y, dx, dy, WALL_MID, WALL_MID)

    @staticmethod
    def _can_support(o: Int) -> Bool:
        return o == WALL_MID  # is_wall or oob (== WALL_MID)

    @staticmethod
    def _has_collision(
        ax: Float32, ay: Float32, arx: Float32, ary: Float32,
        bx: Float32, by: Float32, brx: Float32, bry: Float32,
    ) -> Bool:
        return abs(ax - bx) < (arx + brx) and abs(ay - by) < (ary + bry)

    def _out_of_bounds(self, e: Entity) -> Bool:
        return (
            e.x + e.rx < 0.0 or e.y + e.ry < 0.0
            or e.x - e.rx > Float32(W) or e.y - e.ry > Float32(H)
        )

    # --- physics (grid-only; stars stick to walls via is_blocked side effect) ---
    def _sub_step(self, mut obj: Entity, vx: Float32, vy: Float32) -> Bool:
        if obj.will_erase:
            return False
        var nx = obj.x + vx
        var ny = obj.y + vy
        var margin: Float32 = 0.98
        var is_h = vx != 0.0
        var block = False
        for i in range(2):
            for j in range(2):
                var t = self._obj_ff(
                    nx + obj.rx * margin * Float32(2 * i - 1),
                    ny + obj.ry * margin * Float32(2 * j - 1),
                )
                if t == WALL_MID or t == WALL_OBJ:  # WALL_MID == OOB
                    block = True
                    if obj.type == THROWING_STAR:
                        obj.vx = 0.0
                        obj.vy = 0.0
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
            self.jump_charge += self.jump_charge_inc
            if self.jump_charge > 1.0:
                self.jump_charge = 1.0
        else:
            self.action_vy = 0.0
        if not self.has_support:
            self.jump_charge = 0.0

    def _update_agent_velocity(mut self):
        var a = self.agent.copy()
        var mx = self.mixrate if self.has_support else (self.mixrate * self.air_control)
        a.vx = (1.0 - mx) * a.vx + mx * self.maxspeed * self.action_vx
        if self.action_vy < 1.0 and self.jump_charge > 0.0:
            a.vy = self.jump_charge * self.max_jump
            self.jump_charge = 0.0
        if not self.has_support:
            if a.vy > -2.0:
                a.vy -= self.gravity
        self.agent = a^

    def _check_grid_collisions_agent(mut self):
        var minx = Int(self.agent.x - (self.agent.rx + POS_EPS))
        var maxx = Int(self.agent.x + (self.agent.rx + POS_EPS))
        var miny = Int(self.agent.y - (self.agent.ry + POS_EPS))
        var maxy = Int(self.agent.y + (self.agent.ry + POS_EPS))
        for x in range(minx, maxx + 1):
            for y in range(miny, maxy + 1):
                var gt = self._obj_ff(Float32(x), Float32(y))
                if gt == FIRE or gt == BOMB:
                    self.done = True

    def _star_grid_collisions(mut self, i: Int):
        # entities[i] is a THROWING_STAR: detonate bombs, erase on walls.
        var minx = Int(self.entities[i].x - (self.entities[i].rx + POS_EPS))
        var maxx = Int(self.entities[i].x + (self.entities[i].rx + POS_EPS))
        var miny = Int(self.entities[i].y - (self.entities[i].ry + POS_EPS))
        var maxy = Int(self.entities[i].y + (self.entities[i].ry + POS_EPS))
        for x in range(minx, maxx + 1):
            for y in range(miny, maxy + 1):
                var gt = self._obj_ff(Float32(x), Float32(y))
                if gt == SPACE:
                    continue
                if gt == BOMB:
                    self.entities[i].will_erase = True
                    self._gset(x, y, SPACE)
                    self.entities.append(
                        Entity(Float32(x) + 0.5, Float32(y) + 0.5, 0.0, 0.0, 0.5, 0.5, EXPLOSION)
                    )
                if gt == WALL_MID:
                    self.entities[i].will_erase = True

    # --- level gen ---
    def _gen_section(
        mut self, mut curr_x: Int, mut curr_y: Int, mut min_y: Int, diff: Int,
        min_gap: Int, min_plat_w: Int, inc_dy: Int, max_dy: Int,
        max_gap_inc: Int, bomb_prob: Float32,
    ):
        # One section: an edge loop laying platforms, then bomb + ceiling.
        # curr_x/curr_y/min_y are updated in place (mut params avoid a flaky tuple
        # return). Split out of _generate to keep each function body small (the
        # compiler's analysis is superlinear in function size).
        var prev_x = curr_x
        var prev_y = curr_y
        var num_edges = self.rand_gen.randn(2) + 1
        var max_y = -1
        var last_edge_y = -1
        for j in range(num_edges):
            curr_x = prev_x + j
            if curr_x + 15 >= W:
                break
            curr_y = prev_y
            var dy = self.rand_gen.randn(inc_dy) + 1 + diff // 3
            if dy > max_dy:
                dy = max_dy
            if curr_y >= H - 15:
                dy *= -1
            elif curr_y >= 5 and self.rand_gen.rand01() < 0.4:
                dy *= -1
            curr_y += dy
            if curr_y < 3:
                curr_y = 3
            if abs(curr_y - last_edge_y) <= 1:
                curr_y = last_edge_y + 2
            var dx = min_plat_w + self.rand_gen.randn(3)
            self._fill_ground(curr_x, curr_y - 1, dx, 1)
            curr_x += dx
            curr_x += min_gap + self.rand_gen.randn(max_gap_inc + 1)
            if curr_y > max_y:
                max_y = curr_y
            if curr_y < min_y:
                min_y = curr_y
            last_edge_y = curr_y
        if self.rand_gen.rand01() < bomb_prob:
            self._gset(
                self.rand_gen.randn(curr_x - prev_x + 1) + prev_x, max_y + 2, BOMB
            )
        var ceiling_start = max_y - 1 + 11
        self._fill_ground(prev_x, ceiling_start, curr_x - prev_x, H - ceiling_start)

    def _generate(mut self, diff: Int):
        var min_gap = diff - 1
        var min_plat_w = 1
        var inc_dy = 4
        if self.dist_mode == DIST_EASY:
            min_gap -= 1
            if min_gap < 0:
                min_gap = 0
            min_plat_w = 3
            inc_dy = 2
        var bomb_prob: Float32 = 0.25 * Float32(diff - 1)
        var max_gap_inc = 1 if diff == 1 else 2
        var num_sections = self.rand_gen.randn(diff) + diff
        var start_x = 5
        var curr_x = 5
        var curr_y = H // 2
        var min_y = curr_y
        var _max_dy = self.max_jump * self.max_jump / (2 * self.gravity)
        var max_dy = Int(_max_dy - 0.5)
        self._fill_ground(0, 0, start_x, curr_y)
        self._fill_elem(0, curr_y + 8, start_x, H - curr_y - 8, WALL_MID)
        for _ in range(num_sections):
            self._gen_section(
                curr_x, curr_y, min_y, diff, min_gap, min_plat_w, inc_dy,
                max_dy, max_gap_inc, bomb_prob,
            )
        var g = Entity(Float32(curr_x) + 0.5, Float32(curr_y) + 0.5, 0.0, 0.0, 0.5, 0.5, GOAL)
        g.image_theme = self.rand_gen.randn(NUM_GOAL_THEMES)
        self.entities.append(g^)
        self.goal_x = curr_x
        self.goal_y = curr_y
        self._fill_ground(curr_x, curr_y - 1, 1, 1)
        self._fill_elem(curr_x, curr_y + 6, 1, H - curr_y - 6, WALL_MID)
        var fire_y = min_y - 2
        if fire_y < 1:
            fire_y = 1
        self._fill_ground(start_x, 0, W - start_x, fire_y)
        self._fill_elem(start_x, fire_y, W - start_x, 1, FIRE)
        self._fill_elem(curr_x + 1, 0, W - curr_x - 1, H, WALL_MID)


    def _init_walls(mut self):
        self._fill_elem(0, 0, W, 1, WALL_MID)
        self._fill_elem(0, 0, 1, H, WALL_MID)
        self._fill_elem(W - 1, 0, 1, H, WALL_MID)
        self._fill_elem(0, H - 1, W, 1, WALL_MID)

    def reset(mut self, level_seed: Int):
        self.rand_gen.seed(level_seed)
        for i in range(W * H):
            self.grid[i] = SPACE
        self.entities = List[Entity]()
        self.reward = 0.0
        self.episode_reward = 0.0
        self.done = False
        self.level_complete = False
        self.cur_time = 0
        self.last_fire_time = 0

        self.bg_pct_x = self.rand_gen.rand01()
        self.background_index = self.rand_gen.randn(BG_COUNT)

        self.gravity = 0.2
        self.max_jump = 1.5
        self.air_control = 0.15
        self.maxspeed = 0.5
        self.mixrate = 0.5
        self.jump_charge = 0.0
        self.jump_charge_inc = 0.25
        self.has_support = False
        self.facing_right = True

        self.agent = Entity(1.5, Float32(H // 2) + 0.5, 0.0, 0.0, 0.5, 0.5, PLAYER)
        self.agent.smart_step = True
        self.agent.render_z = 1
        if self.dist_mode == DIST_EASY:
            self.max_jump = 1.25
            self.jump_charge_inc = 1.0

        self.difficulty = self.rand_gen.randn(3) + 1
        self.wall_theme = self.rand_gen.randn(NUM_WALL_THEMES)

        self._init_walls()
        self._generate(self.difficulty)

    def step(mut self, action: Int) -> Float32:
        self.cur_time += 1
        self.step_rand_int = self.rand_gen.randint(0, 1000000)
        var mv = action % 9
        self.special_action = 0
        if action >= 9:
            self.special_action = action - 8
            mv = 4
        self.action_vx = 0.0
        self.action_vy = 0.0
        self.reward = 0.0
        self.done = False
        self.level_complete = False

        self._set_action_xy(mv)
        self._update_agent_velocity()

        # step_entities: entities (reverse) then agent (idx 0, stepped last).
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

        # collisions: agent-death/win + star grid detonation.
        for i in range(len(self.entities) - 1, -1, -1):
            var t = self.entities[i].type
            if t != PLAYER and self._has_collision(
                self.entities[i].x, self.entities[i].y,
                self.entities[i].rx, self.entities[i].ry,
                self.agent.x, self.agent.y, self.agent.rx, self.agent.ry,
            ):
                if t == EXPLOSION:
                    self.done = True
                elif t == GOAL:
                    self.reward += GOAL_REWARD
                    self.level_complete = True
                    self.done = True
            if self.entities[i].smart_step and self.entities[i].type == THROWING_STAR:
                self._star_grid_collisions(i)
        self._check_grid_collisions_agent()

        for i in range(len(self.entities) - 1, -1, -1):
            if self.entities[i].will_erase or (
                self.entities[i].auto_erase and self._out_of_bounds(self.entities[i])
            ):
                _ = self.entities.pop(i)
        if self._out_of_bounds(self.agent):
            self.done = True

        # ninja tail: facing + throwing-star fire.
        if self.action_vx > 0.0:
            self.agent.is_reflected = False
        if self.action_vx < 0.0:
            self.agent.is_reflected = True
        if self.special_action > 0 and (self.cur_time - self.last_fire_time) >= 3:
            var theta: Float32 = 0.0
            if self.special_action == 2:
                theta = PI / 4
            elif self.special_action == 3:
                theta = PI / 2
            elif self.special_action == 4:
                theta = -1.0 * PI / 4
            if self.agent.is_reflected:
                theta = PI - theta
            var b = Entity(
                self.agent.x, self.agent.y, cos(theta), sin(theta), 0.25, 0.25, THROWING_STAR
            )
            b.collides_with_entities = True
            b.expire_time = 15
            b.smart_step = True
            self.entities.append(b^)
            self.last_fire_time = self.cur_time

        self.episode_reward += self.reward
        return self.reward

    # --- rendering (visual-approx; assets passed in) ---
    def render_obs(
        self, assets: NinjaAssets, res: Int = RES, ss: Int = OBS_SS
    ) -> List[UInt8]:
        return downscale(self.render(assets, res * ss), res * ss, res)

    def render(self, assets: NinjaAssets, out_res: Int = RES) -> List[UInt8]:
        var canvas = Canvas(out_res)
        canvas.fill(0, 0, 0)

        # Camera: centered on the agent (visibility 10 Easy / 16 Hard).
        var visibility: Float32 = 10.0 if self.dist_mode == DIST_EASY else 16.0
        var center_x = self.agent.x
        var center_y = self.agent.y
        var view_dim = visibility
        var unit = Float32(out_res) / view_dim
        var x_off = unit * (center_x - view_dim / 2)
        var y_off = unit * (center_y - view_dim / 2)

        # Background (platform, panned by bg_pct_x; anchored to world).
        ref bg = assets.backgrounds[self.background_index]
        var main_w = Float32(W) * unit
        var main_h = Float32(H) * unit
        var main_y = (view_dim - Float32(H)) * unit + y_off
        var bg_ar = Float32(bg.w) / Float32(bg.h)
        var world_ar = Float32(W) / Float32(H)
        var offset_x = self.bg_pct_x * (bg_ar - world_ar)
        canvas.blit(
            bg, -x_off + main_w * (-offset_x), main_y, main_w * (bg_ar / world_ar), main_h
        )

        # Grid tiles inside the camera window (walls, fire, bombs).
        var wt = self.wall_theme
        var vx0 = Int(floor(center_x - view_dim / 2)) - 1
        var vx1 = Int(floor(center_x + view_dim / 2)) + 1
        var vy0 = Int(floor(center_y - view_dim / 2)) - 1
        var vy1 = Int(floor(center_y + view_dim / 2)) + 1
        for x in range(vx0, vx1 + 1):
            for y in range(vy0, vy1 + 1):
                if x < 0 or x >= W or y < 0 or y >= H:
                    continue
                var t = self.grid[y * W + x]
                if t == SPACE:
                    continue
                var sx = (Float32(x) - RENDER_EPS) * unit - x_off
                var sy = (view_dim - Float32(y + 1) - RENDER_EPS) * unit + y_off
                var sz = (1.0 + 2 * RENDER_EPS) * unit
                if t == WALL_MID:
                    canvas.blit(assets.walls[wt], sx, sy, sz, sz)
                elif t == FIRE or t == BOMB:
                    canvas.blit(assets.bomb, sx, sy, sz, sz)

        # Entities (goal, stars, explosions).
        for k in range(len(self.entities)):
            ref e = self.entities[k]
            var ex = (e.x - e.rx) * unit - x_off
            var ey = (view_dim - (e.y + e.ry)) * unit + y_off
            var ew = 2 * e.rx * unit
            var eh = 2 * e.ry * unit
            if e.type == GOAL:
                canvas.blit(assets.goals[e.image_theme], ex, ey, ew, eh)
            elif e.type == THROWING_STAR:
                canvas.blit(assets.star, ex, ey, ew, eh)
            elif e.type == EXPLOSION:
                canvas.blit(assets.explosion, ex, ey, ew, eh)

        # Player (zombie; tall sprite, feet-anchored).
        var pw = 2 * self.agent.rx * unit
        var feet = (view_dim - (self.agent.y - self.agent.ry)) * unit + y_off
        var px = (self.agent.x - self.agent.rx) * unit - x_off
        var still = abs(self.agent.vx) < 0.01 and self.action_vx == 0.0 and self.has_support
        if still:
            ref ps = assets.player
            var ph = pw * (Float32(ps.h) / Float32(ps.w))
            canvas.blit(ps, px, feet - ph, pw, ph, self.agent.is_reflected)
        else:
            ref ps = assets.player_walk1
            var ph = pw * (Float32(ps.h) / Float32(ps.w))
            canvas.blit(ps, px, feet - ph, pw, ph, self.agent.is_reflected)

        return canvas.px.copy()
