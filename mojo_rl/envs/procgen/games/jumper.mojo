"""Jumper game — double-jump cave platformer (port of `games/jumper.cpp`).

A bunny double-jumps through a maze-seeded roomgen cave to a carrot goal (+10),
dodging spikes (touch = death). Reuses the platformer substrate (gravity + jump +
wall collision) with a DOUBLE JUMP + the roomgen + MazeGen substrates. Level-exact
+ visual-approx. See `docs/PROCGEN_JUMPER_SCOPE.md`. P0+P1 parity; render P2.

NOTE: `game_reset` is split into small helper methods (`_maze_fill`, `_carve`,
`_place_goal_agent`, `_place_spikes`, `_thin_walls`, `_finalize`) — a single large
generation function hangs the Mojo compiler (superlinear per-function analysis).
"""

from std.math import floor, ceil, sqrt

from ..core.entity import Entity
from ..core.randgen import RandGen
from ..core.grid import Grid
from ..core.mazegen import MazeGen, MAZE_OFFSET
from ..core.roomgen import (
    roomgen_update, roomgen_find_best_room, roomgen_expand_room, roomgen_find_path
)
from ..core.assets import Sprite, load_sprite, load_sprites
from ..core.rasterizer import Canvas, RES, downscale
from ..core.object_ids import PLAYER, SPACE, WALL_OBJ, TRAIL

comptime GOAL = 1
comptime SPIKE = 2
comptime CAVEWALL = 6
comptime CAVEWALL_TOP = 7
comptime PLAYER_JUMP = 9
comptime PLAYER_LEFT1 = 10
comptime PLAYER_LEFT2 = 11
comptime PLAYER_RIGHT1 = 12
comptime PLAYER_RIGHT2 = 13

comptime GOAL_REWARD: Float32 = 10.0
comptime MAZE_SCALE = 3
comptime JUMP_COOLDOWN = 3
comptime NUM_WALL_THEMES = 4
comptime BG_COUNT = 37  # platform_backgrounds (value only; not in RNG-stream parity)
comptime OBS_SS = 4
comptime RENDER_EPS: Float32 = 0.0

comptime DIST_EASY = 0
comptime DIST_HARD = 1


struct JumperAssets(Movable):
    """Read-only sprite set (shared via ArcPointer; passed into render())."""

    var player: Sprite
    var player_jump: Sprite
    var player_walk1: Sprite
    var player_walk2: Sprite
    var spike: Sprite
    var goal: Sprite
    var wall_top: List[Sprite]  # 4 themes
    var wall_mid: List[Sprite]  # 4 themes
    var backgrounds: List[Sprite]  # platform_backgrounds (37)

    def __init__(out self, asset_root: String) raises:
        self.player = load_sprite(asset_root, "misc_assets/bunny2_ready.png")
        self.player_jump = load_sprite(asset_root, "misc_assets/bunny2_jump.png")
        self.player_walk1 = load_sprite(asset_root, "misc_assets/bunny2_walk1.png")
        self.player_walk2 = load_sprite(asset_root, "misc_assets/bunny2_walk2.png")
        self.spike = load_sprite(asset_root, "misc_assets/spikeMan_stand.png")
        self.goal = load_sprite(asset_root, "misc_assets/carrot.png")
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


struct JumperGame(Copyable, Movable):
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
    var has_support: Bool
    var facing_right: Bool
    var jump_count: Int
    var jump_delta: Int
    var jump_time: Int
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
    var agent_cell: Int
    var goal_cell: Int
    var _free_cells: List[Int]
    var _goal_path: List[Int]

    def __init__(out self, dist_mode: Int = DIST_EASY):
        self.rand_gen = RandGen()
        self.dist_mode = dist_mode
        var wd = 20 if dist_mode == DIST_EASY else 40
        self.grid = Grid()
        self.grid.resize(wd, wd)
        self.w = wd
        self.h = wd
        self.oob = CAVEWALL
        self.agent = Entity(1.5, 1.5, 0.0, 0.0, 0.254, 0.4, PLAYER)
        self.entities = List[Entity]()
        self.mixrate = 0.5
        self.maxspeed = 0.5
        self.has_support = False
        self.facing_right = True
        self.jump_count = 0
        self.jump_delta = 0
        self.jump_time = 0
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
        self.agent_cell = 0
        self.goal_cell = 0
        self._free_cells = List[Int]()
        self._goal_path = List[Int]()

    # --- grid helpers ---
    def _gset(mut self, x: Int, y: Int, v: Int):
        if x >= 0 and x < self.w and y >= 0 and y < self.h:
            self.grid.data[y * self.w + x] = v

    def _gget(self, x: Int, y: Int) -> Int:
        if x < 0 or x >= self.w or y < 0 or y >= self.h:
            return self.oob
        return self.grid.data[y * self.w + x]

    def _obj_ff(self, fi: Float32, fj: Float32) -> Int:
        if fi < 0.0 or fj < 0.0:
            return self.oob
        return self._gget(Int(floor(fi)), Int(floor(fj)))

    @staticmethod
    def _is_wall(t: Int) -> Bool:
        return t == CAVEWALL or t == CAVEWALL_TOP

    def _can_support(self, o: Int) -> Bool:
        return self._is_wall(o) or o == self.oob

    def _is_blocked(self, src_type: Int, target: Int) -> Bool:
        if target == WALL_OBJ or target == self.oob:
            return True
        if src_type == PLAYER and self._is_wall(target):
            return True
        return False

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

    def _is_space_on_ground(self, x: Int, y: Int) -> Bool:
        if self._gget(x, y) != SPACE:
            return False
        if self._gget(x, y + 1) != SPACE:
            return False
        var b = self._gget(x, y - 1)
        return b == CAVEWALL or b == self.oob

    def _is_top_wall(self, x: Int, y: Int) -> Bool:
        return self._gget(x, y) == CAVEWALL and self._gget(x, y + 1) == SPACE

    def _is_left_wall(self, x: Int, y: Int) -> Bool:
        return self._gget(x, y) == CAVEWALL and self._gget(x + 1, y) == SPACE

    def _is_right_wall(self, x: Int, y: Int) -> Bool:
        return self._gget(x, y) == CAVEWALL and self._gget(x - 1, y) == SPACE

    # --- physics (grid-only sub_step) ---
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
                if self._is_blocked(obj.type, t):
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
        self.jump_delta = 0
        self.has_support = self._can_support(b1) or self._can_support(b2)
        if self.has_support:
            self.jump_count = 2
        if (
            self.action_vy == 1.0 and self.jump_count > 0
            and (self.cur_time - self.jump_time > JUMP_COOLDOWN)
        ):
            self.jump_count -= 1
            self.jump_delta = -1
        else:
            self.action_vy = 0.0
        if self.action_vy > 0.0:
            self.jump_time = self.cur_time

    def _update_agent_velocity(mut self):
        var a = self.agent.copy()
        a.vx = (1.0 - self.mixrate) * a.vx + self.mixrate * self.maxspeed * self.action_vx
        if self.action_vy != 0.0:
            a.vy = self.maxspeed * self.action_vy * 2
        self.agent = a^

    # --- level gen (split into small helpers to avoid the compiler hang) ---
    def _maze_fill(mut self):
        var maze_dim = self.w // MAZE_SCALE
        var mg = MazeGen(maze_dim)
        mg.generate_maze_no_dead_ends(self.rand_gen)
        for i in range(self.w * self.h):
            var sx = (i % self.w) // MAZE_SCALE + MAZE_OFFSET
            var sy = (i // self.w) // MAZE_SCALE + MAZE_OFFSET
            var obj = mg.grid.get(sx, sy)
            var prob: Float32 = 0.8 if obj == WALL_OBJ else 0.2
            self.grid.data[i] = WALL_OBJ if self.rand_gen.rand01() < prob else SPACE

    def _carve(mut self):
        for _ in range(2):
            roomgen_update(self.grid, self.oob)
        for i in range(self.w):
            self._gset(i, 0, CAVEWALL)
            self._gset(i, self.h - 1, CAVEWALL)
        for i in range(self.h):
            self._gset(0, i, CAVEWALL)
            self._gset(self.w - 1, i, CAVEWALL)
        var best = roomgen_find_best_room(self.grid)
        for i in range(self.w * self.h):
            self.grid.data[i] = CAVEWALL
        self._free_cells = List[Int]()
        for i in range(self.w * self.h):
            if best[i]:
                self.grid.data[i] = SPACE
                self._free_cells.append(i)

    def _place_goal_agent(mut self):
        self.goal_cell = self.rand_gen.choose_one(self._free_cells)
        var cand = List[Int]()
        for i in range(self.w * self.h):
            if self._is_space_on_ground(i % self.w, i // self.w):
                cand.append(i)
        self.agent_cell = self.rand_gen.choose_one(cand)
        self._goal_path = roomgen_find_path(self.grid, self.agent_cell, self.goal_cell)
        if self.dist_mode != 2:  # not MemoryMode → prune
            var wide = List[Bool]()
            wide.resize(self.w * self.h, False)
            for i in range(len(self._goal_path)):
                wide[self._goal_path[i]] = True
            roomgen_expand_room(self.grid, wide, 4)
            for i in range(self.w * self.h):
                self.grid.data[i] = CAVEWALL
            for i in range(self.w * self.h):
                if wide[i]:
                    self.grid.data[i] = SPACE

    def _place_spikes(mut self):
        var spike_prob: Float32 = 0.0 if self.dist_mode == 2 else 0.2
        for i in range(self.w * self.h):
            var x = i % self.w
            var y = i // self.w
            if (
                self._is_space_on_ground(x, y)
                and self._is_space_on_ground(x - 1, y)
                and self._is_space_on_ground(x + 1, y)
            ):
                if self.rand_gen.rand01() < spike_prob:
                    self._gset(x, y, SPIKE)

    def _thin_walls(mut self):
        for i in range(self.w * self.h):
            var x = i % self.w
            var y = i // self.w
            if (
                self._is_left_wall(x, y) and self._is_left_wall(x, y + 1)
                and self._is_left_wall(x, y + 2)
            ):
                self._gset(x, y + self.rand_gen.randn(3), SPACE)
            if (
                self._is_right_wall(x, y) and self._is_right_wall(x, y + 1)
                and self._is_right_wall(x, y + 2)
            ):
                self._gset(x, y + self.rand_gen.randn(3), SPACE)

    def _finalize(mut self):
        # Spike cells → spike entities; then mark top walls.
        for i in range(self.w * self.h):
            if self.grid.data[i] == SPIKE:
                self.grid.data[i] = SPACE
                var e = Entity(
                    Float32(i % self.w) + 0.5, Float32(i // self.w) + 0.4,
                    0.0, 0.0, 0.23, 0.4, SPIKE,
                )
                self.entities.append(e^)
        for i in range(self.w * self.h):
            var x = i % self.w
            var y = i // self.w
            if self._is_top_wall(x, y):
                self.grid.data[i] = CAVEWALL_TOP

    def reset(mut self, level_seed: Int):
        self.rand_gen.seed(level_seed)
        for i in range(self.w * self.h):
            self.grid.data[i] = SPACE
        self.entities = List[Entity]()
        self.reward = 0.0
        self.episode_reward = 0.0
        self.done = False
        self.level_complete = False
        self.cur_time = 0

        self.bg_pct_x = self.rand_gen.rand01()
        self.background_index = self.rand_gen.randn(BG_COUNT)
        self.oob = WALL_OBJ
        self.wall_theme = self.rand_gen.randn(NUM_WALL_THEMES)
        self.jump_count = 0
        self.jump_delta = 0
        self.jump_time = 0
        self.has_support = False
        self.facing_right = True

        self._maze_fill()
        self._carve()
        self._place_goal_agent()
        # GOAL entity at goal_cell.
        var g = Entity(
            Float32(self.goal_cell % self.w) + 0.5,
            Float32(self.goal_cell // self.w) + 0.5,
            0.0, 0.0, 0.5, 0.5, GOAL,
        )
        self.entities.append(g^)
        self._place_spikes()
        self._thin_walls()
        # Agent placement (before spike→entity conversion, matching the reference).
        self.agent = Entity(
            Float32(self.agent_cell % self.w) + 0.5,
            Float32(self.agent_cell // self.w) + 0.4,
            0.0, 0.0, 0.254, 0.4, PLAYER,
        )
        self.agent.smart_step = True
        self.agent.render_z = 1
        self._finalize()
        self.oob = CAVEWALL

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

        # agent-collision: goal → win, spike → death.
        for i in range(len(self.entities) - 1, -1, -1):
            var t = self.entities[i].type
            if t == PLAYER:
                continue
            if self._has_collision(
                self.entities[i].x, self.entities[i].y,
                self.entities[i].rx, self.entities[i].ry,
                self.agent.x, self.agent.y, self.agent.rx, self.agent.ry,
            ):
                if t == GOAL:
                    self.reward += GOAL_REWARD
                    self.level_complete = True
                    self.done = True
                elif t == SPIKE:
                    self.done = True

        for i in range(len(self.entities) - 1, -1, -1):
            if self.entities[i].will_erase or (
                self.entities[i].auto_erase and self._out_of_bounds(self.entities[i])
            ):
                _ = self.entities.pop(i)
        if self._out_of_bounds(self.agent):
            self.done = True

        # jumper tail: facing + movement trail + gravity.
        if self.action_vx > 0.0:
            self.agent.is_reflected = False
        if self.action_vx < 0.0:
            self.agent.is_reflected = True
        if abs(self.agent.vx) + abs(self.agent.vy) > 0.05:
            var tr = Entity(
                self.agent.x, self.agent.y - self.agent.ry * 0.5, 0.0, 0.01, 0.3, 0.2, TRAIL
            )
            tr.expire_time = 8
            tr.alpha = 0.5
            self.entities.append(tr^)
        if self.agent.vy > -2.0:
            self.agent.vy -= 0.15

        self.episode_reward += self.reward
        return self.reward

    # --- rendering (visual-approx; assets passed in) ---
    def render_obs(
        self, assets: JumperAssets, res: Int = RES, ss: Int = OBS_SS
    ) -> List[UInt8]:
        return downscale(self.render(assets, res * ss), res * ss, res)

    def render(self, assets: JumperAssets, out_res: Int = RES) -> List[UInt8]:
        var canvas = Canvas(out_res)
        canvas.fill(0, 0, 0)
        var visibility: Float32 = 12.0 if self.dist_mode == DIST_EASY else 16.0
        var center_x = self.agent.x
        var center_y = self.agent.y
        var view_dim = visibility
        var unit = Float32(out_res) / view_dim
        var x_off = unit * (center_x - view_dim / 2)
        var y_off = unit * (center_y - view_dim / 2)

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
        var vx0 = Int(floor(center_x - view_dim / 2)) - 1
        var vx1 = Int(floor(center_x + view_dim / 2)) + 1
        var vy0 = Int(floor(center_y - view_dim / 2)) - 1
        var vy1 = Int(floor(center_y + view_dim / 2)) + 1
        for x in range(vx0, vx1 + 1):
            for y in range(vy0, vy1 + 1):
                if x < 0 or x >= self.w or y < 0 or y >= self.h:
                    continue
                var t = self.grid.data[y * self.w + x]
                if t != CAVEWALL and t != CAVEWALL_TOP:
                    continue
                var sx = (Float32(x) - RENDER_EPS) * unit - x_off
                var sy = (view_dim - Float32(y + 1) - RENDER_EPS) * unit + y_off
                var sz = (1.0 + 2 * RENDER_EPS) * unit
                if t == CAVEWALL_TOP:
                    canvas.blit(assets.wall_top[wt], sx, sy, sz, sz)
                else:
                    canvas.blit(assets.wall_mid[wt], sx, sy, sz, sz)

        for k in range(len(self.entities)):
            ref e = self.entities[k]
            var ex = (e.x - e.rx) * unit - x_off
            var ey = (view_dim - (e.y + e.ry)) * unit + y_off
            var ew = 2 * e.rx * unit
            var eh = 2 * e.ry * unit
            if e.type == GOAL:
                canvas.blit(assets.goal, ex, ey, ew, eh)
            elif e.type == SPIKE:
                canvas.blit(assets.spike, ex, ey, ew, eh)

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
