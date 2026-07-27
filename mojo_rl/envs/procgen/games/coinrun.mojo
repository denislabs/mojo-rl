"""Coinrun game — platformer (port of `games/coinrun.cpp`).

Side-scroller that ESTABLISHES THE PLATFORMER SUBSTRATE (gravity + jump +
has_support + one-way land-on-top collision). Run right along generated ground,
jump gaps + over saws/enemies/lava, grab the coin at the far right (+10). Any
touch of an ENEMY/SAW, falling in LAVA, or going out-of-bounds = death.

Reuses the Entity substrate + `basic_step_object`/`sub_step` (now with the vertical
axis live under gravity). The agent is a separate field (entities[0] in the
reference, stepped last; enemies never block/reflect the agent and crates don't
move, so keeping it separate is behavior-neutral). `game_reset`/`game_step` replay
the exact RNG order. Level-exact + visual-approx. See `docs/PROCGEN_COINRUN_SCOPE.md`.
P0+P1 = reset+step parity; render/env in P2.
"""

from std.memory import ArcPointer

from .procgen_env import ProcgenGame
from std.math import floor, ceil, sqrt

from ..core.entity import Entity
from ..core.randgen import RandGen
from ..core.assets import Sprite, load_sprite, load_sprites
from ..core.rasterizer import Canvas, RES, downscale
from ..core.object_ids import PLAYER, SPACE, WALL_OBJ, TRAIL

comptime GOAL = 1
comptime SAW = 2
comptime SAW2 = 3
comptime ENEMY = 5
comptime ENEMY1 = 6
comptime ENEMY2 = 7
comptime PLAYER_JUMP = 9
comptime PLAYER_RIGHT1 = 12
comptime PLAYER_RIGHT2 = 13
comptime WALL_MID = 15
comptime WALL_TOP = 16
comptime LAVA_MID = 17
comptime LAVA_TOP = 18
comptime ENEMY_BARRIER = 19
comptime CRATE = 20

comptime W = 64
comptime H = 64
comptime OOB = WALL_MID  # out_of_bounds_object
comptime GOAL_REWARD: Float32 = 10.0
comptime POS_EPS: Float32 = -0.001
comptime A_R: Float32 = 0.4
comptime NUM_GROUND_THEMES = 6
comptime NUM_PLAYER_THEMES = 5
comptime NUM_ENEMY_THEMES = 9
comptime NUM_CRATE_THEMES = 4
comptime BG_COUNT = 37  # platform_backgrounds (value only; not in RNG-stream parity)
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


def _clip_abs(x: Float32, y: Float32) -> Float32:
    if x > y:
        return y
    if x < -y:
        return -y
    return x


struct CoinrunAssets(Movable):
    """Read-only sprite set (shared via ArcPointer; passed into render())."""

    var players: List[Sprite]  # alien{color}_stand, 5 themes
    var players_walk: List[Sprite]  # alien{color}_walk1, 5 themes
    var enemies: List[Sprite]  # 9 walking-enemy themes
    var coin: Sprite
    var ground_top: List[Sprite]  # {theme}Mid.png (WALL_TOP), 6 themes
    var ground_mid: List[Sprite]  # {theme}Center.png (WALL_MID), 6 themes
    var lava: Sprite
    var lava_top: Sprite
    var saw: Sprite
    var crates: List[Sprite]  # 4 crate variants
    var backgrounds: List[Sprite]  # platform_backgrounds (37)

    def __init__(out self, asset_root: String) raises:
        var colors: List[String] = ["Beige", "Blue", "Green", "Pink", "Yellow"]
        var pstand = List[String]()
        var pwalk = List[String]()
        for i in range(len(colors)):
            var c = colors[i]
            pstand.append("kenney/Players/128x256/" + c + "/alien" + c + "_stand.png")
            pwalk.append("kenney/Players/128x256/" + c + "/alien" + c + "_walk1.png")
        self.players = load_sprites(asset_root, pstand)
        self.players_walk = load_sprites(asset_root, pwalk)
        var enames: List[String] = [
            "slimeBlock", "slimePurple", "slimeBlue", "slimeGreen", "mouse",
            "snail", "ladybug", "wormGreen", "wormPink",
        ]
        var ep = List[String]()
        for i in range(len(enames)):
            ep.append("kenney/Enemies/" + enames[i] + ".png")
        self.enemies = load_sprites(asset_root, ep)
        self.coin = load_sprite(asset_root, "kenney/Items/coinGold.png")
        var gthemes: List[String] = ["dirt", "grass", "planet", "sand", "snow", "stone"]
        var gdirs: List[String] = ["Dirt", "Grass", "Planet", "Sand", "Snow", "Stone"]
        var gtop = List[String]()
        var gmid = List[String]()
        for i in range(len(gthemes)):
            gtop.append("kenney/Ground/" + gdirs[i] + "/" + gthemes[i] + "Mid.png")
            gmid.append("kenney/Ground/" + gdirs[i] + "/" + gthemes[i] + "Center.png")
        self.ground_top = load_sprites(asset_root, gtop)
        self.ground_mid = load_sprites(asset_root, gmid)
        self.lava = load_sprite(asset_root, "kenney/Tiles/lava.png")
        self.lava_top = load_sprite(asset_root, "kenney/Tiles/lavaTop_low.png")
        self.saw = load_sprite(asset_root, "kenney/Enemies/sawHalf.png")
        var cp: List[String] = [
            "kenney/Tiles/boxCrate.png", "kenney/Tiles/boxCrate_double.png",
            "kenney/Tiles/boxCrate_single.png", "kenney/Tiles/boxCrate_warning.png",
        ]
        self.crates = load_sprites(asset_root, cp)
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


struct CoinrunGame(Copyable, Movable, ProcgenGame):
    # ─── ProcgenGame conformance glue (see games/procgen_env.mojo) ──────
    comptime AssetsT = CoinrunAssets
    comptime DEFAULT_DIST = DIST_EASY
    comptime GYM_MAX_STEPS = 1000

    @staticmethod
    def load_assets(asset_root: String) raises -> CoinrunAssets:
        return CoinrunAssets(asset_root)

    @staticmethod
    def make(assets: ArcPointer[CoinrunAssets], dist_mode: Int) -> Self:
        # The env owns the assets and passes them into the render calls.
        return Self(dist_mode)

    def is_done(self) -> Bool:
        return self.done

    def is_level_complete(self) -> Bool:
        return self.level_complete

    def gym_terminated(self) -> Bool:
        return self.done

    def pg_render_obs(self, assets: CoinrunAssets) -> List[UInt8]:
        return self.render_obs(assets)

    def pg_render_obs_train(
        self, assets: CoinrunAssets, res: Int, ss: Int
    ) -> List[UInt8]:
        return self.render_obs(assets, res, ss)

    def pg_render(self, assets: CoinrunAssets, res: Int) -> List[UInt8]:
        return self.render(assets, res)

    var rand_gen: RandGen
    var dist_mode: Int
    var grid: List[Int]
    var agent: Entity
    var entities: List[Entity]
    var gravity: Float32
    var max_jump: Float32
    var air_control: Float32
    var maxspeed: Float32
    var mixrate: Float32
    var has_support: Bool
    var facing_right: Bool
    var is_on_crate: Bool
    var last_agent_y: Float32
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
        self.grid = List[Int]()
        self.grid.resize(W * H, 0)
        self.agent = Entity(1.5, 1.5787, 0.0, 0.0, 0.5, 0.5787, PLAYER)
        self.entities = List[Entity]()
        self.gravity = 0.2
        self.max_jump = 1.5
        self.air_control = 0.15
        self.maxspeed = 0.5
        self.mixrate = 0.2
        self.has_support = False
        self.facing_right = True
        self.is_on_crate = False
        self.last_agent_y = 0.0
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

    def _gget(self, x: Int, y: Int) -> Int:
        if x < 0 or x >= W or y < 0 or y >= H:
            return OOB
        return self.grid[y * W + x]

    def _obj_ff(self, fi: Float32, fj: Float32) -> Int:
        if fi < 0.0 or fj < 0.0:
            return OOB
        return self._gget(Int(floor(fi)), Int(floor(fj)))

    def _fill_elem(mut self, x: Int, y: Int, dx: Int, dy: Int, e: Int):
        for j in range(dx):
            for k in range(dy):
                self._gset(x + j, y + k, e)

    @staticmethod
    def _is_wall(t: Int) -> Bool:
        return t == WALL_MID or t == WALL_TOP

    @staticmethod
    def _is_lava(t: Int) -> Bool:
        return t == LAVA_MID or t == LAVA_TOP

    @staticmethod
    def _can_support(o: Int) -> Bool:
        return CoinrunGame._is_wall(o) or o == OOB

    @staticmethod
    def _is_blocked(src_type: Int, target: Int, is_h: Bool) -> Bool:
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
        bx: Float32, by: Float32, brx: Float32, bry: Float32, m: Float32,
    ) -> Bool:
        return abs(ax - bx) < (arx + brx + m) and abs(ay - by) < (ary + bry + m)

    def _out_of_bounds(self, e: Entity) -> Bool:
        return (
            e.x + e.rx < 0.0 or e.y + e.ry < 0.0
            or e.x - e.rx > Float32(W) or e.y - e.ry > Float32(H)
        )

    # --- physics (platformer substrate) ---
    def _push_obj(
        mut self, src_x: Float32, src_y: Float32, src_rx: Float32, src_ry: Float32,
        mut obj: Entity, is_h: Bool, depth: Int, self_idx: Int,
    ) -> Bool:
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
            block = self._sub_step(obj, t_vx, t_vy, depth + 1, self_idx)
        if is_h:
            obj.vx = 0.0
        else:
            obj.vy = 0.0
        return block

    def _sub_step(
        mut self, mut obj: Entity, vx: Float32, vy: Float32, depth: Int, self_idx: Int
    ) -> Bool:
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
                if self._is_blocked(obj.type, t, is_h):
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

        # Entity-collision pass: crate one-way platform (agent), enemy reflect.
        var block2 = False
        for k in range(len(self.entities) - 1, -1, -1):
            if k == self_idx or self.entities[k].will_erase:
                continue
            var mt = self.entities[k].type
            var mx = self.entities[k].x
            var my = self.entities[k].y
            var mrx = self.entities[k].rx
            var mry = self.entities[k].ry
            if not self._has_collision(
                obj.x, obj.y, obj.rx, obj.ry, mx, my, mrx, mry, POS_EPS
            ):
                continue
            var curr_block = False
            # is_blocked_ents: crate one-way (references agent state) else base.
            if mt == CRATE and not is_h:
                var av = obj.vy if obj.type == PLAYER else self.agent.vy
                var blocked_crate = True
                if av >= 0.0:
                    blocked_crate = False
                elif self.action_vy < 0.0:
                    blocked_crate = False
                elif self.last_agent_y < (my + mry + self.agent.ry):
                    blocked_crate = False
                if blocked_crate:
                    self.is_on_crate = True
                    curr_block = True
            elif self._is_blocked(obj.type, mt, is_h):
                curr_block = True
            elif self._will_reflect(obj.type, mt):
                if is_h:
                    var rsum = mrx + obj.rx
                    if vx > 0.0:
                        obj.x += -2 * (rsum - (mx - obj.x))
                    else:
                        obj.x += 2 * (rsum + (mx - obj.x))
                    obj.vx = -obj.vx
                else:
                    var rsum = mry + obj.ry
                    if vy > 0.0:
                        obj.y += -2 * (rsum - (my - obj.y))
                    else:
                        obj.y += 2 * (rsum + (my - obj.y))
                    obj.vy = -obj.vy
            if curr_block:
                _ = self._push_obj(mx, my, mrx, mry, obj, is_h, depth, self_idx)
                block2 = True
        return block or block2

    def _basic_step_object(mut self, mut obj: Entity, self_idx: Int):
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
                bx = self._sub_step(obj, obj.vx * pct, 0.0, 0, self_idx)
                by = self._sub_step(obj, 0.0, obj.vy * pct, 0, self_idx)
            else:
                by = self._sub_step(obj, 0.0, obj.vy * pct, 0, self_idx)
                bx = self._sub_step(obj, obj.vx * pct, 0.0, 0, self_idx)
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
        self.has_support = (
            self.is_on_crate or self._can_support(b1) or self._can_support(b2)
        ) and self.agent.vy == 0.0
        self.is_on_crate = False
        if self.action_vy == 1.0 and not self.has_support:
            self.action_vy = 0.0

    def _update_agent_velocity(mut self):
        var a = self.agent.copy()
        var mx = self.mixrate if self.has_support else (self.mixrate * self.air_control)
        a.vx = (1.0 - mx) * a.vx + mx * self.maxspeed * self.action_vx
        if abs(a.vx) < mx * self.maxspeed:
            a.vx = 0.0
        if self.action_vy > 0.0:
            a.vy = self.max_jump
        elif self.has_support:
            a.vy += 0.2 * self.action_vy
        if not (self.has_support and self.action_vy > 0.0):
            a.vy -= self.gravity
            a.vy = _clip_abs(a.vy, self.max_jump)
        self.agent = a^

    def _check_grid_collisions_agent(mut self):
        var minx = Int(self.agent.x - (self.agent.rx + POS_EPS))
        var maxx = Int(self.agent.x + (self.agent.rx + POS_EPS))
        var miny = Int(self.agent.y - (self.agent.ry + POS_EPS))
        var maxy = Int(self.agent.y + (self.agent.ry + POS_EPS))
        for x in range(minx, maxx + 1):
            for y in range(miny, maxy + 1):
                var gt = self._obj_ff(Float32(x), Float32(y))
                if gt == SPACE:
                    continue
                if gt == GOAL:
                    self.reward += GOAL_REWARD
                    self.done = True
                    self.level_complete = True
                elif self._is_lava(gt):
                    self.done = True

    # --- level gen ---
    def _create_saw(mut self, x: Int, y: Int):
        self.entities.append(
            Entity(Float32(x) + 0.5, Float32(y) + 0.5, 0.0, 0.0, 0.5, 0.5, SAW)
        )

    def _create_enemy(mut self, x: Int, y: Int):
        var s = 2 * self.rand_gen.randn(2) - 1
        var e = Entity(
            Float32(x) + 0.5, Float32(y) + 0.5, 0.15 * Float32(s), 0.0, 0.5, 0.5, ENEMY
        )
        e.smart_step = True
        e.image_type = ENEMY1
        e.render_z = 1
        e.image_theme = self.rand_gen.randn(NUM_ENEMY_THEMES)
        self.entities.append(e^)

    def _create_crate(mut self, x: Int, y: Int):
        var e = Entity(
            Float32(x) + 0.5, Float32(y) + 0.5, 0.0, 0.0, 0.5, 0.5, CRATE
        )
        e.image_theme = self.rand_gen.randn(NUM_CRATE_THEMES)
        self.entities.append(e^)

    def _fill_ground(mut self, x: Int, y: Int, dx: Int, dy: Int):
        self._fill_elem(x, y, dx, dy - 1, WALL_MID)
        self._fill_elem(x, y + dy - 1, dx, 1, WALL_TOP)

    def _fill_lava(mut self, x: Int, y: Int, dx: Int, dy: Int):
        self._fill_elem(x, y, dx, dy - 1, LAVA_MID)
        self._fill_elem(x, y + dy - 1, dx, 1, LAVA_TOP)

    def _init_floor_and_walls(mut self):
        self._fill_elem(0, 0, W, 1, WALL_TOP)
        self._fill_elem(0, 0, 1, H, WALL_MID)
        self._fill_elem(W - 1, 0, 1, H, WALL_MID)
        self._fill_elem(0, H - 1, W, 1, WALL_MID)

    def _generate(mut self):
        var dif = self.rand_gen.randn(3) + 1
        var num_sections = self.rand_gen.randn(dif) + dif
        var curr_x = 5
        var curr_y = 1
        var pit_threshold = dif
        var danger_type = self.rand_gen.randn(3)
        var allow_monsters = self.dist_mode != DIST_EASY

        var _max_dy = self.max_jump * self.max_jump / (2 * self.gravity)
        var _max_dx = self.maxspeed * 2 * self.max_jump / self.gravity
        var max_dy = Int(_max_dy - 0.5)
        var max_dx = Int(_max_dx - 0.5)

        for _ in range(num_sections):
            if curr_x + 15 >= W:
                break
            var dy = self.rand_gen.randn(4) + 1 + dif // 3
            if dy > max_dy:
                dy = max_dy
            if curr_y >= 20:
                dy *= -1
            elif curr_y >= 5 and self.rand_gen.randn(2) == 1:
                dy *= -1
            var dx = self.rand_gen.randn(2 * dif) + 3 + dif // 3
            curr_y += dy
            if curr_y < 1:
                curr_y = 1
            var use_pit = (
                dx > 7 and curr_y > 3 and self.rand_gen.randn(20) >= pit_threshold
            )
            if use_pit:
                var x1 = self.rand_gen.randn(3) + 1
                var x2 = self.rand_gen.randn(3) + 1
                var pit_width = dx - x1 - x2
                if pit_width > max_dx:
                    pit_width = max_dx
                    x2 = dx - x1 - pit_width
                self._fill_ground(curr_x, 0, x1, curr_y)
                self._fill_ground(curr_x + dx - x2, 0, x2, curr_y)
                var lava_height = self.rand_gen.randn(curr_y - 3) + 1
                if danger_type == 0:
                    self._fill_lava(curr_x + x1, 1, pit_width, lava_height)
                elif danger_type == 1:
                    for ei in range(pit_width):
                        self._create_saw(curr_x + x1 + ei, 1)
                elif danger_type == 2:
                    for ei in range(pit_width):
                        self._create_enemy(curr_x + x1 + ei, 1)
                if pit_width > 4:
                    var x3: Int
                    var w1: Int
                    if pit_width == 5:
                        x3 = 1 + self.rand_gen.randn(2)
                        w1 = 1 + self.rand_gen.randn(2)
                    elif pit_width == 6:
                        x3 = 2 + self.rand_gen.randn(2)
                        w1 = 1 + self.rand_gen.randn(2)
                    else:
                        x3 = 2 + self.rand_gen.randn(2)
                        var x4 = 2 + self.rand_gen.randn(2)
                        w1 = pit_width - x3 - x4
                    self._fill_ground(curr_x + x1 + x3, curr_y - 1, w1, 1)
            else:
                self._fill_ground(curr_x, 0, dx, curr_y)
                var ob1_x = -1
                var ob2_x = -1
                if self.rand_gen.randn(10) < (2 * dif) and dx > 3:
                    ob1_x = curr_x + self.rand_gen.randn(dx - 2) + 1
                    self._create_saw(ob1_x, curr_y)
                if (
                    self.rand_gen.randn(10) < dif and dx > 3
                    and max_dx >= 4 and allow_monsters
                ):
                    ob2_x = curr_x + self.rand_gen.randn(dx - 2) + 1
                    self._create_enemy(ob2_x, curr_y)
                for _ in range(2):
                    var cx = curr_x + self.rand_gen.randn(dx - 2) + 1
                    if self.rand_gen.randn(2) == 1 and ob1_x != cx and ob2_x != cx:
                        var ph = self.rand_gen.randn(3) + 1
                        for jj in range(ph):
                            self._create_crate(cx, curr_y + jj)
            if not self._is_wall(self._gget(curr_x - 1, curr_y)):
                self._gset(curr_x - 1, curr_y, ENEMY_BARRIER)
            curr_x += dx
            self._gset(curr_x, curr_y, ENEMY_BARRIER)

        self._gset(curr_x, curr_y, GOAL)
        self.goal_x = curr_x
        self.goal_y = curr_y
        self._fill_ground(curr_x, 0, 1, curr_y)
        self._fill_elem(curr_x + 1, 0, W - curr_x - 1, H, WALL_MID)

    def reset(mut self, level_seed: Int):
        self.rand_gen.seed(level_seed)
        for i in range(W * H):
            self.grid[i] = 0
        self.entities = List[Entity]()
        self.reward = 0.0
        self.episode_reward = 0.0
        self.done = False
        self.level_complete = False
        self.cur_time = 0

        self.bg_pct_x = self.rand_gen.rand01()
        self.background_index = self.rand_gen.randn(BG_COUNT)
        self._fill_elem(0, 0, W, H, SPACE)

        self.gravity = 0.2
        self.max_jump = 1.5
        self.air_control = 0.15
        self.maxspeed = 0.5
        self.has_support = False
        self.facing_right = True

        if self.dist_mode == DIST_EASY:
            self.agent.image_theme = 0
            self.wall_theme = 0
            self.background_index = 0
        else:
            self.agent.image_theme = self.rand_gen.randn(NUM_PLAYER_THEMES)
            self.wall_theme = self.rand_gen.randn(NUM_GROUND_THEMES)

        self.agent.rx = 0.5
        self.agent.ry = 0.5787
        self.agent.x = 1.0 + self.agent.rx
        self.agent.y = 1.0 + self.agent.ry
        self.agent.vx = 0.0
        self.agent.vy = 0.0
        self.agent.smart_step = True
        self.last_agent_y = self.agent.y
        self.is_on_crate = False

        self._init_floor_and_walls()
        self._generate()

    def step(mut self, action: Int) -> Float32:
        self.cur_time += 1
        self.step_rand_int = self.rand_gen.randint(0, 1000000)
        var move = action % 9
        self.special_action = 0
        if action >= 9:
            self.special_action = action - 8
            move = 4
        self.action_vx = 0.0
        self.action_vy = 0.0
        self.reward = 0.0
        self.done = False
        self.level_complete = False

        self._set_action_xy(move)
        self._update_agent_velocity()

        # step_entities: entities (reverse) then the agent (idx 0, stepped last).
        for i in range(len(self.entities) - 1, -1, -1):
            if self.entities[i].smart_step:
                var e = self.entities[i].copy()
                self._basic_step_object(e, i)
                self.entities[i] = e^
            var e2 = self.entities[i].copy()
            e2.step()
            self.entities[i] = e2^
        var a = self.agent.copy()
        self._basic_step_object(a, -1)
        a.step()
        self.agent = a^

        # collisions: agent-death (enemy/saw) + grid (goal/lava).
        for i in range(len(self.entities) - 1, -1, -1):
            var mt = self.entities[i].type
            if mt != ENEMY and mt != SAW:
                continue
            if self._has_collision(
                self.entities[i].x, self.entities[i].y,
                self.entities[i].rx, self.entities[i].ry,
                self.agent.x, self.agent.y, self.agent.rx, self.agent.ry,
                self.entities[i].collision_margin,
            ):
                self.done = True
        self._check_grid_collisions_agent()

        # erase (expired trails, off-screen entities).
        for i in range(len(self.entities) - 1, -1, -1):
            if self.entities[i].will_erase or (
                self.entities[i].auto_erase and self._out_of_bounds(self.entities[i])
            ):
                _ = self.entities.pop(i)
        if self._out_of_bounds(self.agent):
            self.done = True

        # coinrun tail: facing + enemy trails.
        if self.action_vx > 0.0:
            self.agent.is_reflected = False
        if self.action_vx < 0.0:
            self.agent.is_reflected = True
        var n = len(self.entities)
        for i in range(n - 1, -1, -1):
            if self.entities[i].type == ENEMY:
                var tr = Entity(
                    self.entities[i].x, self.entities[i].y - self.entities[i].ry * 0.5,
                    0.0, 0.01, 0.3, 0.2, TRAIL,
                )
                tr.expire_time = 8
                tr.alpha = 0.5
                self.entities.append(tr^)
        self.last_agent_y = self.agent.y

        self.episode_reward += self.reward
        return self.reward

    # --- rendering (visual-approx; assets passed in) ---
    def render_obs(
        self, assets: CoinrunAssets, res: Int = RES, ss: Int = OBS_SS
    ) -> List[UInt8]:
        return downscale(self.render(assets, res * ss), res * ss, res)

    def render(self, assets: CoinrunAssets, out_res: Int = RES) -> List[UInt8]:
        var canvas = Canvas(out_res)
        canvas.fill(0, 0, 0)

        # Camera: side-scroller centered on the agent (visibility 13).
        var visibility: Float32 = 13.0
        var center_x = self.agent.x
        var center_y = self.agent.y
        var view_dim = visibility
        var unit = Float32(out_res) / view_dim
        var x_off = unit * (center_x - view_dim / 2)
        var y_off = unit * (center_y - view_dim / 2)

        # Background (platform, panned by bg_pct_x; fixed to the world, follows camera).
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

        # Grid tiles (only those inside the camera window).
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
                    canvas.blit(assets.ground_mid[wt], sx, sy, sz, sz)
                elif t == WALL_TOP:
                    canvas.blit(assets.ground_top[wt], sx, sy, sz, sz)
                elif t == LAVA_MID:
                    canvas.blit(assets.lava, sx, sy, sz, sz)
                elif t == LAVA_TOP:
                    canvas.blit(assets.lava_top, sx, sy, sz, sz)
                elif t == GOAL:
                    canvas.blit(assets.coin, sx, sy, sz, sz)

        # Entities (crates, saws, enemies).
        for k in range(len(self.entities)):
            ref e = self.entities[k]
            var t = e.type
            var ex = (e.x - e.rx) * unit - x_off
            var ey = (view_dim - (e.y + e.ry)) * unit + y_off
            var ew = 2 * e.rx * unit
            var eh = 2 * e.ry * unit
            if t == CRATE:
                canvas.blit(assets.crates[e.image_theme], ex, ey, ew, eh)
            elif t == SAW:
                canvas.blit(assets.saw, ex, ey, ew, eh)
            elif t == ENEMY:
                canvas.blit(assets.enemies[e.image_theme], ex, ey, ew, eh, e.is_reflected)

        # Player (tall alien; preserve sprite aspect, anchored at the feet).
        var still = abs(self.agent.vx) < 0.01 and self.action_vx == 0.0 and self.has_support
        var idx = self.agent.image_theme
        var pw = 2 * self.agent.rx * unit
        var feet = (view_dim - (self.agent.y - self.agent.ry)) * unit + y_off
        var px = (self.agent.x - self.agent.rx) * unit - x_off
        if still:
            ref ps = assets.players[idx]
            var ph = pw * (Float32(ps.h) / Float32(ps.w))
            canvas.blit(ps, px, feet - ph, pw, ph, self.agent.is_reflected)
        else:
            ref ps = assets.players_walk[idx]
            var ph = pw * (Float32(ps.h) / Float32(ps.w))
            canvas.blit(ps, px, feet - ph, pw, ph, self.agent.is_reflected)

        return canvas.px.copy()
