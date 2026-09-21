"""Bigfish game — eat smaller fish to grow, avoid bigger ones (port of `bigfish.cpp`).

Open-water game (no maze): the player fish swims in a 20×20 world, eats fish
smaller than itself (grows a little each time), and dies if it touches a bigger
one. Eat `FISH_QUOTA=30` to complete the level. Reuses the entity substrate; the
new corner it exercises is free-swimming **drifting** fish (non-smart `Entity.step`
horizontal motion, auto-erased off-screen), agent **growth**, and a sprite
**aspect ratio** that feeds the eat-collision (fish are wider than tall).

`game_reset`/`game_step` replay the exact RNG order (see `docs/PROCGEN_BIGFISH_SCOPE.md`).
Level-exact + visual-approx. P0+P1 = reset+step parity; render/env in P2.
"""

from std.math import floor, ceil, sqrt, pow
from std.memory import ArcPointer

from .procgen_env import ProcgenGame

from ..core.entity import Entity
from ..core.randgen import RandGen
from ..core.assets import Sprite, load_sprite, load_sprites
from ..core.rasterizer import Canvas, RES, downscale
from ..core.object_ids import SPACE, WALL_OBJ, PLAYER, INVALID_OBJ

comptime FISH = 2
comptime FISH_MIN_R: Float32 = 0.25
comptime FISH_MAX_R: Float32 = 2.0
comptime FISH_QUOTA = 30
comptime POSITIVE_REWARD: Float32 = 1.0
comptime COMPLETION_BONUS: Float32 = 10.0
comptime A_R: Float32 = 0.4  # base agent radius (base-reset spawn draws)
comptime MIXRATE: Float32 = 0.5
comptime MAXSPEED: Float32 = 0.5
comptime WATER_BG = 7  # water_backgrounds (resources.cpp)
comptime WORLD = 20
comptime RENDER_EPS: Float32 = 0.02
comptime OBS_SS = 4

# DistributionMode: bigfish uses start_r 1 (Easy) vs .5 (Hard/…).
comptime DIST_EASY = 0
comptime DIST_HARD = 1


def _fish_aspect(theme: Int) -> Float32:
    # match_aspect_ratio uses float(double(w)/double(h)) of the theme's sprite:
    # fishTile_074 78×46, _078 112×88, _080 108×58.
    if theme == 0:
        return Float32(78.0 / 46.0)
    if theme == 1:
        return Float32(112.0 / 88.0)
    return Float32(108.0 / 58.0)


struct BigfishAssets(Movable):
    """Read-only sprite set (loaded once, shared via ArcPointer). Passed into
    render() so BigfishGame stays asset-free."""

    var player: Sprite  # fishTile_072
    var fish: List[Sprite]  # FISH themes: fishTile_074/078/080
    var backgrounds: List[Sprite]  # water_backgrounds (7)

    def __init__(out self, asset_root: String) raises:
        self.player = load_sprite(asset_root, "misc_assets/fishTile_072.png")
        var fp = List[String]()
        fp.append("misc_assets/fishTile_074.png")
        fp.append("misc_assets/fishTile_078.png")
        fp.append("misc_assets/fishTile_080.png")
        self.fish = load_sprites(asset_root, fp)
        var bp = List[String]()
        bp.append("water_backgrounds/water1.png")
        bp.append("water_backgrounds/water2.png")
        bp.append("water_backgrounds/water3.png")
        bp.append("water_backgrounds/water4.png")
        bp.append("water_backgrounds/underwater1.png")
        bp.append("water_backgrounds/underwater2.png")
        bp.append("water_backgrounds/underwater3.png")
        self.backgrounds = load_sprites(asset_root, bp)


struct BigfishGame(Copyable, Movable, ProcgenGame):
    # ─── ProcgenGame conformance glue (see games/procgen_env.mojo) ──────
    comptime AssetsT = BigfishAssets
    comptime DEFAULT_DIST = DIST_EASY
    comptime GYM_MAX_STEPS = 1000

    @staticmethod
    def load_assets(asset_root: String) raises -> BigfishAssets:
        return BigfishAssets(asset_root)

    @staticmethod
    def make(assets: ArcPointer[BigfishAssets], dist_mode: Int) -> Self:
        # The env owns the assets and passes them into the render calls.
        return Self(dist_mode)

    def is_done(self) -> Bool:
        return self.done

    def is_level_complete(self) -> Bool:
        return self.level_complete

    def gym_terminated(self) -> Bool:
        return self.done

    def pg_render_obs(self, assets: BigfishAssets) -> List[UInt8]:
        return self.render_obs(assets)

    def pg_render_obs_train(
        self, assets: BigfishAssets, res: Int, ss: Int
    ) -> List[UInt8]:
        return self.render_obs(assets, res, ss)

    def pg_render(self, assets: BigfishAssets, res: Int) -> List[UInt8]:
        return self.render(assets, res)

    var rand_gen: RandGen
    var w: Int
    var h: Int
    var dist_mode: Int
    var agent: Entity
    var entities: List[Entity]  # drifting fish
    var fish_eaten: Int
    var r_inc: Float32
    var bg_pct_x: Float32
    var background_index: Int
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
        self.w = WORLD
        self.h = WORLD
        self.dist_mode = dist_mode
        self.agent = Entity.make(0.5, 0.5, 1.0, PLAYER)
        self.entities = List[Entity]()
        self.fish_eaten = 0
        self.r_inc = 0.0
        self.bg_pct_x = 0.0
        self.background_index = 0
        self.action_vx = 0.0
        self.action_vy = 0.0
        self.step_rand_int = 0
        self.reward = 0.0
        self.episode_reward = 0.0
        self.done = False
        self.level_complete = False
        self.cur_time = 0

    # --- physics (grid-only sub_step; out_of_bounds_object == INVALID_OBJ) ---
    def _obj_from_floats(self, fi: Float32, fj: Float32) -> Int:
        if fi < 0.0 or fj < 0.0:
            return INVALID_OBJ
        var xi = Int(floor(fi))
        var yj = Int(floor(fj))
        if xi < 0 or xi >= self.w or yj < 0 or yj >= self.h:
            return INVALID_OBJ
        return SPACE  # interior is all water

    def _sub_step(self, mut obj: Entity, vx: Float32, vy: Float32) -> Bool:
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
                if t == WALL_OBJ or t == INVALID_OBJ:
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

    def _out_of_bounds(self, e: Entity) -> Bool:
        return (
            e.x + e.rx < 0.0
            or e.y + e.ry < 0.0
            or e.x - e.rx > Float32(self.w)
            or e.y - e.ry > Float32(self.h)
        )

    def reset(mut self, level_seed: Int):
        self.rand_gen.seed(level_seed)
        self.reward = 0.0
        self.episode_reward = 0.0
        self.done = False
        self.level_complete = False
        self.cur_time = 0
        self.fish_eaten = 0
        self.entities = List[Entity]()

        # BasicAbstractGame::game_reset base draws (bg = water_backgrounds).
        self.bg_pct_x = self.rand_gen.rand01()
        self.background_index = self.rand_gen.randn(WATER_BG)
        var ax = self.rand_gen.rand01() * (Float32(self.w) - 2 * A_R) + A_R
        _ = self.rand_gen.rand01() * (Float32(self.h) - 2 * A_R) + A_R  # ay (consumed)

        var start_r: Float32 = 1.0 if self.dist_mode == DIST_EASY else 0.5
        self.r_inc = (FISH_MAX_R - start_r) / Float32(FISH_QUOTA)
        self.agent = Entity.make(ax, 1.0 + start_r, start_r, PLAYER)

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

        # update_agent_velocity (base momentum + 0.9 decay).
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

        # Fish drift (non-smart Entity::step: x += vx).
        for i in range(len(self.entities)):
            self.entities[i].x += self.entities[i].vx

        # Agent-collision pass: eat smaller / die to bigger.
        for i in range(len(self.entities) - 1, -1, -1):
            if self.entities[i].type != FISH:
                continue
            var frx = self.entities[i].rx
            var fry = self.entities[i].ry
            if (
                abs(self.entities[i].x - self.agent.x) < (frx + self.agent.rx)
                and abs(self.entities[i].y - self.agent.y) < (fry + self.agent.ry)
            ):
                if frx > self.agent.rx:
                    self.done = True
                else:
                    self.reward += POSITIVE_REWARD
                    self.entities[i].will_erase = True
                    self.agent.rx += self.r_inc
                    self.agent.ry += self.r_inc
                    self.fish_eaten += 1

        # erase_if_needed: will_erase or (auto_erase && out of bounds).
        for i in range(len(self.entities) - 1, -1, -1):
            if self.entities[i].will_erase or (
                self.entities[i].auto_erase and self._out_of_bounds(self.entities[i])
            ):
                _ = self.entities.pop(i)

        if self._out_of_bounds(self.agent):
            self.done = True

        # --- BigFish::game_step: spawn a fish (~1/10 steps), quota, reflection ---
        if self.rand_gen.randn(10) == 1:
            var u = self.rand_gen.rand01()
            var ent_r = Float32(
                Float64(FISH_MAX_R - FISH_MIN_R) * pow(Float64(u), 1.4)
                + Float64(FISH_MIN_R)
            )
            var ent_y = self.rand_gen.rand01() * (Float32(self.h) - 2 * ent_r)
            var moves_right = self.rand_gen.rand01() < 0.5
            var sign_r: Float32 = 1.0 if moves_right else -1.0
            var ent_vx = (0.15 + self.rand_gen.rand01() * 0.25) * sign_r
            var ent_x = -ent_r if moves_right else Float32(self.w) + ent_r
            var theme = self.rand_gen.randn(3)
            var fish = Entity(ent_x, ent_y, ent_vx, 0.0, ent_r, ent_r, FISH)
            fish.image_theme = theme
            fish.ry = ent_r / _fish_aspect(theme)  # match_aspect_ratio
            fish.is_reflected = not moves_right
            self.entities.append(fish^)

        if self.fish_eaten >= FISH_QUOTA:
            self.done = True
            self.reward += COMPLETION_BONUS
            self.level_complete = True

        if self.action_vx > 0.0:
            self.agent.is_reflected = False
        if self.action_vx < 0.0:
            self.agent.is_reflected = True

        self.episode_reward += self.reward
        return self.reward

    # --- rendering (visual-approx; assets passed in) ---
    def render_obs(
        self, assets: BigfishAssets, res: Int = RES, ss: Int = OBS_SS
    ) -> List[UInt8]:
        return downscale(self.render(assets, res * ss), res * ss, res)

    def render(self, assets: BigfishAssets, out_res: Int = RES) -> List[UInt8]:
        var canvas = Canvas(out_res)
        canvas.fill(0, 0, 0)

        # Camera: center_agent=false → whole world (x_off == y_off == 0).
        var view_dim = Float32(self.w if self.w > self.h else self.h)
        var unit = Float32(out_res) / view_dim

        # Water background (panned by bg_pct_x).
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

        # Fish entities (themed, reflected by heading).
        for k in range(len(self.entities)):
            ref e = self.entities[k]
            if e.type != FISH:
                continue
            var ex = (e.x - e.rx) * unit
            var ey = (view_dim - (e.y + e.ry)) * unit
            canvas.blit(
                assets.fish[e.image_theme],
                ex,
                ey,
                2 * e.rx * unit,
                2 * e.ry * unit,
                e.is_reflected,
            )

        # Agent fish.
        var ax = (self.agent.x - self.agent.rx) * unit
        var ay = (view_dim - (self.agent.y + self.agent.ry)) * unit
        canvas.blit(
            assets.player,
            ax,
            ay,
            2 * self.agent.rx * unit,
            2 * self.agent.ry * unit,
            self.agent.is_reflected,
        )

        return canvas.px.copy()
