"""Plunder game — cannon shooter (port of `games/plunder.cpp`).

A cannon ship at the bottom fires cannonballs UP at target ships sailing across
lanes; hit a target (+1, refill "juice") and avoid the decoys (juice penalty).
Juice decays over time — if it hits 0 you lose. Hit `target_quota=20` targets → win.

Reuses the projectile substrate from starpilot (firing, collides_with_entities loop,
handle_collision, drifting ships, explosions). `game_reset`/`game_step` replay the
exact RNG order (incl reposition collision-retry loops). Level-exact + visual-approx.
See `docs/PROCGEN_PLUNDER_SCOPE.md`. P0+P1 = reset+step parity; render/env in P2.
"""

from std.math import floor, ceil, sqrt
from std.memory import ArcPointer

from .procgen_env import ProcgenGame

from ..core.entity import Entity
from ..core.randgen import RandGen
from ..core.assets import Sprite, load_sprite, load_sprites
from ..core.rasterizer import Canvas, RES, downscale
from ..core.object_ids import PLAYER, EXPLOSION

comptime PLAYER_BULLET = 1
comptime TARGET_LEGEND = 2
comptime TARGET_BACKGROUND = 3
comptime PANEL = 6
comptime SHIP = 7

comptime COMPLETION_BONUS: Float32 = 10.0
comptime POSITIVE_REWARD: Float32 = 1.0
comptime A_R: Float32 = 0.4
comptime BG_COUNT = 4  # water_surface_backgrounds
comptime MIXRATE: Float32 = 0.5
comptime MAXSPEED: Float32 = 0.85
comptime SHIP_ASPECT: Float32 = 113.0 / 66.0
comptime PI: Float32 = 3.14159265358979
comptime WORLD = 20
comptime RENDER_EPS: Float32 = 0.02
comptime OBS_SS = 4

comptime DIST_EASY = 0
comptime DIST_HARD = 1


struct PlunderAssets(Movable):
    """Read-only sprite set (shared via ArcPointer; passed into render())."""

    var ships: List[Sprite]  # ship_1..6 (themes; also the cannon ship + legend)
    var cannonball: Sprite
    var panel: Sprite
    var target_bg: Sprite
    var explosion: Sprite
    var backgrounds: List[Sprite]  # water_surface_backgrounds (4)

    def __init__(out self, asset_root: String) raises:
        var sp = List[String]()
        for i in range(1, 7):
            sp.append("misc_assets/ship_" + String(i) + ".png")
        self.ships = load_sprites(asset_root, sp)
        self.cannonball = load_sprite(asset_root, "misc_assets/cannonBall.png")
        self.panel = load_sprite(asset_root, "misc_assets/panel_wood.png")
        self.target_bg = load_sprite(asset_root, "misc_assets/target_red2.png")
        self.explosion = load_sprite(asset_root, "misc_assets/explosion1.png")
        var bp = List[String]()
        bp.append("water_backgrounds/water1.png")
        bp.append("water_backgrounds/water2.png")
        bp.append("water_backgrounds/water3.png")
        bp.append("water_backgrounds/water4.png")
        self.backgrounds = load_sprites(asset_root, bp)


struct PlunderGame(Copyable, Movable, ProcgenGame):
    # ─── ProcgenGame conformance glue (see games/procgen_env.mojo) ──────
    comptime AssetsT = PlunderAssets
    comptime DEFAULT_DIST = DIST_EASY
    comptime GYM_MAX_STEPS = 1000

    @staticmethod
    def load_assets(asset_root: String) raises -> PlunderAssets:
        return PlunderAssets(asset_root)

    @staticmethod
    def make(assets: ArcPointer[PlunderAssets], dist_mode: Int) -> Self:
        # The env owns the assets and passes them into the render calls.
        return Self(dist_mode)

    def is_done(self) -> Bool:
        return self.done

    def is_level_complete(self) -> Bool:
        return self.level_complete

    def gym_terminated(self) -> Bool:
        return self.done

    def pg_render_obs(self, assets: PlunderAssets) -> List[UInt8]:
        return self.render_obs(assets)

    def pg_render_obs_train(
        self, assets: PlunderAssets, res: Int, ss: Int
    ) -> List[UInt8]:
        return self.render_obs(assets, res, ss)

    def pg_render(self, assets: PlunderAssets, res: Int) -> List[UInt8]:
        return self.render(assets, res)

    var rand_gen: RandGen
    var w: Int
    var h: Int
    var dist_mode: Int
    var agent: Entity
    var entities: List[Entity]
    var image_perm: List[Int]
    var target_bools: List[Bool]
    var lane_dirs: List[Bool]
    var lane_vels: List[Float32]
    var num_lanes: Int
    var num_cur: Int
    var targets_hit: Int
    var target_quota: Int
    var last_fire: Int
    var juice: Float32
    var r_scale: Float32
    var spawn_prob: Float32
    var legend_r: Float32
    var min_agent_x: Float32
    var action_vx: Float32
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
        self.w = WORLD
        self.h = WORLD
        self.agent = Entity.make(0.5, 0.5, A_R, PLAYER)
        self.entities = List[Entity]()
        self.image_perm = List[Int]()
        self.target_bools = List[Bool]()
        self.lane_dirs = List[Bool]()
        self.lane_vels = List[Float32]()
        self.num_lanes = 5
        self.num_cur = 2
        self.targets_hit = 0
        self.target_quota = 20
        self.last_fire = 0
        self.juice = 1.0
        self.r_scale = 1.0
        self.spawn_prob = 0.06
        self.legend_r = 2.0
        self.min_agent_x = 0.0
        self.action_vx = 0.0
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

    def _agent_has_coll(self) -> Bool:
        for i in range(len(self.entities)):
            ref e = self.entities[i]
            if e.type != PLAYER and self._coll(
                e.x, e.y, e.rx, e.ry, self.agent.x, self.agent.y, self.agent.rx, self.agent.ry
            ):
                return True
        return False

    def _has_any_coll(self, ex: Float32, ey: Float32, erx: Float32, ery: Float32) -> Bool:
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
        self.entities[i].rx *= self.entities[i].grow_rate
        self.entities[i].ry *= self.entities[i].grow_rate

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
        var ay = self.rand_gen.rand01() * (Float32(self.h) - 2 * A_R) + A_R
        self.agent = Entity.make(ax, ay, A_R, PLAYER)

        self.juice = 1.0
        self.targets_hit = 0
        self.target_quota = 20
        self.spawn_prob = 0.06
        self.r_scale = 1.5 if self.dist_mode == DIST_EASY else 1.0
        var nts = 6

        var idxs = List[Int]()
        for i in range(nts):
            idxs.append(i)
        self.image_perm = self.rand_gen.choose_n(idxs, nts)
        self.num_cur = 2
        self.target_bools = List[Bool]()
        for _ in range(nts):
            self.target_bools.append(False)
        for i in range(self.num_cur // 2):
            self.target_bools[self.image_perm[i]] = True

        self.lane_dirs = List[Bool]()
        self.lane_vels = List[Float32]()
        for _ in range(self.num_lanes):
            self.lane_dirs.append(self.rand_gen.rand01() < 0.5)
            self.lane_vels.append(0.15 + 0.1 * self.rand_gen.rand01())

        var num_panels = 0 if self.dist_mode == DIST_EASY else self.rand_gen.randn(4)
        for _ in range(num_panels):
            var px = PlunderGame._rand_pos(self.rand_gen, 1.2, 0.0, Float32(self.w))
            var py = PlunderGame._rand_pos(self.rand_gen, 0.5, 5.0, 10.0)
            var c = 0
            while (
                self._agent_has_coll_at(px, py, 1.2, 0.5)
                or self._has_any_coll(px, py, 1.2, 0.5)
            ) and c < 100:
                px = PlunderGame._rand_pos(self.rand_gen, 1.2, 0.0, Float32(self.w))
                py = PlunderGame._rand_pos(self.rand_gen, 0.5, 5.0, 10.0)
                c += 1
            self.entities.append(Entity(px, py, 0.0, 0.0, 1.2, 0.5, PANEL))

        self.legend_r = 2.0
        self.entities.append(
            Entity(self.legend_r, self.legend_r, 0.0, 0.0, self.legend_r, self.legend_r, TARGET_BACKGROUND)
        )
        var lg = Entity(
            self.legend_r, self.legend_r, 0.0, 0.0,
            self.r_scale * 1.5, (self.r_scale * 1.5) / SHIP_ASPECT, TARGET_LEGEND,
        )
        lg.image_theme = self.image_perm[0]
        self.entities.append(lg^)

        self.last_fire = 0
        self.agent.rx = self.r_scale
        self.agent.ry = self.r_scale / SHIP_ASPECT
        self.agent.rotation = -1 * PI / 2
        self.agent.image_theme = self.image_perm[
            self.rand_gen.randn(self.num_cur // 2) + self.num_cur // 2
        ]
        # reposition_agent (draw x/y, re-draw while colliding).
        self.agent.x = self.rand_gen.rand01() * (Float32(self.w) - 2 * self.agent.rx) + self.agent.rx
        self.agent.y = self.rand_gen.rand01() * (Float32(self.h) - 2 * self.agent.ry) + self.agent.ry
        var ac = 0
        while self._agent_has_coll() and ac < 100:
            self.agent.x = self.rand_gen.rand01() * (Float32(self.w) - 2 * self.agent.rx) + self.agent.rx
            self.agent.y = self.rand_gen.rand01() * (Float32(self.h) - 2 * self.agent.ry) + self.agent.ry
            ac += 1
        self.agent.y = 1.0 + self.agent.ry
        self.min_agent_x = 2 * self.legend_r + self.agent.rx
        if self.agent.x < self.min_agent_x:
            self.agent.x = self.min_agent_x

    def _agent_has_coll_at(
        self, px: Float32, py: Float32, prx: Float32, pry: Float32
    ) -> Bool:
        # has_agent_collision for a candidate entity vs the current agent.
        return self._coll(
            px, py, prx, pry, self.agent.x, self.agent.y, self.agent.rx, self.agent.ry
        )

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
        self.action_vx = Float32(move // 3 - 1)  # set_action_xy: vy = 0

        self.agent.vx = (
            (1.0 - MIXRATE) * self.agent.vx + MIXRATE * MAXSPEED * self.action_vx
        )
        self.agent.vy = (1.0 - MIXRATE) * self.agent.vy
        self.agent.vx *= 0.9
        self.agent.vy *= 0.9
        var a = self.agent.copy()
        self._basic_step_object(a)
        self.agent = a^

        for i in range(len(self.entities)):
            self._drift(i)

        # collides_with_entities: bullet vs ship/panel.
        var n = len(self.entities)
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
                        if tt == SHIP:
                            var th = self.entities[j].image_theme
                            self.entities[j].will_erase = True
                            self.entities[i].will_erase = True
                            if self.target_bools[th]:
                                self.targets_hit += 1
                                self.reward += POSITIVE_REWARD
                                self.juice += 0.1
                            else:
                                self.juice -= 0.1
                        elif tt == PANEL:
                            self.entities[i].will_erase = True
                        if self.entities[j].will_erase:
                            var tx = self.entities[j].x
                            var ty = self.entities[j].y
                            var tvx = self.entities[j].vx
                            var tvy = self.entities[j].vy
                            var tr = self.entities[j].rx
                            var ex = Entity(tx, ty, tvx / 2, tvy / 2, 0.5 * tr, 0.5 * tr, EXPLOSION)
                            ex.grow_rate = 1.4
                            ex.expire_time = 4
                            self.entities.append(ex^)

        for i in range(len(self.entities) - 1, -1, -1):
            if self.entities[i].will_erase or (
                self.entities[i].auto_erase and self._out_of_bounds(self.entities[i])
            ):
                _ = self.entities.pop(i)
        if self._out_of_bounds(self.agent):
            self.done = True

        # --- plunder game_step tail ---
        self.juice -= 0.0015
        if self.rand_gen.rand01() < self.spawn_prob:
            var er = self.r_scale
            var lane = self.rand_gen.randn(self.num_lanes)
            var ey = (
                (Float32(lane) * 0.11 + 0.4) * (Float32(self.h) / 2 - er)
                + Float32(self.h) / 2
            )
            var mr = self.lane_dirs[lane]
            var mdir: Float32 = 1.0 if mr else -1.0
            var evx = self.lane_vels[lane] * mdir
            var theme = self.image_perm[self.rand_gen.randn(self.num_cur)]
            var sx = -er if mr else Float32(self.w) + er
            if not self._has_any_coll(sx, ey, er, er / SHIP_ASPECT):
                var s = Entity(sx, ey, evx, 0.0, er, er / SHIP_ASPECT, SHIP)
                s.image_theme = theme
                s.is_reflected = not mr
                self.entities.append(s^)

        if self.special_action == 1 and (self.cur_time - self.last_fire) >= 3:
            var b = Entity(self.agent.x, self.agent.y, 0.0, 1.0, 0.25, 0.25, PLAYER_BULLET)
            b.collides_with_entities = True
            b.expire_time = 50
            self.entities.append(b^)
            self.last_fire = self.cur_time
            self.juice -= 0.02

        if self.juice <= 0.0:
            self.done = True
        elif self.juice >= 1.0:
            self.juice = 1.0

        if self.targets_hit >= self.target_quota:
            self.done = True
            self.reward += COMPLETION_BONUS
            self.level_complete = True

        if self.agent.x < self.min_agent_x:
            self.agent.x = self.min_agent_x

        self.episode_reward += self.reward
        return self.reward

    # --- rendering (visual-approx; assets passed in) ---
    def render_obs(
        self, assets: PlunderAssets, res: Int = RES, ss: Int = OBS_SS
    ) -> List[UInt8]:
        return downscale(self.render(assets, res * ss), res * ss, res)

    def render(self, assets: PlunderAssets, out_res: Int = RES) -> List[UInt8]:
        var canvas = Canvas(out_res)
        canvas.fill(0, 0, 0)

        var view_dim = Float32(self.w if self.w > self.h else self.h)
        var unit = Float32(out_res) / view_dim

        # Water-surface background (panned by bg_pct_x).
        ref bg = assets.backgrounds[self.background_index]
        var bg_ar = Float32(bg.w) / Float32(bg.h)
        var world_ar = Float32(self.w) / Float32(self.h)
        var offset_x = self.bg_pct_x * (bg_ar - world_ar)
        var main_w = Float32(self.w) * unit
        canvas.blit(bg, main_w * (-offset_x), 0.0, main_w * (bg_ar / world_ar), Float32(self.h) * unit)

        # Entities (panels, legend, ships, cannonballs, explosions).
        for k in range(len(self.entities)):
            ref e = self.entities[k]
            var t = e.type
            var ex = (e.x - e.rx) * unit
            var ey = (view_dim - (e.y + e.ry)) * unit
            var ew = 2 * e.rx * unit
            var eh = 2 * e.ry * unit
            if t == SHIP or t == TARGET_LEGEND:
                canvas.blit(assets.ships[e.image_theme], ex, ey, ew, eh, e.is_reflected)
            elif t == PANEL:
                canvas.blit(assets.panel, ex, ey, ew, eh)
            elif t == TARGET_BACKGROUND:
                canvas.blit(assets.target_bg, ex, ey, ew, eh)
            elif t == PLAYER_BULLET:
                canvas.blit(assets.cannonball, ex, ey, ew, eh)
            else:  # EXPLOSION
                canvas.blit(assets.explosion, ex, ey, ew, eh)

        # Cannon ship (agent).
        var ax = (self.agent.x - self.agent.rx) * unit
        var ay = (view_dim - (self.agent.y + self.agent.ry)) * unit
        canvas.blit(
            assets.ships[self.agent.image_theme], ax, ay,
            2 * self.agent.rx * unit, 2 * self.agent.ry * unit,
        )

        return canvas.px.copy()
