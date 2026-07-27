"""Bossfight game — boss shooter (port of `games/bossfight.cpp`).

A player ship fights a large moving boss that cycles shields up/down and fires
bullet patterns; shoot the boss when its shields are down (+1 per health chunk),
dodge its bullets + meteor barriers, kill it (+10). Reuses the projectile
substrate; adds boss AI (movement + shields + 4 attack modes), reflected bullets,
laser trails, and health/rounds. boss/shields/agent are separate; entities hold
bullets/trails/barriers/explosions. `game_reset`/`game_step` replay the exact RNG
order (4 rand01/step). Level-exact + visual-approx.
See `docs/PROCGEN_BOSSFIGHT_SCOPE.md`. P0+P1 = reset+step parity; render/env in P2.
"""

from std.math import floor, ceil, sqrt, cos, sin
from std.memory import ArcPointer

from .procgen_env import ProcgenGame

from ..core.entity import Entity
from ..core.randgen import RandGen
from ..core.assets import Sprite, load_sprite, load_sprites
from ..core.rasterizer import Canvas, RES, downscale
from ..core.object_ids import PLAYER, EXPLOSION

comptime PLAYER_BULLET = 1
comptime BOSS = 2
comptime SHIELDS = 3
comptime ENEMY_BULLET = 4
comptime LASER_TRAIL = 5
comptime REFLECTED_BULLET = 6
comptime BARRIER = 7

comptime BOSS_R: Float32 = 3.0
comptime NUM_ATTACK_MODES = 4
comptime NUM_LASER_THEMES = 3
comptime PLAYER_BULLET_VEL: Float32 = 1.0
comptime BOTTOM_MARGIN: Float32 = 6.0
comptime BOSS_VEL_TIMEOUT = 20
comptime BOSS_DAMAGED_TIMEOUT = 40
comptime COMPLETION_BONUS: Float32 = 10.0
comptime POSITIVE_REWARD: Float32 = 1.0
comptime A_R: Float32 = 0.4
comptime BG_COUNT = 13  # space_backgrounds
comptime MIXRATE: Float32 = 0.5
comptime MAXSPEED: Float32 = 0.85
comptime PI: Float32 = 3.14159265358979
comptime WORLD = 20
comptime RENDER_EPS: Float32 = 0.02
comptime OBS_SS = 4

comptime DIST_EASY = 0
comptime DIST_HARD = 1


def _boss_asp(t: Int) -> Float32:
    var w: List[Float32] = [93.0, 104.0, 103.0, 82.0]
    return w[t] / 84.0


def _player_asp(t: Int) -> Float32:
    var w: List[Float32] = [99.0, 99.0, 112.0, 98.0]
    return w[t] / 75.0


def _barrier_asp(t: Int) -> Float32:
    var w: List[Float32] = [215.0, 212.0, 214.0, 220.0, 101.0, 120.0, 89.0, 98.0]
    var h: List[Float32] = [211.0, 218.0, 227.0, 221.0, 84.0, 98.0, 82.0, 96.0]
    return w[t] / h[t]


struct BossfightAssets(Movable):
    """Read-only sprite set (shared via ArcPointer; passed into render())."""

    var players: List[Sprite]  # 4 themes (playerShip1_blue/green, ship2_orange, ship3_red)
    var bosses: List[Sprite]  # 4 themes (enemyShipBlack1/Blue2/Green3/Red4)
    var lasers: List[Sprite]  # 3 themes (green/red/blue) — enemy + player bullets
    var shield: Sprite
    var meteors: List[Sprite]  # 8 themes (spaceMeteors_001..004 + meteorGrey_big1..4)
    var explosion: Sprite
    var backgrounds: List[Sprite]  # space_backgrounds (13)

    def __init__(out self, asset_root: String) raises:
        var pp: List[String] = [
            "misc_assets/playerShip1_blue.png",
            "misc_assets/playerShip1_green.png",
            "misc_assets/playerShip2_orange.png",
            "misc_assets/playerShip3_red.png",
        ]
        self.players = load_sprites(asset_root, pp)
        var bp: List[String] = [
            "misc_assets/enemyShipBlack1.png",
            "misc_assets/enemyShipBlue2.png",
            "misc_assets/enemyShipGreen3.png",
            "misc_assets/enemyShipRed4.png",
        ]
        self.bosses = load_sprites(asset_root, bp)
        var lp: List[String] = [
            "misc_assets/laserGreen14.png",
            "misc_assets/laserRed11.png",
            "misc_assets/laserBlue09.png",
        ]
        self.lasers = load_sprites(asset_root, lp)
        self.shield = load_sprite(asset_root, "misc_assets/shield2.png")
        var mp = List[String]()
        for i in range(1, 5):
            mp.append("misc_assets/spaceMeteors_00" + String(i) + ".png")
        for i in range(1, 5):
            mp.append("misc_assets/meteorGrey_big" + String(i) + ".png")
        self.meteors = load_sprites(asset_root, mp)
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


struct BossfightGame(Copyable, Movable, ProcgenGame):
    # ─── ProcgenGame conformance glue (see games/procgen_env.mojo) ──────
    comptime AssetsT = BossfightAssets
    comptime DEFAULT_DIST = DIST_EASY
    comptime GYM_MAX_STEPS = 1000

    @staticmethod
    def load_assets(asset_root: String) raises -> BossfightAssets:
        return BossfightAssets(asset_root)

    @staticmethod
    def make(assets: ArcPointer[BossfightAssets], dist_mode: Int) -> Self:
        # The env owns the assets and passes them into the render calls.
        return Self(dist_mode)

    def is_done(self) -> Bool:
        return self.done

    def is_level_complete(self) -> Bool:
        return self.level_complete

    def gym_terminated(self) -> Bool:
        return self.done

    def pg_render_obs(self, assets: BossfightAssets) -> List[UInt8]:
        return self.render_obs(assets)

    def pg_render_obs_train(
        self, assets: BossfightAssets, res: Int, ss: Int
    ) -> List[UInt8]:
        return self.render_obs(assets, res, ss)

    def pg_render(self, assets: BossfightAssets, res: Int) -> List[UInt8]:
        return self.render(assets, res)

    var rand_gen: RandGen
    var w: Int
    var h: Int
    var dist_mode: Int
    var agent: Entity
    var boss: Entity
    var shields: Entity
    var entities: List[Entity]
    var attack_modes: List[Int]
    var last_fire: Int
    var time_to_swap: Int
    var invuln: Int
    var vuln: Int
    var num_rounds: Int
    var round_num: Int
    var round_health: Int
    var boss_vel_timeout: Int
    var curr_vel_timeout: Int
    var attack_mode: Int
    var player_laser: Int
    var boss_laser: Int
    var damaged_until: Int
    var shields_up: Bool
    var base_fire_prob: Float32
    var boss_bullet_vel: Float32
    var rand_pct: Float32
    var rand_fire_pct: Float32
    var rand_pct_x: Float32
    var rand_pct_y: Float32
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
        self.w = WORLD
        self.h = WORLD
        self.agent = Entity.make(0.5, 0.5, A_R, PLAYER)
        self.boss = Entity.make(10.0, 10.0, BOSS_R, BOSS)
        self.shields = Entity.make(10.0, 10.0, BOSS_R, SHIELDS)
        self.entities = List[Entity]()
        self.attack_modes = List[Int]()
        self.last_fire = 0
        self.time_to_swap = 0
        self.invuln = 0
        self.vuln = 0
        self.num_rounds = 0
        self.round_num = 0
        self.round_health = 1
        self.boss_vel_timeout = BOSS_VEL_TIMEOUT
        self.curr_vel_timeout = 0
        self.attack_mode = 0
        self.player_laser = 0
        self.boss_laser = 0
        self.damaged_until = 0
        self.shields_up = False
        self.base_fire_prob = 0.1
        self.boss_bullet_vel = 0.5
        self.rand_pct = 0.0
        self.rand_fire_pct = 0.0
        self.rand_pct_x = 0.0
        self.rand_pct_y = 0.0
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

    def _coll(
        self, ax: Float32, ay: Float32, arx: Float32, ary: Float32,
        bx: Float32, by: Float32, brx: Float32, bry: Float32,
    ) -> Bool:
        return abs(ax - bx) < (arx + brx) and abs(ay - by) < (ary + bry)

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
        # Probe tie-break: sxf = false on cmp==0 (step consumes one randint for RNG
        # parity but does not use it here); action overrides below for the agent.
        var sxf = cmp > 0.0
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

    def _boss_fire(mut self, r: Float32, vel: Float32, theta: Float32):
        var b = Entity(
            self.boss.x, self.boss.y, vel * cos(theta), vel * sin(theta), r, r, ENEMY_BULLET
        )
        b.image_theme = self.boss_laser
        b.expire_time = 50
        self.entities.append(b^)

    def _prepare_boss(mut self):
        self.shields_up = True
        self.curr_vel_timeout = self.boss_vel_timeout
        self.time_to_swap = self.invuln
        self.attack_mode = self.attack_modes[self.round_num % len(self.attack_modes)]
        self.boss.vx = 0.0
        self.boss.vy = 0.0

    def _active_attack(mut self):
        if self.attack_mode == 0:
            if self.cur_time % 8 == 0:
                for i in range(5):
                    self._boss_fire(0.5, self.boss_bullet_vel, PI * 1.5 + Float32(i - 2) * PI / 8)
        elif self.attack_mode == 1:
            if self.cur_time % 5 == 0:
                var k = self.cur_time // 5
                k = abs(8 - (k % 16))
                for i in range(4):
                    self._boss_fire(
                        0.5, self.boss_bullet_vel,
                        PI * (1.25 + 0.5 * Float32(k) / 8.0) + Float32(i) * PI / 2,
                    )
        elif self.attack_mode == 2:
            if self.cur_time % 10 == 0:
                var off = self.rand_pct * 2 * PI
                for i in range(8):
                    self._boss_fire(0.5, self.boss_bullet_vel, 2 * PI / 8 * Float32(i) + off)
        elif self.attack_mode == 3:
            if self.cur_time % 4 == 0:
                self._boss_fire(0.5, self.boss_bullet_vel, PI * (1 + self.rand_pct))

    def _has_any_coll(self, ex: Float32, ey: Float32, erx: Float32, ery: Float32) -> Bool:
        if self._coll(ex, ey, erx, ery, self.agent.x, self.agent.y, self.agent.rx, self.agent.ry):
            return True
        if self._coll(ex, ey, erx, ery, self.boss.x, self.boss.y, self.boss.rx, self.boss.ry):
            return True
        for i in range(len(self.entities)):
            ref o = self.entities[i]
            if not o.avoids_collisions and self._coll(ex, ey, erx, ery, o.x, o.y, o.rx, o.ry):
                return True
        return False

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

        self.damaged_until = 0
        self.last_fire = 0
        self.boss_bullet_vel = 0.5 if self.dist_mode == DIST_EASY else 0.75
        var max_extra = 1 if self.dist_mode == DIST_EASY else 3

        self.boss = Entity.make(Float32(self.w) / 2, Float32(self.h) / 2, BOSS_R, BOSS)
        self.boss.image_theme = self.rand_gen.randn(4)
        self.boss.ry = self.boss.rx / _boss_asp(self.boss.image_theme)
        self.shields = Entity(
            self.boss.x, self.boss.y, 0.0, 0.0, 1.2 * self.boss.rx, 1.2 * self.boss.ry, SHIELDS
        )

        self.boss_vel_timeout = BOSS_VEL_TIMEOUT
        self.base_fire_prob = 0.1
        self.round_health = self.rand_gen.randn(9) + 1
        self.num_rounds = 1 + self.rand_gen.randn(5)
        self.invuln = 2 + self.rand_gen.randn(max_extra + 1)
        self.vuln = 500
        self.boss.health = Float32(self.round_health * self.num_rounds)

        self.agent.image_theme = self.rand_gen.randn(4)
        self.player_laser = self.rand_gen.randn(NUM_LASER_THEMES)
        self.boss_laser = self.rand_gen.randn(NUM_LASER_THEMES)

        self.attack_modes = List[Int]()
        for _ in range(self.num_rounds):
            self.attack_modes.append(self.rand_gen.randn(NUM_ATTACK_MODES))
        self.round_num = 0
        self._prepare_boss()

        self.agent.rx = 0.75
        self.agent.ry = 0.75 / _player_asp(self.agent.image_theme)
        # reposition_agent: avoid boss + shields (both centered on the boss).
        self.agent.x = self.rand_gen.rand01() * (Float32(self.w) - 2 * self.agent.rx) + self.agent.rx
        self.agent.y = self.rand_gen.rand01() * (Float32(self.h) - 2 * self.agent.ry) + self.agent.ry
        var rc = 0
        while (
            self._coll(self.boss.x, self.boss.y, self.boss.rx, self.boss.ry, self.agent.x, self.agent.y, self.agent.rx, self.agent.ry)
            or self._coll(self.shields.x, self.shields.y, self.shields.rx, self.shields.ry, self.agent.x, self.agent.y, self.agent.rx, self.agent.ry)
        ) and rc < 100:
            self.agent.x = self.rand_gen.rand01() * (Float32(self.w) - 2 * self.agent.rx) + self.agent.rx
            self.agent.y = self.rand_gen.rand01() * (Float32(self.h) - 2 * self.agent.ry) + self.agent.ry
            rc += 1
        self.agent.y = self.agent.ry

        _ = self.rand_gen.rand01() > 0.5  # barriers_moves_right (randbool)

        var nb = self.rand_gen.randn(3) + 1
        for _ in range(nb):
            var br: Float32 = 0.6
            var minby = 2 * self.agent.ry + br + 0.5
            var ey = self.rand_gen.rand01() * (BOTTOM_MARGIN - minby - br) + minby
            var ex = self.rand_gen.rand01() * (Float32(self.w) - 2 * br) + br
            var theme = self.rand_gen.randn(8)
            var ry = br / _barrier_asp(theme)
            if not self._has_any_coll(ex, ey, br, ry):
                var e = Entity(ex, ey, 0.0, 0.0, br, ry, BARRIER)
                e.image_theme = theme
                e.health = 3.0
                e.collides_with_entities = True
                self.entities.append(e^)

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
        self.action_vy = Float32(move % 3 - 1)

        self.agent.vx = (1.0 - MIXRATE) * self.agent.vx + MIXRATE * MAXSPEED * self.action_vx
        self.agent.vy = (1.0 - MIXRATE) * self.agent.vy + MIXRATE * MAXSPEED * self.action_vy
        self.agent.vx *= 0.9
        self.agent.vy *= 0.9
        var a = self.agent.copy()
        self._basic_step_object(a)
        self.agent = a^

        self.boss.x += self.boss.vx
        self.boss.y += self.boss.vy
        for i in range(len(self.entities)):
            self.entities[i].x += self.entities[i].vx
            self.entities[i].y += self.entities[i].vy
            self.entities[i].life_time += 1
            if (
                self.entities[i].expire_time > 0
                and self.entities[i].life_time > self.entities[i].expire_time
            ):
                self.entities[i].will_erase = True

        # agent-collision: boss / barrier / enemy bullet → done.
        if self._coll(self.boss.x, self.boss.y, self.boss.rx, self.boss.ry, self.agent.x, self.agent.y, self.agent.rx, self.agent.ry):
            self.done = True
        for i in range(len(self.entities) - 1, -1, -1):
            var t = self.entities[i].type
            if (t == BARRIER or t == ENEMY_BULLET) and self._coll(
                self.entities[i].x, self.entities[i].y, self.entities[i].rx, self.entities[i].ry,
                self.agent.x, self.agent.y, self.agent.rx, self.agent.ry,
            ):
                self.done = True

        # cwe collision: player bullets vs shields/boss; barriers vs bullets/trails.
        var n = len(self.entities)
        for i in range(n - 1, -1, -1):
            if not self.entities[i].collides_with_entities or self.entities[i].will_erase:
                continue
            if self.entities[i].type == PLAYER_BULLET:
                # vs shields (reflect) then vs boss (damage).
                if not self.entities[i].will_erase and self._coll(
                    self.entities[i].x, self.entities[i].y, self.entities[i].rx, self.entities[i].ry,
                    self.shields.x, self.shields.y, self.shields.rx, self.shields.ry,
                ):
                    if self.shields_up:
                        var th = PI * (1.25 + 0.5 * self.rand_pct)
                        self.entities[i].type = REFLECTED_BULLET
                        self.entities[i].vy = PLAYER_BULLET_VEL * sin(th) * 0.5
                        self.entities[i].vx = PLAYER_BULLET_VEL * cos(th) * 0.5
                        self.entities[i].expire_time = 4
                        self.entities[i].life_time = 0
                if not self.entities[i].will_erase and self.entities[i].type == PLAYER_BULLET and self._coll(
                    self.entities[i].x, self.entities[i].y, self.entities[i].rx, self.entities[i].ry,
                    self.boss.x, self.boss.y, self.boss.rx, self.boss.ry,
                ):
                    if not self.shields_up:
                        self.boss.health -= 1.0
                        self.entities[i].will_erase = True
                        if Int(self.boss.health) % self.round_health == 0:
                            self.reward += POSITIVE_REWARD
                            if self.boss.health == 0.0:
                                self.done = True
                                self.reward += COMPLETION_BONUS
                                self.level_complete = True
                            else:
                                self.round_num += 1
                                self._prepare_boss()
                                self.curr_vel_timeout = BOSS_DAMAGED_TIMEOUT
                                self.damaged_until = self.cur_time + BOSS_DAMAGED_TIMEOUT
                        var bvx = self.boss.vx
                        var bvy = self.boss.vy
                        var sx = self.entities[i].x
                        var sy = self.entities[i].y
                        var sr = self.entities[i].rx
                        var ex = Entity(sx, sy, bvx, bvy, 0.5 * sr, 0.5 * sr, EXPLOSION)
                        self.entities.append(ex^)
            elif self.entities[i].type == BARRIER:
                for j in range(n - 1, -1, -1):
                    if i == j:
                        continue
                    var tt = self.entities[j].type
                    if self.entities[j].will_erase:
                        continue
                    if (tt == ENEMY_BULLET or tt == PLAYER_BULLET) and self._coll(
                        self.entities[i].x, self.entities[i].y, self.entities[i].rx, self.entities[i].ry,
                        self.entities[j].x, self.entities[j].y, self.entities[j].rx, self.entities[j].ry,
                    ):
                        self.entities[j].will_erase = True
                        var jx = self.entities[j].x
                        var jy = self.entities[j].y
                        var jr = self.entities[j].rx
                        var ex = Entity(jx, jy, 0.0, 0.0, 0.5 * jr, 0.5 * jr, EXPLOSION)
                        self.entities.append(ex^)
                    elif tt == LASER_TRAIL and self._coll(
                        self.entities[i].x, self.entities[i].y, self.entities[i].rx, self.entities[i].ry,
                        self.entities[j].x, self.entities[j].y, self.entities[j].rx, self.entities[j].ry,
                    ):
                        self.entities[j].will_erase = True

        for i in range(len(self.entities) - 1, -1, -1):
            if self.entities[i].will_erase or (
                self.entities[i].auto_erase and self._out_of_bounds(self.entities[i])
            ):
                _ = self.entities.pop(i)
        if self._out_of_bounds(self.agent):
            self.done = True

        # --- bossfight game_step tail ---
        self.shields.x = self.boss.x
        self.shields.y = self.boss.y
        self.rand_pct = self.rand_gen.rand01()
        self.rand_fire_pct = self.rand_gen.rand01()
        self.rand_pct_x = self.rand_gen.rand01()
        self.rand_pct_y = self.rand_gen.rand01()

        if self.curr_vel_timeout <= 0:
            var dx = self.rand_pct_x * (Float32(self.w) - 2 * BOSS_R) + BOSS_R
            var dy = self.rand_pct_y * (Float32(self.h) - 2 * BOSS_R - BOTTOM_MARGIN) + BOSS_R + BOTTOM_MARGIN
            self.boss.vx = (dx - self.boss.x) / Float32(self.boss_vel_timeout)
            self.boss.vy = (dy - self.boss.y) / Float32(self.boss_vel_timeout)
            self.curr_vel_timeout = self.boss_vel_timeout
            if self.time_to_swap > 0:
                self.time_to_swap -= 1
            else:
                self.time_to_swap = self.vuln if self.shields_up else self.invuln
                self.shields_up = not self.shields_up
        else:
            self.curr_vel_timeout -= 1

        if self.special_action == 1 and (self.cur_time - self.last_fire) >= 3:
            var b = Entity(self.agent.x, self.agent.y, 0.0, PLAYER_BULLET_VEL, 0.25, 0.25, PLAYER_BULLET)
            b.image_theme = self.player_laser
            b.collides_with_entities = True
            b.expire_time = 25
            self.entities.append(b^)
            self.last_fire = self.cur_time

        if self.damaged_until >= self.cur_time:
            if self.cur_time % 3 == 0:
                var px = self.boss.x + (2 * self.rand_pct_x - 1) * self.boss.rx
                var py = self.boss.y + (2 * self.rand_pct_y - 1) * self.boss.ry
                self.entities.append(Entity(px, py, 0.0, 0.0, 0.75, 0.75, EXPLOSION))
        elif self.shields_up:
            self._active_attack()
        else:
            if self.rand_fire_pct < self.base_fire_prob:
                self._boss_fire(0.5, self.boss_bullet_vel, PI * (1 + self.rand_pct))

        # laser trails behind each enemy bullet.
        var m2 = len(self.entities)
        for i in range(m2 - 1, -1, -1):
            if self.entities[i].type == ENEMY_BULLET:
                var tr = Entity(
                    self.entities[i].x, self.entities[i].y,
                    self.entities[i].vx * 0.5, self.entities[i].vy * 0.5,
                    self.entities[i].rx, self.entities[i].ry, LASER_TRAIL,
                )
                tr.image_theme = self.boss_laser
                tr.expire_time = 8
                self.entities.append(tr^)

        self.episode_reward += self.reward
        return self.reward

    # --- rendering (visual-approx; assets passed in) ---
    def render_obs(
        self, assets: BossfightAssets, res: Int = RES, ss: Int = OBS_SS
    ) -> List[UInt8]:
        return downscale(self.render(assets, res * ss), res * ss, res)

    def render(self, assets: BossfightAssets, out_res: Int = RES) -> List[UInt8]:
        var canvas = Canvas(out_res)
        canvas.fill(0, 0, 0)

        # Space background: tile wide + slow horizontal parallax scroll by cur_time.
        ref bg = assets.backgrounds[self.background_index]
        var bg_ar = Float32(bg.w) / Float32(bg.h)
        var bg_w = Float32(out_res) * bg_ar
        var scroll = Float32(self.cur_time) * 2.0
        var start = -(scroll % bg_w) - bg_w
        var bx = start
        while bx < Float32(out_res):
            canvas.blit(bg, bx, 0.0, bg_w, Float32(out_res))
            bx += bg_w

        var view_dim = Float32(self.w if self.w > self.h else self.h)
        var unit = Float32(out_res) / view_dim

        # Boss (behind shields + bullets).
        var bxo = (self.boss.x - self.boss.rx) * unit
        var byo = (view_dim - (self.boss.y + self.boss.ry)) * unit
        canvas.blit(
            assets.bosses[self.boss.image_theme], bxo, byo,
            2 * self.boss.rx * unit, 2 * self.boss.ry * unit,
        )
        # Shields (only when up).
        if self.shields_up:
            var sxo = (self.shields.x - self.shields.rx) * unit
            var syo = (view_dim - (self.shields.y + self.shields.ry)) * unit
            canvas.blit(
                assets.shield, sxo, syo,
                2 * self.shields.rx * unit, 2 * self.shields.ry * unit,
            )

        # Entities (bullets, trails, meteors, explosions).
        for k in range(len(self.entities)):
            ref e = self.entities[k]
            var t = e.type
            var ex = (e.x - e.rx) * unit
            var ey = (view_dim - (e.y + e.ry)) * unit
            var ew = 2 * e.rx * unit
            var eh = 2 * e.ry * unit
            if t == BARRIER:
                canvas.blit(assets.meteors[e.image_theme], ex, ey, ew, eh)
            elif t == ENEMY_BULLET or t == PLAYER_BULLET or t == LASER_TRAIL or t == REFLECTED_BULLET:
                canvas.blit(assets.lasers[e.image_theme], ex, ey, ew, eh)
            else:  # EXPLOSION
                canvas.blit(assets.explosion, ex, ey, ew, eh)

        # Player ship (on top).
        var ax = (self.agent.x - self.agent.rx) * unit
        var ay = (view_dim - (self.agent.y + self.agent.ry)) * unit
        canvas.blit(
            assets.players[self.agent.image_theme], ax, ay,
            2 * self.agent.rx * unit, 2 * self.agent.ry * unit,
        )

        return canvas.px.copy()
