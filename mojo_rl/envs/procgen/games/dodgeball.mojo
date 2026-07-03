"""Dodgeball game — top-down arena shooter (port of `games/dodgeball.cpp`).

Navigate an arena carved by LAVA_WALL walls (own recursive room-split gen), throw
soccer balls to kill all enemies (+2 each), then reach the door (+10). Death on
touching an enemy, an enemy ball, or a lava wall. Reuses the projectile substrate +
entity-reflection sub_step (enemies bounce off lava walls). `game_reset`/`game_step`
replay the exact RNG order. Level-exact + visual-approx.
See `docs/PROCGEN_DODGEBALL_SCOPE.md`. Reset is split into helpers (compiler-hang lesson).
"""

from std.math import floor, ceil, sqrt, atan2

from ..core.entity import Entity
from ..core.randgen import RandGen
from ..core.assets import Sprite, load_sprite, load_sprites
from ..core.rasterizer import Canvas, RES, downscale
from ..core.object_ids import PLAYER, SPACE, EXPLOSION

comptime LAVA_WALL = 1
comptime PLAYER_BALL = 3
comptime ENEMY = 4
comptime DOOR = 5
comptime ENEMY_BALL = 6
comptime DOOR_OPEN = 7
comptime DUST_CLOUD = 8
comptime OOB_WALL = 10

comptime PI: Float32 = 3.14159265358979
comptime ENEMY_VEL: Float32 = 0.05
comptime BALL_V_ROT: Float32 = PI * 0.23
comptime ENEMY_REWARD: Float32 = 2.0
comptime COMPLETION_BONUS: Float32 = 10.0
comptime NUM_ENEMY_THEMES = 7
comptime ENEMY_FIRE_DELAY = 50
comptime BG_COUNT = 9  # topdown_backgrounds (value only; not in RNG-stream parity)
comptime OBS_SS = 4
comptime RENDER_EPS: Float32 = 0.0

comptime DIST_EASY = 0
comptime DIST_HARD = 1


@fieldwise_init
struct Rect(ImplicitlyCopyable, Movable):
    var x: Float32
    var y: Float32
    var w: Float32
    var h: Float32


struct DodgeballAssets(Movable):
    """Read-only sprite set (shared via ArcPointer; passed into render())."""

    var player: Sprite
    var player_ball: Sprite
    var enemies: List[Sprite]  # 11 character themes
    var enemy_ball: Sprite
    var door: Sprite
    var door_open: Sprite
    var lava: Sprite
    var dust: List[Sprite]  # 9 spaceEffect themes
    var backgrounds: List[Sprite]  # topdown_backgrounds (9)

    def __init__(out self, asset_root: String) raises:
        self.player = load_sprite(asset_root, "misc_assets/character12.png")
        self.player_ball = load_sprite(asset_root, "misc_assets/ball_soccer1.png")
        var ep = List[String]()
        for i in range(1, 12):
            ep.append("misc_assets/character" + String(i) + ".png")
        self.enemies = load_sprites(asset_root, ep)
        self.enemy_ball = load_sprite(asset_root, "misc_assets/ball_soccer2.png")
        self.door = load_sprite(asset_root, "misc_assets/blockRed.png")
        self.door_open = load_sprite(asset_root, "misc_assets/blockGreen.png")
        self.lava = load_sprite(asset_root, "misc_assets/tileStone_slope2.png")
        var dp = List[String]()
        for i in range(1, 10):
            dp.append("misc_assets/spaceEffect" + String(i) + ".png")
        self.dust = load_sprites(asset_root, dp)
        var names: List[String] = [
            "floortiles", "backgrounddetailed1", "backgrounddetailed2",
            "backgrounddetailed3", "backgrounddetailed4", "backgrounddetailed5",
            "backgrounddetailed6", "backgrounddetailed7", "backgrounddetailed8",
        ]
        var bp = List[String]()
        for i in range(len(names)):
            bp.append("topdown_backgrounds/" + names[i] + ".png")
        self.backgrounds = load_sprites(asset_root, bp)


struct DodgeballGame(Copyable, Movable):
    var rand_gen: RandGen
    var dist_mode: Int
    var w: Int
    var h: Int
    var oob: Int
    var agent: Entity
    var entities: List[Entity]
    var rooms: List[Rect]
    var mixrate: Float32
    var maxspeed: Float32
    var min_dim: Float32
    var hard_min_dim: Float32
    var ball_vscale: Float32
    var ball_r: Float32
    var last_fire_time: Int
    var num_enemies: Int
    var last_move_action: Int
    var special_action: Int
    var step_rand_int: Int
    var action_vx: Float32
    var action_vy: Float32
    var bg_pct_x: Float32
    var background_index: Int
    var exit_wall_choice: Int
    var reward: Float32
    var episode_reward: Float32
    var done: Bool
    var level_complete: Bool
    var cur_time: Int

    def __init__(out self, dist_mode: Int = DIST_EASY):
        self.rand_gen = RandGen()
        self.dist_mode = dist_mode
        self.w = 20
        self.h = 20
        self.oob = OOB_WALL
        self.agent = Entity(0.4, 0.4, 0.0, 0.0, 1.0, 1.0, PLAYER)
        self.entities = List[Entity]()
        self.rooms = List[Rect]()
        self.mixrate = 0.5
        self.maxspeed = 0.75
        self.min_dim = 0.0
        self.hard_min_dim = 0.0
        self.ball_vscale = 0.0
        self.ball_r = 0.0
        self.last_fire_time = 0
        self.num_enemies = 0
        self.last_move_action = 4
        self.special_action = 0
        self.step_rand_int = 0
        self.action_vx = 0.0
        self.action_vy = 0.0
        self.bg_pct_x = 0.0
        self.background_index = 0
        self.exit_wall_choice = 0
        self.reward = 0.0
        self.episode_reward = 0.0
        self.done = False
        self.level_complete = False
        self.cur_time = 0

    # --- collision / geometry helpers ---
    def _obj_ff(self, fi: Float32, fj: Float32) -> Int:
        if fi < 0.0 or fj < 0.0:
            return self.oob
        var x = Int(floor(fi))
        var y = Int(floor(fj))
        if x < 0 or x >= self.w or y < 0 or y >= self.h:
            return self.oob
        return SPACE

    @staticmethod
    def _will_reflect(src_type: Int, target: Int, oob: Int) -> Bool:
        return src_type == ENEMY and (target == LAVA_WALL or target == oob)

    @staticmethod
    def _has_collision(
        ax: Float32, ay: Float32, arx: Float32, ary: Float32,
        bx: Float32, by: Float32, brx: Float32, bry: Float32, m: Float32,
    ) -> Bool:
        return abs(ax - bx) < (arx + brx + m) and abs(ay - by) < (ary + bry + m)

    def _out_of_bounds(self, e: Entity) -> Bool:
        return (
            e.x + e.rx < 0.0 or e.y + e.ry < 0.0
            or e.x - e.rx > Float32(self.w) or e.y - e.ry > Float32(self.h)
        )

    def _rand_pos(mut self, r: Float32, mn: Float32, mx: Float32) -> Float32:
        if mx - mn <= 2 * r:
            return (mx + mn) / 2
        return (mx - mn - 2 * r) * self.rand_gen.rand01() + r + mn

    def _coll_agent(self, ex: Float32, ey: Float32, erx: Float32, ery: Float32) -> Bool:
        return self._has_collision(
            ex, ey, erx, ery, self.agent.x, self.agent.y, self.agent.rx, self.agent.ry, 0.0
        )

    def _coll_any(self, ex: Float32, ey: Float32, erx: Float32, ery: Float32) -> Bool:
        for i in range(len(self.entities)):
            if self._has_collision(
                ex, ey, erx, ery, self.entities[i].x, self.entities[i].y,
                self.entities[i].rx, self.entities[i].ry, 0.0
            ):
                return True
        return False

    def _agent_has_coll(self) -> Bool:
        for i in range(len(self.entities)):
            if self.entities[i].type != PLAYER and self._has_collision(
                self.entities[i].x, self.entities[i].y,
                self.entities[i].rx, self.entities[i].ry,
                self.agent.x, self.agent.y, self.agent.rx, self.agent.ry, 0.0
            ):
                return True
        return False

    # --- physics ---
    def _sub_step(mut self, mut obj: Entity, vx: Float32, vy: Float32, self_idx: Int) -> Bool:
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
                if t == self.oob:
                    block = True
                if self._will_reflect(obj.type, t, self.oob):
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
        # Entity reflection: enemies bounce off LAVA_WALL entities.
        for k in range(len(self.entities) - 1, -1, -1):
            if k == self_idx or self.entities[k].will_erase:
                continue
            var mt = self.entities[k].type
            var mx = self.entities[k].x
            var my = self.entities[k].y
            var mrx = self.entities[k].rx
            var mry = self.entities[k].ry
            if not self._has_collision(
                obj.x, obj.y, obj.rx, obj.ry, mx, my, mrx, mry, -0.001
            ):
                continue
            if self._will_reflect(obj.type, mt, self.oob):
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
        return block

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
                bx = self._sub_step(obj, obj.vx * pct, 0.0, self_idx)
                by = self._sub_step(obj, 0.0, obj.vy * pct, self_idx)
            else:
                by = self._sub_step(obj, 0.0, obj.vy * pct, self_idx)
                bx = self._sub_step(obj, obj.vx * pct, 0.0, self_idx)
            if not bx:
                vx_pct += 1.0
            if not by:
                vy_pct += 1.0
            if bx and by:
                break
        obj.vx *= vx_pct / Float32(nsub)
        obj.vy *= vy_pct / Float32(nsub)

    def _update_agent_velocity(mut self):
        var a = self.agent.copy()
        a.vx = (1.0 - self.mixrate) * a.vx
        a.vy = (1.0 - self.mixrate) * a.vy
        a.vx += self.mixrate * self.maxspeed * self.action_vx
        a.vy += self.mixrate * self.maxspeed * self.action_vy
        a.vx *= 0.9
        a.vy *= 0.9
        self.agent = a^

    def _choose_vel(mut self, mut e: Entity):
        var vel = ENEMY_VEL * Float32(self.rand_gen.randn(2) * 2 - 1)
        if self.rand_gen.randn(2) == 0:
            e.vx = vel
            e.vy = 0.0
        else:
            e.vy = vel
            e.vx = 0.0
        e.spawn_time = self.rand_gen.randn(50) + 25

    # --- level gen (split into helpers) ---
    def _add_room(mut self, r: Rect):
        if (
            (r.w >= self.min_dim or r.h >= self.min_dim)
            and r.w >= self.hard_min_dim and r.h >= self.hard_min_dim
        ):
            self.rooms.append(r)

    def _split_room(mut self, room: Rect, thickness: Float32):
        var wsw = self.rand_gen.rand01() < 0.5
        var c2 = self.rand_gen.rand01() < 0.5
        if room.w < self.min_dim:
            wsw = False
        if room.h < self.min_dim:
            wsw = True
        var rx = room.x
        var ry = room.y
        var rw = room.w
        var rh = room.h
        var gap = 0.25 * Float32(self.rand_gen.randn(3) + 1)
        var pct = 1 - gap
        if not wsw:
            var wy: Float32
            var wh: Float32
            var remy: Float32
            if c2:
                wy = ry
                remy = ry + pct * rh
                wh = pct * rh
            else:
                wy = ry + (1 - pct) * rh
                remy = ry
                wh = pct * rh
            self.entities.append(
                Entity(rx + rw / 2, wy + wh / 2, 0.0, 0.0, thickness, wh / 2, LAVA_WALL)
            )
            var nextw = rw / 2 - thickness
            self._add_room(Rect(rx, wy, nextw, wh))
            self._add_room(Rect(rx + rw / 2 + thickness, wy, nextw, wh))
            self._add_room(Rect(rx, remy, rw, rh - wh))
        else:
            var wx: Float32
            var ww: Float32
            var remx: Float32
            if c2:
                wx = rx
                remx = rx + pct * rw
                ww = pct * rw
            else:
                wx = rx + (1 - pct) * rw
                remx = rx
                ww = pct * rw
            self.entities.append(
                Entity(wx + ww / 2, ry + rh / 2, 0.0, 0.0, ww / 2, thickness, LAVA_WALL)
            )
            var nexth = rh / 2 - thickness
            self._add_room(Rect(wx, ry, ww, nexth))
            self._add_room(Rect(wx, ry + rh / 2 + thickness, ww, nexth))
            self._add_room(Rect(remx, ry, rw - ww, rh))

    def _split_rooms(mut self, num_iterations: Int, thickness: Float32):
        for _ in range(num_iterations):
            if len(self.rooms) == 0:
                break
            var idx = self.rand_gen.randn(len(self.rooms))
            var room = self.rooms[idx]
            _ = self.rooms.pop(idx)
            self._split_room(room, thickness)

    def _reposition(mut self, mut e: Entity, x: Float32, y: Float32, w: Float32, h: Float32):
        e.x = self._rand_pos(e.rx, x, x + w)
        e.y = self._rand_pos(e.ry, y, y + h)
        var c = 0
        while (
            self._coll_agent(e.x, e.y, e.rx, e.ry) or self._coll_any(e.x, e.y, e.rx, e.ry)
        ) and c < 100:
            e.x = self._rand_pos(e.rx, x, x + w)
            e.y = self._rand_pos(e.ry, y, y + h)
            c += 1

    def _reposition_agent(mut self):
        var a = self.agent.copy()
        a.x = self.rand_gen.rand01() * (Float32(self.w) - 2 * a.rx) + a.rx
        a.y = self.rand_gen.rand01() * (Float32(self.h) - 2 * a.ry) + a.ry
        self.agent = a^
        var c = 0
        while self._agent_has_coll() and c < 100:
            var a2 = self.agent.copy()
            a2.x = self.rand_gen.rand01() * (Float32(self.w) - 2 * a2.rx) + a2.rx
            a2.y = self.rand_gen.rand01() * (Float32(self.h) - 2 * a2.ry) + a2.ry
            self.agent = a2^
            c += 1

    def _spawn_door(mut self, exit_r: Float32):
        var doorlen = 2 * exit_r
        self.exit_wall_choice = self.rand_gen.randn(4)
        var d = Entity(0.0, 0.0, 0.0, 0.0, doorlen / 2, exit_r, DOOR)
        var fw = Float32(self.w)
        var fh = Float32(self.h)
        if self.exit_wall_choice == 0:
            d.rx = doorlen / 2
            d.ry = exit_r
            self._reposition(d, 0.0, 0.0, fw, 2 * exit_r)
        elif self.exit_wall_choice == 1:
            d.rx = doorlen / 2
            d.ry = exit_r
            self._reposition(d, 0.0, fh - 2 * exit_r, fw, 2 * exit_r)
        elif self.exit_wall_choice == 2:
            d.rx = exit_r
            d.ry = doorlen / 2
            self._reposition(d, 0.0, 0.0, 2 * exit_r, fh)
        else:
            d.rx = exit_r
            d.ry = doorlen / 2
            self._reposition(d, fw - 2 * exit_r, 0.0, 2 * exit_r, fh)
        self.entities.append(d^)

    def _spawn_enemies(mut self, count: Int, enemy_r: Float32):
        for _ in range(count):
            var e = Entity(0.0, 0.0, 0.0, 0.0, enemy_r, enemy_r, ENEMY)
            self._reposition(e, 0.0, 0.0, Float32(self.w), Float32(self.h))
            self.entities.append(e^)

    def _setup_enemies(mut self):
        var enemy_theme = self.rand_gen.randn(NUM_ENEMY_THEMES)
        for i in range(len(self.entities)):
            if self.entities[i].type == ENEMY:
                var e = self.entities[i].copy()
                e.image_theme = enemy_theme
                e.health = 1.0
                e.spawn_time = 0
                e.fire_time = 10
                e.collides_with_entities = True
                e.smart_step = True
                self._choose_vel(e)
                if e.vx != 0.0 or e.vy != 0.0:
                    e.rotation = -1.0 * atan2(e.vy, e.vx)
                self.entities[i] = e^
            elif self.entities[i].type == LAVA_WALL:
                self.entities[i].collides_with_entities = True

    def reset(mut self, level_seed: Int):
        self.rand_gen.seed(level_seed)
        self.entities = List[Entity]()
        self.rooms = List[Rect]()
        self.reward = 0.0
        self.episode_reward = 0.0
        self.done = False
        self.level_complete = False
        self.cur_time = 0
        self.last_fire_time = 0
        self.last_move_action = 4

        # Fresh agent each reset (base game_reset makes a new Entity: vx=vy=0,
        # rotation=0, at a_r=0.4). Mode branch below overrides rx/ry.
        self.agent = Entity(0.4, 0.4, 0.0, 0.0, 0.4, 0.4, PLAYER)

        self.bg_pct_x = self.rand_gen.rand01()
        self.background_index = self.rand_gen.randn(BG_COUNT)
        self.oob = OOB_WALL
        self.rooms.append(Rect(0.0, 0.0, Float32(self.w), Float32(self.h)))

        var thickness: Float32 = 0.3
        var enemy_r: Float32 = 0.5
        var exit_r: Float32 = 0.75
        self.ball_r = 0.25
        self.ball_vscale = 0.25
        var num_iterations: Int
        var max_extra = 3
        if self.dist_mode == DIST_EASY:
            num_iterations = 2
            thickness *= 2
            enemy_r *= 2
            self.ball_r *= 2
            self.ball_vscale *= 2
            self.maxspeed = 0.75
            self.agent.rx = 1.0
            self.agent.ry = 1.0
            exit_r *= 2
        else:
            num_iterations = 4
            thickness *= 1.5
            enemy_r *= 1.5
            self.ball_r *= 1.5
            self.ball_vscale *= 1.5
            self.maxspeed = 0.5
            self.agent.rx = 0.75
            self.agent.ry = 0.75

        self.hard_min_dim = 4 * self.agent.rx + 2 * thickness + 0.5
        self.min_dim = self.agent.rx * 8 + 0.5

        self._split_rooms(num_iterations, thickness)
        self._spawn_door(exit_r)
        self._reposition_agent()
        self.num_enemies = self.rand_gen.randn(max_extra + 1) + 3
        self._spawn_enemies(self.num_enemies, enemy_r)
        self._setup_enemies()
        # agent.face_direction(1, 0) → rotation 0.
        self.agent.rotation = 0.0

    def _fire_ball(mut self, mut e: Entity, vx: Float32, vy: Float32):
        var b = Entity(
            e.x, e.y, vx * self.ball_vscale, vy * self.ball_vscale, self.ball_r, self.ball_r, ENEMY_BALL
        )
        e.fire_time = self.cur_time + self.rand_gen.randn(4)
        b.vrot = BALL_V_ROT
        b.expire_time = 50
        self.entities.append(b^)

    def step(mut self, action: Int) -> Float32:
        self.cur_time += 1
        self.step_rand_int = self.rand_gen.randint(0, 1000000)
        var mv = action % 9
        self.special_action = 0
        if action >= 9:
            self.special_action = action - 8
            mv = 4
        if mv != 4:
            self.last_move_action = mv
        self.action_vx = Float32(mv // 3 - 1)
        self.action_vy = Float32(mv % 3 - 1)
        self.reward = 0.0
        self.done = False
        self.level_complete = False

        self._update_agent_velocity()

        for i in range(len(self.entities) - 1, -1, -1):
            if self.entities[i].smart_step:
                var e = self.entities[i].copy()
                self._basic_step_object(e, i)
                self.entities[i] = e^
            var e2 = self.entities[i].copy()
            e2.step()
            self.entities[i] = e2^
        var ag = self.agent.copy()
        self._basic_step_object(ag, -1)
        ag.step()
        self.agent = ag^

        # collisions: agent-death/win + collides_with_entities → handle_collision.
        var n = len(self.entities)
        for i in range(n - 1, -1, -1):
            var t = self.entities[i].type
            if t != PLAYER and self._has_collision(
                self.entities[i].x, self.entities[i].y,
                self.entities[i].rx, self.entities[i].ry,
                self.agent.x, self.agent.y, self.agent.rx, self.agent.ry, 0.0
            ):
                if t == ENEMY or t == ENEMY_BALL or t == LAVA_WALL:
                    self.done = True
                elif t == DOOR:
                    if self.num_enemies == 0:
                        self.done = True
                        self.reward += COMPLETION_BONUS
                        self.level_complete = True
            if self.entities[i].collides_with_entities:
                for j in range(n - 1, -1, -1):
                    if i == j:
                        continue
                    if (
                        not self.entities[i].will_erase and not self.entities[j].will_erase
                        and self._has_collision(
                            self.entities[i].x, self.entities[i].y,
                            self.entities[i].rx, self.entities[i].ry,
                            self.entities[j].x, self.entities[j].y,
                            self.entities[j].rx, self.entities[j].ry, 0.0
                        )
                    ):
                        self._handle_collision(i, j)

        for i in range(len(self.entities) - 1, -1, -1):
            if self.entities[i].will_erase or (
                self.entities[i].auto_erase and self._out_of_bounds(self.entities[i])
            ):
                _ = self.entities.pop(i)
        if self._out_of_bounds(self.agent):
            self.done = True

        # dodgeball tail: facing + fire + enemy AI + ball edge-erase.
        var fvx = Float32(self.last_move_action // 3 - 1)
        var fvy = Float32(self.last_move_action % 3 - 1)
        if fvx != 0.0 or fvy != 0.0:
            self.agent.rotation = -1.0 * atan2(fvy, fvx)
        if self.special_action == 1 and (self.cur_time - self.last_fire_time) >= 7:
            var b = Entity(
                self.agent.x, self.agent.y, fvx * self.ball_vscale, fvy * self.ball_vscale,
                self.ball_r, self.ball_r, PLAYER_BALL,
            )
            b.collides_with_entities = True
            b.expire_time = 50
            b.vrot = BALL_V_ROT
            self.entities.append(b^)
            self.last_fire_time = self.cur_time

        self.num_enemies = 0
        var m2 = len(self.entities)
        for i in range(m2 - 1, -1, -1):
            var t = self.entities[i].type
            if t == ENEMY:
                self.num_enemies += 1
                if self.entities[i].spawn_time == 0:
                    var e = self.entities[i].copy()
                    self._choose_vel(e)
                    self.entities[i] = e^
                else:
                    self.entities[i].spawn_time -= 1
                var can_fire = (self.cur_time - self.entities[i].fire_time) >= ENEMY_FIRE_DELAY
                if can_fire:
                    var dx = self.entities[i].x - self.agent.x
                    var dy = self.entities[i].y - self.agent.y
                    var bvelx: Float32 = 1.0 if self.entities[i].x < self.agent.x else -1.0
                    var bvely: Float32 = 1.0 if self.entities[i].y < self.agent.y else -1.0
                    if abs(dx) < 1.0:
                        var e = self.entities[i].copy()
                        self._fire_ball(e, 0.0, bvely)
                        e.vx = 0.0
                        e.vy = bvely * ENEMY_VEL
                        self.entities[i] = e^
                    elif abs(dy) < 1.0:
                        var e = self.entities[i].copy()
                        self._fire_ball(e, bvelx, 0.0)
                        e.vx = bvelx * ENEMY_VEL
                        e.vy = 0.0
                        self.entities[i] = e^
                var ex2 = self.entities[i].vx
                var ey2 = self.entities[i].vy
                if ex2 != 0.0 or ey2 != 0.0:
                    self.entities[i].rotation = -1.0 * atan2(ey2, ex2)
            elif t == PLAYER_BALL or t == ENEMY_BALL:
                var bx = self.entities[i].x
                var by = self.entities[i].y
                var br = self.entities[i].rx
                if bx < br or bx > (Float32(self.w) - br):
                    self.entities[i].will_erase = True
                elif by < br or by > (Float32(self.h) - br):
                    self.entities[i].will_erase = True
        for i in range(len(self.entities) - 1, -1, -1):
            if self.entities[i].will_erase or (
                self.entities[i].auto_erase and self._out_of_bounds(self.entities[i])
            ):
                _ = self.entities.pop(i)

        self.episode_reward += self.reward
        return self.reward

    def _handle_collision(mut self, i: Int, j: Int):
        # entities[i] is a collides_with_entities src; act on player/enemy balls (target=j).
        var tt = self.entities[j].type
        var st = self.entities[i].type
        if tt == PLAYER_BALL:
            if st == LAVA_WALL:
                self.entities[j].will_erase = True
            elif st == ENEMY:
                self.entities[i].health -= 1.0
                self.entities[j].will_erase = True
                if self.entities[i].health <= 0.0 and not self.entities[i].will_erase:
                    var sx = self.entities[i].x
                    var sy = self.entities[i].y
                    var sr = self.entities[i].rx
                    self.entities[i].will_erase = True
                    self.reward += ENEMY_REWARD
                    var d = Entity(sx, sy, 0.0, 0.0, sr, sr, DUST_CLOUD)
                    d.vrot = PI / 0.3
                    d.grow_rate = 1.0 / 1.2
                    d.expire_time = 4
                    d.alpha_decay = 0.9
                    d.image_theme = self.step_rand_int % 9
                    self.entities.append(d^)
        elif tt == ENEMY_BALL:
            if st == LAVA_WALL:
                self.entities[j].will_erase = True

    # --- rendering (visual-approx; assets passed in) ---
    def render_obs(
        self, assets: DodgeballAssets, res: Int = RES, ss: Int = OBS_SS
    ) -> List[UInt8]:
        return downscale(self.render(assets, res * ss), res * ss, res)

    def render(self, assets: DodgeballAssets, out_res: Int = RES) -> List[UInt8]:
        var canvas = Canvas(out_res)
        canvas.fill(0, 0, 0)
        var view_dim = Float32(self.w)
        var unit = Float32(out_res) / view_dim

        ref bg = assets.backgrounds[self.background_index]
        var bg_ar = Float32(bg.w) / Float32(bg.h)
        var world_ar = Float32(self.w) / Float32(self.h)
        var offset_x = self.bg_pct_x * (bg_ar - world_ar)
        var main_w = Float32(self.w) * unit
        canvas.blit(
            bg, main_w * (-offset_x), 0.0, main_w * (bg_ar / world_ar), Float32(self.h) * unit
        )

        var door_open = self.num_enemies == 0
        for k in range(len(self.entities)):
            ref e = self.entities[k]
            var t = e.type
            var ex = (e.x - e.rx) * unit
            var ey = (view_dim - (e.y + e.ry)) * unit
            var ew = 2 * e.rx * unit
            var eh = 2 * e.ry * unit
            if t == LAVA_WALL:
                canvas.blit(assets.lava, ex, ey, ew, eh)
            elif t == DOOR:
                if door_open:
                    canvas.blit(assets.door_open, ex, ey, ew, eh)
                else:
                    canvas.blit(assets.door, ex, ey, ew, eh)
            elif t == ENEMY:
                canvas.blit(assets.enemies[e.image_theme], ex, ey, ew, eh)
            elif t == PLAYER_BALL:
                canvas.blit(assets.player_ball, ex, ey, ew, eh)
            elif t == ENEMY_BALL:
                canvas.blit(assets.enemy_ball, ex, ey, ew, eh)
            elif t == DUST_CLOUD:
                canvas.blit(assets.dust[e.image_theme], ex, ey, ew, eh)

        var ax = (self.agent.x - self.agent.rx) * unit
        var ay = (view_dim - (self.agent.y + self.agent.ry)) * unit
        canvas.blit(
            assets.player, ax, ay, 2 * self.agent.rx * unit, 2 * self.agent.ry * unit
        )

        return canvas.px.copy()
