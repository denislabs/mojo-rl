"""Starpilot game — space shooter (port of `games/starpilot.cpp`).

The ship flies on the left of a 16×16 field; enemy waves fly in from the right (and
sometimes left); you shoot them (+1 each) and survive to `SHOOTER_WIN_TIME=500`, when
a finish line spawns → touch it to win (+10). Any lethal enemy/bullet ends the run.

This is the entry point for the **projectile substrate** (fire → bullet entities →
collide/destroy) shared by the shooter family. `game_reset` pre-schedules the whole
episode's enemy waves (`add_spawners`) with exact RNG order. Level-exact + visual-approx.
See `docs/PROCGEN_STARPILOT_SCOPE.md`. P0 = reset (add_spawners) parity; step/render in P1/P2.
"""

from std.math import floor, cos, sin, sqrt
from std.memory import ArcPointer

from ..core.entity import Entity
from ..core.randgen import RandGen
from ..core.object_ids import PLAYER

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

        self.agent.rotation = PI / 2
        self.agent.image_theme = self.rand_gen.randn(NUM_SHIP_THEMES)
