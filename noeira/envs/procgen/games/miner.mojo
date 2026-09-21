"""Miner game — Boulder Dash / dig-collect-exit (port of `games/miner.cpp`).

A robot digs through dirt in a grid, collects diamonds, avoids falling boulders,
and exits once all diamonds are gone. The "physics" is a **grid cellular automaton**
(integer cell rules — falling / rolling), NOT rigid-body; the agent is grid-locked
(`grid_step`, like maze). No physics engine, no continuous entities (the EXIT is a
static entity). Miner defines ENEMY handling but never spawns any → skipped.

`game_reset`/`game_step` replay the exact RNG order. Level-exact + visual-approx.
See `docs/PROCGEN_MINER_SCOPE.md`. P0+P1 = reset+step parity; render/env in P2.
"""

from std.math import floor
from std.memory import ArcPointer

from .procgen_env import ProcgenGame

from ..core.entity import Entity
from ..core.randgen import RandGen
from ..core.assets import Sprite, load_sprite, load_sprites
from ..core.rasterizer import Canvas, RES, downscale
from ..core.object_ids import SPACE, WALL_OBJ, PLAYER

# Miner object ids (miner.cpp).
comptime BOULDER = 1
comptime DIAMOND = 2
comptime MOVING_BOULDER = 3
comptime MOVING_DIAMOND = 4
comptime EXIT = 6
comptime DIRT = 9
comptime OOB_WALL = 10

comptime COMPLETION_BONUS: Float32 = 10.0
comptime DIAMOND_REWARD: Float32 = 1.0
comptime A_R: Float32 = 0.4
comptime BG_COUNT = 37  # platform_backgrounds (resources.cpp)
comptime RENDER_EPS: Float32 = 0.02
comptime OBS_SS = 4
comptime MEMORY_VISIBILITY: Float32 = 8.0

comptime DIST_EASY = 0
comptime DIST_HARD = 1
comptime DIST_MEMORY = 10


def miner_world_dim(dist_mode: Int) -> Int:
    if dist_mode == DIST_MEMORY:
        return 35
    if dist_mode == DIST_HARD:
        return 20
    return 10  # EasyMode


def _moving_type(t: Int) -> Int:
    if t == DIAMOND:
        return MOVING_DIAMOND
    if t == BOULDER:
        return MOVING_BOULDER
    return t


def _stat_type(t: Int) -> Int:
    if t == MOVING_DIAMOND:
        return DIAMOND
    if t == MOVING_BOULDER:
        return BOULDER
    return t


def _is_round(t: Int) -> Bool:
    return (
        t == BOULDER or t == MOVING_BOULDER or t == DIAMOND or t == MOVING_DIAMOND
    )


def _is_moving(t: Int) -> Bool:
    return t == MOVING_BOULDER or t == MOVING_DIAMOND


struct MinerAssets(Movable):
    """Read-only sprite set (shared via ArcPointer; passed into render())."""

    var player: Sprite
    var boulder: Sprite
    var diamond: Sprite
    var exit: Sprite
    var dirt: Sprite
    var wall: Sprite
    var backgrounds: List[Sprite]

    def __init__(out self, asset_root: String) raises:
        self.player = load_sprite(asset_root, "misc_assets/robot_greenDrive1.png")
        self.boulder = load_sprite(asset_root, "misc_assets/elementStone007.png")
        self.diamond = load_sprite(asset_root, "misc_assets/gemBlue.png")
        self.exit = load_sprite(asset_root, "misc_assets/window.png")
        self.dirt = load_sprite(asset_root, "misc_assets/dirt.png")
        self.wall = load_sprite(asset_root, "misc_assets/tile_bricksGrey.png")
        # platform_backgrounds (resources.cpp order) — background_index = randn(37).
        var names: List[String] = [
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
        for i in range(len(names)):
            bp.append("platform_backgrounds/" + names[i] + ".png")
        self.backgrounds = load_sprites(asset_root, bp)


struct MinerGame(Copyable, Movable, ProcgenGame):
    # ─── ProcgenGame conformance glue (see games/procgen_env.mojo) ──────
    comptime AssetsT = MinerAssets
    comptime DEFAULT_DIST = DIST_EASY
    comptime GYM_MAX_STEPS = 1000

    @staticmethod
    def load_assets(asset_root: String) raises -> MinerAssets:
        return MinerAssets(asset_root)

    @staticmethod
    def make(assets: ArcPointer[MinerAssets], dist_mode: Int) -> Self:
        # The env owns the assets and passes them into the render calls.
        return Self(dist_mode)

    def is_done(self) -> Bool:
        return self.done

    def is_level_complete(self) -> Bool:
        return self.level_complete

    def gym_terminated(self) -> Bool:
        return self.done

    def pg_render_obs(self, assets: MinerAssets) -> List[UInt8]:
        return self.render_obs(assets)

    def pg_render_obs_train(
        self, assets: MinerAssets, res: Int, ss: Int
    ) -> List[UInt8]:
        return self.render_obs(assets, res, ss)

    def pg_render(self, assets: MinerAssets, res: Int) -> List[UInt8]:
        return self.render(assets, res)

    var rand_gen: RandGen
    var w: Int
    var h: Int
    var world_dim: Int
    var dist_mode: Int
    var grid: List[Int]
    var agent: Entity
    var entities: List[Entity]  # the static EXIT entity
    var diamonds_remaining: Int
    var exit_x: Int
    var exit_y: Int
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
        self.world_dim = miner_world_dim(dist_mode)
        self.w = self.world_dim
        self.h = self.world_dim
        self.grid = List[Int]()
        self.agent = Entity.make(0.5, 0.5, 0.5, PLAYER)
        self.entities = List[Entity]()
        self.diamonds_remaining = 0
        self.exit_x = 0
        self.exit_y = 0
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

    # --- grid helpers (out_of_bounds_object == OOB_WALL) ---
    def _gobj(self, idx: Int) -> Int:
        if idx < 0 or idx >= self.w * self.h:
            return OOB_WALL
        return self.grid[idx]

    def _gxy(self, x: Int, y: Int) -> Int:
        if x < 0 or x >= self.w or y < 0 or y >= self.h:
            return OOB_WALL
        return self.grid[y * self.w + x]

    def _agent_index(self) -> Int:
        return Int(self.agent.y) * self.w + Int(self.agent.x)

    def _is_free(self, idx: Int) -> Bool:
        return self._gobj(idx) == SPACE and self._agent_index() != idx

    def _player_blocked(self, t: Int) -> Bool:
        return (
            t == WALL_OBJ or t == OOB_WALL or t == BOULDER or t == MOVING_BOULDER
        )

    def _blocked_at(self, fi: Float32, fj: Float32) -> Bool:
        return self._player_blocked(self._gxy(Int(floor(fi)), Int(floor(fj))))

    def _sub_step(self, mut o: Entity, vx: Float32, vy: Float32) -> Bool:
        # grid_step single sub-step: block => stay put on that axis.
        var nx = o.x + vx
        var ny = o.y + vy
        var margin: Float32 = 0.98
        var is_h = vx != 0.0
        var block = False
        for i in range(2):
            for j in range(2):
                if self._blocked_at(
                    nx + 0.5 * margin * Float32(2 * i - 1),
                    ny + 0.5 * margin * Float32(2 * j - 1),
                ):
                    block = True
        if block:
            if is_h:
                nx = o.x
            else:
                ny = o.y
        o.x = nx
        o.y = ny
        return block

    def reset(mut self, level_seed: Int):
        self.rand_gen.seed(level_seed)
        self.reward = 0.0
        self.episode_reward = 0.0
        self.done = False
        self.level_complete = False
        self.cur_time = 0
        self.entities = List[Entity]()
        var area = self.w * self.h
        self.grid = List[Int]()
        self.grid.resize(area, DIRT)

        self.bg_pct_x = self.rand_gen.rand01()
        self.background_index = self.rand_gen.randn(BG_COUNT)
        var ax = self.rand_gen.rand01() * (Float32(self.w) - 2 * A_R) + A_R
        _ = self.rand_gen.rand01() * (Float32(self.h) - 2 * A_R) + A_R  # ay
        _ = ax  # base agent spawn draws consumed; agent placed from obj_idxs below

        var diamond_pct = Float32(12) / Float32(400)
        var boulder_pct = Float32(80) / Float32(400)
        var num_diamonds = Int(diamond_pct * Float32(area))
        var num_boulders = Int(boulder_pct * Float32(area))

        var idxs = self.rand_gen.simple_choose(
            area, num_diamonds + num_boulders + 1
        )
        var agx = idxs[0] % self.w
        var agy = idxs[0] // self.w
        self.agent = Entity.make(Float32(agx) + 0.5, Float32(agy) + 0.5, 0.5, PLAYER)

        for i in range(num_diamonds):
            self.grid[idxs[i + 1]] = DIAMOND
        for i in range(num_boulders):
            self.grid[idxs[i + 1 + num_diamonds]] = BOULDER

        var dirt_cells = List[Int]()
        for i in range(area):
            if self.grid[i] == DIRT:
                dirt_cells.append(i)

        self.grid[agy * self.w + agx] = SPACE
        for i in range(-1, 2):
            for j in range(-1, 2):
                var ox = agx + i
                var oy = agy + j
                if self._gxy(ox, oy) == BOULDER:
                    self.grid[oy * self.w + ox] = DIRT

        var cand = List[Int]()
        for ci in range(len(dirt_cells)):
            var cell = dirt_cells[ci]
            var above = self._gobj(cell + self.w)
            if above == DIRT or above == OOB_WALL:
                cand.append(cell)
        var exit_cell = cand[self.rand_gen.randn(len(cand))]
        self.grid[exit_cell] = SPACE
        self.exit_x = exit_cell % self.w
        self.exit_y = exit_cell // self.w
        var ex = Entity.make(Float32(self.exit_x) + 0.5, Float32(self.exit_y) + 0.5, 0.5, EXIT)
        ex.render_z = -1
        self.entities.append(ex^)

        var dc = 0
        for i in range(area):
            if self.grid[i] == DIAMOND:
                dc += 1
        self.diamonds_remaining = dc

    def _handle_push(mut self):
        var ai = self._agent_index()
        var ax = ai % self.w
        if (
            self.action_vx == 1.0
            and self.agent.vx == 0.0
            and ax < self.w - 2
            and self._gobj(ai + 1) == BOULDER
            and self._gobj(ai + 2) == SPACE
        ):
            self.grid[ai + 1] = SPACE
            self.grid[ai + 2] = BOULDER
            self.agent.x += 1.0
        elif (
            self.action_vx == -1.0
            and self.agent.vx == 0.0
            and ax > 1
            and self._gobj(ai - 1) == BOULDER
            and self._gobj(ai - 2) == SPACE
        ):
            self.grid[ai - 1] = SPACE
            self.grid[ai - 2] = BOULDER
            self.agent.x -= 1.0

    def step(mut self, action: Int) -> Float32:
        self.cur_time += 1
        self.reward = 0.0
        self.done = False
        self.level_complete = False

        # --- base game_step: grid-step agent move ---
        self.step_rand_int = self.rand_gen.randint(0, 1000000)
        var move = action % 9
        if action >= 9:
            move = 4
        self.action_vx = Float32(move // 3 - 1)
        self.action_vy = Float32(move % 3 - 1)
        if self.action_vx != 0.0:
            self.action_vy = 0.0  # set_action_xy override (no diagonal)

        self.agent.vx = self.action_vx
        self.agent.vy = self.action_vy
        var a = self.agent.copy()
        var sxf = self.action_vx != 0.0  # PLAYER step_x_first by action
        var bx: Bool
        var by: Bool
        if sxf:
            bx = self._sub_step(a, a.vx, 0.0)
            by = self._sub_step(a, 0.0, a.vy)
        else:
            by = self._sub_step(a, 0.0, a.vy)
            bx = self._sub_step(a, a.vx, 0.0)
        if bx:
            a.vx = 0.0
        if by:
            a.vy = 0.0
        self.agent = a^

        # agent-collision with the static EXIT entity (base collision pass).
        for i in range(len(self.entities)):
            if self.entities[i].type == EXIT:
                if (
                    abs(self.entities[i].x - self.agent.x) < 1.0
                    and abs(self.entities[i].y - self.agent.y) < 1.0
                ):
                    if self.diamonds_remaining == 0:
                        self.reward += COMPLETION_BONUS
                        self.level_complete = True
                        self.done = True

        if self.action_vx > 0.0:
            self.agent.is_reflected = False
        if self.action_vx < 0.0:
            self.agent.is_reflected = True

        self._handle_push()

        # Dig / collect at the agent's cell.
        var acx = Int(self.agent.x)
        var acy = Int(self.agent.y)
        var aobj = self._gxy(acx, acy)
        if aobj == DIAMOND:
            self.reward += DIAMOND_REWARD
        if aobj == DIRT or aobj == DIAMOND:
            self.grid[acy * self.w + acx] = SPACE

        # --- Boulder-Dash falling / rolling scan ---
        var area = self.w * self.h
        var dcount = 0
        var ai = self._agent_index()
        for idx in range(area):
            var obj = self.grid[idx]
            var ox = idx % self.w
            if _stat_type(obj) == DIAMOND:
                dcount += 1
            if (
                obj == BOULDER
                or obj == MOVING_BOULDER
                or obj == DIAMOND
                or obj == MOVING_DIAMOND
            ):
                var bi = idx - self.w
                var o2 = self._gobj(bi)
                var ab = ai == bi
                if o2 == SPACE and not ab:
                    self.grid[idx] = SPACE
                    self.grid[bi] = _moving_type(obj)
                elif ab and _is_moving(obj):
                    self.done = True
                elif (
                    _is_round(o2)
                    and ox > 0
                    and self._is_free(idx - 1)
                    and self._is_free(idx - self.w - 1)
                ):
                    self.grid[idx] = SPACE
                    self.grid[idx - 1] = _stat_type(obj)
                elif (
                    _is_round(o2)
                    and ox < self.w - 1
                    and self._is_free(idx + 1)
                    and self._is_free(idx - self.w + 1)
                ):
                    self.grid[idx] = SPACE
                    self.grid[idx + 1] = _stat_type(obj)
                else:
                    self.grid[idx] = _stat_type(obj)
        self.diamonds_remaining = dcount

        self.episode_reward += self.reward
        return self.reward

    # --- rendering (visual-approx; assets passed in) ---
    def render_obs(
        self, assets: MinerAssets, res: Int = RES, ss: Int = OBS_SS
    ) -> List[UInt8]:
        return downscale(self.render(assets, res * ss), res * ss, res)

    def render(self, assets: MinerAssets, out_res: Int = RES) -> List[UInt8]:
        var canvas = Canvas(out_res)
        canvas.fill(0, 0, 0)

        # Camera: Memory centers on the agent (visibility 8); Easy/Hard whole world.
        var center_agent = self.dist_mode == DIST_MEMORY
        var visibility: Float32
        var center_x: Float32
        var center_y: Float32
        if center_agent:
            visibility = MEMORY_VISIBILITY
            center_x = self.agent.x
            center_y = self.agent.y
        else:
            visibility = Float32(self.w if self.w > self.h else self.h)
            center_x = Float32(self.w) * 0.5
            center_y = Float32(self.h) * 0.5
        var view_dim = visibility
        var unit = Float32(out_res) / view_dim
        var x_off = unit * (center_x - view_dim / 2)
        var y_off = unit * (center_y - view_dim / 2)

        # Background (platform, panned by bg_pct_x).
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

        # Grid tiles: dirt, boulders, diamonds (MOVING_* drawn as their base).
        for x in range(self.w):
            for y in range(self.h):
                var t = self.grid[y * self.w + x]
                if t == SPACE:
                    continue
                var sx = (Float32(x) - RENDER_EPS) * unit - x_off
                var sy = (view_dim - Float32(y + 1) - RENDER_EPS) * unit + y_off
                var sz = (1.0 + 2 * RENDER_EPS) * unit
                if t == DIRT:
                    canvas.blit(assets.dirt, sx, sy, sz, sz)
                elif t == BOULDER or t == MOVING_BOULDER:
                    canvas.blit(assets.boulder, sx, sy, sz, sz)
                elif t == DIAMOND or t == MOVING_DIAMOND:
                    canvas.blit(assets.diamond, sx, sy, sz, sz)

        # EXIT entity (render_z=-1 → drawn under the agent; grid cell there is SPACE).
        for k in range(len(self.entities)):
            ref e = self.entities[k]
            if e.type == EXIT:
                var ex = (e.x - e.rx) * unit - x_off
                var ey = (view_dim - (e.y + e.ry)) * unit + y_off
                canvas.blit(assets.exit, ex, ey, 2 * e.rx * unit, 2 * e.ry * unit)

        # Robot player.
        var ax = (self.agent.x - self.agent.rx) * unit - x_off
        var ay = (view_dim - (self.agent.y + self.agent.ry)) * unit + y_off
        canvas.blit(
            assets.player,
            ax,
            ay,
            2 * self.agent.rx * unit,
            2 * self.agent.ry * unit,
            self.agent.is_reflected,
        )

        return canvas.px.copy()
