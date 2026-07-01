"""Maze game — one procedurally-generated level (given a level seed).

Faithful port of `games/maze.cpp`'s grid construction + grid-step movement +
goal logic, on the proven `MazeGen`. `reset(level_seed)` seeds `rand_gen` and
replays the exact `BasicAbstractGame::game_reset` + `MazeGame::game_reset` RNG
order (bg_pct_x, background_index, maze_dim, gen), so a level seed reproduces
reference Procgen's maze exactly (gated by `test_maze_game_parity.mojo`).

Rendering is visual-approx (§ rasterizer): the selected topdown background is
drawn first (panned by `bg_pct_x`), then sand walls / cheese goal / mouse agent
on top, matching Procgen's draw order.

Level *selection* (train/test splits via num_levels/start_level/rand_seed) lives
in the `MazeEnv` wrapper (`maze_env.mojo`). See `docs/PROCGEN_PORT.md`.
"""

from std.math import floor

from ..core.entity import Entity
from ..core.randgen import RandGen
from ..core.mazegen import MazeGen, MAZE_OFFSET
from ..core.assets import Sprite, load_sprite, load_topdown_backgrounds
from ..core.rasterizer import Canvas, RES
from ..core.object_ids import SPACE, WALL_OBJ, PLAYER, INVALID_OBJ

comptime GOAL = 2
comptime REWARD: Float32 = 10.0
comptime WORLD_DIM = 25  # HardMode
comptime RENDER_EPS: Float32 = 0.02
comptime BG_COUNT = 9  # topdown_backgrounds (resources.cpp)


struct MazeGame(Copyable, Movable):
    var rand_gen: RandGen
    var grid: List[Int]
    var w: Int
    var h: Int
    var agent: Entity
    var sand: Sprite
    var cheese: Sprite
    var mouse: Sprite
    var backgrounds: List[Sprite]
    var episode_reward: Float32
    var done: Bool
    var level_complete: Bool
    var bg_pct_x: Float32
    var background_index: Int
    var maze_dim: Int

    def __init__(out self, asset_root: String) raises:
        self.rand_gen = RandGen()
        self.grid = List[Int]()
        self.w = WORLD_DIM
        self.h = WORLD_DIM
        self.agent = Entity.make(0.5, 0.5, 0.5, PLAYER)
        self.sand = load_sprite(asset_root, "kenney/Ground/Sand/sandCenter.png")
        self.cheese = load_sprite(asset_root, "misc_assets/cheese.png")
        self.mouse = load_sprite(asset_root, "kenney/Enemies/mouse_move.png")
        self.backgrounds = load_topdown_backgrounds(asset_root)
        self.episode_reward = 0.0
        self.done = False
        self.level_complete = False
        self.bg_pct_x = 0.0
        self.background_index = 0
        self.maze_dim = 0

    # --- grid helpers (BasicAbstractGame subset, out_of_bounds_object=WALL) ---
    def _get_obj(self, x: Int, y: Int) -> Int:
        if x < 0 or x >= self.w or y < 0 or y >= self.h:
            return WALL_OBJ
        return self.grid[y * self.w + x]

    def _set_obj(mut self, x: Int, y: Int, v: Int):
        self.grid[y * self.w + x] = v

    def _obj_from_floats(self, fi: Float32, fj: Float32) -> Int:
        if fi < 0 or fj < 0:
            return WALL_OBJ
        return self._get_obj(Int(floor(fi)), Int(floor(fj)))

    def _sub_step(mut self, vx: Float32, vy: Float32):
        # grid_step single sub-step: block => stay put on that axis.
        var nx = self.agent.x + vx
        var ny = self.agent.y + vy
        var margin: Float32 = 0.98
        var block = False
        for i in range(2):
            for j in range(2):
                var t = self._obj_from_floats(
                    nx + self.agent.rx * margin * Float32(2 * i - 1),
                    ny + self.agent.ry * margin * Float32(2 * j - 1),
                )
                if t == WALL_OBJ:
                    block = True
        if block:
            if vx != 0:
                nx = self.agent.x
            else:
                ny = self.agent.y
        self.agent.x = nx
        self.agent.y = ny

    def reset(mut self, level_seed: Int):
        self.rand_gen.seed(level_seed)
        self.episode_reward = 0.0
        self.done = False
        self.level_complete = False

        # BasicAbstractGame::game_reset base draws (maze path), in exact order:
        # bg_pct_x then background_index. random_agent_start=false and
        # use_procgen_background=false → no further base draws. This makes the
        # reset RNG stream match reference Procgen for a given level seed.
        self.bg_pct_x = self.rand_gen.rand01()
        self.background_index = self.rand_gen.randn(BG_COUNT)

        var maze_dim = self.rand_gen.randn((WORLD_DIM - 1) // 2) * 2 + 3
        self.maze_dim = maze_dim
        var margin = (WORLD_DIM - maze_dim) // 2

        # Fill the whole game grid with wall, then overlay the maze region.
        self.grid = List[Int]()
        self.grid.resize(self.w * self.h, WALL_OBJ)

        var mg = MazeGen(maze_dim)
        mg.generate_maze(self.rand_gen)
        mg.place_objects(self.rand_gen, GOAL, 1)

        for i in range(maze_dim):
            for j in range(maze_dim):
                var t = mg.grid.get(i + MAZE_OFFSET, j + MAZE_OFFSET)
                self._set_obj(margin + i, margin + j, t)

        if margin > 0:
            for i in range(maze_dim + 2):
                self._set_obj(margin - 1, margin + i - 1, WALL_OBJ)
                self._set_obj(margin + maze_dim, margin + i - 1, WALL_OBJ)
                self._set_obj(margin + i - 1, margin - 1, WALL_OBJ)
                self._set_obj(margin + i - 1, margin + maze_dim, WALL_OBJ)

        self.agent = Entity.make(
            Float32(margin) + 0.5, Float32(margin) + 0.5, 0.5, PLAYER
        )

    def step(mut self, action: Int) -> Float32:
        # base game_step consumes one RNG draw per step (step_rand_int).
        _ = self.rand_gen.randint(0, 1000000)

        var move_action = action % 9
        if action >= 9:
            move_action = 4  # special action → stand still

        var action_vx = Float32(move_action // 3 - 1)
        var action_vy = Float32(move_action % 3 - 1)
        # maze set_action_xy: no diagonal movement.
        if action_vx != 0:
            action_vy = 0

        # grid_step: velocity is the action directly; PLAYER steps x-first when
        # moving horizontally, else y-first.
        if action_vx != 0:
            self._sub_step(action_vx, 0)
            self._sub_step(0, action_vy)
        else:
            self._sub_step(0, action_vy)
            self._sub_step(action_vx, 0)

        if action_vx > 0:
            self.agent.is_reflected = True
        if action_vx < 0:
            self.agent.is_reflected = False

        var reward: Float32 = 0.0
        var ix = Int(self.agent.x)
        var iy = Int(self.agent.y)
        if self._get_obj(ix, iy) == GOAL:
            self._set_obj(ix, iy, SPACE)
            reward += REWARD
            self.level_complete = True

        self.episode_reward += reward
        self.done = reward > 0
        return reward

    def render(self, out_res: Int = RES) -> List[UInt8]:
        # out_res=64 = the agent observation; pass a larger value (e.g. 512) for
        # a crisp human-play / debug frame where small sprites stay visible.
        var canvas = Canvas(out_res)
        canvas.fill(0, 0, 0)

        var visibility = Float32(self.w if self.w > self.h else self.h)
        var view_dim = visibility
        var unit = Float32(out_res) / view_dim
        var center_x = Float32(self.w) * 0.5
        var center_y = Float32(self.h) * 0.5
        var x_off = unit * (center_x - view_dim / 2)
        var y_off = unit * (center_y - view_dim / 2)

        # Background (draw_background, topdown, bg_tile_ratio=0). The main_rect
        # is the whole world; the bg is scaled to cover its height and panned
        # horizontally by bg_pct_x over the extra width. main_rect for maze is
        # (0,0,RES,RES) since x_off==y_off==0 and view_dim==main_height.
        ref bg = self.backgrounds[self.background_index]
        var main_w = Float32(self.w) * unit
        var main_h = Float32(self.h) * unit
        var main_x = -x_off
        var main_y = (view_dim - Float32(self.h)) * unit + y_off
        var bg_ar = Float32(bg.w) / Float32(bg.h)
        var world_ar = Float32(self.w) / Float32(self.h)
        var offset_x = self.bg_pct_x * (bg_ar - world_ar)
        # adjust_rect(main_rect, (-offset_x, 0, bg_ar/world_ar, 1)).
        var bg_x = main_x + main_w * (-offset_x)
        var bg_w = main_w * (bg_ar / world_ar)
        canvas.blit(bg, bg_x, main_y, bg_w, main_h)

        for x in range(self.w):
            for y in range(self.h):
                var t = self._get_obj(x, y)
                if t == SPACE or t == INVALID_OBJ:
                    continue
                var sx = (Float32(x) - RENDER_EPS) * unit - x_off
                var sy = (view_dim - Float32(y + 1) - RENDER_EPS) * unit + y_off
                var sz = (1.0 + 2 * RENDER_EPS) * unit
                if t == WALL_OBJ:
                    canvas.blit(self.sand, sx, sy, sz, sz)
                elif t == GOAL:
                    canvas.blit(self.cheese, sx, sy, sz, sz)

        # agent (mouse), get_object_rect(agent).
        var ax = (self.agent.x - self.agent.rx) * unit - x_off
        var ay = (view_dim - (self.agent.y + self.agent.ry)) * unit + y_off
        var aw = 2 * self.agent.rx * unit
        var ah = 2 * self.agent.ry * unit
        canvas.blit(self.mouse, ax, ay, aw, ah, self.agent.is_reflected)

        return canvas.px.copy()
