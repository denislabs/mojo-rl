"""`MazeGen` — Kruskal maze generator (port of Procgen `mazegen.cpp`).

Reproduces `generate_maze`, `generate_maze_no_dead_ends`, and `place_objects`
draw-for-draw so a seeded `RandGen` yields the exact reference layout (used by
maze / chaser / heist). `generate_maze_with_doors` (heist) is deferred to Phase 1.

Merge note: the reference maintains `std::set` cell-sets and relabels the merged
set to `s1_idx`. Because `lookup()` is only ever called on maze cells (even/even)
and never on wall centers, an equivalent — and RNG-order-identical — union is to
relabel every cell currently tagged `s0_idx` to `s1_idx`. No randomness is
consumed during the merge, so this is bit-faithful for level generation.
"""

from .grid import Grid
from .randgen import RandGen
from .object_ids import SPACE, WALL_OBJ, INVALID_OBJ

comptime MAZE_OFFSET = 1


@fieldwise_init
struct Wall(ImplicitlyCopyable, Movable):
    var x1: Int
    var y1: Int
    var x2: Int
    var y2: Int


struct MazeGen(Copyable, Movable):
    var grid: Grid
    var maze_dim: Int
    var array_dim: Int
    var num_free_cells: Int
    var cell_sets_idxs: List[Int]
    var free_cells: List[Int]
    var free_cell_set: List[Bool]

    def __init__(out self, maze_dim: Int):
        self.maze_dim = maze_dim
        self.array_dim = maze_dim + 2
        self.num_free_cells = 0
        var n = self.array_dim * self.array_dim
        self.cell_sets_idxs = List[Int]()
        self.cell_sets_idxs.resize(n, 0)
        self.free_cells = List[Int]()
        self.free_cells.resize(n, 0)
        self.free_cell_set = List[Bool]()
        self.free_cell_set.resize(maze_dim * maze_dim, False)
        self.grid = Grid()
        self.grid.resize(self.array_dim, self.array_dim)

    def lookup(self, x: Int, y: Int) -> Int:
        return self.cell_sets_idxs[self.maze_dim * y + x]

    def set_free_cell(mut self, x: Int, y: Int):
        self.grid.set(x + MAZE_OFFSET, y + MAZE_OFFSET, SPACE)
        var cell = self.maze_dim * y + x
        if not self.free_cell_set[cell]:
            self.free_cells[self.num_free_cells] = cell
            self.free_cell_set[cell] = True
            self.num_free_cells += 1

    def get_obj(self, idx: Int) -> Int:
        var x = idx % self.array_dim
        var y = idx // self.array_dim
        if x <= 0 or x >= self.array_dim - 1:
            return INVALID_OBJ
        if y <= 0 or y >= self.array_dim - 1:
            return INVALID_OBJ
        return self.grid.get(x, y)

    def get_neighbors(self, idx: Int, type: Int) -> List[Int]:
        var x = idx % self.array_dim
        var y = idx // self.array_dim
        var neighbors = List[Int]()
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                if dx != 0 and dy != 0:
                    continue
                var n_idx = self.grid.to_index(x + dx, y + dy)
                if self.get_obj(n_idx) == type:
                    neighbors.append(n_idx)
        return neighbors^

    def generate_maze(mut self, mut rand_gen: RandGen):
        for i in range(self.array_dim):
            for j in range(self.array_dim):
                self.grid.set(i, j, WALL_OBJ)

        self.grid.set(MAZE_OFFSET, MAZE_OFFSET, 0)

        var walls = List[Wall]()

        self.num_free_cells = 0
        for i in range(len(self.free_cell_set)):
            self.free_cell_set[i] = False

        self.cell_sets_idxs[0] = 0
        for i in range(1, self.maze_dim * self.maze_dim):
            self.cell_sets_idxs[i] = i

        var md = self.maze_dim
        var i = 1
        while i < md:
            var j = 0
            while j < md:
                if i > 0 and i < md - 1:
                    walls.append(Wall(i - 1, j, i + 1, j))
                j += 2
            i += 2

        i = 0
        while i < md:
            var j = 1
            while j < md:
                if j > 0 and j < md - 1:
                    walls.append(Wall(i, j - 1, i, j + 1))
                j += 2
            i += 2

        while len(walls) > 0:
            var n = rand_gen.randn(len(walls))
            var wall = walls[n]

            var s0_idx = self.lookup(wall.x1, wall.y1)
            var s1_idx = self.lookup(wall.x2, wall.y2)

            var x0 = (wall.x1 + wall.x2) // 2
            var y0 = (wall.y1 + wall.y2) // 2
            var center = md * y0 + x0

            var can_remove = (
                self.grid.get(x0 + MAZE_OFFSET, y0 + MAZE_OFFSET) == WALL_OBJ
            ) and (s0_idx != s1_idx)

            if can_remove:
                self.set_free_cell(wall.x1, wall.y1)
                self.set_free_cell(x0, y0)
                self.set_free_cell(wall.x2, wall.y2)

                # Relabel s0's component into s1 (see module docstring).
                for c in range(md * md):
                    if self.cell_sets_idxs[c] == s0_idx:
                        self.cell_sets_idxs[c] = s1_idx
                self.cell_sets_idxs[center] = s1_idx

            _ = walls.pop(n)

    def generate_maze_no_dead_ends(mut self, mut rand_gen: RandGen):
        self.generate_maze(rand_gen)
        for i in range(self.array_dim * self.array_dim):
            if self.get_obj(i) == SPACE:
                var adj_space = self.get_neighbors(i, SPACE)
                if len(adj_space) == 1:
                    var adj_wall = self.get_neighbors(i, WALL_OBJ)
                    if len(adj_wall) > 0:
                        var n = rand_gen.randn(len(adj_wall))
                        self.grid.set_index(adj_wall[n], SPACE)

    def place_objects(mut self, mut rand_gen: RandGen, start_obj: Int, num_objs: Int):
        for j in range(num_objs):
            var m = rand_gen.randn(self.num_free_cells)
            while self.free_cells[m] == -1 or self.free_cells[m] == 0:
                m = rand_gen.randn(self.num_free_cells)
            var coin_cell = self.free_cells[m]
            self.free_cells[m] = -1
            self.grid.set(
                coin_cell % self.maze_dim + MAZE_OFFSET,
                coin_cell // self.maze_dim + MAZE_OFFSET,
                start_obj + j,
            )
