"""`Grid` — flat row-major integer grid (port of Procgen `grid.h`).

Int-typed for the Phase-0 spike (Procgen's `Grid<int>` is the only instantiation
the maze path uses). Generalize to `Grid[T]` later if a game needs it.
"""


struct Grid(Copyable, Movable):
    var w: Int
    var h: Int
    var data: List[Int]

    def __init__(out self):
        self.w = 0
        self.h = 0
        self.data = List[Int]()

    def resize(mut self, width: Int, height: Int):
        self.w = width
        self.h = height
        self.data = List[Int]()
        self.data.resize(width * height, 0)

    def contains(self, x: Int, y: Int) -> Bool:
        return 0 <= y and y < self.h and 0 <= x and x < self.w

    def get(self, x: Int, y: Int) -> Int:
        return self.data[y * self.w + x]

    def get_index(self, index: Int) -> Int:
        return self.data[index]

    def to_index(self, x: Int, y: Int) -> Int:
        return y * self.w + x

    def set(mut self, x: Int, y: Int, v: Int):
        self.data[y * self.w + x] = v

    def set_index(mut self, index: Int, v: Int):
        self.data[index] = v
