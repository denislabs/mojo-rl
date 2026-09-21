"""`RandGen` — faithful port of Procgen's `randgen.cpp`.

Every helper draws from the exact `std::mt19937` in `mt19937.mojo` in the same
order as the reference, so a given seed reproduces identical level layouts
(level-exact fidelity — see `docs/PROCGEN_PORT.md`). Integer-modulo bias and the
`rand01` double-division are reproduced deliberately.

MT-state serialize/deserialize (`use_sequential_levels` save/restore) is out of
scope for the Phase-0 spike.
"""

from .mt19937 import MT19937


struct RandGen(Copyable, Movable):
    var gen: MT19937
    var is_seeded: Bool

    def __init__(out self):
        self.gen = MT19937()
        self.is_seeded = False

    def seed(mut self, s: Int):
        self.gen.seed(UInt32(s))
        self.is_seeded = True

    # --- raw ---
    def randint(mut self) -> UInt32:
        return self.gen.next_u32()

    # --- integer ranges ---
    def randn(mut self, high: Int) -> Int:
        # C++: (x % high) with x uint32 → unsigned modulo.
        return Int(self.gen.next_u32() % UInt32(high))

    def randint(mut self, low: Int, high: Int) -> Int:
        # C++: range = high - low; low + (x % range).
        var rng = UInt32(high - low)
        return low + Int(self.gen.next_u32() % rng)

    # --- floats ---
    def rand01(mut self) -> Float32:
        # C++: (float)((double)x / ((double)max() + 1)) == x / 2^32.
        return Float32(Float64(self.gen.next_u32()) / 4294967296.0)

    def randrange(mut self, low: Float32, high: Float32) -> Float32:
        return self.rand01() * (high - low) + low

    def randbool(mut self) -> Bool:
        return self.rand01() > 0.5

    # --- composites ---
    def partition(mut self, x: Int, n: Int) -> List[Int]:
        var part = List[Int]()
        part.resize(n, 0)
        for _ in range(x):
            part[self.randn(n)] += 1
        return part^

    def choose_one(mut self, mut elems: List[Int]) -> Int:
        return elems[self.randn(len(elems))]

    def choose_n(mut self, elems: List[Int], n: Int) -> List[Int]:
        var rem = elems.copy()
        if n > len(rem):
            return rem^
        var chosen = List[Int]()
        while len(chosen) < n:
            var idx = self.randn(len(rem))
            chosen.append(rem[idx])
            _ = rem.pop(idx)
        return chosen^

    def simple_choose(mut self, n: Int, k: Int) -> List[Int]:
        var chosen = List[Int]()
        chosen.resize(k, 0)
        var seen = List[Bool]()
        seen.resize(n, False)
        for i in range(k):
            var nxt = self.randn(n)
            while seen[nxt]:
                nxt = self.randn(n)
            chosen[i] = nxt
            seen[nxt] = True
        return chosen^
