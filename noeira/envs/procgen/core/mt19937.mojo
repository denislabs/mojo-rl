"""Exact `std::mt19937` (32-bit Mersenne Twister).

Bit-for-bit reproduction of the C++ standard `std::mt19937` used by Procgen's
`RandGen` (`references/procgen-master/procgen/src/randgen.{h,cpp}`). Seeding and
tempering follow the reference MT19937 (Matsumoto & Nishimura, 1998), which is
what libstdc++/libc++ implement. This is the reproducibility backbone for
level-exact procedural generation — see `docs/PROCGEN_PORT.md`.

Validated in `tests/envs/procgen/test_mt19937_parity.mojo` against the C++
ground-truth probe (e.g. seed 5489 → 10000th draw == 4123659995).
"""

comptime N = 624
comptime M = 397
comptime MATRIX_A: UInt32 = 0x9908B0DF
comptime UPPER_MASK: UInt32 = 0x80000000  # most significant w-r bits
comptime LOWER_MASK: UInt32 = 0x7FFFFFFF  # least significant r bits
comptime INIT_MULT: UInt32 = 1812433253


struct MT19937(Copyable, Movable):
    """32-bit Mersenne Twister matching `std::mt19937`."""

    var mt: List[UInt32]
    var mti: Int

    def __init__(out self, seed: UInt32 = 5489):
        self.mt = List[UInt32]()
        self.mt.resize(N, 0)
        self.mti = N + 1
        self.seed(seed)

    def seed(mut self, seed: UInt32):
        # init_genrand: mt[0]=s; mt[i]=1812433253*(mt[i-1]^(mt[i-1]>>30))+i
        self.mt[0] = seed
        for i in range(1, N):
            var prev = self.mt[i - 1]
            self.mt[i] = INIT_MULT * (prev ^ (prev >> 30)) + UInt32(i)
        self.mti = N

    def next_u32(mut self) -> UInt32:
        var y: UInt32
        if self.mti >= N:
            # Regenerate the whole state array in one pass.
            var kk = 0
            while kk < N - M:
                y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK)
                self.mt[kk] = self.mt[kk + M] ^ (y >> 1) ^ (
                    MATRIX_A if (y & 1) != 0 else UInt32(0)
                )
                kk += 1
            while kk < N - 1:
                y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK)
                self.mt[kk] = self.mt[kk + (M - N)] ^ (y >> 1) ^ (
                    MATRIX_A if (y & 1) != 0 else UInt32(0)
                )
                kk += 1
            y = (self.mt[N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK)
            self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ (
                MATRIX_A if (y & 1) != 0 else UInt32(0)
            )
            self.mti = 0

        y = self.mt[self.mti]
        self.mti += 1

        # Tempering.
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y

    @staticmethod
    def max() -> UInt32:
        return 0xFFFFFFFF
