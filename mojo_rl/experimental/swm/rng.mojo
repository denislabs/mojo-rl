"""A self-contained deterministic RNG for the SWM environments.

Not `std.random`: the gates must be reproducible from a seed carried in the
environment itself, independent of whatever else in a process called `seed()`.
xorshift64* — small, well-mixed enough for planting worlds and drawing noise,
and emphatically not for cryptography.
"""

from std.math import sqrt, log, cos, sin, pi


struct Rng(Copyable, ImplicitlyCopyable, Movable):
    var state: UInt64

    def __init__(out self, seed_value: UInt64 = 0x2026_0904_5357_4D48):
        # A zero state is a fixed point of xorshift; fold it away.
        self.state = seed_value if seed_value != 0 else 0x9E3779B97F4A7C15

    def next_u64(mut self) -> UInt64:
        var x = self.state
        x ^= x >> 12
        x ^= x << 25
        x ^= x >> 27
        self.state = x
        return x * 0x2545F4914F6CDD1D

    def uniform(mut self) -> Float64:
        """Uniform on [0, 1) from the top 53 bits."""
        return Float64(self.next_u64() >> 11) * (1.0 / 9007199254740992.0)

    def uniform_range(mut self, lo: Float64, hi: Float64) -> Float64:
        return lo + (hi - lo) * self.uniform()

    def normal(mut self) -> Float64:
        """Standard normal by Box-Muller."""
        var u1 = self.uniform()
        if u1 < 1e-300:
            u1 = 1e-300
        var u2 = self.uniform()
        return sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)
