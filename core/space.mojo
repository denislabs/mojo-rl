from random import random_si64


trait Space:
    """Base trait for action/observation spaces."""

    fn sample(self) -> Int:
        """Sample a random element from the space."""
        ...

    fn contains(self, element: Int) -> Bool:
        """Check if element is within the space."""
        ...


struct DiscreteSpace(Space):
    """A discrete space of n possible values: {0, 1, ..., n-1}."""

    var n: Int

    fn __init__(out self, n: Int):
        self.n = n

    fn sample(self) -> Int:
        """Sample a random integer from [0, n)."""
        # random_si64 is inclusive on both ends, so use n - 1
        return Int(random_si64(0, self.n - 1))

    fn contains(self, element: Int) -> Bool:
        """Check if element is in [0, n)."""
        return element >= 0 and element < self.n


struct BoxSpace[dim: Int]:
    """A continuous space represented as a box in R^dim."""

    var low: SIMD[DType.float64, Self.dim]
    var high: SIMD[DType.float64, Self.dim]

    fn __init__(out self, low: SIMD[DType.float64, Self.dim], high: SIMD[DType.float64, Self.dim]):
        self.low = low
        self.high = high

    fn sample(self) -> SIMD[DType.float64, Self.dim]:
        """Sample a random point uniformly from the box."""
        var result = SIMD[DType.float64, Self.dim]()
        for i in range(Self.dim):
            # Generate uniform random in [low[i], high[i]]
            var rand_val = Float64(random_si64(0, 1000000)) / 1000000.0
            result[i] = self.low[i] + rand_val * (self.high[i] - self.low[i])
        return result

    fn contains(self, element: SIMD[DType.float64, Self.dim]) -> Bool:
        """Check if element is within the box bounds."""
        for i in range(Self.dim):
            if element[i] < self.low[i] or element[i] > self.high[i]:
                return False
        return True
