"""BinaryElementOp — per-element binary op for `BinaryElementwise[DIM, OP]`.

Phase 4.5. Mirror of `ElementOp` but for 2-input → 1-output element-wise
operations. The `BinaryElementwise[DIM, OP]` template provides one CPU
SIMD body + one GPU kernel per (cached / uncached) direction; OP supplies
the per-lane math.

Current implementations:
  - `BinarySubOp`     (owns_cache=False) — output = x - y
  - `BinaryElemMinOp` (owns_cache=True)  — output = min(x, y), cache=mask

Future binaries (`BinaryMulOp` / `BinaryDivOp` / `BinaryWhereOp` / …) drop
in as one OP impl + one alias each. (Addition lives in `primitives/add.mojo`
as a variadic `Add[DIM, N]` primitive — it doesn't go through this trait.)

  - `forward_scalar(x, y)` / `forward_simd[W](x, y)`: output = f(x, y).
  - `cache_scalar(x, y)` / `cache_simd[W](x, y)`: per-element carry value
    stored at forward time (called only when `owns_cache = True`). For
    `BinaryElemMin` this is the mask (1.0 if in0 wins, 0.0 if in1 wins).
    Implementations for `owns_cache = False` ops still must define this
    method (trait requires it) but the template never calls it.
  - `backward_scalar_x(c, go)` / `backward_simd_x[W](c, go)`: gi0 = ∂f/∂x · go.
  - `backward_scalar_y(c, go)` / `backward_simd_y[W](c, go)`: gi1 = ∂f/∂y · go.
    For `owns_cache = False` ops, `c` is junk — these ops compute
    gradients from `go` alone (and the constant ∂f/∂x = ∂f/∂y = ±1 for
    Sub). The trait surface stays uniform so the template doesn't
    branch on cache mode at the inner-loop level.

  - `comptime owns_cache: Bool`: tells `BinaryElementwise[DIM, OP]`
    whether to allocate a cache buffer (`cache: List + cache_dev:
    Optional[DeviceBuffer]`) and dispatch the cached forward/backward
    kernel pair. `False` skips both allocation and the cache reads —
    `BinarySub` needs only `grad_output` to compute both grad-inputs.
"""

from ..constants import DT


trait BinaryElementOp(Movable & ImplicitlyDestructible):
    """Per-element binary math (scalar + SIMD specialisations) for the
    `BinaryElementwise[DIM, OP]` template."""

    comptime owns_cache: Bool
    """If True, forward writes a per-element carry to an owned cache
    buffer and backward reads it. If False, backward needs only
    `grad_output`."""

    @staticmethod
    def display_label() -> String:
        """Short display name for graph exporters. Default 'Binary'; ops
        override with their symbol (e.g. 'Min', 'Sub')."""
        return String("Binary")

    @staticmethod
    def forward_scalar(x: Scalar[DT], y: Scalar[DT]) -> Scalar[DT]:
        ...

    @staticmethod
    def forward_simd[W: Int](
        x: SIMD[DT, W], y: SIMD[DT, W]
    ) -> SIMD[DT, W]:
        ...

    @staticmethod
    def cache_scalar(x: Scalar[DT], y: Scalar[DT]) -> Scalar[DT]:
        """Per-element carry; called only when `owns_cache = True`. For
        `owns_cache = False` ops the body is unreachable — return any
        constant."""
        ...

    @staticmethod
    def cache_simd[W: Int](
        x: SIMD[DT, W], y: SIMD[DT, W]
    ) -> SIMD[DT, W]:
        ...

    @staticmethod
    def backward_scalar_x(c: Scalar[DT], go: Scalar[DT]) -> Scalar[DT]:
        """Compute `gi0 = ∂f/∂x · go`. `c` is the cached carry when
        `owns_cache`, unused otherwise."""
        ...

    @staticmethod
    def backward_simd_x[W: Int](
        c: SIMD[DT, W], go: SIMD[DT, W]
    ) -> SIMD[DT, W]:
        ...

    @staticmethod
    def backward_scalar_y(c: Scalar[DT], go: Scalar[DT]) -> Scalar[DT]:
        """Compute `gi1 = ∂f/∂y · go`."""
        ...

    @staticmethod
    def backward_simd_y[W: Int](
        c: SIMD[DT, W], go: SIMD[DT, W]
    ) -> SIMD[DT, W]:
        ...
