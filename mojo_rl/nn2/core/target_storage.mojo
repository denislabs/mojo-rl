"""TargetStorage + free helpers (NN2_AUDIT retrofit).

Bundles the (`_target_tag`, `_inference`, `ctx`) field cluster that
every nn2 leaf currently declares as three separate fields into a
single composable field. Replaces the per-leaf `_assert_tag[target]`
method with a module-level free function and the per-leaf
`_ensure_cache_cpu` / `_ensure_cache_gpu` helpers with module-level
free helpers.

Per-leaf savings: ~3 fields → 1 field, ~16 LOC of boilerplate methods
removed. Across 22 leaves: ~350 LOC of pure boilerplate.

Migration: a leaf that previously declared
    var _target_tag: Int8
    var _inference: Bool
    var ctx: Optional[DeviceContext]
    var cache: List[Scalar[DT]]
    var cache_dev: Optional[DeviceBuffer[DT]]
    var cache_dev_n: Int

now declares
    var ts: TargetStorage
    var cache: List[Scalar[DT]]                 # if it owns a cache
    var cache_dev: Optional[DeviceBuffer[DT]]
    var cache_dev_n: Int

(unified-buffer design — Spike #1 — also eliminates the cache fields
for input-caching layers; this struct only handles the tag/inference/ctx
cluster.)

The previous per-leaf `_assert_tag[target](self)` method becomes:
    assert_tag_for[name="Linear", target=target](self.ts.target_tag)

CPU + GPU buffer-grow helpers are extracted similarly. See sibling
`nn2/core/target_tag.mojo` for the underlying constants and
`target_tag_for[target]()` mapping.
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from ..constants import DT
from .target_tag import (
    TARGET_UNINIT,
    TARGET_CPU,
    TARGET_GPU,
    target_tag_for,
)


# ──────────────────────────────────────────────────────────────────────
# TargetStorage — the per-leaf bookkeeping cluster.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct TargetStorage(Movable & ImplicitlyDestructible):
    """Bundles `target_tag` + `inference` + `ctx` into one composable
    field. Every nn2 leaf retrofit owns exactly one of these.

    `ctx` is `Some` when the leaf was built via `make[target='gpu', INIT](ctx)`
    and `None` for CPU leaves."""

    var target_tag: Int8
    var inference: Bool
    var ctx: Optional[DeviceContext]

    @staticmethod
    def make_uninit() -> Self:
        """Default-constructible factory — produces UNINIT-tagged storage.
        Used by `Defaultable.__init__` overloads on retrofit leaves."""
        return Self(target_tag=TARGET_UNINIT, inference=False, ctx=None)

    @staticmethod
    def make_cpu() -> Self:
        """CPU-tagged storage. No DeviceContext."""
        return Self(target_tag=TARGET_CPU, inference=False, ctx=None)

    @staticmethod
    def make_gpu(ctx: DeviceContext) -> Self:
        """GPU-tagged storage with the DeviceContext stamped in."""
        return Self(target_tag=TARGET_GPU, inference=False, ctx=ctx)


# ──────────────────────────────────────────────────────────────────────
# assert_tag_for — replaces the per-leaf `_assert_tag[target]` method.
# ──────────────────────────────────────────────────────────────────────


def assert_tag_for[name: StaticString, target: StaticString](
    tag: Int8,
) raises:
    """Raise if `tag` does not match the comptime-resolved `target`.

    Call from method bodies:
        assert_tag_for["Linear", target](self.ts.target_tag)

    `name` appears in the error message so the caller knows which
    leaf failed the check."""
    comptime expected = target_tag_for[target]()
    if tag != expected:
        raise Error(
            "[" + String(name) + "] method called with [target='"
            + String(target)
            + "'] but module was make'd for a different target "
            + "(tag=" + String(Int(tag)) + ")"
        )


# ──────────────────────────────────────────────────────────────────────
# ensure_cpu_buffer / ensure_gpu_buffer — replace per-leaf
# `_ensure_cache_cpu` / `_ensure_cache_gpu` helpers.
# ──────────────────────────────────────────────────────────────────────


def ensure_cpu_buffer(mut buf: List[Scalar[DT]], needed: Int):
    """Lazy-grow `buf` to at least `needed` elements, zero-filled."""
    if len(buf) < needed:
        buf.resize(needed, Scalar[DT](0.0))


def ensure_gpu_buffer(
    mut buf: Optional[DeviceBuffer[DT]],
    mut cap: Int,
    needed: Int,
    ctx: DeviceContext,
) raises:
    """Lazy-grow GPU buffer to `needed`. Updates `cap` if reallocated."""
    if cap < needed:
        buf = ctx.enqueue_create_buffer[DT](needed)
        cap = needed
