"""TargetStorage + free helpers.

Bundles a leaf's `(target_tag, ctx)` cluster into a single composable
field. Every leaf has exactly one `var ts: TargetStorage`. Methods
gate on `assert_tag_for[name, target](self.ts.target_tag)` to catch
make-CPU / call-GPU misuse at the entry point.

Input-caching leaves alias the orchestrator's input slab through a
pointer field — no `cache: List[Scalar[DT]]` is needed.

Module-level helpers:

  - `assert_tag_for[name, target](tag)` — raise if `tag` does not match
    the comptime-resolved `target`.
  - `ensure_cpu_buffer` / `ensure_gpu_buffer` — lazy-grow CPU `List` /
    GPU `DeviceBuffer` scratch to a needed length.

See `nn2/core/target_tag.mojo` for the underlying constants and
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
    """Bundles `target_tag` + `ctx` into one composable field. Every
    nn2 leaf retrofit owns exactly one of these.

    `ctx` is `Some` when the leaf was built via `make[target='gpu', INIT](ctx)`
    and `None` for CPU leaves.

    Earlier revisions carried an `inference: Bool` flag for train/eval
    gating, but no consumer ever read it — `set_inference` was dropped
    when the slim Module trait landed. The field is gone."""

    var target_tag: Int8
    var ctx: Optional[DeviceContext]

    @staticmethod
    def make_uninit() -> Self:
        """Default-constructible factory — produces UNINIT-tagged storage.
        Used by `Defaultable.__init__` overloads on retrofit leaves."""
        return Self(target_tag=TARGET_UNINIT, ctx=None)

    @staticmethod
    def make_cpu() -> Self:
        """CPU-tagged storage. No DeviceContext."""
        return Self(target_tag=TARGET_CPU, ctx=None)

    @staticmethod
    def make_gpu(ctx: DeviceContext) -> Self:
        """GPU-tagged storage with the DeviceContext stamped in."""
        return Self(target_tag=TARGET_GPU, ctx=ctx)

    @staticmethod
    def make[target: StaticString](ctx: Optional[DeviceContext] = None) -> Self:
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "TargetStorage.make[target='cpu' or 'gpu']"
        comptime if target == "cpu":
            return Self(target_tag=TARGET_CPU, ctx=None)
        else:
            return Self(target_tag=TARGET_GPU, ctx=ctx)


# ──────────────────────────────────────────────────────────────────────
# assert_tag_for — replaces the per-leaf `_assert_tag[target]` method.
# ──────────────────────────────────────────────────────────────────────


def assert_tag_for[
    name: StaticString, target: StaticString
](tag: Int8,) raises:
    """Raise if `tag` does not match the comptime-resolved `target`.

    Call from method bodies:
        assert_tag_for["Linear", target](self.ts.target_tag)

    `name` appears in the error message so the caller knows which
    leaf failed the check."""
    comptime expected = target_tag_for[target]()
    if tag != expected:
        raise Error(
            "["
            + String(name)
            + "] method called with [target='"
            + String(target)
            + "'] but module was make'd for a different target "
            + "(tag="
            + String(Int(tag))
            + ")"
        )


# ──────────────────────────────────────────────────────────────────────
# require_ctx[where](ctx) — unwrap a GPU factory's Optional[DeviceContext]
# or raise a uniform "<where>: ctx required" error (S3-DRY, 2026-06-07).
# Collapses the repeated 3-line guard
#     if not ctx:
#         raise Error("X.make[target='gpu']: ctx required")
#     var ctx_v = ctx.value()
# to one line: `var ctx_v = require_ctx["X.make[target='gpu']"](ctx)`.
# NOTE: this is the *runtime* check — the compile-time variant is
# architecturally blocked (combinators are target-generic and thread an
# Optional ctx down, so a leaf's GPU make can't require a non-optional
# ctx without duplicating the whole make API; see the improvement audit
# §spike-table S3). Pass the full call-site string as `where`.
# ──────────────────────────────────────────────────────────────────────


def require_ctx[
    where: StaticString
](ctx: Optional[DeviceContext]) raises -> DeviceContext:
    if not ctx:
        raise Error(String(where) + ": ctx required")
    return ctx.value()


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
