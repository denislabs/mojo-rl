"""Scratch[NAME, SIZE, STAGING=False] — non-trainable scratch buffer + lifecycle.

Phase 1.4. Mirrors `Param[NAME, DECAY, SIZE]` design for the second
class of field clusters every block declares — the working scratches
(activations, gradient pieces, packed buffers, …) that aren't
parameters and don't get visited by the optimizer.

A block that previously declared

    var _mb_ao:     List[Scalar[DT]]
    var _mb_ao_dev: Optional[DeviceBuffer[DT]]
    # ...times 16

and initialised every one inside `make[cpu]` / `make[gpu]` will, post-
migration, declare

    var ao:  Scratch["ao",  BATCH * 2 * ACT_DIM]
    # ...times 16

and call `init_scratch_auto[Self, target](self, ctx)` once. Reflection
in `core/scratch_walkers.mojo` walks the `Scratch`-typed fields, filters
by `conforms_to(_, IsScratch)`, and dispatches `init_with[target]`.

Same CPU + GPU dual storage pattern as `Param` (both lists/buffers
always present in the struct; only the matching one is populated on
init, selected by `target`).

**STAGING flag (Phase 2.5):** When `STAGING=True`, `init_with["gpu"]`
also allocates the CPU list (in addition to the device buffer). This is
the "host staging" pattern used by `ActionSamplingBlock` — the block
runs the actor on GPU but downloads sampler output to host for the
final clamp. Default `STAGING=False` preserves the strict CPU-OR-GPU
allocation behavior every Phase 2 block relies on.
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from ..constants import DT


# ──────────────────────────────────────────────────────────────────────
# IsScratch — marker trait the walker filters on. Minimal surface:
# scratch_name + init_with(target, ctx). No visit/visitor since
# scratches are write-only from the block's perspective; consumers grab
# the raw pointer via `cpu_ptr()` / `dev_ptr()`.
# ──────────────────────────────────────────────────────────────────────


trait IsScratch(Movable & ImplicitlyDestructible):
    """Marker — a field-type the scratch-walker should initialise."""

    def scratch_name(self) -> StaticString:
        ...

    def scratch_size(self) -> Int:
        ...

    def init_with[target: StaticString](
        mut self, ctx: Optional[DeviceContext]
    ) raises:
        ...


# ──────────────────────────────────────────────────────────────────────
# Scratch[NAME, SIZE, STAGING=False] — flat 1D buffer. Each (NAME, SIZE,
# STAGING) triple is a distinct type, so reflection sees fields
# distinctly. CPU `List` + GPU `Optional[DeviceBuffer]` always present;
# selection via the owning block's `TargetStorage.target_tag`.
# ──────────────────────────────────────────────────────────────────────


struct Scratch[NAME: StaticString, SIZE: Int, STAGING: Bool = False](IsScratch):
    var cpu: List[Scalar[DT]]
    var dev: Optional[DeviceBuffer[DT]]

    def __init__(out self):
        self.cpu = List[Scalar[DT]]()
        self.dev = None

    @staticmethod
    def make_cpu() raises -> Self:
        var s = Self()
        s.cpu = List[Scalar[DT]](length=Self.SIZE, fill=Scalar[DT](0.0))
        return s^

    @staticmethod
    def make_gpu(ctx: DeviceContext) raises -> Self:
        var s = Self()
        s.dev = ctx.enqueue_create_buffer[DT](Self.SIZE)
        comptime if Self.STAGING:
            # Staging scratch: keep a CPU mirror alongside the device
            # buffer for host-side upload/download bookkeeping.
            s.cpu = List[Scalar[DT]](
                length=Self.SIZE, fill=Scalar[DT](0.0),
            )
        return s^

    def init_with[target: StaticString](
        mut self, ctx: Optional[DeviceContext]
    ) raises:
        """Walker entry point. Called once per scratch field by
        `init_scratch_auto[T, target]`. Populates the matching storage
        and leaves the other in its default state.

        When `Self.STAGING == True`, GPU init also allocates the CPU
        list — see the struct docstring."""
        comptime if target == "cpu":
            self.cpu = List[Scalar[DT]](
                length=Self.SIZE, fill=Scalar[DT](0.0),
            )
        else:
            self.dev = ctx.value().enqueue_create_buffer[DT](Self.SIZE)
            comptime if Self.STAGING:
                self.cpu = List[Scalar[DT]](
                    length=Self.SIZE, fill=Scalar[DT](0.0),
                )

    def scratch_name(self) -> StaticString:
        return Self.NAME

    def scratch_size(self) -> Int:
        return Self.SIZE

    # ----- Pointer accessors --------------------------------------------

    def cpu_ptr(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        """Returns the raw CPU pointer. Caller must know the target is CPU
        (or that this is a STAGING scratch — both populate `self.cpu`)."""
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.cpu.unsafe_ptr()
        )

    def dev_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        """Returns the raw GPU pointer. Caller must know the target is GPU
        and that `init_with["gpu"]` has been called."""
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.dev.value().unsafe_ptr()
        )
