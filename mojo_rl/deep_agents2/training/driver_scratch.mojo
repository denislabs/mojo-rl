"""DriverScratch[NAME, N_ENVS, DIM] — driver-owned, N_ENVS-sized buffer.

Tier-1 unification: the off-policy driver currently owns ~9 distinct
device buffers in the N_ENVS GPU path (`states`, `prev_obs`, `obs_buf`,
`actions`, `rewards`, `dones`, `terminated`, `ao_scratch`, `alp_scratch`)
plus per-call `List` mirrors in the CPU + single-env paths. DriverScratch
collapses both behind one typed wrapper, exactly mirroring nn2's existing
trainer-side `Scratch[NAME, SIZE, STAGING]` (see `nn2/core/scratch.mojo`)
but with the N_ENVS dimension pulled into a comptime parameter so the
total size is `N_ENVS * DIM`.

Symmetry with `Scratch`:

  * `Scratch[NAME, SIZE]`         — trainer-owned, BATCH baked in by
                                    the owning block.
  * `DriverScratch[NAME, N, DIM]` — driver-owned, N_ENVS exposed.

Both carry CPU `List` + GPU `Optional[DeviceBuffer]`, populated by
`make[target]`. Both expose `target_ptr[target]() -> UnsafePointer` for
generic access plus explicit `host_ptr()` / `dev_ptr()` for the staging
paths that need both at once.

The `with_host_mirror` knob on `make["gpu"]` mirrors `Scratch`'s
`STAGING` parameter (kept runtime here — driver knows at construction
whether it needs a host shadow for episode tracking D2H, no value in
fanning the type out comptime).
"""

from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import mptr


struct DriverScratch[NAME: StaticString, N_ENVS: Int, DIM: Int](
    Movable & ImplicitlyDestructible
):
    comptime SIZE: Int = Self.N_ENVS * Self.DIM

    var cpu: List[Scalar[DT]]
    var dev: Optional[DeviceBuffer[DT]]

    def __init__(out self):
        self.cpu = List[Scalar[DT]]()
        self.dev = None

    @staticmethod
    def make[
        target: StaticString
    ](
        ctx: Optional[DeviceContext] = None,
        with_host_mirror: Bool = False,
    ) raises -> Self:
        """Allocate storage for `target`. On GPU, `with_host_mirror=True`
        also allocates a CPU `List` of the same size for H2D/D2H staging
        — the analogue of `Scratch[..., STAGING=True]`."""
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "DriverScratch.make: target must be 'cpu' or 'gpu'"
        var s = Self()
        comptime if target == "cpu":
            s.cpu = List[Scalar[DT]](
                length=Self.SIZE, fill=Scalar[DT](0.0)
            )
        else:
            if not ctx:
                raise Error(
                    "DriverScratch.make[target='gpu']: ctx required"
                )
            s.dev = ctx.value().enqueue_create_buffer[DT](Self.SIZE)
            if with_host_mirror:
                s.cpu = List[Scalar[DT]](
                    length=Self.SIZE, fill=Scalar[DT](0.0)
                )
        return s^

    # ----- Pointer accessors --------------------------------------------

    def host_ptr(ref self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        """Raw CPU pointer. Valid for `target="cpu"` and for
        `target="gpu"` builds that requested `with_host_mirror=True`."""
        return mptr(self.cpu.unsafe_ptr())

    def dev_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        """Raw GPU pointer. Valid only after `make["gpu"]`."""
        return mptr(self.dev.value().unsafe_ptr())

    def target_ptr[
        target: StaticString
    ](self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        """Unified pointer accessor. CPU build returns the host pointer;
        GPU build returns the device pointer (host mirror, if present,
        is reachable via `host_ptr()`)."""
        comptime if target == "cpu":
            return self.host_ptr()
        elif target == "gpu":
            return self.dev_ptr()
        else:
            comptime assert False, "target must be 'cpu' or 'gpu'"

    # ----- Buffer accessors (for enqueue_copy / kernel launch) ----------
    #
    # The Optional[DeviceBuffer] field is left publicly accessible so
    # call sites that need a `mut DeviceBuffer` (env step kernels,
    # selective_reset_kernel_gpu, etc.) can write `obs_buf.dev.value()`
    # and have the mutability propagate from the local. This mirrors
    # the existing nn2 pattern (e.g. `Scratch.dev.value()` is used the
    # same way in sac_trainer / sac_actor_loss).

    # ----- Sizes --------------------------------------------------------

    def size(self) -> Int:
        return Self.SIZE
