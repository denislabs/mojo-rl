"""`map_params` / `polyak_update` — two-tree parameter walker.

`polyak_update` mutates a target model toward an online model:

    target = (1 - tau) * target + tau * online

elementwise across every leaf parameter. Used for SAC's soft-update of
target critic nets each gradient step (tau ≈ 0.005).

Implementation builds on `named_params`: walks both models, validates
the resulting `List[NamedParam]`s match leaf-for-leaf (same count, same
names, same sizes), then runs the linear interpolation.

CPU path: scalar loop. GPU path: one-thread-per-element kernel launched
per leaf, using the named-params raw pointers (which point at the live
Param storage — CPU `List` or GPU `DeviceBuffer` depending on how the
Param was made).
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor, TileTensor

from ..constants import DT, TPB
from .module import Module
from .named_params import NamedParam, named_params, named_states
from .target_tag import TARGET_GPU, target_tag_for


def _polyak_kernel(
    online: UnsafePointer[Scalar[DT], MutAnyOrigin],
    target_net: UnsafePointer[Scalar[DT], MutAnyOrigin],
    one_minus_tau: Scalar[DT],
    tau: Scalar[DT],
    n: Int,
):
    var idx = Int(global_idx.x)
    if idx < n:
        target_net[idx] = one_minus_tau * target_net[idx] + tau * online[idx]


def polyak_update[
    target: StaticString,
    M: Module,
](
    mut online: M,
    mut target_net: M,
    tau: Scalar[DT],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Mutate `target_net` toward `online` by `tau`.

    `target = (1 - tau) * target + tau * online` per leaf parameter.

    Validates structure parity via `named_params`. Raises if the two
    walks disagree on count, name, or size (a sign of a typo or a model
    architecture mismatch).

    Block A: GPU path. The named-params raw pointers come from the
    Param wrapper's `param_ptr_for[target]` which returns the right
    storage's `.unsafe_ptr()`. The polyak kernel reads/writes through
    those pointers directly.
    """
    var online_ps = named_params[target, M](online)
    var target_ps = named_params[target, M](target_net)

    if len(online_ps) != len(target_ps):
        raise Error(
            "polyak_update: param count mismatch (online="
            + String(len(online_ps))
            + ", target="
            + String(len(target_ps))
            + ")"
        )

    var one_minus_tau: Scalar[DT] = Scalar[DT](1.0) - tau

    for i in range(len(online_ps)):
        ref op = online_ps[i]
        ref tp = target_ps[i]
        if op.n_elems != tp.n_elems:
            raise Error(
                "polyak_update: param size mismatch at index "
                + String(i)
                + " (online '"
                + op.name
                + "' n="
                + String(op.n_elems)
                + ", target '"
                + tp.name
                + "' n="
                + String(tp.n_elems)
                + ")"
            )
        if op.name != tp.name:
            raise Error(
                "polyak_update: param name mismatch at index "
                + String(i)
                + " (online '"
                + op.name
                + "', target '"
                + tp.name
                + "')"
            )

        comptime if target == "cpu":
            var n = op.n_elems
            var op_ptr = op.param_ptr
            var tp_ptr = tp.param_ptr
            for k in range(n):
                tp_ptr[k] = one_minus_tau * tp_ptr[k] + tau * op_ptr[k]
        else:
            # GPU: launch one-thread-per-element via a kernel. The
            # named-params pointers are already on-device because the
            # Param wrapper's `param_ptr_for["gpu"]` returns the
            # DeviceBuffer's underlying pointer.
            comptime assert target == "gpu", (
                "polyak_update: target must be 'cpu' or 'gpu'"
            )
            if not ctx:
                raise Error(
                    "polyak_update[target='gpu']: ctx is required — "
                    "thread the SAC trainer's DeviceContext through "
                    "polyak_step / polyak_update to avoid per-call "
                    "DeviceContext() construction (Apple Metal command-"
                    "queue exhaustion)."
                )
            _polyak_launch_gpu(op, tp, one_minus_tau, tau, ctx.value())


def _polyak_launch_gpu(
    op: NamedParam,
    tp: NamedParam,
    one_minus_tau: Scalar[DT],
    tau: Scalar[DT],
    ctx: DeviceContext,
) raises:
    """Helper to launch the polyak kernel for one leaf. Extracted from
    the loop body so the comptime-for body doesn't carry a kernel
    constructor (keeps the inliner happy).

    Takes ctx explicitly: constructing a fresh `DeviceContext()` per leaf
    per train step exhausts Apple Metal command-queue resources within a
    few hundred SAC train steps."""
    var n = op.n_elems
    var n_blocks = (n + TPB - 1) // TPB
    ctx.enqueue_function[_polyak_kernel](
        op.param_ptr, tp.param_ptr, one_minus_tau, tau, n,
        grid_dim=n_blocks, block_dim=TPB,
    )


# ──────────────────────────────────────────────────────────────────────
# Grouped (multi-tensor) polyak — NVIDIA only.
#
# `polyak_update` launches one `_polyak_kernel` per leaf (~3/critic →
# ~6/update for twin target critics). Like the grouped Adam path, collapse
# them into ONE launch: a 1-D grid over all elements where each thread maps its
# flat index to (param, local) via the dense per-param offset table and
# reconstructs the online/target pointers from device-resident address arrays.
#
# NVIDIA-only (host-captured device-address deref is invalid on Apple Metal —
# see adam.mojo / tests/nn/test_grouped_adam_prototype.mojo). Cached on
# `OnlineTargetPair`; CPU + Apple keep the per-leaf `polyak_update`.
# ──────────────────────────────────────────────────────────────────────


def _grouped_polyak_kernel(
    online_addrs: UnsafePointer[UInt64, MutAnyOrigin],
    target_addrs: UnsafePointer[UInt64, MutAnyOrigin],
    offs: UnsafePointer[Int32, MutAnyOrigin],
    n_params: Int,
    total: Int,
    one_minus_tau: Scalar[DT],
    tau: Scalar[DT],
):
    var flat = Int(global_idx.x)
    if flat < total:
        var p = 0
        while p + 1 < n_params and Int(offs[p + 1]) <= flat:
            p += 1
        var local = flat - Int(offs[p])
        var online = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=Int(online_addrs[p])
        )
        var target_net = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=Int(target_addrs[p])
        )
        target_net[local] = (
            one_minus_tau * target_net[local] + tau * online[local]
        )


@fieldwise_init
struct GroupedPolyakCache(Movable & ImplicitlyDeletable):
    """Device-resident descriptor arrays for the grouped polyak launch, built
    ONCE from an (online, target) model pair (params are stable per-Param
    DeviceBuffers). `apply` launches one `_grouped_polyak_kernel`. NVIDIA only."""

    var online_addrs: DeviceBuffer[DType.uint64]
    var target_addrs: DeviceBuffer[DType.uint64]
    var offs: DeviceBuffer[DType.int32]
    var n_params: Int
    var total: Int

    @staticmethod
    def build[target: StaticString, M: Module](
        mut online: M, mut target_net: M, ctx: DeviceContext
    ) raises -> Self:
        var ops = named_params[target, M](online)
        var tps = named_params[target, M](target_net)
        if len(ops) != len(tps):
            raise Error("GroupedPolyakCache: param count mismatch")
        var oa = List[UInt64]()
        var ta = List[UInt64]()
        var offs_h = List[Int32]()
        var running = 0
        for i in range(len(ops)):
            if ops[i].n_elems != tps[i].n_elems or ops[i].name != tps[i].name:
                raise Error(
                    "GroupedPolyakCache: param mismatch at index " + String(i)
                )
            oa.append(UInt64(Int(ops[i].param_ptr)))
            ta.append(UInt64(Int(tps[i].param_ptr)))
            offs_h.append(Int32(running))
            running += ops[i].n_elems
        var n = len(ops)
        var cap = n if n > 0 else 1
        var oad = ctx.enqueue_create_buffer[DType.uint64](cap)
        var tad = ctx.enqueue_create_buffer[DType.uint64](cap)
        var ofd = ctx.enqueue_create_buffer[DType.int32](cap)
        var oah = ctx.enqueue_create_host_buffer[DType.uint64](cap)
        var tah = ctx.enqueue_create_host_buffer[DType.uint64](cap)
        var ofh = ctx.enqueue_create_host_buffer[DType.int32](cap)
        ctx.synchronize()
        for i in range(n):
            oah.unsafe_ptr()[i] = oa[i]
            tah.unsafe_ptr()[i] = ta[i]
            ofh.unsafe_ptr()[i] = offs_h[i]
        ctx.enqueue_copy(oad, oah)
        ctx.enqueue_copy(tad, tah)
        ctx.enqueue_copy(ofd, ofh)
        return Self(
            online_addrs=oad^,
            target_addrs=tad^,
            offs=ofd^,
            n_params=n,
            total=running,
        )

    def apply(
        self, one_minus_tau: Scalar[DT], tau: Scalar[DT], ctx: DeviceContext
    ) raises:
        if self.n_params > 0:
            var n_blocks = (self.total + TPB - 1) // TPB
            ctx.enqueue_function[_grouped_polyak_kernel](
                rebind[UnsafePointer[UInt64, MutAnyOrigin]](
                    self.online_addrs.unsafe_ptr()
                ),
                rebind[UnsafePointer[UInt64, MutAnyOrigin]](
                    self.target_addrs.unsafe_ptr()
                ),
                rebind[UnsafePointer[Int32, MutAnyOrigin]](
                    self.offs.unsafe_ptr()
                ),
                self.n_params,
                self.total,
                one_minus_tau,
                tau,
                grid_dim=n_blocks,
                block_dim=TPB,
            )


def hard_copy_params[
    target: StaticString,
    M: Module,
](
    mut online: M,
    mut target_net: M,
    ctx: Optional[DeviceContext] = None,
) raises:
    """Copy `online` → `target_net` verbatim (tau=1.0): every `Param` AND
    every `IsState` buffer (BatchNorm running stats). Used to initialize
    target nets and to promote arena winners.

    The State copy is essential for BatchNorm nets: a hard copy that moves
    weights/γ/β but not running_mean/var produces a net whose EVAL-mode
    forward runs trained weights under stale (init: mean 0 / var 1)
    normalization constants — activations explode and the policy head
    emits non-finite outputs (the AlphaZero post-promotion collapse).
    Stateless nets (MLPs) have an empty state walk — bit-identical no-op."""
    polyak_update[target, M](online, target_net, Scalar[DT](1.0), ctx)
    hard_copy_states[target, M](online, target_net, ctx)


def hard_copy_states[
    target: StaticString,
    M: Module,
](
    mut online: M,
    mut target_net: M,
    ctx: Optional[DeviceContext] = None,
) raises:
    """Copy every `IsState` buffer `online` → `target_net` (the
    `for_each_state` twin of the `polyak_update(tau=1)` param copy).
    Validates structure parity like `polyak_update`. No-op for
    stateless models."""
    var online_ss = named_states[target, M](online)
    var target_ss = named_states[target, M](target_net)

    if len(online_ss) != len(target_ss):
        raise Error(
            "hard_copy_states: state count mismatch (online="
            + String(len(online_ss))
            + ", target="
            + String(len(target_ss))
            + ")"
        )

    for i in range(len(online_ss)):
        ref os = online_ss[i]
        ref ts = target_ss[i]
        if os.n_elems != ts.n_elems or os.name != ts.name:
            raise Error(
                "hard_copy_states: state mismatch at index "
                + String(i)
                + " (online '"
                + os.name
                + "' n="
                + String(os.n_elems)
                + ", target '"
                + ts.name
                + "' n="
                + String(ts.n_elems)
                + ")"
            )
        comptime if target == "cpu":
            var os_ptr = os.param_ptr
            var ts_ptr = ts.param_ptr
            for k in range(os.n_elems):
                ts_ptr[k] = os_ptr[k]
        else:
            comptime assert target == "gpu", (
                "hard_copy_states: target must be 'cpu' or 'gpu'"
            )
            if not ctx:
                raise Error("hard_copy_states[target='gpu']: ctx is required")
            # Reuse the polyak kernel with tau=1 → target = online.
            _polyak_launch_gpu(
                os, ts, Scalar[DT](0.0), Scalar[DT](1.0), ctx.value()
            )
