"""DQNTargetYBlock — target-Y compute for DQN (standard + Double).

Forward-only block — produces `y = r + γ · max_a Q_target(s', a) · (1 − d)`.
No gradient flows through this path; the trainer only uses `y` as a
target for the critic update.

Linear chain (Q_target.forward → ReduceMax.forward → finalize kernel),
not a ComputeGraph. Matches the design rationale in
`loss/critic_update_block.mojo`: "the chain is a single Module forward
+ MSELoss + backward + opt step, no fan-out, no Slice/Min, so a full
ComputeGraph would just add overhead. The win here is **scratch
ownership**, not DAG topology."

Double DQN (`DOUBLE=True`):

    y = r + γ · Q_target(s', argmax_a Q_online(s', a)) · (1 − d)

uses an additional `Q_online.forward(sp) → q_on`, an inline argmax
helper kernel that emits Scalar[DT] indices, then GatherCols[NA] to
pick `q_t[b, argmax]`. Tail finalize kernel is identical to the
standard branch.

CPU + GPU. Self-contained scratch ownership (init_scratch_auto).
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn2.constants import DT, TPB
from mojo_rl.nn2.core.tensor_pack import TensorPack
from mojo_rl.nn2.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.core.scratch import Scratch
from mojo_rl.nn2.core.scratch_walkers import init_scratch_auto
from mojo_rl.nn2.core.target_storage import TargetStorage, assert_tag_for
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.primitives.reduce_max import ReduceMax
from mojo_rl.nn2.primitives.gather_cols import GatherCols


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — argmax-index emission and the finalize fuse.
# ──────────────────────────────────────────────────────────────────────


def _argmax_idx_kernel[
    BATCH: Int,
    NA: Int,
](
    q: LayoutTensor[DT, Layout.row_major(BATCH, NA), MutAnyOrigin],
    idx_out: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
):
    """`idx_out[b, 0] = argmax_a q[b, a]`, emitted as Scalar[DT]."""
    var b = Int(global_idx.x)
    if b < BATCH:
        var best_a: Int = 0
        var best_q: Scalar[DT] = rebind[Scalar[DT]](q[b, 0])
        for a in range(1, NA):
            var v = rebind[Scalar[DT]](q[b, a])
            if v > best_q:
                best_q = v
                best_a = a
        idx_out[b, 0] = Scalar[DT](best_a)


def _target_y_finalize_kernel[
    BATCH: Int,
](
    max_q: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    r: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    d: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    y_out: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    gamma: Scalar[DT],
):
    """`y_out[b] = r[b] + γ · max_q[b, 0] · (1 − d[b])`."""
    var b = Int(global_idx.x)
    if b < BATCH:
        var nonterm = Scalar[DT](1.0) - rebind[Scalar[DT]](d[b])
        var mq = rebind[Scalar[DT]](max_q[b, 0])
        y_out[b] = rebind[Scalar[DT]](r[b]) + gamma * mq * nonterm


struct DQNTargetYBlock[
    Q_NET: Module,
    BATCH: Int,
    OBS: Int,
    NA: Int,
    DOUBLE: Bool = False,
](Defaultable & Movable & ImplicitlyDestructible):
    """Owns:
    - `ReduceMax[NA]` (standard branch reducer)
    - `GatherCols[NA]` (Double branch gather; constructed but never
      used on the standard path — keeps the field set uniform across
      DOUBLE configurations, matching how `BinaryElementwise` keeps
      its `cache` field unconditionally even when `OP.owns_cache=False`)
    - scratch for `q_all` ([B, NA]), `q_on_all` ([B, NA]; Double only),
      `idx` ([B, 1]; Double only), `max_q` ([B, 1]).
    """

    var reduce_max: ReduceMax[Self.NA]
    var gather_cols: GatherCols[Self.NA]

    var _q_all: Scratch["q_all", Self.BATCH * Self.NA, True]
    var _q_on_all: Scratch["q_on_all", Self.BATCH * Self.NA, True]
    var _idx: Scratch["idx", Self.BATCH, True]
    var _max_q: Scratch["max_q", Self.BATCH, True]

    var gamma: Scalar[DT]
    var ts: TargetStorage

    def __init__(out self):
        self.reduce_max = ReduceMax[Self.NA]()
        self.gather_cols = GatherCols[Self.NA]()
        self._q_all = Scratch["q_all", Self.BATCH * Self.NA, True]()
        self._q_on_all = Scratch["q_on_all", Self.BATCH * Self.NA, True]()
        self._idx = Scratch["idx", Self.BATCH, True]()
        self._max_q = Scratch["max_q", Self.BATCH, True]()
        self.gamma = Scalar[DT](0.99)
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString
    ](
        gamma: Scalar[DT] = Scalar[DT](0.99),
        nstep: Int = 1,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory.

        `nstep` (default 1) bakes `γ^nstep` into the effective discount
        used by the finalize fuse. For n-step DQN the sample block emits
        compressed transitions where `mb_r` is the n-step return
        `Σ_{i<N} γ^i r_i` and `mb_sp` is the state at `t+N`; the bootstrap
        therefore uses `γ^N`. For uniform / 1-step replay, `nstep=1`
        keeps the discount at γ (bit-identical to pre-N-step).
        """
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "DQNTargetYBlock: target must be 'cpu' or 'gpu'"
        comptime assert (
            Self.Q_NET.IN_DIMS[0] == Self.OBS
        ), "DQNTargetYBlock: Q_NET.IN_DIM must equal OBS"
        comptime assert (
            Self.Q_NET.OUT_DIM == Self.NA
        ), "DQNTargetYBlock: Q_NET.OUT_DIM must equal NA"
        var b = Self()
        # Bake γ^nstep so the finalize kernel multiplies by the right
        # discount regardless of replay flavor. CPU loop — nstep is small
        # (typically 1..10) so a runtime pow is fine.
        var gamma_n: Scalar[DT] = Scalar[DT](1.0)
        for _ in range(nstep):
            gamma_n = gamma_n * gamma
        b.gamma = gamma_n
        b.ts = TargetStorage.make[target](ctx=ctx)
        b.reduce_max = ReduceMax[Self.NA].make[target, INIT=Zero](ctx=ctx)
        b.gather_cols = GatherCols[Self.NA].make[target, INIT=Zero](ctx=ctx)
        init_scratch_auto[Self, target=target](b, ctx)
        return b^

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut q_target: Self.Q_NET,
        mut q_online: Self.Q_NET,
        mb_sp_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_r_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_d_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_y_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Standard: `mb_y[b] = mb_r[b] + γ·max_a Q_target(sp)[b,a]·(1−mb_d[b])`.
        Double:    `mb_y[b] = mb_r[b] + γ·Q_target(sp)[b, argmax_a Q_online(sp)[b,a]]·(1−mb_d[b])`.

        `q_online` is ignored on the standard path (DOUBLE=False). For
        Double, both nets are forwarded on `sp`."""
        assert_tag_for["DQNTargetYBlock", target](self.ts.target_tag)

        var q_all_p = self._q_all.target_ptr[target]()
        var max_q_p = self._max_q.target_ptr[target]()

        # Step 1: Q_target(sp) → _q_all.
        var sp_t = TileTensor(mb_sp_ptr, row_major[Self.BATCH, Self.OBS]())
        var q_all_t = TileTensor(q_all_p, row_major[Self.BATCH, Self.NA]())
        q_target.forward[target, Self.BATCH, POLICY](sp_t, output=q_all_t)

        # Step 2: pick best_q = either max or gather-by-online-argmax.
        comptime if Self.DOUBLE:
            # Q_online(sp) → _q_on_all, argmax → _idx, gather Q_target at idx → _max_q.
            var q_on_p = self._q_on_all.target_ptr[target]()
            var idx_p = self._idx.target_ptr[target]()
            var q_on_t = TileTensor(q_on_p, row_major[Self.BATCH, Self.NA]())
            q_online.forward[target, Self.BATCH, POLICY](sp_t, output=q_on_t)

            comptime if target == "cpu":
                for bb in range(Self.BATCH):
                    var best_a: Int = 0
                    var best_q: Scalar[DT] = q_on_p[bb * Self.NA]
                    for a in range(1, Self.NA):
                        var v = q_on_p[bb * Self.NA + a]
                        if v > best_q:
                            best_q = v
                            best_a = a
                    idx_p[bb] = Scalar[DT](best_a)
            else:
                var q_on_lt = LayoutTensor[
                    DT,
                    Layout.row_major(Self.BATCH, Self.NA),
                    MutAnyOrigin,
                ](q_on_p)
                var idx_lt = LayoutTensor[
                    DT,
                    Layout.row_major(Self.BATCH, 1),
                    MutAnyOrigin,
                ](idx_p)
                comptime n_blocks = (Self.BATCH + TPB - 1) // TPB
                comptime kernel = _argmax_idx_kernel[Self.BATCH, Self.NA]
                self.ts.ctx.value().enqueue_function[kernel](
                    q_on_lt,
                    idx_lt,
                    grid_dim=n_blocks,
                    block_dim=TPB,
                )

            # GatherCols(_q_all, _idx) → _max_q. Hetero-variadic: same
            # carrier Layout for both inputs (row_major[BATCH, NA]); the
            # leaf typed_view recovers the real shape.
            var max_q_t_g = TileTensor(max_q_p, row_major[Self.BATCH, 1]())
            var q_all_carrier = TileTensor(
                q_all_p,
                row_major[Self.BATCH, Self.NA](),
            )
            var idx_carrier = TileTensor(
                idx_p,
                row_major[Self.BATCH, Self.NA](),
            )
            self.gather_cols.forward[target, Self.BATCH, POLICY](
                TensorPack[2].of(q_all_carrier, idx_carrier),
                output=max_q_t_g,
            )
        else:
            # Standard DQN: max_a Q_target(sp).
            var max_q_t = TileTensor(max_q_p, row_major[Self.BATCH, 1]())
            self.reduce_max.forward[target, Self.BATCH, POLICY](
                q_all_t,
                output=max_q_t,
            )

        # Step 3: y = r + γ·max_q·(1 − d).
        comptime if target == "cpu":
            var mb_r = mb_r_ptr
            var mb_d = mb_d_ptr
            var mb_y = mb_y_ptr
            for bb in range(Self.BATCH):
                var nonterm = Scalar[DT](1.0) - mb_d[bb]
                mb_y[bb] = mb_r[bb] + self.gamma * max_q_p[bb] * nonterm
        else:
            var max_q_lt = LayoutTensor[
                DT,
                Layout.row_major(Self.BATCH, 1),
                MutAnyOrigin,
            ](max_q_p)
            var r_lt = LayoutTensor[
                DT,
                Layout.row_major(Self.BATCH),
                MutAnyOrigin,
            ](mb_r_ptr)
            var d_lt = LayoutTensor[
                DT,
                Layout.row_major(Self.BATCH),
                MutAnyOrigin,
            ](mb_d_ptr)
            var y_lt = LayoutTensor[
                DT,
                Layout.row_major(Self.BATCH),
                MutAnyOrigin,
            ](mb_y_ptr)
            comptime n_blocks = (Self.BATCH + TPB - 1) // TPB
            comptime kernel = _target_y_finalize_kernel[Self.BATCH]
            self.ts.ctx.value().enqueue_function[kernel](
                max_q_lt,
                r_lt,
                d_lt,
                y_lt,
                self.gamma,
                grid_dim=n_blocks,
                block_dim=TPB,
            )
