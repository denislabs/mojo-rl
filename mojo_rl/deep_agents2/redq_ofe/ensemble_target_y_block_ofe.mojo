"""EnsembleTargetYBlockOFE — REDQ-OFE target-y computation.

Phase O.2.b.1 (CPU). Mirrors `EnsembleTargetYBlock` from `redq/` but
reads φ(s') (precomputed by `OFEFeatureStep`) instead of raw next-obs
and runs the OFE `action_branch` forward to obtain φ(s', a') before
the N target-critic forwards.

Pipeline (per step) — differences from the non-OFE block are flagged
with ▌:

  1.   actor.forward(φ(s')) → raw [B, 2·ACT]                ▌ φ in
  2.   rsample.forward(raw) → packed [B, ACT+1] = (a' | logπ)
  3.   sa_in = concat(φ(s'), a')                            ▌ wider
       (dim PHI_S_DIM + ACT, on the heap as `_mb_sa_in`)
  4. ▌ action_branch.forward(sa_in) → φ(s', a')              ▌ NEW
       (dim PHI_SA_DIM, on the heap as `_mb_phi_spap`)
  5.   for i in 0..N:
         target_net_i.forward(φ(s', a')) → row i of stacked Q
  6.   redq_ensemble_target_cpu → mb_y[b]

Gradient policy
===============
Forward-only block. The OFE state-branch was already run (in
`OFEFeatureStep`) before this block. The action-branch forward here
populates its cache, but the cache will be clobbered by later
forwards (critic step, actor step, aux step) and the RL path NEVER
calls `action_branch.vjp` — so the clobber is harmless.

The `actor` and target critics are also read-only here (target_y
seeds the critic update; no gradient flows back through actor / target
critics on this path).

Subset bookkeeping (subset_idxs, set_subset_idxs, resample_subset_idxs):
identical to `EnsembleTargetYBlock` — copy-pasted, since the
MODE=MIN combine still selects N_MIN of N stacked Qs.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor, TileTensor, row_major
from std.random import random_float64

from mojo_rl.nn2.constants import DT, TPB
from mojo_rl.nn2.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.core.scratch import Scratch
from mojo_rl.nn2.core.scratch_walkers import init_scratch_auto
from mojo_rl.nn2.core.target_storage import TargetStorage, assert_tag_for
from mojo_rl.nn2.initializer import Zero

from ..primitives.rsample import RSample
from ..redq.ensemble import CriticEnsemble
from ..redq.ensemble_target_y_block import _redq_concat_sa_extract_lp_kernel
from ..redq.kernels import (
    redq_ensemble_target_cpu,
    redq_ensemble_target_gpu,
    REDQ_TARGET_MIN,
    REDQ_TARGET_AVE,
)


struct EnsembleTargetYBlockOFE[
    ACTOR: Module,          # IN=PHI_S_DIM, OUT=2·ACT
    AB: Module,             # IN=PHI_S_DIM+ACT, OUT=PHI_SA_DIM
    CRITIC: Module,         # IN=PHI_SA_DIM, OUT=1
    N_: Int,
    BATCH_: Int,
    PHI_S_DIM_: Int,
    ACT_: Int,
    N_MIN_: Int,
    MODE_: Int,
](Movable & ImplicitlyDestructible):
    comptime N = Self.N_
    comptime BATCH = Self.BATCH_
    comptime PHI_S_DIM = Self.PHI_S_DIM_
    comptime ACT = Self.ACT_
    comptime N_MIN = Self.N_MIN_
    comptime MODE = Self.MODE_
    comptime SA_IN_DIM = Self.PHI_S_DIM + Self.ACT
    comptime PHI_SA_DIM = Self.AB.OUT_DIM
    comptime ALP_DIM = Self.ACT + 1

    var rsample: RSample[Self.ACT]

    var _mb_ao: Scratch["of_y_mb_ao", Self.BATCH * (2 * Self.ACT)]
    var _mb_alp: Scratch["of_y_mb_alp", Self.BATCH * (Self.ACT + 1)]
    var _mb_sa_in: Scratch["of_y_mb_sa_in", Self.BATCH * Self.SA_IN_DIM]
    var _mb_phi_spap: Scratch[
        "of_y_mb_phi_spap", Self.BATCH * Self.PHI_SA_DIM,
    ]
    var _mb_stacked_q: Scratch["of_y_mb_stacked_q", Self.N * Self.BATCH]
    var _mb_lp: Scratch["of_y_mb_lp", Self.BATCH]

    var subset_idxs: List[Int]
    # GPU mirror of `subset_idxs` — uploaded once per `step["gpu"]` call.
    # None on CPU.
    var _subset_dev: Optional[DeviceBuffer[DType.uint32]]

    var action_scale: Scalar[DT]
    var gamma: Scalar[DT]
    var ts: TargetStorage

    def __init__(out self):
        self.rsample = RSample[Self.ACT]()
        self._mb_ao = Scratch[
            "of_y_mb_ao", Self.BATCH * (2 * Self.ACT),
        ]()
        self._mb_alp = Scratch[
            "of_y_mb_alp", Self.BATCH * (Self.ACT + 1),
        ]()
        self._mb_sa_in = Scratch[
            "of_y_mb_sa_in", Self.BATCH * Self.SA_IN_DIM,
        ]()
        self._mb_phi_spap = Scratch[
            "of_y_mb_phi_spap", Self.BATCH * Self.PHI_SA_DIM,
        ]()
        self._mb_stacked_q = Scratch[
            "of_y_mb_stacked_q", Self.N * Self.BATCH,
        ]()
        self._mb_lp = Scratch["of_y_mb_lp", Self.BATCH]()
        self.subset_idxs = List[Int](length=Self.N_MIN, fill=0)
        for k in range(Self.N_MIN):
            self.subset_idxs[k] = k
        self._subset_dev = None
        self.action_scale = Scalar[DT](1.0)
        self.gamma = Scalar[DT](0.99)
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString](
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "EnsembleTargetYBlockOFE: target must be 'cpu' or 'gpu'"
        )
        comptime if target == "gpu":
            if not ctx:
                raise Error(
                    "EnsembleTargetYBlockOFE.make[target='gpu']: ctx required"
                )
        comptime assert (
            Self.MODE == REDQ_TARGET_MIN or Self.MODE == REDQ_TARGET_AVE
        ), (
            "EnsembleTargetYBlockOFE: MODE must be MIN (0) or AVE (1)"
        )
        comptime assert Self.ACTOR.IN_DIMS[0] == Self.PHI_S_DIM, (
            "EnsembleTargetYBlockOFE: ACTOR.IN_DIM must equal PHI_S_DIM"
        )
        comptime assert Self.ACTOR.OUT_DIM == 2 * Self.ACT, (
            "EnsembleTargetYBlockOFE: ACTOR.OUT_DIM must equal 2·ACT"
        )
        comptime assert Self.AB.IN_DIMS[0] == Self.SA_IN_DIM, (
            "EnsembleTargetYBlockOFE: AB.IN must equal PHI_S_DIM + ACT"
        )
        comptime assert Self.CRITIC.IN_DIMS[0] == Self.PHI_SA_DIM, (
            "EnsembleTargetYBlockOFE: CRITIC.IN_DIM must equal PHI_SA_DIM"
        )
        comptime assert Self.CRITIC.OUT_DIM == 1, (
            "EnsembleTargetYBlockOFE: CRITIC.OUT_DIM must equal 1"
        )
        var blk = Self()
        blk.rsample = RSample[Self.ACT].make[target, Zero](ctx=ctx)
        blk.rsample.action_scale = action_scale
        blk.action_scale = action_scale
        blk.gamma = gamma
        blk.ts = TargetStorage.make[target](ctx=ctx)
        init_scratch_auto[Self, target](blk, ctx)
        comptime if target == "gpu":
            blk._subset_dev = ctx.value().enqueue_create_buffer[
                DType.uint32
            ](Self.N_MIN)
        return blk^

    def set_subset_idxs(mut self, idxs: List[Int]) raises:
        """Pin the MODE=MIN subset deterministically (test hook)."""
        if len(idxs) != Self.N_MIN:
            raise Error(
                "set_subset_idxs: expected length " + String(Self.N_MIN)
                + ", got " + String(len(idxs))
            )
        for k in range(Self.N_MIN):
            if idxs[k] < 0 or idxs[k] >= Self.N:
                raise Error(
                    "set_subset_idxs: index out of range [0, "
                    + String(Self.N) + ")"
                )
            self.subset_idxs[k] = idxs[k]

    def resample_subset_idxs(mut self) raises:
        """Fisher-Yates partial shuffle: pick N_MIN distinct indices
        from {0..N-1}. Production drivers call this every train step
        before `step`."""
        var picks = List[Int](length=Self.N, fill=0)
        for i in range(Self.N):
            picks[i] = i
        for i in range(Self.N_MIN):
            var j = i + Int(random_float64() * Float64(Self.N - i))
            if j >= Self.N:
                j = Self.N - 1
            var tmp = picks[i]
            picks[i] = picks[j]
            picks[j] = tmp
            self.subset_idxs[i] = picks[i]

    def step[
        target: StaticString = "cpu",
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut actor: Self.ACTOR,
        mut action_branch: Self.AB,
        mut ensemble: CriticEnsemble[Self.CRITIC, Self.N],
        mb_phi_sp_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_r_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_term_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        alpha: Scalar[DT],
        mb_y_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Writes `mb_y[b] = r[b] + (1 − term[b]) · γ · (combined_Q[b]
        − α · log_π(a'|s')[b])` into `mb_y_ptr`. Forward-only."""
        comptime assert target == "cpu" or target == "gpu", (
            "EnsembleTargetYBlockOFE.step: target must be 'cpu' or 'gpu'"
        )
        assert_tag_for["EnsembleTargetYBlockOFE", target](self.ts.target_tag)

        # 1. actor.forward(φ(s')) → _mb_ao [BATCH, 2·ACT].
        var ao_p = self._mb_ao.target_ptr[target]()
        var phi_sp_t = TileTensor(
            mb_phi_sp_ptr, row_major[Self.BATCH, Self.PHI_S_DIM](),
        )
        var ao_t = TileTensor(
            ao_p, row_major[Self.BATCH, 2 * Self.ACT](),
        )
        actor.forward[target, Self.BATCH, POLICY](phi_sp_t, output=ao_t)

        # 2. rsample → _mb_alp [BATCH, ACT+1] = (a' | log_prob).
        var alp_p = self._mb_alp.target_ptr[target]()
        var alp_t = TileTensor(
            alp_p, row_major[Self.BATCH, Self.ALP_DIM](),
        )
        self.rsample.forward[target, Self.BATCH, POLICY](
            ao_t, output=alp_t,
        )

        # 3. sa_in = concat(φ(s'), a') + extract log_prob into _mb_lp.
        var sa_in_p = self._mb_sa_in.target_ptr[target]()
        var lp_p = self._mb_lp.target_ptr[target]()
        comptime if target == "cpu":
            for b in range(Self.BATCH):
                for d in range(Self.PHI_S_DIM):
                    sa_in_p[b * Self.SA_IN_DIM + d] = mb_phi_sp_ptr[
                        b * Self.PHI_S_DIM + d
                    ]
                for j in range(Self.ACT):
                    sa_in_p[b * Self.SA_IN_DIM + Self.PHI_S_DIM + j] = (
                        alp_p[b * Self.ALP_DIM + j]
                    )
                lp_p[b] = alp_p[b * Self.ALP_DIM + Self.ACT]
        else:
            # Reuse REDQ's `_redq_concat_sa_extract_lp_kernel` — its
            # "OBS" param is just the first-input width, so passing
            # PHI_S_DIM in that slot is correct.
            var ctx = self.ts.ctx.value()
            var phi_sp_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH, Self.PHI_S_DIM),
                MutAnyOrigin,
            ](mb_phi_sp_ptr)
            var alp_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH, Self.ALP_DIM),
                MutAnyOrigin,
            ](alp_p)
            var sa_in_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH, Self.SA_IN_DIM),
                MutAnyOrigin,
            ](sa_in_p)
            var lp_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH), MutAnyOrigin,
            ](lp_p)
            comptime total_sa = Self.BATCH * Self.SA_IN_DIM
            comptime n_blocks = (total_sa + TPB - 1) // TPB
            comptime kernel = _redq_concat_sa_extract_lp_kernel[
                Self.PHI_S_DIM, Self.ACT, Self.BATCH,
                Self.SA_IN_DIM, Self.ALP_DIM,
            ]
            ctx.enqueue_function[kernel](
                phi_sp_lt, alp_lt, sa_in_lt, lp_lt,
                grid_dim=n_blocks, block_dim=TPB,
            )

        # 4. action_branch.forward(sa_in) → φ(s', a') [BATCH, PHI_SA_DIM].
        var sa_in_t = TileTensor(
            sa_in_p, row_major[Self.BATCH, Self.SA_IN_DIM](),
        )
        var phi_spap_p = self._mb_phi_spap.target_ptr[target]()
        var phi_spap_t = TileTensor(
            phi_spap_p, row_major[Self.BATCH, Self.PHI_SA_DIM](),
        )
        action_branch.forward[target, Self.BATCH, POLICY](
            sa_in_t, output=phi_spap_t,
        )

        # 5. Loop N target critic forwards on φ(s', a'). Each writes
        # its [BATCH, 1] output directly into row i of _mb_stacked_q.
        var stacked_p = self._mb_stacked_q.target_ptr[target]()
        for i in range(Self.N):
            var row_p = stacked_p + i * Self.BATCH
            var q_i_t = TileTensor(row_p, row_major[Self.BATCH, 1]())
            ensemble.pairs[i].target_net.forward[
                target, Self.BATCH, POLICY,
            ](phi_spap_t, output=q_i_t)

        # 6. Combine + α·logπ + γ + terminal mask.
        comptime if target == "cpu":
            redq_ensemble_target_cpu[
                Self.N, Self.N_MIN, Self.MODE, Self.BATCH,
            ](
                mb_r_ptr,
                stacked_p,
                mb_term_ptr,
                lp_p,
                self.subset_idxs.unsafe_ptr(),
                self.gamma,
                alpha,
                mb_y_ptr,
            )
        else:
            # Upload subset_idxs (host List[Int] → device uint32 buffer)
            # once per step. Same pattern as REDQ's GPU target-y.
            var ctx = self.ts.ctx.value()
            var subset_host = ctx.enqueue_create_host_buffer[
                DType.uint32
            ](Self.N_MIN)
            var subset_host_p = subset_host.unsafe_ptr()
            for k in range(Self.N_MIN):
                subset_host_p[k] = UInt32(self.subset_idxs[k])
            ctx.enqueue_copy(self._subset_dev.value(), subset_host_p)
            redq_ensemble_target_gpu[
                Self.N, Self.N_MIN, Self.MODE, Self.BATCH,
            ](
                ctx,
                mb_y_ptr,
                mb_r_ptr,
                stacked_p,
                mb_term_ptr,
                lp_p,
                self._subset_dev.value().unsafe_ptr(),
                self.gamma,
                alpha,
            )
