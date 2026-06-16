"""EnsembleTargetYBlock — REDQ target-y computation.

Phase R.1 (CPU). Computes the SAC-style entropy-regularized TD target
using a randomized ensemble of N target critics:

    y[b] = r[b] + (1 - term[b]) * γ * (combined_Q[b] - α * log_π(a'|s')[b])

`combined_Q[b]` is the per-sample reduction of N target-Q evaluations
selected by `MODE`:
  - MODE = MIN — min over `subset_idxs[0..N_MIN]` (paper-faithful REDQ)
  - MODE = AVE — mean over all N critics
  - MODE = REM — GPU-only (Philox per-(b, n) draws)

Pipeline (per step):
  1. `actor.forward(s')` → raw [B, 2·ACT] (mean | log_std)
  2. `rsample.forward(raw)` → packed [B, ACT+1] (action | log_prob)
  3. `sa = concat(s', action)` — host loop into `_mb_sa [B, SA_DIM]`
  4. For i in 0..N: `target_net_i.forward(sa)` into `_mb_stacked_q[i, :]`
  5. `redq_ensemble_target_cpu` → `mb_y[b]` (combine + α·logp + γ + mask)

Subset bookkeeping:
  - `subset_idxs: List[Int]` of length N_MIN holds the chosen target
    critics for MODE=MIN. The default `__init__` populates `[0, 1, …,
    N_MIN-1]` so a fresh block runs without explicit setup (matches
    the legacy REDQ first-step ordering before its Fisher-Yates draw).
  - `resample_subset_idxs()` does a Fisher-Yates partial shuffle of
    {0..N-1}; production drivers call it per train step.
  - `set_subset_idxs(idxs)` pins the subset deterministically — used
    by the R.1 smoke test to verify the combine formula.

R.1 is CPU-only. The combine kernel is target-parametric in shape but
not in implementation (GPU follows in a later phase).

Forward-only — this block computes a target the critic update reads;
no gradient flows through actor or target critics here.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor, TileTensor, row_major
from std.random import random_float64

from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.scratch import Scratch
from mojo_rl.nn.core.scratch_walkers import init_scratch_auto
from mojo_rl.nn.core.target_storage import TargetStorage, assert_tag_for
from mojo_rl.nn.initializer import Zero

from ..primitives.rsample import RSample
from .ensemble import CriticEnsemble
from .kernels import (
    redq_ensemble_target_cpu,
    redq_ensemble_target_gpu,
    REDQ_TARGET_MIN,
    REDQ_TARGET_AVE,
)


# ────────────────────────────────────────────────────────────────────
# GPU helper kernel — concat(sp, action) + extract log_prob from alp.
# `alp` is the packed RSample output [BATCH, ACT+1] = (action | log_prob).
# Splits in one pass:
#   sa[b, :OBS]   = sp[b, :]
#   sa[b, OBS:]   = alp[b, :ACT]      (action portion)
#   lp[b]         = alp[b, ACT]       (log_prob column)
# One thread per output element in `sa`; the lp[] write is gated on d==0
# so each batch index writes lp[b] exactly once.
# ────────────────────────────────────────────────────────────────────


def _redq_concat_sa_extract_lp_kernel[
    OBS: Int, ACT: Int, BATCH: Int, SA_DIM: Int, ALP_DIM: Int,
](
    sp: LayoutTensor[DT, Layout.row_major(BATCH, OBS), MutAnyOrigin],
    alp: LayoutTensor[DT, Layout.row_major(BATCH, ALP_DIM), MutAnyOrigin],
    sa: LayoutTensor[DT, Layout.row_major(BATCH, SA_DIM), MutAnyOrigin],
    lp: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = BATCH * SA_DIM
    if idx >= total:
        return
    var b = idx // SA_DIM
    var d = idx % SA_DIM
    if d < OBS:
        sa[b, d] = rebind[Scalar[DT]](sp[b, d])
    else:
        sa[b, d] = rebind[Scalar[DT]](alp[b, d - OBS])
    if d == 0:
        lp[b] = rebind[Scalar[DT]](alp[b, ACT])


struct EnsembleTargetYBlock[
    ACTOR: Module,
    CRITIC: Module,
    N_: Int,
    BATCH_: Int,
    OBS_: Int,
    ACT_: Int,
    N_MIN_: Int,
    MODE_: Int,
](Movable & ImplicitlyDeletable):
    comptime N = Self.N_
    comptime BATCH = Self.BATCH_
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime N_MIN = Self.N_MIN_
    comptime MODE = Self.MODE_
    comptime SA_DIM = Self.OBS + Self.ACT
    comptime ALP_DIM = Self.ACT + 1

    var rsample: RSample[Self.ACT]

    var _mb_ao: Scratch["ens_mb_ao", Self.BATCH * (2 * Self.ACT)]
    var _mb_alp: Scratch["ens_mb_alp", Self.BATCH * (Self.ACT + 1)]
    var _mb_sa: Scratch["ens_mb_sa", Self.BATCH * Self.SA_DIM]
    var _mb_q_i: Scratch["ens_mb_q_i", Self.BATCH]
    var _mb_stacked_q: Scratch["ens_mb_stacked_q", Self.N * Self.BATCH]
    var _mb_lp: Scratch["ens_mb_lp", Self.BATCH]

    var subset_idxs: List[Int]
    # GPU mirror of `subset_idxs` — uploaded once per `step[target="gpu"]`
    # call. None on CPU.
    var _subset_dev: Optional[DeviceBuffer[DType.uint32]]

    var action_scale: Scalar[DT]
    var gamma: Scalar[DT]
    var ts: TargetStorage

    def __init__(out self):
        self.rsample = RSample[Self.ACT]()
        self._mb_ao = Scratch["ens_mb_ao", Self.BATCH * (2 * Self.ACT)]()
        self._mb_alp = Scratch["ens_mb_alp", Self.BATCH * (Self.ACT + 1)]()
        self._mb_sa = Scratch["ens_mb_sa", Self.BATCH * Self.SA_DIM]()
        self._mb_q_i = Scratch["ens_mb_q_i", Self.BATCH]()
        self._mb_stacked_q = Scratch[
            "ens_mb_stacked_q", Self.N * Self.BATCH
        ]()
        self._mb_lp = Scratch["ens_mb_lp", Self.BATCH]()
        # Deterministic first subset: [0, 1, …, N_MIN-1]. The trainer
        # will call resample_subset_idxs() each step in production.
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
            "EnsembleTargetYBlock: target must be 'cpu' or 'gpu'"
        )
        comptime if target == "gpu":
            if not ctx:
                raise Error(
                    "EnsembleTargetYBlock.make[target='gpu']: ctx required"
                )
        comptime assert (
            Self.MODE == REDQ_TARGET_MIN or Self.MODE == REDQ_TARGET_AVE
        ), (
            "EnsembleTargetYBlock: MODE must be MIN (0) or AVE (1)."
            " REM (random ensemble mixture) is not on R.5's surface."
        )
        comptime assert Self.ACTOR.IN_DIMS[0] == Self.OBS, (
            "EnsembleTargetYBlock: ACTOR.IN_DIM must equal OBS"
        )
        comptime assert Self.ACTOR.OUT_DIM == 2 * Self.ACT, (
            "EnsembleTargetYBlock: ACTOR.OUT_DIM must equal 2·ACT"
        )
        comptime assert Self.CRITIC.IN_DIMS[0] == Self.SA_DIM, (
            "EnsembleTargetYBlock: CRITIC.IN_DIM must equal OBS+ACT"
        )
        comptime assert Self.CRITIC.OUT_DIM == 1, (
            "EnsembleTargetYBlock: CRITIC.OUT_DIM must equal 1"
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
        """Pin the MODE=MIN subset deterministically (test hook).
        Skips the per-step Fisher-Yates resample."""
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
        from {0..N-1}, store into `self.subset_idxs`. Production
        drivers call this every train step before `step`. CPU only
        (uses host `random_float64`); GPU port will use Philox."""
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
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut actor: Self.ACTOR,
        mut ensemble: CriticEnsemble[Self.CRITIC, Self.N],
        mb_sp_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_r_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_term_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        alpha: Scalar[DT],
        mb_y_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Write `mb_y[b] = r[b] + (1 − term[b]) · γ · (combined_Q[b]
        − α · log_prob[b])` in-place into `mb_y_ptr`.

        Forward-only. The actor + N target critics are read but never
        receive gradient on this path.
        """
        assert_tag_for["EnsembleTargetYBlock", target](self.ts.target_tag)

        # 1. actor.forward(s') → _mb_ao [BATCH, 2·ACT]
        var ao_p = self._mb_ao.target_ptr[target]()
        var sp_t = TileTensor(mb_sp_ptr, row_major[Self.BATCH, Self.OBS]())
        var ao_t = TileTensor(
            ao_p, row_major[Self.BATCH, 2 * Self.ACT](),
        )
        actor.forward[target, Self.BATCH, POLICY](sp_t, output=ao_t)

        # 2. rsample → _mb_alp [BATCH, ACT+1] (packed action | log_prob)
        var alp_p = self._mb_alp.target_ptr[target]()
        var alp_t = TileTensor(
            alp_p, row_major[Self.BATCH, Self.ALP_DIM](),
        )
        self.rsample.forward[target, Self.BATCH, POLICY](
            ao_t, output=alp_t,
        )

        # 3. sa = concat(s', action) + extract log_prob.
        var sa_p = self._mb_sa.target_ptr[target]()
        var lp_p = self._mb_lp.target_ptr[target]()
        comptime if target == "cpu":
            for b in range(Self.BATCH):
                for d in range(Self.OBS):
                    sa_p[b * Self.SA_DIM + d] = mb_sp_ptr[b * Self.OBS + d]
                for j in range(Self.ACT):
                    sa_p[b * Self.SA_DIM + Self.OBS + j] = alp_p[
                        b * Self.ALP_DIM + j
                    ]
                lp_p[b] = alp_p[b * Self.ALP_DIM + Self.ACT]
        else:
            # GPU: one kernel concats sp into the [:, :OBS] half of sa,
            # alp into the [:, OBS:] half, and writes lp[b] = alp[b, ACT].
            var ctx = self.ts.ctx.value()
            var sp_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH, Self.OBS), MutAnyOrigin,
            ](mb_sp_ptr)
            var alp_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH, Self.ALP_DIM), MutAnyOrigin,
            ](alp_p)
            var sa_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH, Self.SA_DIM), MutAnyOrigin,
            ](sa_p)
            var lp_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH), MutAnyOrigin,
            ](lp_p)
            comptime total_sa = Self.BATCH * Self.SA_DIM
            comptime n_blocks = (total_sa + TPB - 1) // TPB
            comptime kernel = _redq_concat_sa_extract_lp_kernel[
                Self.OBS, Self.ACT, Self.BATCH, Self.SA_DIM, Self.ALP_DIM,
            ]
            ctx.enqueue_function[kernel](
                sp_lt, alp_lt, sa_lt, lp_lt,
                grid_dim=n_blocks, block_dim=TPB,
            )

        # 4. Loop N target critic forwards. Each writes its [BATCH, 1]
        # output directly into row i of `_mb_stacked_q` [N, BATCH] —
        # `q_i_t` is a TileTensor over `stacked + i * BATCH` so no
        # per-iter copy is needed on CPU or GPU.
        var sa_t = TileTensor(
            sa_p, row_major[Self.BATCH, Self.SA_DIM](),
        )
        var stacked_p = self._mb_stacked_q.target_ptr[target]()
        for i in range(Self.N):
            var row_p = stacked_p + i * Self.BATCH
            var q_i_t = TileTensor(row_p, row_major[Self.BATCH, 1]())
            ensemble.pairs[i].target_net.forward[
                target, Self.BATCH, POLICY,
            ](sa_t, output=q_i_t)

        # 5. Combine + α·lp + γ + terminal mask.
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
            # once per step. N_MIN is small (paper-default 2); the copy
            # cost is negligible vs the N critic forwards above.
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
