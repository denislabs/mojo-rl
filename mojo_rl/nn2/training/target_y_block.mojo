"""TargetYBlock — SAC target-y computation as a self-contained block. Phase 10F + Block D.

Encapsulates the off-policy target-value compute used by both critic
losses (and shared by SAC, TD3, and DDPG variants with minor tweaks).

Forward formula (CleanRL-style continuous SAC):
    a'           ~ squashed-Gaussian(actor(s'))      (no grad — fresh z)
    log_prob(a') = Σ log_N(z) - log_std - 0.5·log(2π)
                   - log(action_scale·(1 - tanh²) + ε)
    sa'          = concat(s', a')
    q1_tgt       = critic1_target.forward(sa')
    q2_tgt       = critic2_target.forward(sa')
    qmin         = min(q1_tgt, q2_tgt)
    y[b]         = r[b] + γ·nonterm·(qmin - α·log_prob)
        nonterm = 1.0 for time-limit-only envs (Pendulum); see
        `feedback_ppo_pendulum_timelimit_gae`.

Uses the **free-function** `squashed_gaussian_sample` rather than the
`RSample` Module — no gradient flow back through this path, so the
caching the Module would do is wasted.

Surface:
    TargetYBlock[ACTOR, CRITIC, BATCH, OBS, ACT]
        - `make[target](action_scale, gamma) raises -> Self`          (CPU)
        - `make[target](ctx, action_scale, gamma) raises -> Self`      (GPU)
        - `step[target](mut actor, mut critic1_target, mut critic2_target,
                        mb_sp_ptr, mb_r_ptr, alpha, mb_y_ptr) raises`
            Writes `mb_y_ptr` ([BATCH]) in-place.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor, TileTensor, TensorLayout, row_major

from ..constants import DT
from ..core import TARGET_GPU
from ..core.module import Module
from ..core.target_storage import (
    TargetStorage, assert_tag_for, ensure_gpu_buffer,
)
from ..loss.sac_actor_loss import squashed_gaussian_sample
from ..loss.squashed_gaussian import squashed_gaussian_forward_gpu
from ..random.box_muller import box_muller_normal, box_muller_normal_gpu
from .off_policy_critic import concat_sa, concat_sa_gpu


def _target_y_kernel[BATCH: Int](
    r: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    q1: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    q2: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    lp: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    alpha: Scalar[DT],
    gamma: Scalar[DT],
    y: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
):
    var b = Int(global_idx.x)
    if b >= BATCH:
        return
    var q1_v = rebind[Scalar[DT]](q1[b, 0])
    var q2_v = rebind[Scalar[DT]](q2[b, 0])
    var qmin = q1_v if q1_v < q2_v else q2_v
    var nonterm: Scalar[DT] = 1.0
    y[b, 0] = (
        rebind[Scalar[DT]](r[b])
        + gamma * nonterm * (qmin - alpha * rebind[Scalar[DT]](lp[b]))
    )


struct TargetYBlock[
    ACTOR: Module,
    CRITIC: Module,
    BATCH: Int,
    OBS: Int,
    ACT: Int,
](Movable & ImplicitlyDestructible):
    comptime SA_DIM = Self.OBS + Self.ACT

    # CPU scratch.
    var _mb_ao_sp: List[Scalar[DT]]    # [BATCH, 2*ACT]   actor(s') output
    var _mb_z_sp: List[Scalar[DT]]     # [BATCH, ACT]     z noise
    var _mb_act_sp: List[Scalar[DT]]   # [BATCH, ACT]     sampled a'
    var _mb_lp_sp: List[Scalar[DT]]    # [BATCH]          log_prob(a')
    var _mb_sa: List[Scalar[DT]]       # [BATCH, SA_DIM]  concat(s', a')
    var _mb_q1_tgt: List[Scalar[DT]]   # [BATCH, 1]
    var _mb_q2_tgt: List[Scalar[DT]]   # [BATCH, 1]

    # GPU scratch (block D).
    var _mb_ao_sp_dev: Optional[DeviceBuffer[DT]]
    var _mb_z_sp_dev: Optional[DeviceBuffer[DT]]
    var _mb_act_sp_dev: Optional[DeviceBuffer[DT]]
    var _mb_lp_sp_dev: Optional[DeviceBuffer[DT]]
    var _mb_sa_dev: Optional[DeviceBuffer[DT]]
    var _mb_q1_tgt_dev: Optional[DeviceBuffer[DT]]
    var _mb_q2_tgt_dev: Optional[DeviceBuffer[DT]]
    var _gpu_n: Int   # 1 = allocated, 0 = uninit

    var action_scale: Scalar[DT]
    var gamma: Scalar[DT]
    var rng_seed: UInt64
    var _rng_offset: UInt64
    var ts: TargetStorage

    def __init__(out self):
        self._mb_ao_sp = List[Scalar[DT]]()
        self._mb_z_sp = List[Scalar[DT]]()
        self._mb_act_sp = List[Scalar[DT]]()
        self._mb_lp_sp = List[Scalar[DT]]()
        self._mb_sa = List[Scalar[DT]]()
        self._mb_q1_tgt = List[Scalar[DT]]()
        self._mb_q2_tgt = List[Scalar[DT]]()
        self._mb_ao_sp_dev = None
        self._mb_z_sp_dev = None
        self._mb_act_sp_dev = None
        self._mb_lp_sp_dev = None
        self._mb_sa_dev = None
        self._mb_q1_tgt_dev = None
        self._mb_q2_tgt_dev = None
        self._gpu_n = 0
        self.action_scale = Scalar[DT](1.0)
        self.gamma = Scalar[DT](0.99)
        self.rng_seed = UInt64(7919)
        self._rng_offset = UInt64(0)
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString](
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        gamma: Scalar[DT] = Scalar[DT](0.99),
    ) raises -> Self:
        comptime assert target == "cpu", (
            "TargetYBlock.make[target='gpu'] requires a DeviceContext"
        )
        comptime assert Self.ACTOR.IN_DIM == Self.OBS, (
            "TargetYBlock: ACTOR.IN_DIM must equal OBS"
        )
        comptime assert Self.ACTOR.OUT_DIM == 2 * Self.ACT, (
            "TargetYBlock: ACTOR.OUT_DIM must equal 2·ACT"
        )
        comptime assert Self.CRITIC.IN_DIM == Self.SA_DIM, (
            "TargetYBlock: CRITIC.IN_DIM must equal OBS + ACT"
        )
        comptime assert Self.CRITIC.OUT_DIM == 1, (
            "TargetYBlock: CRITIC.OUT_DIM must equal 1"
        )
        var blk = Self()
        var zero: Scalar[DT] = 0.0
        blk._mb_ao_sp.resize(Self.BATCH * 2 * Self.ACT, zero)
        blk._mb_z_sp.resize(Self.BATCH * Self.ACT, zero)
        blk._mb_act_sp.resize(Self.BATCH * Self.ACT, zero)
        blk._mb_lp_sp.resize(Self.BATCH, zero)
        blk._mb_sa.resize(Self.BATCH * Self.SA_DIM, zero)
        blk._mb_q1_tgt.resize(Self.BATCH, zero)
        blk._mb_q2_tgt.resize(Self.BATCH, zero)
        blk.action_scale = action_scale
        blk.gamma = gamma
        blk.ts = TargetStorage.make_cpu()
        return blk^

    @staticmethod
    def make[target: StaticString](
        ctx: DeviceContext,
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        gamma: Scalar[DT] = Scalar[DT](0.99),
    ) raises -> Self:
        """GPU factory. Pre-allocates all device scratch."""
        comptime assert target == "gpu", (
            "TargetYBlock.make[target='cpu'](ctx) — drop ctx for CPU"
        )
        comptime assert Self.ACTOR.IN_DIM == Self.OBS, (
            "TargetYBlock: ACTOR.IN_DIM must equal OBS"
        )
        comptime assert Self.ACTOR.OUT_DIM == 2 * Self.ACT, (
            "TargetYBlock: ACTOR.OUT_DIM must equal 2·ACT"
        )
        comptime assert Self.CRITIC.IN_DIM == Self.SA_DIM, (
            "TargetYBlock: CRITIC.IN_DIM must equal OBS + ACT"
        )
        comptime assert Self.CRITIC.OUT_DIM == 1, (
            "TargetYBlock: CRITIC.OUT_DIM must equal 1"
        )
        var blk = Self()
        blk._mb_ao_sp_dev = ctx.enqueue_create_buffer[DT](Self.BATCH * 2 * Self.ACT)
        blk._mb_z_sp_dev = ctx.enqueue_create_buffer[DT](Self.BATCH * Self.ACT)
        blk._mb_act_sp_dev = ctx.enqueue_create_buffer[DT](Self.BATCH * Self.ACT)
        blk._mb_lp_sp_dev = ctx.enqueue_create_buffer[DT](Self.BATCH)
        blk._mb_sa_dev = ctx.enqueue_create_buffer[DT](Self.BATCH * Self.SA_DIM)
        blk._mb_q1_tgt_dev = ctx.enqueue_create_buffer[DT](Self.BATCH)
        blk._mb_q2_tgt_dev = ctx.enqueue_create_buffer[DT](Self.BATCH)
        blk._gpu_n = 1
        blk.action_scale = action_scale
        blk.gamma = gamma
        blk.ts = TargetStorage.make_gpu(ctx)
        return blk^

    def step[target: StaticString](
        mut self,
        mut actor: Self.ACTOR,
        mut critic1_target: Self.CRITIC,
        mut critic2_target: Self.CRITIC,
        mb_sp_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_r_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        alpha: Scalar[DT],
        mb_y_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Compute `mb_y[b] = r[b] + γ·(min(Q1_t, Q2_t)(s', a') − α·log_prob(a'|s'))`
        in-place into `mb_y_ptr`. `nonterm=1.0` for time-limit-only envs."""
        assert_tag_for["TargetYBlock", target](self.ts.target_tag)

        comptime if target == "cpu":
            self._step_cpu(
                actor, critic1_target, critic2_target,
                mb_sp_ptr, mb_r_ptr, alpha, mb_y_ptr,
            )
        else:
            self._step_gpu(
                actor, critic1_target, critic2_target,
                mb_sp_ptr, mb_r_ptr, alpha, mb_y_ptr,
            )

    def _step_cpu(
        mut self,
        mut actor: Self.ACTOR,
        mut critic1_target: Self.CRITIC,
        mut critic2_target: Self.CRITIC,
        mb_sp_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_r_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        alpha: Scalar[DT],
        mb_y_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        var mb_sp_t = TileTensor(mb_sp_ptr, row_major[Self.BATCH, Self.OBS]())
        var mb_ao_sp_p = self._mb_ao_sp.unsafe_ptr()
        var mb_ao_sp_t = TileTensor(
            mb_ao_sp_p, row_major[Self.BATCH, 2 * Self.ACT]()
        )
        actor.forward["cpu", Self.BATCH](mb_sp_t, mb_ao_sp_t)

        var mb_z_sp_p = self._mb_z_sp.unsafe_ptr()
        box_muller_normal(mb_z_sp_p, Self.BATCH * Self.ACT)
        var mb_z_sp_t = TileTensor(mb_z_sp_p, row_major[Self.BATCH, Self.ACT]())

        var mb_act_sp_p = self._mb_act_sp.unsafe_ptr()
        var mb_act_sp_t = TileTensor(
            mb_act_sp_p, row_major[Self.BATCH, Self.ACT]()
        )
        var mb_lp_sp_p = self._mb_lp_sp.unsafe_ptr()
        var mb_lp_sp_t = TileTensor(mb_lp_sp_p, row_major[Self.BATCH]())
        squashed_gaussian_sample[Self.ACT, Self.BATCH](
            mb_ao_sp_t, mb_z_sp_t, self.action_scale, mb_act_sp_t, mb_lp_sp_t
        )

        var mb_sa_p = self._mb_sa.unsafe_ptr()
        concat_sa[Self.OBS, Self.ACT, Self.BATCH](
            mb_sp_ptr, mb_act_sp_p, mb_sa_p
        )
        var mb_sa_t = TileTensor(mb_sa_p, row_major[Self.BATCH, Self.SA_DIM]())
        var mb_q1_tgt_p = self._mb_q1_tgt.unsafe_ptr()
        var mb_q2_tgt_p = self._mb_q2_tgt.unsafe_ptr()
        var mb_q1_tgt_t = TileTensor(mb_q1_tgt_p, row_major[Self.BATCH, 1]())
        var mb_q2_tgt_t = TileTensor(mb_q2_tgt_p, row_major[Self.BATCH, 1]())
        critic1_target.forward["cpu", Self.BATCH](mb_sa_t, mb_q1_tgt_t)
        critic2_target.forward["cpu", Self.BATCH](mb_sa_t, mb_q2_tgt_t)

        for b in range(Self.BATCH):
            var q1 = mb_q1_tgt_p[b]
            var q2 = mb_q2_tgt_p[b]
            var qmin = q1 if q1 < q2 else q2
            var nonterm: Scalar[DT] = 1.0
            mb_y_ptr[b] = mb_r_ptr[b] + self.gamma * nonterm * (
                qmin - alpha * mb_lp_sp_p[b]
            )

    def _step_gpu(
        mut self,
        mut actor: Self.ACTOR,
        mut critic1_target: Self.CRITIC,
        mut critic2_target: Self.CRITIC,
        mb_sp_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_r_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        alpha: Scalar[DT],
        mb_y_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        var ctx = self.ts.ctx.value()
        # Resolve device pointers from scratch.
        var ao_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._mb_ao_sp_dev.value().unsafe_ptr()
        )
        var z_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._mb_z_sp_dev.value().unsafe_ptr()
        )
        var act_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._mb_act_sp_dev.value().unsafe_ptr()
        )
        var lp_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._mb_lp_sp_dev.value().unsafe_ptr()
        )
        var sa_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._mb_sa_dev.value().unsafe_ptr()
        )
        var q1_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._mb_q1_tgt_dev.value().unsafe_ptr()
        )
        var q2_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._mb_q2_tgt_dev.value().unsafe_ptr()
        )

        var mb_sp_t = TileTensor(mb_sp_ptr, row_major[Self.BATCH, Self.OBS]())
        var ao_t = TileTensor(ao_p, row_major[Self.BATCH, 2 * Self.ACT]())
        actor.forward["gpu", Self.BATCH](mb_sp_t, ao_t)

        box_muller_normal_gpu[Self.BATCH * Self.ACT](
            ctx, z_p, self.rng_seed, self._rng_offset,
        )
        self._rng_offset += UInt64(2 * Self.BATCH * Self.ACT)

        squashed_gaussian_forward_gpu[Self.ACT, Self.BATCH](
            ctx, ao_p, z_p, self.action_scale, act_p, lp_p,
        )
        concat_sa_gpu[Self.OBS, Self.ACT, Self.BATCH](
            ctx, mb_sp_ptr, act_p, sa_p,
        )
        var sa_t = TileTensor(sa_p, row_major[Self.BATCH, Self.SA_DIM]())
        var q1_t = TileTensor(q1_p, row_major[Self.BATCH, 1]())
        var q2_t = TileTensor(q2_p, row_major[Self.BATCH, 1]())
        critic1_target.forward["gpu", Self.BATCH](sa_t, q1_t)
        critic2_target.forward["gpu", Self.BATCH](sa_t, q2_t)

        # Compute y = r + γ·(min(q1,q2) - α·lp).
        var r_lt = LayoutTensor[
            DT, Layout.row_major(Self.BATCH), MutAnyOrigin,
        ](mb_r_ptr)
        var q1_lt = LayoutTensor[
            DT, Layout.row_major(Self.BATCH, 1), MutAnyOrigin,
        ](q1_p)
        var q2_lt = LayoutTensor[
            DT, Layout.row_major(Self.BATCH, 1), MutAnyOrigin,
        ](q2_p)
        var lp_lt = LayoutTensor[
            DT, Layout.row_major(Self.BATCH), MutAnyOrigin,
        ](lp_p)
        var y_lt = LayoutTensor[
            DT, Layout.row_major(Self.BATCH, 1), MutAnyOrigin,
        ](mb_y_ptr)
        comptime TPB = 64
        comptime n_blocks = (Self.BATCH + TPB - 1) // TPB
        comptime kernel = _target_y_kernel[Self.BATCH]
        ctx.enqueue_function[kernel](
            r_lt, q1_lt, q2_lt, lp_lt, alpha, self.gamma, y_lt,
            grid_dim=n_blocks, block_dim=TPB,
        )
