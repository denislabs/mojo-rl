"""TD-MPC2 MPPI rollout callback (CPU) — bridges the nn2 world model into
the shared `mojo_rl.planners.trajectory.MPPICPU` planner.

Implements `RolloutCallbackCPU` (B=1, List[Float64] views — the planner
loops over samples/timesteps). The three trait methods map to the world
model's forward passes:

  * policy_action_cpu(z) → tanh(mean) of π(z)  — normalized [-1,1] seed for
    the planner's policy-trajectory warm-start.
  * rollout_step_cpu(z, a) → (z' = dynamics(z, a·scale),
    reward = two-hot-decode(reward(z, a·scale)))  — actions are scaled to
    the range the world model was trained on (replay stored a·action_scale);
    the planner samples/refits in normalized [-1,1] and applies action_scale
    to its returned action.
  * terminal_value_cpu(z) → avg of 2 (random) target-Q heads at π(z),
    two-hot-decoded (reference `Q(z, π(z), return_type='avg', target=True)`).

Holds raw pointers to the trainer-owned modules (dynamics / reward / policy
/ target-Q ensemble) + an owned TwoHotDecode. CPU-only (eval / single-env
acting path); the batched GPU planner callback is a later phase.
"""

from std.memory import alloc
from std.math import tanh
from std.random.philox import Random as PhiloxRandom
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor, row_major, TileTensor

from mojo_rl.nn2.constants import DT, TPB
from mojo_rl.nn2.core.module import mptr
from mojo_rl.nn2.initializer import Zero
from mojo_rl.planners.trajectory.rollout_callback import (
    RolloutCallbackCPU, RolloutCallbackGPU,
)

from .nets import TDMPC2Dynamics, TDMPC2Reward, TDMPC2QNet, TDMPC2Policy
from .losses import TwoHotDecode


@always_inline
def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


@fieldwise_init
struct TDMPC2RolloutCallbackCPU[
    ACT: Int,
    LATENT: Int,
    MLP: Int,
    BINS: Int,
    SN: Int,
    VMIN: Int,
    VMAX: Int,
    QP: Float64 = 0.0,
](RolloutCallbackCPU):
    comptime LATENT_DIM: Int = Self.LATENT
    comptime ACTION_DIM: Int = Self.ACT
    comptime DynT = TDMPC2Dynamics[Self.LATENT, Self.ACT, Self.MLP, Self.SN]
    comptime RewT = TDMPC2Reward[Self.LATENT, Self.ACT, Self.MLP, Self.BINS]
    comptime QNetT = TDMPC2QNet[Self.LATENT, Self.ACT, Self.MLP, Self.BINS, Self.QP]
    comptime PolicyT = TDMPC2Policy[Self.LATENT, Self.ACT, Self.MLP]
    comptime ZA = Self.LATENT + Self.ACT

    var dyn: UnsafePointer[Self.DynT, MutAnyOrigin]
    var rew: UnsafePointer[Self.RewT, MutAnyOrigin]
    var pol: UnsafePointer[Self.PolicyT, MutAnyOrigin]
    var qt: UnsafePointer[List[Self.QNetT], MutAnyOrigin]
    var decode: TwoHotDecode[Self.BINS, Self.VMIN, Self.VMAX]
    var action_scale: Float64
    var qi: Int
    var qj: Int

    @staticmethod
    def make(
        mut dyn: Self.DynT,
        mut rew: Self.RewT,
        mut pol: Self.PolicyT,
        mut qt: List[Self.QNetT],
        action_scale: Float64,
        qi: Int,
        qj: Int,
    ) raises -> Self:
        return Self(
            dyn=UnsafePointer(to=dyn),
            rew=UnsafePointer(to=rew),
            pol=UnsafePointer(to=pol),
            qt=UnsafePointer(to=qt),
            decode=TwoHotDecode[
                Self.BINS, Self.VMIN, Self.VMAX
            ].make["cpu", INIT=Zero](),
            action_scale=action_scale,
            qi=qi,
            qj=qj,
        )

    @always_inline
    def _z_buf(self, z: List[Float64]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        var p = _a(Self.LATENT)
        for i in range(Self.LATENT):
            p[i] = Scalar[DT](z[i])
        return p

    def policy_action_cpu(
        mut self, z: List[Float64], mut action_out: List[Float64],
    ) raises:
        var zb = self._z_buf(z)
        var pio = _a(2 * Self.ACT)
        var pio_t = TileTensor(pio, row_major[1, 2 * Self.ACT]())
        self.pol[].forward["cpu", 1](
            TileTensor(zb, row_major[1, Self.LATENT]()), output=pio_t,
        )
        for j in range(Self.ACT):
            action_out[j] = Float64(tanh(pio[j]))   # normalized [-1,1] mean
        zb.free(); pio.free()

    @always_inline
    def _za_buf(
        self, z: List[Float64], a: List[Float64]
    ) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        var p = _a(Self.ZA)
        for i in range(Self.LATENT):
            p[i] = Scalar[DT](z[i])
        for j in range(Self.ACT):
            p[Self.LATENT + j] = Scalar[DT](a[j] * self.action_scale)
        return p

    def rollout_step_cpu(
        mut self,
        z: List[Float64],
        a: List[Float64],
        mut z_next_out: List[Float64],
    ) raises -> Float64:
        var za = self._za_buf(z, a)
        var za_t = TileTensor(za, row_major[1, Self.ZA]())
        # dynamics → z'
        var zn = _a(Self.LATENT)
        var zn_t = TileTensor(zn, row_major[1, Self.LATENT]())
        self.dyn[].forward["cpu", 1](za_t, output=zn_t)
        for i in range(Self.LATENT):
            z_next_out[i] = Float64(zn[i])
        # reward → two-hot decode → scalar
        var rl = _a(Self.BINS)
        var rl_t = TileTensor(rl, row_major[1, Self.BINS]())
        self.rew[].forward["cpu", 1](za_t, output=rl_t)
        var rs = _a(1)
        var rs_t = TileTensor(rs, row_major[1, 1]())
        self.decode.forward["cpu", 1](rl_t, output=rs_t)
        var reward = Float64(rs[0])
        za.free(); zn.free(); rl.free(); rs.free()
        return reward

    def terminal_value_cpu(mut self, z: List[Float64]) raises -> Float64:
        # action = tanh(mean) of π(z), then za = [z, action·scale]
        var act = List[Float64](length=Self.ACT, fill=0.0)
        self.policy_action_cpu(z, act)
        var za = self._za_buf(z, act)
        var za_t = TileTensor(za, row_major[1, Self.ZA]())
        var ql = _a(Self.BINS)
        var ql_t = TileTensor(ql, row_major[1, Self.BINS]())
        var qs = _a(1)
        var qs_t = TileTensor(qs, row_major[1, 1]())
        self.qt[][self.qi].forward["cpu", 1](za_t, output=ql_t)
        self.decode.forward["cpu", 1](ql_t, output=qs_t)
        var qa = Float64(qs[0])
        self.qt[][self.qj].forward["cpu", 1](za_t, output=ql_t)
        self.decode.forward["cpu", 1](ql_t, output=qs_t)
        var qb = Float64(qs[0])
        za.free(); ql.free(); qs.free()
        return (qa + qb) * 0.5


# ──────────────────────────────────────────────────────────────────────
# GPU batched MPPI callback — the practical MPC path (MPPIGPUBatched plans
# all N_ENVS×TOTAL_SAMPLES trajectories per horizon step in one grid).
# ──────────────────────────────────────────────────────────────────────


@always_inline
def _dpg(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return mptr(b.unsafe_ptr())


@always_inline
def _ltg[N: Int](
    p: UnsafePointer[Scalar[DT], MutAnyOrigin]
) -> LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin]:
    return LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](p)


def _extract_tanh_mean_k[B_: Int, ACT_: Int, POL_: Int](
    pio: LayoutTensor[DT, Layout.row_major(B_ * POL_), MutAnyOrigin],
    action: LayoutTensor[DT, Layout.row_major(B_ * ACT_), MutAnyOrigin],
):
    """action[b,j] = tanh(pio[b, j])  (mean = first ACT cols of [mean|log_std])."""
    var i = Int(global_idx.x)
    if i < B_ * ACT_:
        var b = i // ACT_
        var j = i % ACT_
        action[i] = tanh(rebind[Scalar[DT]](pio[b * POL_ + j]))


def _build_za_scaled_k[B_: Int, LATENT_: Int, ACT_: Int, ZA_: Int](
    z: LayoutTensor[DT, Layout.row_major(B_ * LATENT_), MutAnyOrigin],
    a: LayoutTensor[DT, Layout.row_major(B_ * ACT_), MutAnyOrigin],
    za: LayoutTensor[DT, Layout.row_major(B_ * ZA_), MutAnyOrigin],
    scale: Scalar[DT],
):
    """za[b] = [z[b] | a[b]·scale]."""
    var i = Int(global_idx.x)
    if i < B_ * ZA_:
        var b = i // ZA_
        var k = i % ZA_
        if k < LATENT_:
            za[i] = rebind[Scalar[DT]](z[b * LATENT_ + k])
        else:
            za[i] = rebind[Scalar[DT]](a[b * ACT_ + (k - LATENT_)]) * scale


def _avg2_k[B_: Int](
    qa: LayoutTensor[DT, Layout.row_major(B_), MutAnyOrigin],
    qb: LayoutTensor[DT, Layout.row_major(B_), MutAnyOrigin],
    out_v: LayoutTensor[DT, Layout.row_major(B_), MutAnyOrigin],
):
    var b = Int(global_idx.x)
    if b < B_:
        out_v[b] = Scalar[DT](0.5) * (
            rebind[Scalar[DT]](qa[b]) + rebind[Scalar[DT]](qb[b])
        )


@fieldwise_init
struct TDMPC2RolloutCallbackGPU[
    ACT: Int,
    LATENT: Int,
    MLP: Int,
    BINS: Int,
    SN: Int,
    VMIN: Int,
    VMAX: Int,
    NUM_Q: Int,
    BT: Int,   # BATCH_TOTAL = N_ENVS × TOTAL_SAMPLES
    QP: Float64 = 0.0,
](RolloutCallbackGPU):
    comptime LATENT_DIM: Int = Self.LATENT
    comptime ACTION_DIM: Int = Self.ACT
    comptime POL: Int = 2 * Self.ACT
    comptime ZA: Int = Self.LATENT + Self.ACT
    comptime DynT = TDMPC2Dynamics[Self.LATENT, Self.ACT, Self.MLP, Self.SN]
    comptime RewT = TDMPC2Reward[Self.LATENT, Self.ACT, Self.MLP, Self.BINS]
    comptime QNetT = TDMPC2QNet[Self.LATENT, Self.ACT, Self.MLP, Self.BINS, Self.QP]
    comptime PolicyT = TDMPC2Policy[Self.LATENT, Self.ACT, Self.MLP]

    var dyn: UnsafePointer[Self.DynT, MutAnyOrigin]
    var rew: UnsafePointer[Self.RewT, MutAnyOrigin]
    var pol: UnsafePointer[Self.PolicyT, MutAnyOrigin]
    var qt: UnsafePointer[List[Self.QNetT], MutAnyOrigin]
    var decode: TwoHotDecode[Self.BINS, Self.VMIN, Self.VMAX]
    var action_scale: Scalar[DT]
    # device scratch (sized BT)
    var pio: DeviceBuffer[DT]
    var action: DeviceBuffer[DT]
    var za: DeviceBuffer[DT]
    var rlog: DeviceBuffer[DT]
    var qlog: DeviceBuffer[DT]
    var qa: DeviceBuffer[DT]
    var qb: DeviceBuffer[DT]

    @staticmethod
    def make(
        mut dyn: Self.DynT,
        mut rew: Self.RewT,
        mut pol: Self.PolicyT,
        mut qt: List[Self.QNetT],
        action_scale: Scalar[DT],
        ctx: DeviceContext,
    ) raises -> Self:
        return Self(
            dyn=UnsafePointer(to=dyn),
            rew=UnsafePointer(to=rew),
            pol=UnsafePointer(to=pol),
            qt=UnsafePointer(to=qt),
            decode=TwoHotDecode[
                Self.BINS, Self.VMIN, Self.VMAX
            ].make["gpu", INIT=Zero](ctx=ctx),
            action_scale=action_scale,
            pio=ctx.enqueue_create_buffer[DT](Self.BT * Self.POL),
            action=ctx.enqueue_create_buffer[DT](Self.BT * Self.ACT),
            za=ctx.enqueue_create_buffer[DT](Self.BT * Self.ZA),
            rlog=ctx.enqueue_create_buffer[DT](Self.BT * Self.BINS),
            qlog=ctx.enqueue_create_buffer[DT](Self.BT * Self.BINS),
            qa=ctx.enqueue_create_buffer[DT](Self.BT),
            qb=ctx.enqueue_create_buffer[DT](Self.BT),
        )

    def policy_action_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        z: LayoutTensor[
            DT, Layout.row_major(B, Self.LATENT_DIM), MutAnyOrigin
        ],
        action_out: LayoutTensor[
            DT, Layout.row_major(B, Self.ACTION_DIM), MutAnyOrigin
        ],
    ) raises:
        comptime assert B == Self.BT, "callback B must equal BT"
        var zp = mptr(z.ptr)
        var ap = mptr(action_out.ptr)
        var pio_t = TileTensor(_dpg(self.pio), row_major[B, Self.POL]())
        self.pol[].forward["gpu", B](
            TileTensor(zp, row_major[B, Self.LATENT]()), output=pio_t
        )
        comptime k = _extract_tanh_mean_k[B, Self.ACT, Self.POL]
        comptime nb = (B * Self.ACT + TPB - 1) // TPB
        ctx.enqueue_function[k](
            _ltg[B * Self.POL](_dpg(self.pio)), _ltg[B * Self.ACT](ap),
            grid_dim=nb, block_dim=TPB,
        )

    def rollout_step_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        z: LayoutTensor[
            DT, Layout.row_major(B, Self.LATENT_DIM), MutAnyOrigin
        ],
        a: LayoutTensor[
            DT, Layout.row_major(B, Self.ACTION_DIM), MutAnyOrigin
        ],
        z_next_out: LayoutTensor[
            DT, Layout.row_major(B, Self.LATENT_DIM), MutAnyOrigin
        ],
        r_out: LayoutTensor[DT, Layout.row_major(B), MutAnyOrigin],
    ) raises:
        comptime assert B == Self.BT, "callback B must equal BT"
        var zp = mptr(z.ptr)
        var ap = mptr(a.ptr)
        var znp = mptr(z_next_out.ptr)
        var rp = mptr(r_out.ptr)
        comptime bza = _build_za_scaled_k[B, Self.LATENT, Self.ACT, Self.ZA]
        comptime nbz = (B * Self.ZA + TPB - 1) // TPB
        ctx.enqueue_function[bza](
            _ltg[B * Self.LATENT](zp), _ltg[B * Self.ACT](ap),
            _ltg[B * Self.ZA](_dpg(self.za)), self.action_scale,
            grid_dim=nbz, block_dim=TPB,
        )
        var za_t = TileTensor(_dpg(self.za), row_major[B, Self.ZA]())
        var zn_t = TileTensor(znp, row_major[B, Self.LATENT]())
        self.dyn[].forward["gpu", B](za_t, output=zn_t)
        var rl_t = TileTensor(_dpg(self.rlog), row_major[B, Self.BINS]())
        self.rew[].forward["gpu", B](za_t, output=rl_t)
        var r_t = TileTensor(rp, row_major[B, 1]())
        self.decode.forward["gpu", B](rl_t, output=r_t)

    def terminal_value_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        z: LayoutTensor[
            DT, Layout.row_major(B, Self.LATENT_DIM), MutAnyOrigin
        ],
        v_out: LayoutTensor[DT, Layout.row_major(B), MutAnyOrigin],
        seed: UInt32,
    ) raises:
        comptime assert B == Self.BT, "callback B must equal BT"
        # host Philox → 2 distinct Q indices (matches legacy recipe).
        var rng = PhiloxRandom(seed=UInt64(seed) + UInt64(0xA1B2C3D4), offset=0)
        var u = rng.step_uniform()
        var qi = Int(Float64(u[0]) * Float64(Self.NUM_Q)) % Self.NUM_Q
        var qj = (
            qi + 1 + Int(Float64(u[1]) * Float64(Self.NUM_Q - 1))
            % (Self.NUM_Q - 1)
        ) % Self.NUM_Q

        var zp = mptr(z.ptr)
        var vp = mptr(v_out.ptr)
        # action = tanh(mean) of π(z)
        var pio_t = TileTensor(_dpg(self.pio), row_major[B, Self.POL]())
        self.pol[].forward["gpu", B](
            TileTensor(zp, row_major[B, Self.LATENT]()), output=pio_t
        )
        comptime ek = _extract_tanh_mean_k[B, Self.ACT, Self.POL]
        comptime nba = (B * Self.ACT + TPB - 1) // TPB
        ctx.enqueue_function[ek](
            _ltg[B * Self.POL](_dpg(self.pio)),
            _ltg[B * Self.ACT](_dpg(self.action)),
            grid_dim=nba, block_dim=TPB,
        )
        # za = [z, action·scale]
        comptime bza = _build_za_scaled_k[B, Self.LATENT, Self.ACT, Self.ZA]
        comptime nbz = (B * Self.ZA + TPB - 1) // TPB
        ctx.enqueue_function[bza](
            _ltg[B * Self.LATENT](zp), _ltg[B * Self.ACT](_dpg(self.action)),
            _ltg[B * Self.ZA](_dpg(self.za)), self.action_scale,
            grid_dim=nbz, block_dim=TPB,
        )
        var za_t = TileTensor(_dpg(self.za), row_major[B, Self.ZA]())
        # avg of 2 (random) target-Q, two-hot decoded
        var ql_t = TileTensor(_dpg(self.qlog), row_major[B, Self.BINS]())
        var qa_t = TileTensor(_dpg(self.qa), row_major[B, 1]())
        var qb_t = TileTensor(_dpg(self.qb), row_major[B, 1]())
        self.qt[][qi].forward["gpu", B](za_t, output=ql_t)
        self.decode.forward["gpu", B](ql_t, output=qa_t)
        self.qt[][qj].forward["gpu", B](za_t, output=ql_t)
        self.decode.forward["gpu", B](ql_t, output=qb_t)
        comptime ak = _avg2_k[B]
        comptime nbb = (B + TPB - 1) // TPB
        ctx.enqueue_function[ak](
            _ltg[B](_dpg(self.qa)), _ltg[B](_dpg(self.qb)), _ltg[B](vp),
            grid_dim=nbb, block_dim=TPB,
        )
