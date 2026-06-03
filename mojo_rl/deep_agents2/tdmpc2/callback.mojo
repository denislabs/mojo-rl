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
from layout import row_major, TileTensor

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.planners.trajectory.rollout_callback import RolloutCallbackCPU

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
](RolloutCallbackCPU):
    comptime LATENT_DIM: Int = Self.LATENT
    comptime ACTION_DIM: Int = Self.ACT
    comptime DynT = TDMPC2Dynamics[Self.LATENT, Self.ACT, Self.MLP, Self.SN]
    comptime RewT = TDMPC2Reward[Self.LATENT, Self.ACT, Self.MLP, Self.BINS]
    comptime QNetT = TDMPC2QNet[Self.LATENT, Self.ACT, Self.MLP, Self.BINS]
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
