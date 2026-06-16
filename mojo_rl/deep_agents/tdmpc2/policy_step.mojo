"""TD-MPC2 policy (actor) update block — CPU (GPU is P4).

Drives `TDMPC2PolicyGraph`: binds the trainer-owned policy + two (random)
detached Q heads, sets the per-step Scale multipliers (`0.5/scale` for Q,
`entropy_coef·ACT` for the entropy term), forwards, updates RunningScale
from this step's avg-Q, backprops into the policy only, and steps the pi
optimizer.

RunningScale uses a one-step lag: the multiplier for step t uses the value
from step t−1's Q, then is updated from step t's Q for t+1. The EMA
(tau=0.01) drifts slowly so the lag is negligible — and it keeps the scale
out of the autograd graph (reference detaches it anyway).

Block owns the graph + RSample + RunningScale + coefs; the policy, Q heads,
and pi optimizer are passed by ref (trainer-owned).
"""

from std.memory import alloc
from layout import Layout, LayoutTensor, TileTensor, row_major
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.core.module import mptr
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.deep_agents.primitives.rsample import RSample
from mojo_rl.deep_agents.loss.seed_grad_inv_batch import seed_grad_inv_batch

from .nets import TDMPC2Policy, TDMPC2QNet
from .policy_graph import TDMPC2PolicyGraph
from .running_scale import RunningScale


@always_inline
def _dp(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return mptr(b.unsafe_ptr())


@always_inline
def _lt_pol[N: Int](
    p: UnsafePointer[Scalar[DT], MutAnyOrigin]
) -> LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin]:
    return LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](p)


def _scale_copy_k[N: Int](
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    f: Scalar[DT],
):
    var i = Int(global_idx.x)
    if i < N:
        dst[i] = rebind[Scalar[DT]](src[i]) * f


struct PolicyStep[
    LATENT: Int,
    ACT: Int,
    MLP: Int,
    BINS: Int,
    VMIN: Int,
    VMAX: Int,
    B: Int,
    QP: Float64 = 0.0,
](Movable & ImplicitlyDestructible):
    comptime PolicyT = TDMPC2Policy[Self.LATENT, Self.ACT, Self.MLP]
    comptime QNetT = TDMPC2QNet[Self.LATENT, Self.ACT, Self.MLP, Self.BINS, Self.QP]
    comptime GraphT = TDMPC2PolicyGraph[
        Self.LATENT, Self.ACT, Self.MLP, Self.BINS, Self.VMIN, Self.VMAX, Self.QP,
    ]

    var graph: Self.GraphT
    var rsample: RSample[Self.ACT]
    var scale: RunningScale
    var entropy_coef: Scalar[DT]
    # last-step Q diagnostics (avg-of-2 decoded Q at the policy's actions).
    var q_mean: Scalar[DT]
    var q_min: Scalar[DT]
    var q_max: Scalar[DT]
    # Persistent GPU scratch (allocated once in `make`, reused every step —
    # per-step `enqueue_create_buffer` explodes disk on NVIDIA).
    var d_loss: Optional[DeviceBuffer[DT]]
    var d_qavg: Optional[DeviceBuffer[DT]]
    var d_grad: Optional[DeviceBuffer[DT]]
    var h_loss: Optional[HostBuffer[DT]]
    var h_qavg: Optional[HostBuffer[DT]]

    def __init__(out self):
        self.graph = Self.GraphT()
        self.rsample = RSample[Self.ACT]()
        self.scale = RunningScale()
        self.entropy_coef = Scalar[DT](1e-4)
        self.q_mean = Scalar[DT](0.0)
        self.q_min = Scalar[DT](0.0)
        self.q_max = Scalar[DT](0.0)
        self.d_loss = None; self.d_qavg = None; self.d_grad = None
        self.h_loss = None; self.h_qavg = None

    def _set_q_stats(
        mut self, p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int
    ):
        if n <= 0:
            return
        var s: Scalar[DT] = 0.0
        var mn = p[0]
        var mx = p[0]
        for i in range(n):
            var v = p[i]
            s += v
            if v < mn:
                mn = v
            if v > mx:
                mx = v
        self.q_mean = s / Scalar[DT](n)
        self.q_min = mn
        self.q_max = mx

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "PolicyStep: target must be 'cpu' or 'gpu'"
        )
        var s = Self()
        s.graph = Self.GraphT.make[target, INIT=Zero](ctx=ctx)
        s.rsample = RSample[Self.ACT].make[target, INIT=Zero](ctx=ctx)
        comptime if target == "gpu":
            var c = ctx.value()
            s.d_loss = c.enqueue_create_buffer[DT](Self.B)
            s.d_qavg = c.enqueue_create_buffer[DT](Self.B)
            s.d_grad = c.enqueue_create_buffer[DT](Self.B)
            s.h_loss = c.enqueue_create_host_buffer[DT](Self.B)
            s.h_qavg = c.enqueue_create_host_buffer[DT](Self.B)
            c.synchronize()
        return s^

    def _bind[target: StaticString](
        mut self,
        mut policy: Self.PolicyT,
        mut q: List[Self.QNetT],
        qi: Int,
        qj: Int,
        z: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        # Two `mut` subscripts of one List in a single call is rejected by
        # Mojo's aliasing checker — bind in separate statements.
        self.graph.set_external["pi_out", Self.PolicyT](policy)
        self.graph.set_external["alp", RSample[Self.ACT]](self.rsample)
        self.graph.set_external["q1", Self.QNetT](q[qi])
        self.graph.set_external["q2", Self.QNetT](q[qj])
        self.graph.set_input["z", Self.B](
            TileTensor(z, row_major[Self.B, Self.LATENT]())
        )
        self.graph.set_node_attr["alpha_lp", "multiplier"](
            self.entropy_coef * Scalar[DT](Self.ACT)
        )
        self.graph.set_node_attr["qscaled", "multiplier"](
            Scalar[DT](0.5) * self.scale.inv()
        )

    def step[target: StaticString](
        mut self,
        mut policy: Self.PolicyT,
        mut q: List[Self.QNetT],
        qi: Int,
        qj: Int,
        mut pi_opt: Adam,
        z: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B, LATENT]
        ctx: Optional[DeviceContext] = None,
    ) raises -> Scalar[DT]:
        comptime if target == "cpu":
            return self._pol_cpu[target](policy, q, qi, qj, pi_opt, z)
        else:
            return self._pol_gpu[target](policy, q, qi, qj, pi_opt, z, ctx.value())

    def _pol_cpu[target: StaticString](
        mut self,
        mut policy: Self.PolicyT,
        mut q: List[Self.QNetT],
        qi: Int,
        qj: Int,
        mut pi_opt: Adam,
        z: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises -> Scalar[DT]:
        comptime BB = Self.B
        self._bind[target](policy, q, qi, qj, z)
        pi_opt.zero_grad[target, Self.PolicyT](policy)

        var loss = alloc[Scalar[DT]](BB)
        var loss_t = TileTensor(loss, row_major[BB, 1]())
        self.graph.forward[target, BB](loss_t)

        var loss_sum: Scalar[DT] = 0.0
        for b in range(BB):
            loss_sum += loss[b]
        var loss_mean = loss_sum / Scalar[DT](BB)

        var qsum = self.graph.node_out_ptr["qsum"]()
        var qavg = alloc[Scalar[DT]](BB)
        for b in range(BB):
            qavg[b] = qsum[b] * Scalar[DT](0.5)
        self.scale.update_from(qavg, BB)
        self._set_q_stats(qavg, BB)

        var grad = alloc[Scalar[DT]](BB)
        seed_grad_inv_batch[target, BB](grad, ctx=None)
        var grad_t = TileTensor(grad, row_major[BB, 1]())
        self.graph.vjp[target, BB](grad_t)

        pi_opt.step[target, Self.PolicyT](policy)
        loss.free(); qavg.free(); grad.free()
        return loss_mean

    def _pol_gpu[target: StaticString](
        mut self,
        mut policy: Self.PolicyT,
        mut q: List[Self.QNetT],
        qi: Int,
        qj: Int,
        mut pi_opt: Adam,
        z: UnsafePointer[Scalar[DT], MutAnyOrigin],   # device [B, LATENT]
        ctx: DeviceContext,
    ) raises -> Scalar[DT]:
        comptime BB = Self.B
        self._bind[target](policy, q, qi, qj, z)
        pi_opt.zero_grad[target, Self.PolicyT](policy)

        var d_loss = self.d_loss.value()
        var loss_t = TileTensor(_dp(d_loss), row_major[BB, 1]())
        self.graph.forward[target, BB](loss_t)

        # D2H loss + (scaled) qsum → host reductions identical to CPU.
        var d_qavg = self.d_qavg.value()
        comptime sck = _scale_copy_k[BB]
        comptime nb = (BB + TPB - 1) // TPB
        ctx.enqueue_function[sck](
            _lt_pol[BB](self.graph.node_out_ptr["qsum"]()),
            _lt_pol[BB](_dp(d_qavg)),
            Scalar[DT](0.5),
            grid_dim=nb, block_dim=TPB,
        )
        var h_loss = self.h_loss.value()
        var h_qavg = self.h_qavg.value()
        ctx.enqueue_copy(h_loss, d_loss)
        ctx.enqueue_copy(h_qavg, d_qavg)
        ctx.synchronize()
        var loss_sum: Scalar[DT] = 0.0
        for b in range(BB):
            loss_sum += h_loss.unsafe_ptr()[b]
        var loss_mean = loss_sum / Scalar[DT](BB)
        var qavg_p = mptr(h_qavg.unsafe_ptr())
        self.scale.update_from(qavg_p, BB)
        self._set_q_stats(qavg_p, BB)

        var d_grad = self.d_grad.value()
        seed_grad_inv_batch[target, BB](_dp(d_grad), ctx=ctx)
        var grad_t = TileTensor(_dp(d_grad), row_major[BB, 1]())
        self.graph.vjp[target, BB](grad_t)

        pi_opt.step[target, Self.PolicyT](policy)
        return loss_mean
