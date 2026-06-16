"""TD-MPC2 multi-task policy (actor) update block (CPU + GPU) — item C, §14.3.

Clone of `policy_step.mojo` threading a per-row task embedding: the policy graph
gains a `task_emb` input (feeding both `zt = [z|task_emb]` and the detached-Q
`za = [z|action|task_emb]`). After `graph.vjp`, the `task_emb` input-slot grad is
scatter-added into the table (site 3 of the embedding grad flow). The policy
batch `B` here is `PB = (H+1)·B_wm`; the caller passes one task id per row.

Action masking note: per-task action masking (zeroing unused action dims) is
applied at acting + replay-record time (env wrapper + `agent_mt.select_action`),
which is what the lighthouse (`MAX_ACT=1`, mask≡1) and the synthetic-mask test
exercise. In-graph masking of the actor-loss sampled action + masked-dim log-prob
exclusion is deferred-experimental (a no-op at MAX_ACT=1; revisit for a real
heterogeneous-action multi-task suite) — same spirit as the QP-dropout caveats.
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

from .nets_mt import TDMPC2PolicyMT, TDMPC2QNetMT
from .policy_graph_mt import TDMPC2PolicyGraphMT
from .policy_step import _dp, _lt_pol, _scale_copy_k
from .wm_step import _alloc
from .running_scale import RunningScale
from .task_embedding import TaskEmbedding


struct PolicyStepMT[
    LATENT: Int,
    MAX_ACT: Int,
    MLP: Int,
    BINS: Int,
    VMIN: Int,
    VMAX: Int,
    B: Int,
    NUM_TASKS: Int,
    TASK_EMB: Int,
    QP: Float64 = 0.0,
](Movable & ImplicitlyDeletable):
    comptime PolicyT = TDMPC2PolicyMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.TASK_EMB
    ]
    comptime QNetT = TDMPC2QNetMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.BINS, Self.TASK_EMB, Self.QP
    ]
    comptime GraphT = TDMPC2PolicyGraphMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.BINS, Self.VMIN, Self.VMAX,
        Self.TASK_EMB, Self.QP,
    ]
    comptime EmbT = TaskEmbedding[Self.NUM_TASKS, Self.TASK_EMB]

    var graph: Self.GraphT
    var rsample: RSample[Self.MAX_ACT]
    var scale: RunningScale
    var entropy_coef: Scalar[DT]
    var q_mean: Scalar[DT]
    var q_min: Scalar[DT]
    var q_max: Scalar[DT]
    # Persistent GPU scratch (allocated once in `make`, reused every step —
    # per-step `enqueue_create_buffer` explodes disk on NVIDIA).
    var d_tem: Optional[DeviceBuffer[DT]]
    var d_loss: Optional[DeviceBuffer[DT]]
    var d_qavg: Optional[DeviceBuffer[DT]]
    var d_grad: Optional[DeviceBuffer[DT]]
    var h_loss: Optional[HostBuffer[DT]]
    var h_qavg: Optional[HostBuffer[DT]]

    def __init__(out self):
        self.graph = Self.GraphT()
        self.rsample = RSample[Self.MAX_ACT]()
        self.scale = RunningScale()
        self.entropy_coef = Scalar[DT](1e-4)
        self.q_mean = Scalar[DT](0.0)
        self.q_min = Scalar[DT](0.0)
        self.q_max = Scalar[DT](0.0)
        self.d_tem = None; self.d_loss = None; self.d_qavg = None
        self.d_grad = None; self.h_loss = None; self.h_qavg = None

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
            "PolicyStepMT: target must be 'cpu' or 'gpu'"
        )
        var s = Self()
        s.graph = Self.GraphT.make[target, INIT=Zero](ctx=ctx)
        s.rsample = RSample[Self.MAX_ACT].make[target, INIT=Zero](ctx=ctx)
        comptime if target == "gpu":
            var c = ctx.value()
            s.d_tem = c.enqueue_create_buffer[DT](Self.B * Self.TASK_EMB)
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
        tem: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        self.graph.set_external["pi_out", Self.PolicyT](policy)
        self.graph.set_external["alp", RSample[Self.MAX_ACT]](self.rsample)
        self.graph.set_external["q1", Self.QNetT](q[qi])
        self.graph.set_external["q2", Self.QNetT](q[qj])
        self.graph.set_input["z", Self.B](
            TileTensor(z, row_major[Self.B, Self.LATENT]())
        )
        self.graph.set_input["task_emb", Self.B](
            TileTensor(tem, row_major[Self.B, Self.TASK_EMB]())
        )
        self.graph.set_node_attr["alpha_lp", "multiplier"](
            self.entropy_coef * Scalar[DT](Self.MAX_ACT)
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
        mut task_emb: Self.EmbT,
        z: UnsafePointer[Scalar[DT], MutAnyOrigin],          # [B, LATENT]
        task_ids: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B] per-row
        ctx: Optional[DeviceContext] = None,
    ) raises -> Scalar[DT]:
        comptime if target == "cpu":
            return self._pol_cpu[target](
                policy, q, qi, qj, pi_opt, task_emb, z, task_ids
            )
        else:
            return self._pol_gpu[target](
                policy, q, qi, qj, pi_opt, task_emb, z, task_ids, ctx.value()
            )

    def _pol_cpu[target: StaticString](
        mut self,
        mut policy: Self.PolicyT,
        mut q: List[Self.QNetT],
        qi: Int,
        qj: Int,
        mut pi_opt: Adam,
        mut task_emb: Self.EmbT,
        z: UnsafePointer[Scalar[DT], MutAnyOrigin],
        task_ids: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises -> Scalar[DT]:
        comptime BB = Self.B
        comptime EMB = Self.TASK_EMB
        var tem = _alloc(BB * EMB)
        task_emb.gather[target, BB](task_ids, tem)
        self._bind[target](policy, q, qi, qj, z, tem)
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

        # site 3: actor-loss grad w.r.t. the task embedding.
        task_emb.accumulate[target, BB, EMB, 0](
            task_ids, self.graph.grad_input_ptr["task_emb"]()
        )

        pi_opt.step[target, Self.PolicyT](policy)
        loss.free(); qavg.free(); grad.free(); tem.free()
        return loss_mean

    def _pol_gpu[target: StaticString](
        mut self,
        mut policy: Self.PolicyT,
        mut q: List[Self.QNetT],
        qi: Int,
        qj: Int,
        mut pi_opt: Adam,
        mut task_emb: Self.EmbT,
        z: UnsafePointer[Scalar[DT], MutAnyOrigin],          # device [B, LATENT]
        task_ids: UnsafePointer[Scalar[DT], MutAnyOrigin],   # device [B]
        ctx: DeviceContext,
    ) raises -> Scalar[DT]:
        comptime BB = Self.B
        comptime EMB = Self.TASK_EMB
        var d_tem = self.d_tem.value()
        task_emb.gather[target, BB](task_ids, _dp(d_tem), ctx=ctx)
        self._bind[target](policy, q, qi, qj, z, _dp(d_tem))
        pi_opt.zero_grad[target, Self.PolicyT](policy)

        var d_loss = self.d_loss.value()
        var loss_t = TileTensor(_dp(d_loss), row_major[BB, 1]())
        self.graph.forward[target, BB](loss_t)

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

        # site 3: actor-loss grad w.r.t. the task embedding.
        task_emb.accumulate[target, BB, EMB, 0](
            task_ids, self.graph.grad_input_ptr["task_emb"](), ctx=ctx
        )

        pi_opt.step[target, Self.PolicyT](policy)
        return loss_mean
