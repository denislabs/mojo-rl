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
from layout import TileTensor, row_major
from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.deep_agents2.primitives.rsample import RSample
from mojo_rl.deep_agents2.loss.seed_grad_inv_batch import seed_grad_inv_batch

from .nets import TDMPC2Policy, TDMPC2QNet
from .policy_graph import TDMPC2PolicyGraph
from .running_scale import RunningScale


struct PolicyStep[
    LATENT: Int,
    ACT: Int,
    MLP: Int,
    BINS: Int,
    VMIN: Int,
    VMAX: Int,
    B: Int,
](Movable & ImplicitlyDestructible):
    comptime PolicyT = TDMPC2Policy[Self.LATENT, Self.ACT, Self.MLP]
    comptime QNetT = TDMPC2QNet[Self.LATENT, Self.ACT, Self.MLP, Self.BINS]
    comptime GraphT = TDMPC2PolicyGraph[
        Self.LATENT, Self.ACT, Self.MLP, Self.BINS, Self.VMIN, Self.VMAX
    ]

    var graph: Self.GraphT
    var rsample: RSample[Self.ACT]
    var scale: RunningScale
    var entropy_coef: Scalar[DT]

    def __init__(out self):
        self.graph = Self.GraphT()
        self.rsample = RSample[Self.ACT]()
        self.scale = RunningScale()
        self.entropy_coef = Scalar[DT](1e-4)

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu", (
            "PolicyStep: only the CPU path is implemented (GPU is P4)"
        )
        var s = Self()
        s.graph = Self.GraphT.make[target, INIT=Zero](ctx=ctx)
        s.rsample = RSample[Self.ACT].make[target, INIT=Zero](ctx=ctx)
        return s^

    def step[target: StaticString](
        mut self,
        mut policy: Self.PolicyT,
        mut q1: Self.QNetT,
        mut q2: Self.QNetT,
        mut pi_opt: Adam,
        z: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B, LATENT]
    ) raises -> Scalar[DT]:
        comptime assert target == "cpu", "PolicyStep.step: CPU only (P4 = GPU)"
        comptime BB = Self.B

        # ── Bind externals (repeat each call: fields may move). ────────
        self.graph.set_external["pi_out", Self.PolicyT](policy)
        self.graph.set_external["alp", RSample[Self.ACT]](self.rsample)
        self.graph.set_external["q1", Self.QNetT](q1)
        self.graph.set_external["q2", Self.QNetT](q2)

        self.graph.set_input["z", BB](
            TileTensor(z, row_major[BB, Self.LATENT]())
        )

        # entropy term coef = entropy_coef·ACT; Q scaled by 0.5/scale.
        self.graph.set_node_attr["alpha_lp", "multiplier"](
            self.entropy_coef * Scalar[DT](Self.ACT)
        )
        self.graph.set_node_attr["qscaled", "multiplier"](
            Scalar[DT](0.5) * self.scale.inv()
        )

        pi_opt.zero_grad[target, Self.PolicyT](policy)

        var loss = alloc[Scalar[DT]](BB)
        var loss_t = TileTensor(loss, row_major[BB, 1]())
        self.graph.forward[target, BB](loss_t)

        var loss_sum: Scalar[DT] = 0.0
        for b in range(BB):
            loss_sum += loss[b]
        var loss_mean = loss_sum / Scalar[DT](BB)

        # ── Update RunningScale from this step's avg-Q (for next step). ─
        var qsum = self.graph.node_out_ptr["qsum"]()
        var qavg = alloc[Scalar[DT]](BB)
        for b in range(BB):
            qavg[b] = qsum[b] * Scalar[DT](0.5)
        self.scale.update_from(qavg, BB)

        # ── Backward (policy only) + step. ─────────────────────────────
        var grad = alloc[Scalar[DT]](BB)
        seed_grad_inv_batch[target, BB](grad, ctx=None)
        var grad_t = TileTensor(grad, row_major[BB, 1]())
        self.graph.vjp[target, BB](grad_t)

        pi_opt.step[target, Self.PolicyT](policy)

        loss.free()
        qavg.free()
        grad.free()
        return loss_mean
