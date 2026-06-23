"""PPOActorLoss — PPO actor loss as an imperative storage block.

The actor-side loss is a 2-node chain (actor → PPOObjective), so it does NOT use
a ComputeGraph (the storage graph dispatch tops out at arity 3; PPOObjective is
arity 4). Instead it drives the pieces directly, mirroring the off-policy
q-update blocks:

    actor.forward(s) → actor_out [B, 2*ACT]
    loss_per_b = PPOObjective(actor_out, a, old_log_prob, advantage)   [B, 1]
    seed = 1/BATCH ; PPOObjective.vjp → grad_actor_out (+ zero rollout grads)
    actor.vjp(grad_actor_out) → actor param grads ; (optional grad-norm clip)
    actor_opt.step

The four objective inputs are staged into an owned `TensorPack[4]` so they share
one origin (§B0) for the `TensorRefs[4]` the leaf consumes; grad_inputs land in a
second pool. `forward_backward` returns the mean per-batch loss for logging.
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.core.initializer import Zero
from .objective import PPOObjective
from ..loss.loss_block import LossBlock


struct PPOActorLoss[
    ACTOR: Module,
    BATCH: Int,
](LossBlock):
    comptime OBS_DIM = Self.ACTOR.IN_DIMS[0]
    comptime ACT_DIM = Self.ACTOR.OUT_DIM // 2

    var objective: PPOObjective[Self.ACT_DIM]
    var _in: TensorPack[4]   # [actor_out (2*ACT) | a (ACT) | olp (1) | adv (1)]
    var _gin: TensorPack[4]  # grad_inputs (only slot 0 = grad_actor_out used)
    var _loss_out: Tensor    # [B] loss_per_b
    var _grad_seed: Tensor   # [B] = 1/BATCH backward seed
    var _obs_grad: Tensor    # [B*OBS] unused grad sink for actor.vjp

    def __init__(out self):
        self.objective = PPOObjective[Self.ACT_DIM]()
        self._in = TensorPack[4]()
        self._gin = TensorPack[4]()
        self._loss_out = Tensor()
        self._grad_seed = Tensor()
        self._obs_grad = Tensor()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
        clip_eps: Scalar[DT] = Scalar[DT](0.2),
        entropy_coef: Scalar[DT] = Scalar[DT](0.0),
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "PPOActorLoss: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.ACTOR.OUT_DIM == 2 * Self.ACT_DIM, (
            "PPOActorLoss: ACTOR.OUT_DIM must equal 2·ACT_DIM"
        )
        var blk = Self()
        blk.objective = PPOObjective[Self.ACT_DIM].make[target, Zero](ctx)
        blk.objective.set_attr["clip_eps"](clip_eps)
        blk.objective.set_attr["entropy_coef"](entropy_coef)
        blk._loss_out = Tensor.make[target](Self.BATCH, ctx)
        blk._obs_grad = Tensor.make[target](Self.BATCH * Self.OBS_DIM, ctx)
        # Seed = 1/BATCH in every slot (host-fill → upload on GPU).
        blk._grad_seed = Tensor.alloc(Self.BATCH)
        for b in range(Self.BATCH):
            blk._grad_seed.data[b] = Scalar[DT](1.0) / Scalar[DT](Self.BATCH)
        comptime if target == "gpu":
            blk._grad_seed.upload(ctx.value())
        return blk^

    def set_clip_eps(mut self, value: Scalar[DT]):
        self.objective.set_attr["clip_eps"](value)

    def set_entropy_coef(mut self, value: Scalar[DT]):
        self.objective.set_attr["entropy_coef"](value)

    @staticmethod
    def _copy_into[
        target: StaticString
    ](
        mut dst: Tensor, mut src: Tensor, n: Int, ctx: Optional[DeviceContext]
    ) raises:
        """Copy `n` elements src → dst (host element-loop on CPU; device
        enqueue_copy on GPU). Stages an external input into the §B0 input pool."""
        comptime if target == "cpu":
            dst.ensure(n)
            for i in range(n):
                dst.data[i] = src.data[i]
        else:
            var c = ctx.value()
            dst.ensure_gpu(c, n)
            c.enqueue_copy(dst.dev.value(), src.dev.value())

    def forward_backward[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut actor: Self.ACTOR,
        mut actor_opt: Adam,
        mut mb_s: Tensor,
        mut mb_a: Tensor,
        mut mb_olp: Tensor,
        mut mb_adv: Tensor,
        max_grad_norm: Scalar[DT] = Scalar[DT](0.0),
        ctx: Optional[DeviceContext] = None,
    ) raises -> Scalar[DT]:
        comptime BB = Self.BATCH
        comptime ACT = Self.ACT_DIM

        actor_opt.zero_grad[target, M = Self.ACTOR](actor, ctx)

        # actor.forward(s) → _in[0] (the [mu|log_std] actor output).
        actor.forward[target, BB, POLICY=POLICY](
            TensorRefs[Self.ACTOR.ARITY](mb_s), self._in[0], ctx
        )
        # Stage the three rollout-time inputs into the §B0 pool.
        Self._copy_into[target](self._in[1], mb_a, BB * ACT, ctx)
        Self._copy_into[target](self._in[2], mb_olp, BB, ctx)
        Self._copy_into[target](self._in[3], mb_adv, BB, ctx)

        # loss_per_b = PPOObjective(...).
        self.objective.forward[target, BB, POLICY=POLICY](
            TensorRefs[4](self._in[0], self._in[1], self._in[2], self._in[3]),
            self._loss_out,
            ctx,
        )

        # Mean loss for logging (host reduction; D2H on GPU).
        comptime if target == "gpu":
            self._loss_out.download(ctx.value())
        var loss_sum: Scalar[DT] = 0.0
        for b in range(BB):
            loss_sum += self._loss_out.data[b]
        var loss_mean = loss_sum / Scalar[DT](BB)

        # Backward: seed 1/BATCH → grad_actor_out (+ zeroed rollout grads).
        self.objective.vjp[target, BB, POLICY=POLICY](
            TensorRefs[4](self._in[0], self._in[1], self._in[2], self._in[3]),
            self._grad_seed,
            TensorRefs[4](
                self._gin[0], self._gin[1], self._gin[2], self._gin[3]
            ),
            ctx,
        )
        # actor.vjp(grad_actor_out) → actor param grads (obs grad discarded).
        actor.vjp[target, BB, POLICY=POLICY](
            TensorRefs[Self.ACTOR.ARITY](mb_s),
            self._gin[0],
            TensorRefs[Self.ACTOR.ARITY](self._obs_grad),
            ctx,
        )

        if max_grad_norm > Scalar[DT](0.0):
            _ = actor_opt.clip_grads[target, M = Self.ACTOR](
                actor, max_grad_norm, ctx
            )
        actor_opt.step[target, M = Self.ACTOR](actor, ctx)
        return loss_mean
