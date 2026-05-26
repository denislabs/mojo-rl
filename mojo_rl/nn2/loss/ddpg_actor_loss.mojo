"""DDPGActorLoss — deterministic policy gradient block (Block E-4).

Maximizes E_s[critic(s, π_φ(s))] via:

  loss(φ)        = -mean_b critic(s, π_φ(s))
  ∂loss/∂a       = -1/B
  ∂a/∂φ          via π.backward
  ∂critic/∂a     via critic.vjp[mode="input_only"]
                  (NO critic param-grad accumulation — actor update only)

CPU only. GPU mirror is straightforward (every dependency has a GPU
path); deferred until DDPG GPU env work.

Owns:
  _mb_a       [BATCH, ACT]
  _mb_sa      [BATCH, OBS+ACT]
  _mb_q       [BATCH, 1]
  _mb_grad_q  [BATCH, 1]
  _mb_grad_sa [BATCH, OBS+ACT]
  _mb_grad_s_unused [BATCH, OBS]

Shared by TD3 (uses critic1 only, identical math).
"""

from std.memory import alloc
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from ..constants import DT
from ..core import Module, Optimizer
from ..core.target_storage import TargetStorage, assert_tag_for
from ..training.off_policy_critic import concat_sa
from .loss_block import LossBlock


struct DDPGActorLoss[
    ACTOR: Module,
    CRITIC: Module,
    BATCH: Int,
](LossBlock):
    comptime OBS_DIM = Self.ACTOR.IN_DIMS[0]
    comptime ACT_DIM = Self.ACTOR.OUT_DIM
    comptime SA_DIM = Self.OBS_DIM + Self.ACT_DIM

    var _mb_a: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _mb_sa: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _mb_q: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _mb_grad_q: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _mb_grad_sa: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _mb_grad_s_unused: UnsafePointer[Scalar[DT], MutAnyOrigin]

    var ts: TargetStorage

    def __init__(out self):
        comptime assert (
            Self.CRITIC.IN_DIMS[0] == Self.SA_DIM
        ), "DDPGActorLoss: CRITIC.IN_DIM must equal OBS+ACT"
        comptime assert (
            Self.CRITIC.OUT_DIM == 1
        ), "DDPGActorLoss: CRITIC.OUT_DIM must equal 1"
        var null_p = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=0
        )
        self._mb_a = null_p
        self._mb_sa = null_p
        self._mb_q = null_p
        self._mb_grad_q = null_p
        self._mb_grad_sa = null_p
        self._mb_grad_s_unused = null_p
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert (
            target == "cpu"
        ), "DDPGActorLoss CPU only — GPU path deferred"
        var b = Self()
        b._mb_a = alloc[Scalar[DT]](Self.BATCH * Self.ACT_DIM)
        b._mb_sa = alloc[Scalar[DT]](Self.BATCH * Self.SA_DIM)
        b._mb_q = alloc[Scalar[DT]](Self.BATCH)
        b._mb_grad_q = alloc[Scalar[DT]](Self.BATCH)
        b._mb_grad_sa = alloc[Scalar[DT]](Self.BATCH * Self.SA_DIM)
        b._mb_grad_s_unused = alloc[Scalar[DT]](Self.BATCH * Self.OBS_DIM)
        b.ts = TargetStorage.make_cpu()
        return b^

    def __del__(deinit self):
        if Int(self._mb_a) != 0:
            self._mb_a.free()
        if Int(self._mb_sa) != 0:
            self._mb_sa.free()
        if Int(self._mb_q) != 0:
            self._mb_q.free()
        if Int(self._mb_grad_q) != 0:
            self._mb_grad_q.free()
        if Int(self._mb_grad_sa) != 0:
            self._mb_grad_sa.free()
        if Int(self._mb_grad_s_unused) != 0:
            self._mb_grad_s_unused.free()

    def forward_backward[
        target: StaticString,
        OPT: Optimizer,
    ](
        mut self,
        mut actor: Self.ACTOR,
        mut actor_opt: OPT,
        mut critic: Self.CRITIC,
        mb_s_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises -> Scalar[DT]:
        """Single DPG update step. Returns the scalar actor loss (= -mean_b q).

        Caller must hold `mut critic` for the `backward[mode="input_only"]`
        call; critic params are NOT updated by this method. Use the
        critic's own `CriticUpdateBlock.step` for that, prior to this call."""
        comptime assert target == "cpu", "DDPGActorLoss: CPU only"
        assert_tag_for["DDPGActorLoss", target](self.ts.target_tag)

        # Zero actor grads. Critic grads remain whatever the caller set them
        # to (we won't touch critic param grads — input_only backward).
        actor_opt.zero_grad[target, M=Self.ACTOR](actor)

        # Forward: a = actor(s); sa = concat(s, a); q = critic(sa).
        var mb_s_t = TileTensor(mb_s_ptr, row_major[Self.BATCH, Self.OBS_DIM]())
        var mb_a_t = TileTensor(
            self._mb_a, row_major[Self.BATCH, Self.ACT_DIM]()
        )
        actor.forward[target, Self.BATCH](mb_s_t, output=mb_a_t)
        concat_sa[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH](
            mb_s_ptr,
            self._mb_a,
            self._mb_sa,
        )
        var mb_sa_t = TileTensor(
            self._mb_sa, row_major[Self.BATCH, Self.SA_DIM]()
        )
        var mb_q_t = TileTensor(self._mb_q, row_major[Self.BATCH, 1]())
        critic.forward[target, Self.BATCH](mb_sa_t, output=mb_q_t)

        # Loss = -mean_b q.
        var q_sum: Scalar[DT] = 0.0
        for b in range(Self.BATCH):
            q_sum += self._mb_q[b]
        var loss = -q_sum / Scalar[DT](Self.BATCH)

        # ∂loss/∂q[b] = -1/B (broadcast).
        var inv_B = Scalar[DT](1.0) / Scalar[DT](Self.BATCH)
        for b in range(Self.BATCH):
            self._mb_grad_q[b] = -inv_B

        # critic.vjp[mode="input_only"]: write ∂q/∂sa, skip critic params.
        var mb_grad_q_t = TileTensor(
            self._mb_grad_q, row_major[Self.BATCH, 1]()
        )
        var mb_grad_sa_t = TileTensor(
            self._mb_grad_sa, row_major[Self.BATCH, Self.SA_DIM]()
        )
        critic.vjp[target, Self.BATCH, mode="input_only"](
            mb_grad_q_t,
            mb_grad_sa_t,
        )

        # actor.backward: route ∂q/∂a (= grad_sa[:, OBS:]) into actor's grad-out.
        # We pack it into a contiguous [BATCH, ACT] tile (just the tail of grad_sa).
        # Cheaper to read directly from grad_sa via stride? mb_grad_sa is
        # row-major; rows are [grad_obs | grad_act]. Copy into mb_grad_q (reuse)?
        # No — mb_grad_q is [BATCH, 1]. Allocate mb_grad_a is NOT in scratch.
        # Solution: just write through mb_grad_sa with row-stride awareness —
        # actor.backward expects [BATCH, ACT_DIM] tile, so build one over
        # mb_grad_sa starting at column OBS_DIM. Use TileTensor with offset.
        # Simpler: copy into mb_a (clobber forward cache — but actor.backward
        # already consumed it).
        for b in range(Self.BATCH):
            for j in range(Self.ACT_DIM):
                self._mb_a[b * Self.ACT_DIM + j] = self._mb_grad_sa[
                    b * Self.SA_DIM + Self.OBS_DIM + j
                ]
        var mb_grad_a_t = TileTensor(
            self._mb_a, row_major[Self.BATCH, Self.ACT_DIM]()
        )
        var mb_grad_s_unused_t = TileTensor(
            self._mb_grad_s_unused,
            row_major[Self.BATCH, Self.OBS_DIM](),
        )
        actor.vjp[target, Self.BATCH](mb_grad_a_t, mb_grad_s_unused_t)

        # Step actor only.
        actor_opt.step[target, M=Self.ACTOR](actor)
        return loss
