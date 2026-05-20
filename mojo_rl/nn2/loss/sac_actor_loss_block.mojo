"""SACActorLoss — composed-form SAC actor loss as a self-contained block.

Phase 9A. Hides the chain Modules + intermediate scratch + gradient seed
+ optimizer step behind a single `forward_backward(...)` call. Replaces
the ~200 LOC of in-trainer plumbing the Phase 8.4 composed validator
needed.

Loss (mean over batch):
    L = E_b [ α · log_prob_b - min(Q1(s_b, a_b), Q2(s_b, a_b)) ]
        where (a, log_prob) = squashed_gaussian( actor(s), z )

Internal chain (all rsample-then-min Modules owned by the block):
    actor.forward             [BATCH, OBS]    → [BATCH, 2·ACT]   (= [mu | log_std])
    rsample.forward           [BATCH, 2·ACT]  → [BATCH, ACT+1]   (= [action | log_prob])
    split action / log_prob
    concat(s, action)         → [BATCH, OBS+ACT]
    critic1.forward, critic2.forward          → q1, q2  ([BATCH, 1] each)
    pack [q1 | q2]            → [BATCH, 2]
    elem_min.forward          → [BATCH, 1]    (= min_q)
    scale.multiplier = α
    scale.forward(log_prob)   → α·log_prob    ([BATCH, 1])
    pack [α·log_prob | min_q] → [BATCH, 2]
    sub.forward               → L_per_b       ([BATCH, 1])
    mean over BATCH           → scalar L

Backward seeds grad_L_per_b = 1/BATCH and walks the chain in reverse.
Critics use `backward_input` (frozen params, Phase 8.2 contract). Actor
runs full `backward` to accumulate param grads; the block owns the
optimizer zero_grad + step calls (parameterized on `OPT: Optimizer`).

Public field: `rsample`. Kept public so the trainer can reuse it for
env-interaction sampling (single-step `actor.forward` + `rsample.forward`
to get an action). Set `block.rsample.action_scale` after `make` if you
need a non-unit env action range.

CPU only (Phase 9A). GPU `make` raises until the first GPU SAC env
pulls the chain kernels through.
"""

from layout import TileTensor, TensorLayout, row_major

from ..constants import DT
from ..core import (
    Module,
    Optimizer,
    Initializer,
    TARGET_UNINIT,
    TARGET_CPU,
    target_tag_for,
)
from ..initializer import Zero
from ..primitives.rsample import RSample
from ..primitives.scale import Scale
from ..primitives.elem_min import ElemMin
from ..primitives.sub import Sub


@fieldwise_init
struct SACActorLossOut(Movable & ImplicitlyDestructible):
    """Result of one `forward_backward` call.

    `loss` is the mean-batch scalar value (for logging).
    `log_prob_mean` is the mean of log_prob over the batch — caller passes
    `-(log_prob_mean + target_entropy)` to its α optimizer.
    """
    var loss: Scalar[DT]
    var log_prob_mean: Scalar[DT]


struct SACActorLoss[
    ACTOR: Module,
    CRITIC: Module,
    BATCH: Int,
](Movable & ImplicitlyDestructible):
    comptime OBS_DIM = Self.ACTOR.IN_DIM
    comptime ACT_DIM = Self.ACTOR.OUT_DIM // 2
    comptime SA_DIM = Self.OBS_DIM + Self.ACT_DIM

    # Chain modules. `rsample` is public for env-interaction sampling reuse.
    var rsample: RSample[Self.ACT_DIM]
    var _scale: Scale[1]
    var _elem_min: ElemMin[1]
    var _sub: Sub[1]

    # ─── Forward scratch ─────────────────────────────────────────────
    var _mb_ao: List[Scalar[DT]]           # [BATCH, 2·ACT]  actor output
    var _mb_alp: List[Scalar[DT]]          # [BATCH, ACT+1]  [action | log_prob]
    var _mb_act: List[Scalar[DT]]          # [BATCH, ACT]    extracted action
    var _mb_lp: List[Scalar[DT]]           # [BATCH, 1]      extracted log_prob
    var _mb_sa: List[Scalar[DT]]           # [BATCH, SA]
    var _mb_q1: List[Scalar[DT]]           # [BATCH, 1]
    var _mb_q2: List[Scalar[DT]]           # [BATCH, 1]
    var _mb_q12: List[Scalar[DT]]          # [BATCH, 2]      packed [q1 | q2]
    var _mb_min_q: List[Scalar[DT]]        # [BATCH, 1]
    var _mb_alpha_lp: List[Scalar[DT]]     # [BATCH, 1]      α·log_prob
    var _mb_packed_loss: List[Scalar[DT]]  # [BATCH, 2]      [α·lp | min_q]
    var _mb_loss_per_b: List[Scalar[DT]]   # [BATCH, 1]

    # ─── Backward scratch ────────────────────────────────────────────
    var _mb_grad_loss_per_b: List[Scalar[DT]]   # [BATCH, 1]   seed = 1/BATCH
    var _mb_grad_packed_loss: List[Scalar[DT]]  # [BATCH, 2]
    var _mb_grad_alpha_lp: List[Scalar[DT]]     # [BATCH, 1]
    var _mb_grad_min_q: List[Scalar[DT]]        # [BATCH, 1]
    var _mb_grad_lp: List[Scalar[DT]]           # [BATCH, 1]
    var _mb_grad_q12: List[Scalar[DT]]          # [BATCH, 2]
    var _mb_grad_q1: List[Scalar[DT]]           # [BATCH, 1]
    var _mb_grad_q2: List[Scalar[DT]]           # [BATCH, 1]
    var _mb_grad_sa1: List[Scalar[DT]]          # [BATCH, SA]
    var _mb_grad_sa2: List[Scalar[DT]]          # [BATCH, SA]
    var _mb_grad_action_sum: List[Scalar[DT]]   # [BATCH, ACT]
    var _mb_grad_alp: List[Scalar[DT]]          # [BATCH, ACT+1]
    var _mb_grad_ao: List[Scalar[DT]]           # [BATCH, 2·ACT]
    var _mb_grad_obs_unused: List[Scalar[DT]]   # [BATCH, OBS]  thrown away

    var _target_tag: Int8

    def __init__(out self):
        self.rsample = RSample[Self.ACT_DIM]()
        self._scale = Scale[1]()
        self._elem_min = ElemMin[1]()
        self._sub = Sub[1]()
        self._mb_ao = List[Scalar[DT]]()
        self._mb_alp = List[Scalar[DT]]()
        self._mb_act = List[Scalar[DT]]()
        self._mb_lp = List[Scalar[DT]]()
        self._mb_sa = List[Scalar[DT]]()
        self._mb_q1 = List[Scalar[DT]]()
        self._mb_q2 = List[Scalar[DT]]()
        self._mb_q12 = List[Scalar[DT]]()
        self._mb_min_q = List[Scalar[DT]]()
        self._mb_alpha_lp = List[Scalar[DT]]()
        self._mb_packed_loss = List[Scalar[DT]]()
        self._mb_loss_per_b = List[Scalar[DT]]()
        self._mb_grad_loss_per_b = List[Scalar[DT]]()
        self._mb_grad_packed_loss = List[Scalar[DT]]()
        self._mb_grad_alpha_lp = List[Scalar[DT]]()
        self._mb_grad_min_q = List[Scalar[DT]]()
        self._mb_grad_lp = List[Scalar[DT]]()
        self._mb_grad_q12 = List[Scalar[DT]]()
        self._mb_grad_q1 = List[Scalar[DT]]()
        self._mb_grad_q2 = List[Scalar[DT]]()
        self._mb_grad_sa1 = List[Scalar[DT]]()
        self._mb_grad_sa2 = List[Scalar[DT]]()
        self._mb_grad_action_sum = List[Scalar[DT]]()
        self._mb_grad_alp = List[Scalar[DT]]()
        self._mb_grad_ao = List[Scalar[DT]]()
        self._mb_grad_obs_unused = List[Scalar[DT]]()
        self._target_tag = TARGET_UNINIT

    @staticmethod
    def make[target: StaticString](
        action_scale: Scalar[DT] = Scalar[DT](1.0),
    ) raises -> Self:
        """CPU factory. Allocates all scratch + chain Modules."""
        comptime assert target == "cpu", (
            "SACActorLoss.make[target='gpu'] not yet implemented (Phase 9A CPU only)"
        )
        comptime assert Self.ACTOR.OUT_DIM == 2 * Self.ACT_DIM, (
            "SACActorLoss: ACTOR.OUT_DIM must equal 2·ACT_DIM"
        )
        comptime assert Self.CRITIC.IN_DIM == Self.SA_DIM, (
            "SACActorLoss: CRITIC.IN_DIM must equal OBS_DIM + ACT_DIM"
        )
        comptime assert Self.CRITIC.OUT_DIM == 1, (
            "SACActorLoss: CRITIC.OUT_DIM must equal 1"
        )

        var blk = Self()
        blk.rsample = RSample[Self.ACT_DIM].make[target="cpu", INIT=Zero]()
        blk.rsample.action_scale = action_scale
        blk._scale = Scale[1].make[target="cpu", INIT=Zero]()
        blk._elem_min = ElemMin[1].make[target="cpu", INIT=Zero]()
        blk._sub = Sub[1].make[target="cpu", INIT=Zero]()

        var zero: Scalar[DT] = 0.0
        blk._mb_ao.resize(Self.BATCH * 2 * Self.ACT_DIM, zero)
        blk._mb_alp.resize(Self.BATCH * (Self.ACT_DIM + 1), zero)
        blk._mb_act.resize(Self.BATCH * Self.ACT_DIM, zero)
        blk._mb_lp.resize(Self.BATCH, zero)
        blk._mb_sa.resize(Self.BATCH * Self.SA_DIM, zero)
        blk._mb_q1.resize(Self.BATCH, zero)
        blk._mb_q2.resize(Self.BATCH, zero)
        blk._mb_q12.resize(Self.BATCH * 2, zero)
        blk._mb_min_q.resize(Self.BATCH, zero)
        blk._mb_alpha_lp.resize(Self.BATCH, zero)
        blk._mb_packed_loss.resize(Self.BATCH * 2, zero)
        blk._mb_loss_per_b.resize(Self.BATCH, zero)
        blk._mb_grad_loss_per_b.resize(Self.BATCH, zero)
        blk._mb_grad_packed_loss.resize(Self.BATCH * 2, zero)
        blk._mb_grad_alpha_lp.resize(Self.BATCH, zero)
        blk._mb_grad_min_q.resize(Self.BATCH, zero)
        blk._mb_grad_lp.resize(Self.BATCH, zero)
        blk._mb_grad_q12.resize(Self.BATCH * 2, zero)
        blk._mb_grad_q1.resize(Self.BATCH, zero)
        blk._mb_grad_q2.resize(Self.BATCH, zero)
        blk._mb_grad_sa1.resize(Self.BATCH * Self.SA_DIM, zero)
        blk._mb_grad_sa2.resize(Self.BATCH * Self.SA_DIM, zero)
        blk._mb_grad_action_sum.resize(Self.BATCH * Self.ACT_DIM, zero)
        blk._mb_grad_alp.resize(Self.BATCH * (Self.ACT_DIM + 1), zero)
        blk._mb_grad_ao.resize(Self.BATCH * 2 * Self.ACT_DIM, zero)
        blk._mb_grad_obs_unused.resize(Self.BATCH * Self.OBS_DIM, zero)

        blk._target_tag = TARGET_CPU
        return blk^

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "SACActorLoss: method called with [target='"
                + String(target)
                + "'] but block was make'd for a different target (tag="
                + String(Int(self._target_tag)) + ")"
            )

    # ──────────────────────────────────────────────────────────────────
    # Forward chain — actor → rsample → split → concat(s,a) → critics
    #               → pack q12 → elem_min → scale → pack → sub → mean
    # ──────────────────────────────────────────────────────────────────

    def _forward_chain(
        mut self,
        mut actor: Self.ACTOR,
        mut critic1: Self.CRITIC,
        mut critic2: Self.CRITIC,
        mb_s_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        alpha: Scalar[DT],
    ) raises -> SACActorLossOut:
        comptime BB = Self.BATCH
        comptime ACT = Self.ACT_DIM
        comptime OBS = Self.OBS_DIM
        comptime SA = Self.SA_DIM

        # actor.forward: s → [mu | log_std]
        var mb_s_t = TileTensor(mb_s_ptr, row_major[BB, OBS]())
        var mb_ao_p = self._mb_ao.unsafe_ptr()
        var mb_ao_t = TileTensor(mb_ao_p, row_major[BB, 2 * ACT]())
        actor.forward["cpu", BB](mb_s_t, mb_ao_t)

        # rsample.forward: [mu|log_std] → [action | log_prob]
        var mb_alp_p = self._mb_alp.unsafe_ptr()
        var mb_alp_t = TileTensor(mb_alp_p, row_major[BB, ACT + 1]())
        self.rsample.forward["cpu", BB](mb_ao_t, mb_alp_t)

        # Split action / log_prob and concat(s, action).
        var mb_act_p = self._mb_act.unsafe_ptr()
        var mb_lp_p = self._mb_lp.unsafe_ptr()
        var mb_sa_p = self._mb_sa.unsafe_ptr()
        for b in range(BB):
            for j in range(ACT):
                mb_act_p[b * ACT + j] = mb_alp_p[b * (ACT + 1) + j]
            mb_lp_p[b] = mb_alp_p[b * (ACT + 1) + ACT]
            for d in range(OBS):
                mb_sa_p[b * SA + d] = mb_s_ptr[b * OBS + d]
            for j in range(ACT):
                mb_sa_p[b * SA + OBS + j] = mb_act_p[b * ACT + j]

        # Twin critic forwards on (s, π(s)).
        var mb_sa_t = TileTensor(mb_sa_p, row_major[BB, SA]())
        var mb_q1_p = self._mb_q1.unsafe_ptr()
        var mb_q2_p = self._mb_q2.unsafe_ptr()
        var mb_q1_t = TileTensor(mb_q1_p, row_major[BB, 1]())
        var mb_q2_t = TileTensor(mb_q2_p, row_major[BB, 1]())
        critic1.forward["cpu", BB](mb_sa_t, mb_q1_t)
        critic2.forward["cpu", BB](mb_sa_t, mb_q2_t)

        # Pack [q1 | q2] then elem_min, then scale(log_prob), then pack
        # [α·lp | min_q], then sub.
        var mb_q12_p = self._mb_q12.unsafe_ptr()
        for b in range(BB):
            mb_q12_p[b * 2] = mb_q1_p[b]
            mb_q12_p[b * 2 + 1] = mb_q2_p[b]
        var mb_q12_t = TileTensor(mb_q12_p, row_major[BB, 2]())
        var mb_min_q_p = self._mb_min_q.unsafe_ptr()
        var mb_min_q_t = TileTensor(mb_min_q_p, row_major[BB, 1]())
        self._elem_min.forward["cpu", BB](mb_q12_t, mb_min_q_t)

        self._scale.multiplier = alpha
        var mb_lp_t = TileTensor(mb_lp_p, row_major[BB, 1]())
        var mb_alpha_lp_p = self._mb_alpha_lp.unsafe_ptr()
        var mb_alpha_lp_t = TileTensor(mb_alpha_lp_p, row_major[BB, 1]())
        self._scale.forward["cpu", BB](mb_lp_t, mb_alpha_lp_t)

        var mb_packed_loss_p = self._mb_packed_loss.unsafe_ptr()
        for b in range(BB):
            mb_packed_loss_p[b * 2] = mb_alpha_lp_p[b]
            mb_packed_loss_p[b * 2 + 1] = mb_min_q_p[b]
        var mb_packed_loss_t = TileTensor(mb_packed_loss_p, row_major[BB, 2]())
        var mb_loss_per_b_p = self._mb_loss_per_b.unsafe_ptr()
        var mb_loss_per_b_t = TileTensor(mb_loss_per_b_p, row_major[BB, 1]())
        self._sub.forward["cpu", BB](mb_packed_loss_t, mb_loss_per_b_t)

        # Reduce to scalars: mean loss, mean log_prob.
        var loss_sum: Scalar[DT] = 0.0
        var lp_sum: Scalar[DT] = 0.0
        for b in range(BB):
            loss_sum += mb_loss_per_b_p[b]
            lp_sum += mb_lp_p[b]
        var inv_b: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](BB)
        return SACActorLossOut(loss=loss_sum * inv_b, log_prob_mean=lp_sum * inv_b)

    # ──────────────────────────────────────────────────────────────────
    # Backward chain — seed 1/BATCH → sub → scale → elem_min
    #                → critic1/2.backward_input → sum grad_action
    #                → pack grad_alp → rsample.backward → actor.backward
    # ──────────────────────────────────────────────────────────────────

    def _backward_chain(
        mut self,
        mut actor: Self.ACTOR,
        mut critic1: Self.CRITIC,
        mut critic2: Self.CRITIC,
    ) raises:
        comptime BB = Self.BATCH
        comptime ACT = Self.ACT_DIM
        comptime OBS = Self.OBS_DIM
        comptime SA = Self.SA_DIM

        # Seed dL/dL_per_b = 1/BATCH.
        var inv_b: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](BB)
        var g_loss_p = self._mb_grad_loss_per_b.unsafe_ptr()
        for b in range(BB):
            g_loss_p[b] = inv_b
        var g_loss_t = TileTensor(g_loss_p, row_major[BB, 1]())

        # sub.backward → packed [grad_α·lp | grad_min_q]
        var g_pack_p = self._mb_grad_packed_loss.unsafe_ptr()
        var g_pack_t = TileTensor(g_pack_p, row_major[BB, 2]())
        self._sub.backward["cpu", BB](g_loss_t, g_pack_t)

        # Split and route: scale.backward → grad_lp; elem_min.backward → grad_q12.
        var g_alp_p = self._mb_grad_alpha_lp.unsafe_ptr()
        var g_minq_p = self._mb_grad_min_q.unsafe_ptr()
        for b in range(BB):
            g_alp_p[b] = g_pack_p[b * 2]
            g_minq_p[b] = g_pack_p[b * 2 + 1]
        var g_alp_t = TileTensor(g_alp_p, row_major[BB, 1]())
        var g_lp_p = self._mb_grad_lp.unsafe_ptr()
        var g_lp_t = TileTensor(g_lp_p, row_major[BB, 1]())
        self._scale.backward["cpu", BB](g_alp_t, g_lp_t)

        var g_minq_t = TileTensor(g_minq_p, row_major[BB, 1]())
        var g_q12_p = self._mb_grad_q12.unsafe_ptr()
        var g_q12_t = TileTensor(g_q12_p, row_major[BB, 2]())
        self._elem_min.backward["cpu", BB](g_minq_t, g_q12_t)

        # Split grad_q12 into grad_q1, grad_q2; run critic backward_input
        # (frozen params — Phase 8.2 contract).
        var g_q1_p = self._mb_grad_q1.unsafe_ptr()
        var g_q2_p = self._mb_grad_q2.unsafe_ptr()
        for b in range(BB):
            g_q1_p[b] = g_q12_p[b * 2]
            g_q2_p[b] = g_q12_p[b * 2 + 1]
        var g_q1_t = TileTensor(g_q1_p, row_major[BB, 1]())
        var g_q2_t = TileTensor(g_q2_p, row_major[BB, 1]())
        var g_sa1_p = self._mb_grad_sa1.unsafe_ptr()
        var g_sa2_p = self._mb_grad_sa2.unsafe_ptr()
        var g_sa1_t = TileTensor(g_sa1_p, row_major[BB, SA]())
        var g_sa2_t = TileTensor(g_sa2_p, row_major[BB, SA]())
        critic1.backward_input["cpu", BB](g_q1_t, g_sa1_t)
        critic2.backward_input["cpu", BB](g_q2_t, g_sa2_t)

        # Sum grad_action from both critic paths, pack with grad_lp into
        # [grad_action | grad_lp] for rsample.backward.
        var g_action_p = self._mb_grad_action_sum.unsafe_ptr()
        var g_alp_pkg_p = self._mb_grad_alp.unsafe_ptr()
        for b in range(BB):
            for j in range(ACT):
                var ga = g_sa1_p[b * SA + OBS + j] + g_sa2_p[b * SA + OBS + j]
                g_action_p[b * ACT + j] = ga
                g_alp_pkg_p[b * (ACT + 1) + j] = ga
            g_alp_pkg_p[b * (ACT + 1) + ACT] = g_lp_p[b]

        # rsample.backward → grad_ao = [grad_mu | grad_log_std].
        var g_alp_pkg_t = TileTensor(g_alp_pkg_p, row_major[BB, ACT + 1]())
        var g_ao_p = self._mb_grad_ao.unsafe_ptr()
        var g_ao_t = TileTensor(g_ao_p, row_major[BB, 2 * ACT]())
        self.rsample.backward["cpu", BB](g_alp_pkg_t, g_ao_t)

        # actor.backward — accumulates actor param grads; grad_obs is
        # not consumed by the trainer but Module.backward requires the
        # buffer.
        var g_obs_p = self._mb_grad_obs_unused.unsafe_ptr()
        var g_obs_t = TileTensor(g_obs_p, row_major[BB, OBS]())
        actor.backward["cpu", BB](g_ao_t, g_obs_t)

    # ──────────────────────────────────────────────────────────────────
    # Public entry point.
    # ──────────────────────────────────────────────────────────────────

    def forward_backward[
        target: StaticString,
        OPT: Optimizer,
    ](
        mut self,
        mut actor: Self.ACTOR,
        mut actor_opt: OPT,
        mut critic1: Self.CRITIC,
        mut critic2: Self.CRITIC,
        mb_s_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        alpha: Scalar[DT],
    ) raises -> SACActorLossOut:
        """One full SAC actor update: zero_grad → composed-form chain
        → backward chain → optimizer step. Returns the mean-batch loss
        scalar (logging) and the mean log_prob (caller passes to its
        α optimizer as `-(log_prob_mean + target_entropy)`)."""
        comptime assert target == "cpu", (
            "SACActorLoss.forward_backward: GPU path not yet implemented"
        )
        self._assert_tag[target]()
        actor_opt.zero_grad["cpu", M=Self.ACTOR](actor)
        var fwd = self._forward_chain(actor, critic1, critic2, mb_s_ptr, alpha)
        self._backward_chain(actor, critic1, critic2)
        actor_opt.step["cpu", M=Self.ACTOR](actor)
        return fwd^
