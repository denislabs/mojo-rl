"""`FBTrainer` — offline Forward-Backward training.

**There is no environment in this loop.** FB learns from a frozen dataset, which
is the fact that takes dm_control gap G10 off milestone 1's critical path
(`docs/BFM_ZERO_SHOT_RL.md` §5.3): collection is a separate phase, evaluation is
a separate phase, and only those two touch an env. So this is not an
`OffPolicyAgent` and does not go through `driver_offpolicy` — it is fed batches
and told to step.

## Networks

Five, not three. Passed as compile-time `Module` params so the caller picks the
architecture:

    BNET : [OBS]            -> D     backward, one instance + target
    FNET : [OBS + ACT + D]  -> D     forward, TWIN + targets (TD3-style min)
    ANET : [OBS + D]        -> ACT   pi_z, deterministic + truncated noise

Inputs are pre-concatenated into one buffer per net rather than routed through
`Concat`: the trainer already owns the batch assembly, and a flat
`[s | a | z]` row keeps the FNET a plain `Sequential` the caller can write in
one line. The slice offsets are `_A_OFF` / `_Z_OFF` below, and the actor
gradient reads the action slice back out of FNET's input gradient.

## One step

    1  B(s), B(s+), B(s')                          online, three forwards
    2  a' = pi_z_target(s', z) + clipped noise      target-policy smoothing
    3  Mtarget = gamma · min( Fbar1(s',a',z)·Bbar(s+)^T ,
                              Fbar2(s',a',z)·Bbar(s+)^T )     elementwise
    4  L_FB for each twin, L_ortho on B
    5  backprop; Adam; Polyak

⚠ **`B` is forwarded three times per step and its parameter gradients
ACCUMULATE across the three vjps.** That is the framework's contract (`Linear`
does `grad_w += ...`), and it is what makes the three roles of `B` — the ortho
term's `B(s)`, the measure term's `B(s+)`, the anchor's `B(s')` — sum into one
update. It also means `zero_grad` must run exactly once per step, at the top. A
second zeroing anywhere in the middle silently discards whichever contributions
came before it, and the loss would still descend.

⚠ **The two batches must be INDEPENDENT draws.** `s+` is not `s'`. See
`loss.mojo` on why; `train_step` takes them as separate arguments and cannot
enforce it, so the sampler is where that invariant lives.

## Scope

CPU. Milestone 1 validates on `point_mass` (nq = 2), where the successor
measure is traceable by hand and a collapsed `B` is visible — on walker a
collapsed and a correct `B` produce the same loss curve. `PairwiseDot` already
carries its GPU path, so the scale-up is the trainer's plumbing, not the
primitive's.
"""

from std.gpu.host import DeviceContext
from std.math import sqrt
from std.random import random_float64

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.core.call import call_forward, call_vjp
from mojo_rl.nn.core.initializer import Initializer, Xavier
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.random.box_muller import box_muller_normal

from ..core.online_target_pair import OnlineTargetPair
from .loss import fb_measure_loss, fb_ortho_loss, pairwise_matrix


struct FBLosses(Movable & ImplicitlyDeletable):
    """Per-step diagnostics. `measure` and `ortho` are the two published terms;
    `f_norm` and `b_norm` are the collapse detectors — a `B` whose rows shrink
    towards zero is the failure `L_ortho` exists to prevent, and watching the
    loss alone will not show it."""

    var measure: Float64
    var ortho: Float64
    var actor: Float64
    var f_norm: Float64
    var b_norm: Float64

    def __init__(
        out self, measure: Float64, ortho: Float64, actor: Float64,
        f_norm: Float64, b_norm: Float64,
    ):
        self.measure = measure
        self.ortho = ortho
        self.actor = actor
        self.f_norm = f_norm
        self.b_norm = b_norm

    def __init__(out self, *, deinit move: Self):
        self.measure = move.measure
        self.ortho = move.ortho
        self.actor = move.actor
        self.f_norm = move.f_norm
        self.b_norm = move.b_norm


struct FBTrainer[
    FNET: Module,
    BNET: Module,
    ANET: Module,
    OBS: Int,
    ACT: Int,
    D: Int,
    BATCH: Int,
](Movable & ImplicitlyDeletable):
    comptime F_IN: Int = Self.OBS + Self.ACT + Self.D
    comptime A_IN: Int = Self.OBS + Self.D
    comptime _A_OFF: Int = Self.OBS
    comptime _Z_OFF: Int = Self.OBS + Self.ACT

    var f1: OnlineTargetPair[Self.FNET]
    var f2: OnlineTargetPair[Self.FNET]
    var bnet: OnlineTargetPair[Self.BNET]
    var actor: OnlineTargetPair[Self.ANET]

    var opt_f1: Adam
    var opt_f2: Adam
    var opt_b: Adam
    var opt_actor: Adam

    var gamma: Float64
    var tau: Float64
    var ortho_weight: Float64
    var policy_noise: Float64
    var noise_clip: Float64
    var steps: Int

    def __init__(out self):
        self.f1 = OnlineTargetPair[Self.FNET]()
        self.f2 = OnlineTargetPair[Self.FNET]()
        self.bnet = OnlineTargetPair[Self.BNET]()
        self.actor = OnlineTargetPair[Self.ANET]()
        self.opt_f1 = Adam(lr=Scalar[DT](3e-4))
        self.opt_f2 = Adam(lr=Scalar[DT](3e-4))
        self.opt_b = Adam(lr=Scalar[DT](3e-4))
        self.opt_actor = Adam(lr=Scalar[DT](3e-4))
        self.gamma = 0.98
        self.tau = 0.01
        self.ortho_weight = 1.0
        self.policy_noise = 0.2
        self.noise_clip = 0.3
        self.steps = 0

    def __init__(out self, *, deinit move: Self):
        self.f1 = move.f1^
        self.f2 = move.f2^
        self.bnet = move.bnet^
        self.actor = move.actor^
        self.opt_f1 = move.opt_f1^
        self.opt_f2 = move.opt_f2^
        self.opt_b = move.opt_b^
        self.opt_actor = move.opt_actor^
        self.gamma = move.gamma
        self.tau = move.tau
        self.ortho_weight = move.ortho_weight
        self.policy_noise = move.policy_noise
        self.noise_clip = move.noise_clip
        self.steps = move.steps

    @staticmethod
    def make[
        INIT: Initializer = Xavier
    ](
        lr: Float64 = 3e-4,
        gamma: Float64 = 0.98,
        tau: Float64 = 0.01,
        ortho_weight: Float64 = 1.0,
    ) raises -> Self:
        """Defaults are the published FB / Meta Motivo settings: EMA 0.99
        (`tau = 0.01`), `gamma = 0.98`, Adam 3e-4."""
        var t = Self()
        t.f1 = OnlineTargetPair[Self.FNET].make["cpu", INIT](None)
        t.f2 = OnlineTargetPair[Self.FNET].make["cpu", INIT](None)
        t.bnet = OnlineTargetPair[Self.BNET].make["cpu", INIT](None)
        t.actor = OnlineTargetPair[Self.ANET].make["cpu", INIT](None)
        t.opt_f1 = Adam(lr=Scalar[DT](lr))
        t.opt_f2 = Adam(lr=Scalar[DT](lr))
        t.opt_b = Adam(lr=Scalar[DT](lr))
        t.opt_actor = Adam(lr=Scalar[DT](lr))
        t.gamma = gamma
        t.tau = tau
        t.ortho_weight = ortho_weight
        return t^

    # ── input assembly ───────────────────────────────────────────────────
    # These are FREE functions (below the struct), not methods. A method taking
    # `self` immutably cannot be called while one of `self`'s net fields is
    # borrowed mutably for a forward, and every call site here does exactly
    # that.

    def _smoothed_target_action(
        mut self, ref s_next: Tensor, ref z: Tensor, mut out: Tensor
    ) raises:
        """`pi_z_target(s', z)` plus truncated Gaussian noise, clamped to
        [-1, 1].

        TD3's target-policy smoothing, with FB's published sigma = 0.2 and
        clip 0.3. Without it the twin min is computed at a single point and
        the measure loss overfits sharp ridges of `F`.
        """
        var ain = Tensor()
        _pack_a_in[Self.OBS, Self.D, Self.BATCH](s_next, z, ain)
        var pack = TensorPack[1]()
        pack[0].ensure(Self.BATCH * Self.A_IN)
        for i in range(Self.BATCH * Self.A_IN):
            pack[0].data[i] = ain.data[i]
        out.ensure(Self.BATCH * Self.ACT)
        call_forward["cpu", Self.BATCH](
            self.actor.target_net, TensorRefs[1](pack[0]), out, None
        )
        var noise = Tensor.alloc(Self.BATCH * Self.ACT)
        box_muller_normal(noise.data.unsafe_ptr(), Self.BATCH * Self.ACT)
        for i in range(Self.BATCH * Self.ACT):
            var n = Float64(noise.data[i]) * self.policy_noise
            if n > self.noise_clip:
                n = self.noise_clip
            elif n < -self.noise_clip:
                n = -self.noise_clip
            var v = Float64(out.data[i]) + n
            if v > 1.0:
                v = 1.0
            elif v < -1.0:
                v = -1.0
            out.data[i] = Scalar[DT](v)

    # ── the step ─────────────────────────────────────────────────────────

    def train_step(
        mut self,
        ref s: Tensor,
        ref a: Tensor,
        ref s_next: Tensor,
        ref s_plus: Tensor,
        ref z: Tensor,
    ) raises -> FBLosses:
        """One gradient step. All inputs are `[BATCH, ·]` host buffers.

        `s_plus` MUST come from a draw independent of `(s, a, s_next)`.
        """
        self.steps += 1

        # ── zero once, at the top. See the module docstring: B's parameter
        # gradients accumulate over its three forwards, and a second zeroing
        # in the middle would drop whichever came first, silently.
        self.f1.online.zero_grad["cpu"](None)
        self.f2.online.zero_grad["cpu"](None)
        self.bnet.online.zero_grad["cpu"](None)
        self.actor.online.zero_grad["cpu"](None)

        # ── 1. B forwards (online) ───────────────────────────────────────
        var in_b_s = Tensor()
        var b_s = Tensor()
        _forward_net[Self.BNET, Self.OBS, Self.D, Self.BATCH](
            self.bnet.online, s, in_b_s, b_s
        )
        var in_b_sp = Tensor()
        var b_sp = Tensor()
        _forward_net[Self.BNET, Self.OBS, Self.D, Self.BATCH](
            self.bnet.online, s_plus, in_b_sp, b_sp
        )
        var in_b_sn = Tensor()
        var b_sn = Tensor()
        _forward_net[Self.BNET, Self.OBS, Self.D, Self.BATCH](
            self.bnet.online, s_next, in_b_sn, b_sn
        )

        # ── 2-3. the bootstrapped target ─────────────────────────────────
        var a_next = Tensor()
        self._smoothed_target_action(s_next, z, a_next)

        var fin_t = Tensor()
        _pack_f_in[Self.OBS, Self.ACT, Self.D, Self.BATCH](s_next, a_next, z, fin_t)
        var scratch = Tensor()
        var ft1 = Tensor()
        _forward_net[Self.FNET, Self.F_IN, Self.D, Self.BATCH](
            self.f1.target_net, fin_t, scratch, ft1
        )
        var ft2 = Tensor()
        _forward_net[Self.FNET, Self.F_IN, Self.D, Self.BATCH](
            self.f2.target_net, fin_t, scratch, ft2
        )
        var bt_in = Tensor()
        var bt_sp = Tensor()
        _forward_net[Self.BNET, Self.OBS, Self.D, Self.BATCH](
            self.bnet.target_net, s_plus, bt_in, bt_sp
        )

        var mt1 = Tensor.alloc(Self.BATCH * Self.BATCH)
        var mt2 = Tensor.alloc(Self.BATCH * Self.BATCH)
        pairwise_matrix[Self.D, Self.BATCH](ft1, bt_sp, mt1)
        pairwise_matrix[Self.D, Self.BATCH](ft2, bt_sp, mt2)
        var m_target = Tensor.alloc(Self.BATCH * Self.BATCH)
        for i in range(Self.BATCH * Self.BATCH):
            # Elementwise min of the twins, then gamma. Taking the min of the
            # two MATRICES entrywise (not of two scalars per row) is what the
            # successor-measure form asks for: every (i, j) pair is its own
            # value estimate.
            var v1 = Float64(mt1.data[i])
            var v2 = Float64(mt2.data[i])
            var mn = v1 if v1 < v2 else v2
            m_target.data[i] = Scalar[DT](self.gamma * mn)

        # ── 4. online F forwards + losses ────────────────────────────────
        var fin = Tensor()
        _pack_f_in[Self.OBS, Self.ACT, Self.D, Self.BATCH](s, a, z, fin)
        var in_f1 = Tensor()
        var f1o = Tensor()
        _forward_net[Self.FNET, Self.F_IN, Self.D, Self.BATCH](
            self.f1.online, fin, in_f1, f1o
        )
        var in_f2 = Tensor()
        var f2o = Tensor()
        _forward_net[Self.FNET, Self.F_IN, Self.D, Self.BATCH](
            self.f2.online, fin, in_f2, f2o
        )

        var g_f1 = Tensor()
        var g_bsp_1 = Tensor()
        var g_bsn_1 = Tensor()
        var l1 = fb_measure_loss[Self.D, Self.BATCH](
            f1o, b_sp, b_sn, m_target, g_f1, g_bsp_1, g_bsn_1, True
        )
        var g_f2 = Tensor()
        var g_bsp_2 = Tensor()
        var g_bsn_2 = Tensor()
        var l2 = fb_measure_loss[Self.D, Self.BATCH](
            f2o, b_sp, b_sn, m_target, g_f2, g_bsp_2, g_bsn_2, True
        )
        var g_bs_o = Tensor()
        var g_bsp_o = Tensor()
        var l_ortho = fb_ortho_loss[Self.D, Self.BATCH](
            b_s, b_sp, g_bs_o, g_bsp_o
        )

        # ── 5. backprop ──────────────────────────────────────────────────
        var sink = TensorPack[1]()
        var fin_pack = TensorPack[1]()

        fin_pack[0].ensure(Self.BATCH * Self.F_IN)
        for i in range(Self.BATCH * Self.F_IN):
            fin_pack[0].data[i] = in_f1.data[i]
        call_vjp["cpu", Self.BATCH](
            self.f1.online, TensorRefs[1](fin_pack[0]), g_f1,
            TensorRefs[1](sink[0]), None,
        )
        for i in range(Self.BATCH * Self.F_IN):
            fin_pack[0].data[i] = in_f2.data[i]
        call_vjp["cpu", Self.BATCH](
            self.f2.online, TensorRefs[1](fin_pack[0]), g_f2,
            TensorRefs[1](sink[0]), None,
        )

        # B: three vjps, accumulating into one set of parameter grads.
        var g_bsp = Tensor.alloc(Self.BATCH * Self.D)
        for i in range(Self.BATCH * Self.D):
            g_bsp.data[i] = Scalar[DT](
                Float64(g_bsp_1.data[i]) + Float64(g_bsp_2.data[i])
                + self.ortho_weight * Float64(g_bsp_o.data[i])
            )
        var g_bsn = Tensor.alloc(Self.BATCH * Self.D)
        for i in range(Self.BATCH * Self.D):
            g_bsn.data[i] = Scalar[DT](
                Float64(g_bsn_1.data[i]) + Float64(g_bsn_2.data[i])
            )
        var g_bs = Tensor.alloc(Self.BATCH * Self.D)
        for i in range(Self.BATCH * Self.D):
            g_bs.data[i] = Scalar[DT](
                self.ortho_weight * Float64(g_bs_o.data[i])
            )

        var b_pack = TensorPack[1]()
        b_pack[0].ensure(Self.BATCH * Self.OBS)
        for i in range(Self.BATCH * Self.OBS):
            b_pack[0].data[i] = in_b_sp.data[i]
        call_vjp["cpu", Self.BATCH](
            self.bnet.online, TensorRefs[1](b_pack[0]), g_bsp,
            TensorRefs[1](sink[0]), None,
        )
        for i in range(Self.BATCH * Self.OBS):
            b_pack[0].data[i] = in_b_sn.data[i]
        call_vjp["cpu", Self.BATCH](
            self.bnet.online, TensorRefs[1](b_pack[0]), g_bsn,
            TensorRefs[1](sink[0]), None,
        )
        for i in range(Self.BATCH * Self.OBS):
            b_pack[0].data[i] = in_b_s.data[i]
        call_vjp["cpu", Self.BATCH](
            self.bnet.online, TensorRefs[1](b_pack[0]), g_bs,
            TensorRefs[1](sink[0]), None,
        )

        self.opt_f1.step["cpu"](self.f1.online, None)
        self.opt_f2.step["cpu"](self.f2.online, None)
        self.opt_b.step["cpu"](self.bnet.online, None)

        # ── actor: maximise F1(s, pi(s,z), z) · z ────────────────────────
        var l_actor = self._actor_step(s, z)

        # ── Polyak ───────────────────────────────────────────────────────
        self.f1.polyak_step["cpu"](Scalar[DT](self.tau), None)
        self.f2.polyak_step["cpu"](Scalar[DT](self.tau), None)
        self.bnet.polyak_step["cpu"](Scalar[DT](self.tau), None)
        self.actor.polyak_step["cpu"](Scalar[DT](self.tau), None)

        var f_sq = Float64(0)
        var b_sq = Float64(0)
        for i in range(Self.BATCH * Self.D):
            f_sq += Float64(f1o.data[i]) * Float64(f1o.data[i])
            b_sq += Float64(b_s.data[i]) * Float64(b_s.data[i])
        var inv = 1.0 / Float64(Self.BATCH)
        return FBLosses(
            0.5 * (l1 + l2), l_ortho, l_actor,
            sqrt(f_sq * inv), sqrt(b_sq * inv),
        )

    def _actor_step(mut self, ref s: Tensor, ref z: Tensor) raises -> Float64:
        """DPG through `F1`: maximise `F(s, pi_z(s,z), z) · z`.

        `z` is both the actor's conditioning input and the direction the value
        is projected onto — that is the whole point of the latent, and it is
        why the actor loss needs no reward.
        """
        var ain = Tensor()
        _pack_a_in[Self.OBS, Self.D, Self.BATCH](s, z, ain)
        var a_pack = TensorPack[1]()
        a_pack[0].ensure(Self.BATCH * Self.A_IN)
        for i in range(Self.BATCH * Self.A_IN):
            a_pack[0].data[i] = ain.data[i]
        var pi = Tensor()
        pi.ensure(Self.BATCH * Self.ACT)
        call_forward["cpu", Self.BATCH](
            self.actor.online, TensorRefs[1](a_pack[0]), pi, None
        )

        var fin = Tensor()
        _pack_f_in[Self.OBS, Self.ACT, Self.D, Self.BATCH](s, pi, z, fin)
        var f_pack = TensorPack[1]()
        f_pack[0].ensure(Self.BATCH * Self.F_IN)
        for i in range(Self.BATCH * Self.F_IN):
            f_pack[0].data[i] = fin.data[i]
        var fo = Tensor()
        fo.ensure(Self.BATCH * Self.D)
        call_forward["cpu", Self.BATCH](
            self.f1.online, TensorRefs[1](f_pack[0]), fo, None
        )

        var loss = Float64(0)
        for i in range(Self.BATCH):
            for k in range(Self.D):
                loss += Float64(fo.data[i * Self.D + k]) * Float64(
                    z.data[i * Self.D + k]
                )
        loss = -loss / Float64(Self.BATCH)

        # dL/dF = -z / BATCH
        var g_f = Tensor.alloc(Self.BATCH * Self.D)
        for i in range(Self.BATCH * Self.D):
            g_f.data[i] = Scalar[DT](
                -Float64(z.data[i]) / Float64(Self.BATCH)
            )

        # ⚠ Through F1 WITHOUT touching F1's parameter grads: the optimizer
        # already stepped them above, and the actor update must not fold a
        # second, differently-scaled critic gradient into the next step. The
        # vjp accumulates into params, so F1's grads are zeroed after the pass.
        var g_fin = TensorPack[1]()
        call_vjp["cpu", Self.BATCH](
            self.f1.online, TensorRefs[1](f_pack[0]), g_f,
            TensorRefs[1](g_fin[0]), None,
        )
        self.f1.online.zero_grad["cpu"](None)

        # Slice out d/da and push it through the actor.
        var g_pi = Tensor.alloc(Self.BATCH * Self.ACT)
        for i in range(Self.BATCH):
            for k in range(Self.ACT):
                g_pi.data[i * Self.ACT + k] = g_fin[0].data[
                    i * Self.F_IN + Self._A_OFF + k
                ]
        var sink = TensorPack[1]()
        call_vjp["cpu", Self.BATCH](
            self.actor.online, TensorRefs[1](a_pack[0]), g_pi,
            TensorRefs[1](sink[0]), None,
        )
        self.opt_actor.step["cpu"](self.actor.online, None)
        return loss

    # ── inference ────────────────────────────────────────────────────────

    def act[
        N: Int
    ](mut self, ref s: Tensor, ref z: Tensor, mut out: Tensor) raises:
        """`pi_z(s, z)` for `N` rows — the zero-shot policy.

        ⚠ `z` must already be on the radius-sqrt(D) sphere. `z_sampler`'s
        producers all guarantee that; a `z` assembled by hand does not, and the
        symptom is a policy that acts plausibly and optimises nothing.
        """
        var ain = Tensor.alloc(N * Self.A_IN)
        for i in range(N):
            var o = i * Self.A_IN
            for k in range(Self.OBS):
                ain.data[o + k] = s.data[i * Self.OBS + k]
            for k in range(Self.D):
                ain.data[o + Self.OBS + k] = z.data[i * Self.D + k]
        var pack = TensorPack[1]()
        pack[0].ensure(N * Self.A_IN)
        for i in range(N * Self.A_IN):
            pack[0].data[i] = ain.data[i]
        out.ensure(N * Self.ACT)
        call_forward["cpu", N](
            self.actor.online, TensorRefs[1](pack[0]), out, None
        )

    def backward_embed[
        N: Int
    ](mut self, ref s: Tensor, mut out: Tensor) raises:
        """`B(s)` for `N` rows — the input to `z_from_reward`."""
        var pack = TensorPack[1]()
        pack[0].ensure(N * Self.OBS)
        for i in range(N * Self.OBS):
            pack[0].data[i] = s.data[i]
        out.ensure(N * Self.D)
        call_forward["cpu", N](
            self.bnet.online, TensorRefs[1](pack[0]), out, None
        )


# ──────────────────────────────────────────────────────────────────────
# Free helpers — see the note inside the struct on why these are not methods.
# ──────────────────────────────────────────────────────────────────────


def _pack_f_in[
    OBS: Int, ACT: Int, D: Int, BATCH: Int
](ref s: Tensor, ref a: Tensor, ref z: Tensor, mut dst: Tensor) raises:
    """`[s | a | z]` rows, width `OBS + ACT + D`."""
    comptime W = OBS + ACT + D
    dst.ensure(BATCH * W)
    for i in range(BATCH):
        var o = i * W
        for k in range(OBS):
            dst.data[o + k] = s.data[i * OBS + k]
        for k in range(ACT):
            dst.data[o + OBS + k] = a.data[i * ACT + k]
        for k in range(D):
            dst.data[o + OBS + ACT + k] = z.data[i * D + k]


def _pack_a_in[
    OBS: Int, D: Int, BATCH: Int
](ref s: Tensor, ref z: Tensor, mut dst: Tensor) raises:
    """`[s | z]` rows, width `OBS + D`."""
    comptime W = OBS + D
    dst.ensure(BATCH * W)
    for i in range(BATCH):
        var o = i * W
        for k in range(OBS):
            dst.data[o + k] = s.data[i * OBS + k]
        for k in range(D):
            dst.data[o + OBS + k] = z.data[i * D + k]


def _forward_net[
    M: Module, IN_W: Int, OUT_W: Int, BATCH: Int
](mut net: M, ref x: Tensor, mut cache_in: Tensor, mut out: Tensor) raises:
    """Forward `net` on `x`, keeping the input in `cache_in` for the vjp.

    The vjp needs the same values it was forwarded on. Copying into a
    trainer-owned tensor rather than aliasing the caller's keeps a later
    in-place edit of the batch from silently changing what the backward pass
    differentiates.
    """
    cache_in.ensure(BATCH * IN_W)
    for i in range(BATCH * IN_W):
        cache_in.data[i] = x.data[i]
    out.ensure(BATCH * OUT_W)
    var pack = TensorPack[1]()
    pack[0].ensure(BATCH * IN_W)
    for i in range(BATCH * IN_W):
        pack[0].data[i] = cache_in.data[i]
    call_forward["cpu", BATCH](net, TensorRefs[1](pack[0]), out, None)
