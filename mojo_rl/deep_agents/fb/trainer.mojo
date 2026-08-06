"""`FBTrainer` — offline Forward-Backward training, CPU or GPU.

**There is no environment in this loop.** FB learns from a frozen dataset, which
is the fact that takes dm_control gap G10 off the critical path
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
one line. The slice offsets are `_A_OFF` / `_Z_OFF`, and the actor gradient
reads the action slice back out of FNET's input gradient.

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

⚠⚠ **`BNET` should end in a normalisation.** Meta Motivo carries
`"b": {"norm": true}`, and a bare `Linear -> ReLU -> Linear` backward net
DIVERGED on walker at d=128: `L_ortho` went POSITIVE and grew 8x (21 -> 172
over 24 k steps) while `|B|` climbed and the measure loss fell without bound.
At the orthonormality optimum `L_ortho` is NEGATIVE, so its SIGN is the health
check — not `|B|` alone, and certainly not the measure loss, which descends in
both the healthy and the diverging case. `nn/primitives/layer_norm.mojo`.

⚠ **The two batches must be INDEPENDENT draws.** `s+` is not `s'`. See
`loss.mojo` on why; `train_step` takes them as separate arguments and cannot
enforce it, so the sampler is where that invariant lives.

## CPU and GPU are ONE body

`TARGET` is a struct parameter (default `"cpu"`, so callers written for M1 are
unaffected). The step is written once: `nn` `Module.forward`/`vjp`,
`PairwiseDot`, Adam and Polyak already dispatch on target, and the ~8 remaining
elementwise/pack operations go through the `*_t` helpers in `kernels.mojo`.
There is deliberately no second GPU trainer struct — a duplicated 400-line step
body is a drift hazard, and the interesting failures here are silent ones.

⚠ Buffers are trainer-owned FIELDS, sized once on the first step. At
`BATCH = 1024` the `[BATCH, BATCH]` matrices are 4 MB each and several are live
at once; allocating them per step would dominate.

⚠⚠ **`want_loss` gates a device sync, not just a print.** Reading a loss value
back from the GPU is a full pipeline stall. `train_step(..., want_loss=False)`
skips both reduction kernels and the readback and returns zeros in `FBLosses`;
the GRADIENTS are identical either way, because the loss value never enters the
update. Log every few hundred steps, not every step.
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
from mojo_rl.nn.core.checkpoint import (
    CheckpointWriter, CheckpointReader, _split_lines,
)

from ..core.online_target_pair import OnlineTargetPair
from .loss import (
    FBLossWorkspace,
    fb_measure_loss_into,
    fb_ortho_loss_into,
    fb_measure_loss,
    fb_ortho_loss,
    pairwise_matrix,
)
from .kernels import (
    ensure_t,
    pack3_t,
    pack2_t,
    axpy_t,
    scale_t,
    sum3_scaled_t,
    min_scale_t,
    smooth_action_t,
    slice_cols_t,
    mean_sq_t,
    mean_t,
    gaussian_t,
)


struct FBLosses(Movable & ImplicitlyDeletable):
    """Per-step diagnostics. `measure` and `ortho` are the two published terms;
    `f_norm` and `b_norm` are the collapse detectors — a `B` whose rows shrink
    towards zero is the failure `L_ortho` exists to prevent, and watching the
    loss alone will not show it.

    All zero when `want_loss=False`; that is not an error, it is the caller
    declining to pay for a device sync on this step."""

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
    TARGET: StaticString = "cpu",
](Movable & ImplicitlyDeletable):
    comptime F_IN: Int = Self.OBS + Self.ACT + Self.D
    comptime A_IN: Int = Self.OBS + Self.D
    comptime _A_OFF: Int = Self.OBS
    comptime _Z_OFF: Int = Self.OBS + Self.ACT
    comptime _ND: Int = Self.BATCH * Self.D
    comptime _NN: Int = Self.BATCH * Self.BATCH
    comptime _NA: Int = Self.BATCH * Self.ACT

    var f1: OnlineTargetPair[Self.FNET]
    var f2: OnlineTargetPair[Self.FNET]
    var bnet: OnlineTargetPair[Self.BNET]
    var actor: OnlineTargetPair[Self.ANET]

    var opt_f1: Adam
    var opt_f2: Adam
    var opt_b: Adam
    var opt_actor: Adam

    var ws1: FBLossWorkspace[Self.D, Self.BATCH]
    var ws2: FBLossWorkspace[Self.D, Self.BATCH]
    var wso: FBLossWorkspace[Self.D, Self.BATCH]

    # Owned scratch — sized once, reused every step.
    var b_s: Tensor
    var b_sp: Tensor
    var b_sn: Tensor
    var bt_sp: Tensor
    var pi_t: Tensor
    var a_next: Tensor
    var noise: Tensor
    var fin: Tensor
    var fin_t: Tensor
    var ain: Tensor
    var ain_t: Tensor
    var ft1: Tensor
    var ft2: Tensor
    var mt1: Tensor
    var mt2: Tensor
    var m_target: Tensor
    var f1o: Tensor
    var f2o: Tensor
    var g_f1: Tensor
    var g_f2: Tensor
    var g_bsp1: Tensor
    var g_bsp2: Tensor
    var g_bsn1: Tensor
    var g_bsn2: Tensor
    var g_bs_o: Tensor
    var g_bsp_o: Tensor
    var g_bsp: Tensor
    var g_bsn: Tensor
    var g_bs: Tensor
    var acc: Tensor
    # ── the batch itself ────────────────────────────────────────────────
    # Owned, not passed per step. `TensorRefs[N, o]` requires every tensor in
    # a pack to share ONE origin, so a `train_step(s, a, ...)` taking five
    # separately-owned `mut Tensor`s cannot build the packs the nets need —
    # that is why the M1 CPU trainer staged everything through a `TensorPack`
    # and copied. Owning them makes every pack `origin_of(self)` and removes
    # the copy; the GPU gather writes straight into these.
    var bs: Tensor
    var ba: Tensor
    var bsn: Tensor
    var bsp: Tensor
    var bz: Tensor
    # actor path
    var pi: Tensor
    var fin_a: Tensor
    var fo: Tensor
    var g_fa: Tensor
    var g_pi: Tensor
    var rz: Tensor

    var ctx: Optional[DeviceContext]
    var gamma: Float64
    var tau: Float64
    var ortho_weight: Float64
    var policy_noise: Float64
    var noise_clip: Float64
    # ⚠ Global grad-norm clip, 0 = OFF. NOT cosmetic on FB: the measure loss is
    # `E[(F·B+^T - gamma·Fbar·Bbar^T)^2]` with ||B|| pinned at sqrt(d) ~ 11.3,
    # so its scale goes as (||F||·11.3)^2 and a spike to 2.5e3 carries gradients
    # to match. Measured on walker at 1 M rows: stable to ~50 k steps, then
    # excursions to +2559 by 116 k. `F` is the unconstrained half of the pair —
    # `L_ortho` and the LayerNorm bound `B`, nothing bounds `F`.
    var max_grad_norm: Float64
    var steps: Int
    var _rng_seed: UInt64
    var _rng_offset: UInt64
    var _sized: Bool

    def __init__(out self):
        self.f1 = OnlineTargetPair[Self.FNET]()
        self.f2 = OnlineTargetPair[Self.FNET]()
        self.bnet = OnlineTargetPair[Self.BNET]()
        self.actor = OnlineTargetPair[Self.ANET]()
        self.opt_f1 = Adam(lr=Scalar[DT](3e-4))
        self.opt_f2 = Adam(lr=Scalar[DT](3e-4))
        self.opt_b = Adam(lr=Scalar[DT](3e-4))
        self.opt_actor = Adam(lr=Scalar[DT](3e-4))
        self.ws1 = FBLossWorkspace[Self.D, Self.BATCH]()
        self.ws2 = FBLossWorkspace[Self.D, Self.BATCH]()
        self.wso = FBLossWorkspace[Self.D, Self.BATCH]()
        self.b_s = Tensor()
        self.b_sp = Tensor()
        self.b_sn = Tensor()
        self.bt_sp = Tensor()
        self.pi_t = Tensor()
        self.a_next = Tensor()
        self.noise = Tensor()
        self.fin = Tensor()
        self.fin_t = Tensor()
        self.ain = Tensor()
        self.ain_t = Tensor()
        self.ft1 = Tensor()
        self.ft2 = Tensor()
        self.mt1 = Tensor()
        self.mt2 = Tensor()
        self.m_target = Tensor()
        self.f1o = Tensor()
        self.f2o = Tensor()
        self.g_f1 = Tensor()
        self.g_f2 = Tensor()
        self.g_bsp1 = Tensor()
        self.g_bsp2 = Tensor()
        self.g_bsn1 = Tensor()
        self.g_bsn2 = Tensor()
        self.g_bs_o = Tensor()
        self.g_bsp_o = Tensor()
        self.g_bsp = Tensor()
        self.g_bsn = Tensor()
        self.g_bs = Tensor()
        self.acc = Tensor()
        self.bs = Tensor()
        self.ba = Tensor()
        self.bsn = Tensor()
        self.bsp = Tensor()
        self.bz = Tensor()
        self.pi = Tensor()
        self.fin_a = Tensor()
        self.fo = Tensor()
        self.g_fa = Tensor()
        self.g_pi = Tensor()
        self.rz = Tensor()
        self.ctx = None
        self.gamma = 0.98
        self.tau = 0.01
        self.ortho_weight = 1.0
        self.policy_noise = 0.2
        self.noise_clip = 0.3
        self.max_grad_norm = 0.0
        self.steps = 0
        self._rng_seed = UInt64(0x5EED)
        self._rng_offset = UInt64(0)
        self._sized = False

    def __init__(out self, *, deinit move: Self):
        self.f1 = move.f1^
        self.f2 = move.f2^
        self.bnet = move.bnet^
        self.actor = move.actor^
        self.opt_f1 = move.opt_f1^
        self.opt_f2 = move.opt_f2^
        self.opt_b = move.opt_b^
        self.opt_actor = move.opt_actor^
        self.ws1 = move.ws1^
        self.ws2 = move.ws2^
        self.wso = move.wso^
        self.b_s = move.b_s^
        self.b_sp = move.b_sp^
        self.b_sn = move.b_sn^
        self.bt_sp = move.bt_sp^
        self.pi_t = move.pi_t^
        self.a_next = move.a_next^
        self.noise = move.noise^
        self.fin = move.fin^
        self.fin_t = move.fin_t^
        self.ain = move.ain^
        self.ain_t = move.ain_t^
        self.ft1 = move.ft1^
        self.ft2 = move.ft2^
        self.mt1 = move.mt1^
        self.mt2 = move.mt2^
        self.m_target = move.m_target^
        self.f1o = move.f1o^
        self.f2o = move.f2o^
        self.g_f1 = move.g_f1^
        self.g_f2 = move.g_f2^
        self.g_bsp1 = move.g_bsp1^
        self.g_bsp2 = move.g_bsp2^
        self.g_bsn1 = move.g_bsn1^
        self.g_bsn2 = move.g_bsn2^
        self.g_bs_o = move.g_bs_o^
        self.g_bsp_o = move.g_bsp_o^
        self.g_bsp = move.g_bsp^
        self.g_bsn = move.g_bsn^
        self.g_bs = move.g_bs^
        self.acc = move.acc^
        self.bs = move.bs^
        self.ba = move.ba^
        self.bsn = move.bsn^
        self.bsp = move.bsp^
        self.bz = move.bz^
        self.pi = move.pi^
        self.fin_a = move.fin_a^
        self.fo = move.fo^
        self.g_fa = move.g_fa^
        self.g_pi = move.g_pi^
        self.rz = move.rz^
        self.ctx = move.ctx^
        self.gamma = move.gamma
        self.tau = move.tau
        self.ortho_weight = move.ortho_weight
        self.policy_noise = move.policy_noise
        self.noise_clip = move.noise_clip
        self.max_grad_norm = move.max_grad_norm
        self.steps = move.steps
        self._rng_seed = move._rng_seed
        self._rng_offset = move._rng_offset
        self._sized = move._sized

    @staticmethod
    def make[
        INIT: Initializer = Xavier
    ](
        lr: Float64 = 3e-4,
        gamma: Float64 = 0.98,
        tau: Float64 = 0.01,
        ortho_weight: Float64 = 1.0,
        ctx: Optional[DeviceContext] = None,
        seed: UInt64 = UInt64(0x5EED),
        max_grad_norm: Float64 = 0.0,
    ) raises -> Self:
        """Defaults are the published FB / Meta Motivo settings: EMA 0.99
        (`tau = 0.01`), `gamma = 0.98`, Adam 3e-4.

        `ctx` is required when `TARGET == "gpu"` and ignored on CPU.
        """
        comptime assert Self.TARGET == "cpu" or Self.TARGET == "gpu", (
            "FBTrainer: TARGET must be 'cpu' or 'gpu'"
        )
        if Self.TARGET == "gpu" and not ctx:
            raise Error("FBTrainer.make: TARGET='gpu' requires a ctx")
        var t = Self()
        t.ctx = ctx
        t.f1 = OnlineTargetPair[Self.FNET].make[Self.TARGET, INIT](ctx)
        t.f2 = OnlineTargetPair[Self.FNET].make[Self.TARGET, INIT](ctx)
        t.bnet = OnlineTargetPair[Self.BNET].make[Self.TARGET, INIT](ctx)
        t.actor = OnlineTargetPair[Self.ANET].make[Self.TARGET, INIT](ctx)
        t.opt_f1 = Adam(lr=Scalar[DT](lr))
        t.opt_f2 = Adam(lr=Scalar[DT](lr))
        t.opt_b = Adam(lr=Scalar[DT](lr))
        t.opt_actor = Adam(lr=Scalar[DT](lr))
        t.gamma = gamma
        t.tau = tau
        t.ortho_weight = ortho_weight
        t.max_grad_norm = max_grad_norm
        t._rng_seed = seed
        return t^

    def _size_once(mut self) raises:
        """Allocate every scratch buffer on the first step."""
        if self._sized:
            return
        comptime T = Self.TARGET
        var c = self.ctx
        ensure_t[T](self.b_s, Self._ND, c)
        ensure_t[T](self.b_sp, Self._ND, c)
        ensure_t[T](self.b_sn, Self._ND, c)
        ensure_t[T](self.bt_sp, Self._ND, c)
        ensure_t[T](self.pi_t, Self._NA, c)
        ensure_t[T](self.a_next, Self._NA, c)
        ensure_t[T](self.noise, Self._NA, c)
        ensure_t[T](self.fin, Self.BATCH * Self.F_IN, c)
        ensure_t[T](self.fin_t, Self.BATCH * Self.F_IN, c)
        ensure_t[T](self.fin_a, Self.BATCH * Self.F_IN, c)
        ensure_t[T](self.ain, Self.BATCH * Self.A_IN, c)
        ensure_t[T](self.ain_t, Self.BATCH * Self.A_IN, c)
        ensure_t[T](self.ft1, Self._ND, c)
        ensure_t[T](self.ft2, Self._ND, c)
        ensure_t[T](self.mt1, Self._NN, c)
        ensure_t[T](self.mt2, Self._NN, c)
        ensure_t[T](self.m_target, Self._NN, c)
        ensure_t[T](self.f1o, Self._ND, c)
        ensure_t[T](self.f2o, Self._ND, c)
        ensure_t[T](self.g_f1, Self._ND, c)
        ensure_t[T](self.g_f2, Self._ND, c)
        ensure_t[T](self.g_bsp1, Self._ND, c)
        ensure_t[T](self.g_bsp2, Self._ND, c)
        ensure_t[T](self.g_bsn1, Self._ND, c)
        ensure_t[T](self.g_bsn2, Self._ND, c)
        ensure_t[T](self.g_bs_o, Self._ND, c)
        ensure_t[T](self.g_bsp_o, Self._ND, c)
        ensure_t[T](self.g_bsp, Self._ND, c)
        ensure_t[T](self.g_bsn, Self._ND, c)
        ensure_t[T](self.g_bs, Self._ND, c)
        ensure_t[T](self.pi, Self._NA, c)
        ensure_t[T](self.fo, Self._ND, c)
        ensure_t[T](self.g_fa, Self._ND, c)
        ensure_t[T](self.g_pi, Self._NA, c)
        ensure_t[T](self.rz, Self.BATCH, c)
        ensure_t[T](self.acc, 1, c)
        ensure_t[T](self.bs, Self.BATCH * Self.OBS, c)
        ensure_t[T](self.ba, Self._NA, c)
        ensure_t[T](self.bsn, Self.BATCH * Self.OBS, c)
        ensure_t[T](self.bsp, Self.BATCH * Self.OBS, c)
        ensure_t[T](self.bz, Self._ND, c)
        self._sized = True

    def ensure_sized(mut self) raises:
        """Allocate the owned batch + scratch without running a step.

        A GPU caller gathers straight into `bs`/`ba`/`bsn`/`bsp`/`bz`, so it
        needs them sized first. It cannot get there via `load_batch` — passing
        `self`'s own fields to a `mut self` method aliases, and Mojo rejects
        it.
        """
        self._size_once()

    def embed_sp(mut self) raises:
        """`b_sp = B(bsp)` over the OWNED batch, for the `z` mixture.

        `train_step` computes this itself, but the mixture needs `B(s+)` BEFORE
        the step that consumes `z`. Exposed as a method rather than letting the
        caller write `backward_embed(t.bsp, t.b_sp)`, which aliases `self`.
        """
        comptime T = Self.TARGET
        self._size_once()
        call_forward[T, Self.BATCH](
            self.bnet.online, TensorRefs[1, MutAnyOrigin](self.bsp),
            self.b_sp, self.ctx,
        )

    # ── the step ─────────────────────────────────────────────────────────

    def load_batch(
        mut self,
        mut s: Tensor,
        mut a: Tensor,
        mut s_next: Tensor,
        mut s_plus: Tensor,
        mut z: Tensor,
    ) raises:
        """Copy a batch into the owned buffers.

        The convenience path. A GPU caller that gathers straight into
        `self.bs` / `self.ba` / ... should skip this and call `train_step`
        directly — this exists so a host-side caller need not know about the
        origin constraint described on the fields.
        """
        comptime T = Self.TARGET
        var c = self.ctx
        self._size_once()
        scale_t[T, Self.BATCH * Self.OBS](self.bs, s, Scalar[DT](1.0), c)
        scale_t[T, Self._NA](self.ba, a, Scalar[DT](1.0), c)
        scale_t[T, Self.BATCH * Self.OBS](self.bsn, s_next, Scalar[DT](1.0), c)
        scale_t[T, Self.BATCH * Self.OBS](self.bsp, s_plus, Scalar[DT](1.0), c)
        scale_t[T, Self._ND](self.bz, z, Scalar[DT](1.0), c)

    def train_step(mut self, want_loss: Bool = True) raises -> FBLosses:
        """One gradient step over the OWNED batch (`bs`/`ba`/`bsn`/`bsp`/`bz`).

        ⚠ `bsp` MUST come from a draw independent of `(bs, ba, bsn)`.
        """
        comptime T = Self.TARGET
        var c = self.ctx
        self._size_once()
        self.steps += 1

        # ── zero once, at the top. See the module docstring: B's parameter
        # gradients accumulate over its three forwards, and a second zeroing
        # in the middle would drop whichever came first, silently.
        self.f1.online.zero_grad[T](c)
        self.f2.online.zero_grad[T](c)
        self.bnet.online.zero_grad[T](c)
        self.actor.online.zero_grad[T](c)

        # ── 1. B forwards (online) ───────────────────────────────────────
        call_forward[T, Self.BATCH](
            self.bnet.online, TensorRefs[1, MutAnyOrigin](self.bs), self.b_s, c
        )
        call_forward[T, Self.BATCH](
            self.bnet.online, TensorRefs[1, MutAnyOrigin](self.bsp), self.b_sp, c
        )
        call_forward[T, Self.BATCH](
            self.bnet.online, TensorRefs[1, MutAnyOrigin](self.bsn), self.b_sn, c
        )

        # ── 2. a' = pi_target(s', z) + truncated noise ───────────────────
        pack2_t[T, Self.OBS, Self.D, Self.BATCH](
            self.ain_t, self.bsn, self.bz, c
        )
        call_forward[T, Self.BATCH](
            self.actor.target_net, TensorRefs[1, MutAnyOrigin](self.ain_t), self.pi_t, c
        )
        gaussian_t[T, Self._NA](self.noise, self._rng_seed, self._rng_offset, c)
        self._rng_offset += UInt64(Self._NA + (Self._NA % 2))
        smooth_action_t[T, Self._NA](
            self.a_next, self.pi_t, self.noise,
            Scalar[DT](self.policy_noise), Scalar[DT](self.noise_clip), c,
        )

        # ── 3. the bootstrapped target ───────────────────────────────────
        pack3_t[T, Self.OBS, Self.ACT, Self.D, Self.BATCH](
            self.fin_t, self.bsn, self.a_next, self.bz, c
        )
        call_forward[T, Self.BATCH](
            self.f1.target_net, TensorRefs[1, MutAnyOrigin](self.fin_t), self.ft1, c
        )
        call_forward[T, Self.BATCH](
            self.f2.target_net, TensorRefs[1, MutAnyOrigin](self.fin_t), self.ft2, c
        )
        call_forward[T, Self.BATCH](
            self.bnet.target_net, TensorRefs[1, MutAnyOrigin](self.bsp), self.bt_sp, c
        )

        # Both target matrices go through the SAME primitive the online path
        # uses. A hand-inlined target would be one edit away from disagreeing
        # with the online one, and that disagreement is invisible in the loss.
        self.ws1.prepare[T](c)
        self.ws1.pd.forward[T, Self.BATCH](
            TensorRefs[2, MutAnyOrigin](self.ft1, self.bt_sp), self.mt1, c
        )
        self.ws1.pd.forward[T, Self.BATCH](
            TensorRefs[2, MutAnyOrigin](self.ft2, self.bt_sp), self.mt2, c
        )
        min_scale_t[T, Self._NN](
            self.m_target, self.mt1, self.mt2, Scalar[DT](self.gamma), c
        )

        # ── 4. online F forwards + losses ────────────────────────────────
        pack3_t[T, Self.OBS, Self.ACT, Self.D, Self.BATCH](
            self.fin, self.bs, self.ba, self.bz, c
        )
        call_forward[T, Self.BATCH](
            self.f1.online, TensorRefs[1, MutAnyOrigin](self.fin), self.f1o, c
        )
        call_forward[T, Self.BATCH](
            self.f2.online, TensorRefs[1, MutAnyOrigin](self.fin), self.f2o, c
        )

        var l1 = fb_measure_loss_into[T, Self.D, Self.BATCH](
            self.ws1, self.f1o, self.b_sp, self.b_sn, self.m_target,
            self.g_f1, self.g_bsp1, self.g_bsn1, want_loss, c,
        )
        var l2 = fb_measure_loss_into[T, Self.D, Self.BATCH](
            self.ws2, self.f2o, self.b_sp, self.b_sn, self.m_target,
            self.g_f2, self.g_bsp2, self.g_bsn2, want_loss, c,
        )
        var l_ortho = fb_ortho_loss_into[T, Self.D, Self.BATCH](
            self.wso, self.b_s, self.b_sp, self.g_bs_o, self.g_bsp_o,
            want_loss, c,
        )

        # ── 5. backprop ──────────────────────────────────────────────────
        var sink = TensorPack[1]()
        call_vjp[T, Self.BATCH](
            self.f1.online, TensorRefs[1, MutAnyOrigin](self.fin), self.g_f1,
            TensorRefs[1, MutAnyOrigin](sink[0]), c,
        )
        call_vjp[T, Self.BATCH](
            self.f2.online, TensorRefs[1, MutAnyOrigin](self.fin), self.g_f2,
            TensorRefs[1, MutAnyOrigin](sink[0]), c,
        )

        # B: three vjps, accumulating into one set of parameter gradients.
        sum3_scaled_t[T, Self._ND](
            self.g_bsp, self.g_bsp1, self.g_bsp2, self.g_bsp_o,
            Scalar[DT](self.ortho_weight), c,
        )
        scale_t[T, Self._ND](self.g_bsn, self.g_bsn1, Scalar[DT](1.0), c)
        axpy_t[T, Self._ND](self.g_bsn, self.g_bsn2, Scalar[DT](1.0), c)
        scale_t[T, Self._ND](
            self.g_bs, self.g_bs_o, Scalar[DT](self.ortho_weight), c
        )

        call_vjp[T, Self.BATCH](
            self.bnet.online, TensorRefs[1, MutAnyOrigin](self.bsp), self.g_bsp,
            TensorRefs[1, MutAnyOrigin](sink[0]), c,
        )
        call_vjp[T, Self.BATCH](
            self.bnet.online, TensorRefs[1, MutAnyOrigin](self.bsn), self.g_bsn,
            TensorRefs[1, MutAnyOrigin](sink[0]), c,
        )
        call_vjp[T, Self.BATCH](
            self.bnet.online, TensorRefs[1, MutAnyOrigin](self.bs), self.g_bs,
            TensorRefs[1, MutAnyOrigin](sink[0]), c,
        )

        if self.max_grad_norm > 0.0:
            var mgn = Scalar[DT](self.max_grad_norm)
            _ = self.opt_f1.clip_grads[T](self.f1.online, mgn, c)
            _ = self.opt_f2.clip_grads[T](self.f2.online, mgn, c)
            _ = self.opt_b.clip_grads[T](self.bnet.online, mgn, c)
        self.opt_f1.step[T](self.f1.online, c)
        self.opt_f2.step[T](self.f2.online, c)
        self.opt_b.step[T](self.bnet.online, c)

        # ── actor ────────────────────────────────────────────────────────
        var l_actor = self._actor_step(want_loss)

        # ── Polyak ───────────────────────────────────────────────────────
        self.f1.polyak_step[T](Scalar[DT](self.tau), c)
        self.f2.polyak_step[T](Scalar[DT](self.tau), c)
        self.bnet.polyak_step[T](Scalar[DT](self.tau), c)
        self.actor.polyak_step[T](Scalar[DT](self.tau), c)

        if not want_loss:
            return FBLosses(0.0, 0.0, 0.0, 0.0, 0.0)
        var fn2 = mean_sq_t[T, Self._ND](self.f1o, self.acc, c)
        var bn2 = mean_sq_t[T, Self._ND](self.b_s, self.acc, c)
        return FBLosses(
            0.5 * (l1 + l2), l_ortho, l_actor,
            sqrt(fn2 * Float64(Self.D)), sqrt(bn2 * Float64(Self.D)),
        )

    def _actor_step(mut self, want_loss: Bool) raises -> Float64:
        """DPG through `F1`: maximise `F(s, pi_z(s,z), z) · z`.

        `z` is both the actor's conditioning input and the direction the value
        is projected onto — that is the whole point of the latent, and it is
        why the actor loss needs no reward.
        """
        comptime T = Self.TARGET
        var c = self.ctx

        pack2_t[T, Self.OBS, Self.D, Self.BATCH](
            self.ain, self.bs, self.bz, c
        )
        call_forward[T, Self.BATCH](
            self.actor.online, TensorRefs[1, MutAnyOrigin](self.ain), self.pi, c
        )
        pack3_t[T, Self.OBS, Self.ACT, Self.D, Self.BATCH](
            self.fin_a, self.bs, self.pi, self.bz, c
        )
        call_forward[T, Self.BATCH](
            self.f1.online, TensorRefs[1, MutAnyOrigin](self.fin_a), self.fo, c
        )

        var loss = Float64(0)
        if want_loss:
            # rowwise F·z, then the batch mean — reusing RowDot rather than a
            # bespoke kernel keeps the CPU and GPU numbers identical.
            self.ws1.rd.forward[T, Self.BATCH](
                TensorRefs[2, MutAnyOrigin](self.fo, self.bz), self.rz, c
            )
            var m = mean_t[T, Self.BATCH](self.rz, self.acc, c)
            loss = -m

        # dL/dF = -z / BATCH
        scale_t[T, Self._ND](
            self.g_fa, self.bz, Scalar[DT](-1.0 / Float64(Self.BATCH)), c
        )

        # ⚠ Through F1 WITHOUT keeping F1's parameter grads: the optimizer has
        # already stepped them above, and folding a second, differently-scaled
        # critic gradient into the next step would be silent. The vjp
        # accumulates into params, so F1's grads are zeroed right after.
        var g_fin = TensorPack[1]()
        call_vjp[T, Self.BATCH](
            self.f1.online, TensorRefs[1, MutAnyOrigin](self.fin_a), self.g_fa,
            TensorRefs[1, MutAnyOrigin](g_fin[0]), c,
        )
        self.f1.online.zero_grad[T](c)

        slice_cols_t[T, Self.F_IN, Self._A_OFF, Self.ACT, Self.BATCH](
            self.g_pi, g_fin[0], c
        )
        var sink = TensorPack[1]()
        call_vjp[T, Self.BATCH](
            self.actor.online, TensorRefs[1, MutAnyOrigin](self.ain), self.g_pi,
            TensorRefs[1, MutAnyOrigin](sink[0]), c,
        )
        if self.max_grad_norm > 0.0:
            _ = self.opt_actor.clip_grads[T](
                self.actor.online, Scalar[DT](self.max_grad_norm), c
            )
        self.opt_actor.step[T](self.actor.online, c)
        return loss

    # ── checkpoint ───────────────────────────────────────────────────────

    def save_state(mut self, path: String) raises:
        """Write B, both F twins and the actor into ONE `storage-ckpt` file.

        ⚠ Only the ONLINE nets. The targets are EMA copies that re-converge
        within a few thousand Polyak steps, and Adam moments re-warm — neither
        is worth the file size. A resume is therefore not bit-identical, which
        is fine for a 2 M-step run and would not be for a parity gate.

        ⚠⚠ Call this PERIODICALLY, not only at the end. The first version of
        the M2 run script trained for 2 M steps and exited without saving
        anything: hours of GPU time producing a log file and no weights.
        """
        var w = CheckpointWriter(save_moments=False)
        w.mode = 0
        self.bnet.online.for_each_param[Self.TARGET](w, self.ctx, "b")
        self.f1.online.for_each_param[Self.TARGET](w, self.ctx, "f1")
        self.f2.online.for_each_param[Self.TARGET](w, self.ctx, "f2")
        self.actor.online.for_each_param[Self.TARGET](w, self.ctx, "actor")
        w.mode = 1
        self.bnet.online.for_each_state[Self.TARGET](w, self.ctx, "b")
        self.f1.online.for_each_state[Self.TARGET](w, self.ctx, "f1")
        self.f2.online.for_each_state[Self.TARGET](w, self.ctx, "f2")
        self.actor.online.for_each_state[Self.TARGET](w, self.ctx, "actor")
        with open(path, "w") as f:
            f.write(w.content)

    def load_state(mut self, path: String) raises:
        """Restore the online nets and HARD-COPY them onto the targets.

        Without the hard copy the targets stay at their random init while the
        online nets are trained, and the first bootstrapped target is garbage —
        a resume that silently undoes part of the run it is resuming.
        """
        var content: String
        with open(path, "r") as f:
            content = String(f.read())
        var lines = _split_lines(content)
        # The `storage-ckpt vN` header is not a section; the reader expects the
        # first line to BE one, so strip it. (`CheckpointWriter` emits it.)
        var body = List[String]()
        for li in range(len(lines)):
            if lines[li].startswith("storage-ckpt"):
                continue
            body.append(lines[li])
        var r = CheckpointReader(body^)
        r.mode = 0
        self.bnet.online.for_each_param[Self.TARGET](r, self.ctx, "b")
        self.f1.online.for_each_param[Self.TARGET](r, self.ctx, "f1")
        self.f2.online.for_each_param[Self.TARGET](r, self.ctx, "f2")
        self.actor.online.for_each_param[Self.TARGET](r, self.ctx, "actor")
        r.mode = 1
        self.bnet.online.for_each_state[Self.TARGET](r, self.ctx, "b")
        self.f1.online.for_each_state[Self.TARGET](r, self.ctx, "f1")
        self.f2.online.for_each_state[Self.TARGET](r, self.ctx, "f2")
        self.actor.online.for_each_state[Self.TARGET](r, self.ctx, "actor")
        self.bnet.target_net.polyak_from[Self.TARGET](
            self.bnet.online, Scalar[DT](1.0), self.ctx
        )
        self.f1.target_net.polyak_from[Self.TARGET](
            self.f1.online, Scalar[DT](1.0), self.ctx
        )
        self.f2.target_net.polyak_from[Self.TARGET](
            self.f2.online, Scalar[DT](1.0), self.ctx
        )
        self.actor.target_net.polyak_from[Self.TARGET](
            self.actor.online, Scalar[DT](1.0), self.ctx
        )

    # ── inference ────────────────────────────────────────────────────────

    def act[
        N: Int
    ](mut self, mut s: Tensor, mut z_row: Tensor, mut dst: Tensor) raises:
        """`pi_z(s, z)` for `N` rows — the zero-shot policy.

        ⚠ `z_row` must already be on the radius-sqrt(D) sphere and must hold N
        rows (broadcast a single z yourself). `z_sampler`'s producers all
        guarantee the norm; a `z` assembled by hand does not, and the symptom
        is a policy that acts plausibly and optimises nothing.
        """
        comptime T = Self.TARGET
        var c = self.ctx
        # Staged through a TensorPack so the packed input is a single owned
        # buffer of exactly `N * A_IN`, independent of how the caller sized
        # `s` and `z_row`. `N` here is an INFERENCE batch and differs from the
        # training `BATCH`, so none of the trainer's own scratch is the right
        # width.
        var pack = TensorPack[1]()
        var ain = Tensor()
        pack2_t[T, Self.OBS, Self.D, N](ain, s, z_row, c)
        ensure_t[T](pack[0], N * Self.A_IN, c)
        scale_t[T, N * Self.A_IN](pack[0], ain, Scalar[DT](1.0), c)
        ensure_t[T](dst, N * Self.ACT, c)
        call_forward[T, N](
            self.actor.online, TensorRefs[1, MutAnyOrigin](pack[0]), dst, c
        )

    def backward_embed[
        N: Int
    ](mut self, mut s: Tensor, mut dst: Tensor) raises:
        """`B(s)` for `N` rows — the input to `z_from_reward`."""
        comptime T = Self.TARGET
        var c = self.ctx
        ensure_t[T](dst, N * Self.D, c)
        var pack = TensorPack[1]()
        ensure_t[T](pack[0], N * Self.OBS, c)
        scale_t[T, N * Self.OBS](pack[0], s, Scalar[DT](1.0), c)
        call_forward[T, N](
            self.bnet.online, TensorRefs[1, MutAnyOrigin](pack[0]), dst, c
        )

