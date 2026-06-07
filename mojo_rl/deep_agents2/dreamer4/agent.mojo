"""Dreamer4Agent — behavior-cloning agent facade (Phase 3.7).

Wires the four trained components into one BC training step:

  TaskEmbedder ──embed_into──▶ agent_in (task token per sequence, broadcast)
        │                          │
        │                          ▼
        │      Dreamer4Dynamics.set_agent_in → shortcut-forcing forwards
        │      (video-prediction loss on noisy reps) → main-pass h_t
        │                          │
        │                          ▼
        │              MTP policy + reward heads → BC loss (eq. 9) → grad_h
        │                          │
        ▼                          ▼
  accumulate_grad ◀── dyn.grad_agent_in ◀── dyn.vjp(grad_zhat, set_grad_h)

The video-prediction (shortcut-forcing) loss and the BC loss are trained
jointly: the MAIN forward of the shortcut loss produces h_t (agent tokens are
injected into every pass), the BC loss backprops into h_t, and `dyn.vjp`
carries BOTH the video grad (spatial flow columns) and the BC grad (agent
columns) through the transformer in one pass — then the agent-input grad feeds
the TaskEmbedder.

`Dreamer4Agent` is itself a (degenerate) `Module`: its forward/vjp are unused
(the real entry point is `bc_train_step`), but conforming lets a SINGLE Adam
cover every parameter (dynamics + task embedder + both heads) through the
delegated `for_each_param`/`zero_grad`.

v1 uses an UNCONDITIONAL dynamics (ADIM=0) — the action-conditioned video
prediction (ADIM>0, already built) layers in for the real-env lighthouse.
"""

from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core import Initializer, AMPPolicy, NoAMP, ParamVisitor
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.core.tensor_pack import TensorPack
from mojo_rl.nn2.core.target_storage import TargetStorage, assert_tag_for

from .dynamics import Dreamer4Dynamics
from .task_embedder import TaskEmbedder
from .heads import (
    Dreamer4PolicyHead, Dreamer4RewardHead, Dreamer4ValueHead,
    Dreamer4ContinueHead,
)
from .shortcut_loss import dynamics_pretrain_loss
from .bc_loss import bc_mtp_loss
from .imag_rollout import imagine_rollout
from .imag_rl_loss import (
    lambda_returns,
    value_td_loss_cpu,
    value_td_loss_backward,
    pmpo_policy_loss_cpu,
    pmpo_policy_loss_backward,
    continue_pred,
)
from ..dreamerv3.polyak import polyak_module


def _ilog2(n: Int) -> Int:
    var k = 0
    var v = n
    while v > 1:
        v //= 2
        k += 1
    return k


struct Dreamer4Agent[
    DSP: Int, NSP: Int, D: Int, NH: Int, T: Int, NREG: Int, HID: Int,
    DEPTH: Int, KMAX: Int,            # dynamics backbone
    NAGENT: Int, NTASK: Int,          # agent tokens + task table
    HHID: Int, NACT: Int, NBINS: Int, NMTP: Int,   # heads
    B: Int, B_SELF: Int,              # sequences per batch + self rows
    USE_MAX: Bool = True,
    ADIM: Int = 0,                    # action dim (0 ⇒ unconditional; Phase-4
    AHID: Int = 0,                    #  imagination needs ADIM = NACT one-hot)
    K_IMAG: Int = 0,                  # ODE steps for imagination (0 ⇒ KMAX)
    NCTX: Int = 1,                    # clean context frames for the rollout
    DYN_TARGET: StaticString = "cpu", # "gpu" ⇒ dynamics on device (the heavy
                                      #  imagination compute); heads stay CPU
](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=1)
    comptime OUT_DIM = 1

    comptime AGD: Int = Self.NAGENT * Self.D          # agent-token width = h_t dim
    comptime ND: Int = Self.NSP * Self.DSP            # packed latent width
    comptime BF: Int = Self.B * Self.T                # nn2 batch (B·T)
    comptime PLOG: Int = Self.NMTP * Self.NACT
    comptime RLOG: Int = Self.NMTP * Self.NBINS
    comptime EMAX: Int = _ilog2(Self.KMAX)            # clean-BC step index
    comptime KI: Int = Self.K_IMAG if Self.K_IMAG > 0 else Self.KMAX
    comptime TM1: Int = Self.T - 1                    # imagined states with a return

    comptime DYN = Dreamer4Dynamics[
        Self.DSP, Self.NSP, Self.D, Self.NH, Self.T, Self.NREG, Self.HID,
        Self.DEPTH, Self.KMAX, Self.USE_MAX, Self.ADIM, Self.AHID, Self.NAGENT,
    ]
    comptime TE = TaskEmbedder[Self.D, Self.NTASK, Self.NAGENT]
    comptime PH = Dreamer4PolicyHead[Self.AGD, Self.HHID, Self.NACT, Self.NMTP]
    comptime RH = Dreamer4RewardHead[Self.AGD, Self.HHID, Self.NBINS, Self.NMTP]
    comptime VH = Dreamer4ValueHead[Self.AGD, Self.HHID, Self.NBINS]
    comptime CH = Dreamer4ContinueHead[Self.AGD, Self.HHID]

    var dyn: Self.DYN
    var te: Self.TE
    var ph: Self.PH
    var rh: Self.RH
    var vh: Self.VH                  # value head (Phase 4; untrained during BC)
    var ph_prior: Self.PH           # frozen behavioral prior (PMPO reverse-KL)
    var ch: Self.CH                 # continue/termination head (opt-in via
                                    #  use_continue; frozen during imagination)

    # CPU scratch (owned; sized at make)
    var agent_in: List[Scalar[DT]]      # [BF, AGD]
    var grad_zhat: List[Scalar[DT]]     # [BF, ND]
    var zhat: List[Scalar[DT]]          # [BF, ND]
    var grad_zt: List[Scalar[DT]]       # [BF, ND] (grad wrt z̃; discarded)
    var grad_h: List[Scalar[DT]]        # [BF, AGD]
    var grad_h_tmp: List[Scalar[DT]]    # [BF, AGD]
    var plog: List[Scalar[DT]]          # [BF, PLOG]
    var rlog: List[Scalar[DT]]          # [BF, RLOG]
    var gpl: List[Scalar[DT]]           # [BF, PLOG]
    var grl: List[Scalar[DT]]           # [BF, RLOG]
    # clean-BC scratch: a dedicated near-clean forward at the HIGHEST TRAINED
    # signal level (σ=(KMAX-1)/KMAX, sig=KMAX-1) gives h_t a near-un-noised
    # frame so the policy can read the observation. σ=1 (sig=KMAX) is avoided:
    # the video loss never samples it, so that embedding row is untrained ⇒ OOD.
    var clean_sig: List[Scalar[DT]]     # [BF] = KMAX-1 (highest trained σ)
    var clean_step: List[Scalar[DT]]    # [BF] = EMAX
    var bc_in: List[Scalar[DT]]         # [BF, ND] noised BC input at σ_bc
    var gzero: List[Scalar[DT]]         # [BF, ND] zero flow-grad for the BC vjp
    # ── imagination-RL scratch (Phase 4) ────────────────────────────────
    var im_h: List[Scalar[DT]]          # [BF, AGD]  rollout agent tokens
    var im_act: List[Scalar[DT]]        # [B, T-1]   sampled action class
    var im_rew: List[Scalar[DT]]        # [BF]       reward-head pred per state
    var im_val: List[Scalar[DT]]        # [BF]       value-head pred per state
    var im_con: List[Scalar[DT]]        # [BF]       continue (= γ)
    var im_ret: List[Scalar[DT]]        # [B, T-1]   λ-returns
    var im_adv: List[Scalar[DT]]        # [B, T-1]   advantages
    var im_actbt: List[Scalar[DT]]      # [BF]       actions on the [B,T] grid
    var im_vlog: List[Scalar[DT]]       # [BF, NBINS] value logits (grad re-run)
    var im_gvlog: List[Scalar[DT]]      # [BF, NBINS] value-logit grad
    var im_vloss: List[Scalar[DT]]      # [B, T-1]
    var im_plog: List[Scalar[DT]]       # [BF, PLOG]  policy logits (grad re-run)
    var im_prior: List[Scalar[DT]]      # [BF, PLOG]  frozen prior logits
    var im_plog0: List[Scalar[DT]]      # [BF, NACT]  dist-0 policy block
    var im_prior0: List[Scalar[DT]]     # [BF, NACT]  dist-0 prior block
    var im_gplog0: List[Scalar[DT]]     # [BF, NACT]  dist-0 grad
    var im_gplog: List[Scalar[DT]]      # [BF, PLOG]  full policy-logit grad
    var im_clog: List[Scalar[DT]]       # [BF]        continue logits (use_continue)
    var im_chat: List[Scalar[DT]]       # [BF]        continue preds ĉ
    var ts: TargetStorage

    def __init__(out self):
        self.dyn = Self.DYN()
        self.te = Self.TE()
        self.ph = Self.PH()
        self.rh = Self.RH()
        self.vh = Self.VH()
        self.ph_prior = Self.PH()
        self.ch = Self.CH()
        self.im_h = List[Scalar[DT]]()
        self.im_act = List[Scalar[DT]]()
        self.im_rew = List[Scalar[DT]]()
        self.im_val = List[Scalar[DT]]()
        self.im_con = List[Scalar[DT]]()
        self.im_ret = List[Scalar[DT]]()
        self.im_adv = List[Scalar[DT]]()
        self.im_actbt = List[Scalar[DT]]()
        self.im_vlog = List[Scalar[DT]]()
        self.im_gvlog = List[Scalar[DT]]()
        self.im_vloss = List[Scalar[DT]]()
        self.im_plog = List[Scalar[DT]]()
        self.im_prior = List[Scalar[DT]]()
        self.im_plog0 = List[Scalar[DT]]()
        self.im_prior0 = List[Scalar[DT]]()
        self.im_gplog0 = List[Scalar[DT]]()
        self.im_gplog = List[Scalar[DT]]()
        self.im_clog = List[Scalar[DT]]()
        self.im_chat = List[Scalar[DT]]()
        self.agent_in = List[Scalar[DT]]()
        self.grad_zhat = List[Scalar[DT]]()
        self.zhat = List[Scalar[DT]]()
        self.grad_zt = List[Scalar[DT]]()
        self.grad_h = List[Scalar[DT]]()
        self.grad_h_tmp = List[Scalar[DT]]()
        self.plog = List[Scalar[DT]]()
        self.rlog = List[Scalar[DT]]()
        self.gpl = List[Scalar[DT]]()
        self.grl = List[Scalar[DT]]()
        self.clean_sig = List[Scalar[DT]]()
        self.clean_step = List[Scalar[DT]]()
        self.bc_in = List[Scalar[DT]]()
        self.gzero = List[Scalar[DT]]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu", (
            "Dreamer4Agent heads/te/value are CPU; set DYN_TARGET=\"gpu\" to put"
            " the dynamics on device (the heavy imagination compute)."
        )
        comptime assert Self.DYN_TARGET == "cpu" or Self.DYN_TARGET == "gpu", (
            "DYN_TARGET must be 'cpu' or 'gpu'"
        )
        var m = Self()
        # dynamics on DYN_TARGET (GPU for the device rollout); everything else
        # (task embedder, heads, value, prior) stays on host.
        m.dyn = Self.DYN.make[target = Self.DYN_TARGET, INIT=INIT](ctx)
        m.te = Self.TE.make[target=target, INIT=INIT](ctx)
        m.ph = Self.PH.make[target=target, INIT=INIT](ctx)
        m.rh = Self.RH.make[target=target, INIT=INIT](ctx)
        # vh + ph_prior made AFTER rh so the dyn/te/ph/rh RNG draws (hence the
        # BC path) are byte-for-byte unchanged from the Phase-3 agent.
        m.vh = Self.VH.make[target=target, INIT=INIT](ctx)
        m.ph_prior = Self.PH.make[target=target, INIT=INIT](ctx)
        # continue head made LAST so all earlier RNG draws (BC path + Phase-4
        # heads) are byte-for-byte unchanged.
        m.ch = Self.CH.make[target=target, INIT=INIT](ctx)
        m.im_h.resize(Self.BF * Self.AGD, Scalar[DT](0.0))
        m.im_act.resize(Self.B * Self.TM1, Scalar[DT](0.0))
        m.im_rew.resize(Self.BF, Scalar[DT](0.0))
        m.im_val.resize(Self.BF, Scalar[DT](0.0))
        m.im_con.resize(Self.BF, Scalar[DT](0.0))
        m.im_ret.resize(Self.B * Self.TM1, Scalar[DT](0.0))
        m.im_adv.resize(Self.B * Self.TM1, Scalar[DT](0.0))
        m.im_actbt.resize(Self.BF, Scalar[DT](0.0))
        m.im_vlog.resize(Self.BF * Self.NBINS, Scalar[DT](0.0))
        m.im_gvlog.resize(Self.BF * Self.NBINS, Scalar[DT](0.0))
        m.im_vloss.resize(Self.B * Self.TM1, Scalar[DT](0.0))
        m.im_plog.resize(Self.BF * Self.PLOG, Scalar[DT](0.0))
        m.im_prior.resize(Self.BF * Self.PLOG, Scalar[DT](0.0))
        m.im_plog0.resize(Self.BF * Self.NACT, Scalar[DT](0.0))
        m.im_prior0.resize(Self.BF * Self.NACT, Scalar[DT](0.0))
        m.im_gplog0.resize(Self.BF * Self.NACT, Scalar[DT](0.0))
        m.im_gplog.resize(Self.BF * Self.PLOG, Scalar[DT](0.0))
        m.im_clog.resize(Self.BF, Scalar[DT](0.0))
        m.im_chat.resize(Self.BF, Scalar[DT](0.0))
        m.agent_in.resize(Self.BF * Self.AGD, Scalar[DT](0.0))
        m.grad_zhat.resize(Self.BF * Self.ND, Scalar[DT](0.0))
        m.zhat.resize(Self.BF * Self.ND, Scalar[DT](0.0))
        m.grad_zt.resize(Self.BF * Self.ND, Scalar[DT](0.0))
        m.grad_h.resize(Self.BF * Self.AGD, Scalar[DT](0.0))
        m.grad_h_tmp.resize(Self.BF * Self.AGD, Scalar[DT](0.0))
        m.plog.resize(Self.BF * Self.PLOG, Scalar[DT](0.0))
        m.rlog.resize(Self.BF * Self.RLOG, Scalar[DT](0.0))
        m.gpl.resize(Self.BF * Self.PLOG, Scalar[DT](0.0))
        m.grl.resize(Self.BF * Self.RLOG, Scalar[DT](0.0))
        m.clean_sig.resize(Self.BF, Scalar[DT](Float64(Self.KMAX - 1)))  # σ=0.75
        m.clean_step.resize(Self.BF, Scalar[DT](Float64(Self.EMAX)))
        m.bc_in.resize(Self.BF * Self.ND, Scalar[DT](0.0))
        m.gzero.resize(Self.BF * Self.ND, Scalar[DT](0.0))
        m.ts = TargetStorage.make_cpu()
        return m^

    @staticmethod
    def display_label() -> String:
        return String("Dreamer4Agent")

    # ── Module conformance: forward/vjp unused (entry point is bc_train_step) ─
    def forward[
        target: StaticString, BATCH: Int, POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorPack[Self.ARITY],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        raise Error("Dreamer4Agent.forward is unused; call bc_train_step")

    def vjp[
        target: StaticString, BATCH: Int, POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        grad_inputs: TensorPack[Self.ARITY],
    ) raises:
        raise Error("Dreamer4Agent.vjp is unused; call bc_train_step")

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, prefix: String, mut visitor: V) raises:
        comptime assert Self.DYN_TARGET == "cpu", (
            "whole-agent for_each_param spans one target; with DYN_TARGET=\"gpu\""
            " the dynamics is on device — optimize submodules separately"
            " (agent.dyn on GPU, agent.ph/agent.vh on CPU)."
        )
        assert_tag_for["Dreamer4Agent", target](self.ts.target_tag)
        self.dyn.for_each_param[target, V](prefix + ".dyn", visitor)
        self.te.for_each_param[target, V](prefix + ".te", visitor)
        self.ph.for_each_param[target, V](prefix + ".ph", visitor)
        self.rh.for_each_param[target, V](prefix + ".rh", visitor)
        self.vh.for_each_param[target, V](prefix + ".vh", visitor)
        self.ch.for_each_param[target, V](prefix + ".ch", visitor)
        # NOTE: `ph_prior` is the FROZEN behavioral prior — never optimized, so
        # it is deliberately excluded from the param walk.

    def zero_grad[target: StaticString](mut self) raises:
        comptime assert Self.DYN_TARGET == "cpu", (
            "whole-agent zero_grad spans one target; with DYN_TARGET=\"gpu\""
            " zero submodule grads separately."
        )
        assert_tag_for["Dreamer4Agent", target](self.ts.target_tag)
        self.dyn.zero_grad[target]()
        self.te.zero_grad[target]()
        self.ph.zero_grad[target]()
        self.rh.zero_grad[target]()
        self.vh.zero_grad[target]()
        self.ch.zero_grad[target]()

    # ── eval accessors ──────────────────────────────────────────────────
    def agent_out_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        """Return h_t from the last forward (for inspection / eval heads)."""
        return self.dyn.agent_out_ptr_cpu()

    def policy_logits_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        """Return the [BF, NMTP·NACT] policy logits from the last bc_train_step
        (distance n at columns [n·NACT, (n+1)·NACT)) — greedy action = argmax of
        the distance-0 block."""
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.plog.unsafe_ptr()
        )

    def _run_bc_loss(
        mut self,
        ht: UnsafePointer[Scalar[DT], MutAnyOrigin],
        actions: UnsafePointer[Scalar[DT], MutAnyOrigin],
        rewards: UnsafePointer[Scalar[DT], MutAnyOrigin],
        bins: UnsafePointer[Scalar[DT], MutAnyOrigin],
        policy_weight: Scalar[DT],
        reward_weight: Scalar[DT],
    ) raises -> Float64:
        """BC loss on h_t → fills grad_h + head param grads. Returns the loss."""
        return bc_mtp_loss[
            Self.PH, Self.RH, Self.B, Self.T, Self.NMTP, Self.NACT,
            Self.NBINS, Self.AGD,
        ](
            self.ph, self.rh, ht, actions, rewards, bins,
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](self.plog.unsafe_ptr()),
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](self.rlog.unsafe_ptr()),
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](self.gpl.unsafe_ptr()),
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](self.grl.unsafe_ptr()),
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](self.grad_h.unsafe_ptr()),
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.grad_h_tmp.unsafe_ptr()
            ),
            policy_weight=policy_weight,
            reward_weight=reward_weight,
        )

    # ── one joint BC + video-prediction training step (fills all grads) ──
    def bc_train_step(
        mut self,
        z1: UnsafePointer[Scalar[DT], MutAnyOrigin],         # [BF, ND] latents
        z0: UnsafePointer[Scalar[DT], MutAnyOrigin],         # [BF, ND] noise
        sigma: UnsafePointer[Scalar[DT], MutAnyOrigin],      # [BF]
        sigma_idx: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [BF]
        step_idx: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [BF]
        do_boot: Bool,
        task_ids: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B]
        actions: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [BF] class ids
        rewards: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [BF]
        bins: UnsafePointer[Scalar[DT], MutAnyOrigin],       # [NBINS]
        policy_weight: Scalar[DT] = Scalar[DT](1.0),
        reward_weight: Scalar[DT] = Scalar[DT](1.0),
        clean_bc: Bool = True,
    ) raises -> Tuple[Float64, Float64]:
        """Returns (video_loss, bc_loss). Fills the param grads of all four
        components (caller then runs one `optim.step`). Assumes the caller has
        already `optim.zero_grad`'d this agent.

        `clean_bc=True` (default) DECOUPLES the two losses: the shortcut-forcing
        video loss trains the world model on noised reps, then a SEPARATE clean
        (σ=1) forward gives h_t the un-noised frame so the policy can read the
        observation — the straightforward BC. Both backprop into the shared
        transformer (grads accumulate over the two vjp calls). `clean_bc=False`
        is the paper's coupled form (BC reads the noised main-pass h_t); it only
        clones cleanly when the WM is a strong denoiser (large-scale regime)."""
        comptime assert Self.DYN_TARGET == "cpu", (
            "bc_train_step is the CPU world-model path; a DYN_TARGET=\"gpu\""
            " agent trains the dynamics via dynamics_pretrain_loss[FWD=\"gpu\"]"
            " directly (see the imagination lighthouse stage 1)."
        )
        var agp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.agent_in.unsafe_ptr()
        )
        var gzh = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.grad_zhat.unsafe_ptr()
        )
        var zh = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.zhat.unsafe_ptr()
        )
        var ghp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.grad_h.unsafe_ptr()
        )
        var gzt_t = TileTensor(
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.grad_zt.unsafe_ptr()
            ),
            row_major[Self.BF, Self.ND](),
        )

        # 1. task embeddings → agent token input for every (b,t)
        self.te.embed_into["cpu", Self.B, Self.T](task_ids, agp)

        # 2. shortcut-forcing video-prediction loss (injects agent tokens into
        #    every pass; the MAIN pass leaves h_t in dyn.agent_out)
        var loss_v = dynamics_pretrain_loss[
            Self.DYN, Self.B, Self.T, Self.B_SELF, Self.NSP, Self.DSP,
            Self.KMAX, "cpu", 0, Self.AGD,
        ](
            self.dyn, z1, z0, sigma, sigma_idx, step_idx, do_boot, gzh, zh,
            agent_in=agp,
        )

        var loss_bc: Float64 = 0.0
        if not clean_bc:
            # COUPLED: BC reads the noised main-pass h_t; one combined vjp.
            loss_bc = self._run_bc_loss(
                self.dyn.agent_out_ptr_cpu(), actions, rewards, bins,
                policy_weight, reward_weight,
            )
            self.dyn.set_grad_h(ghp, Self.BF)
            var gzh_t = TileTensor(gzh, row_major[Self.BF, Self.ND]())
            self.dyn.vjp["cpu", Self.BF](gzh_t, gzt_t)
            self.te.accumulate_grad["cpu", Self.B, Self.T](
                self.dyn.grad_agent_in_ptr_cpu()
            )
            return Tuple(loss_v, loss_bc)

        # DECOUPLED clean BC.
        # 3. video vjp ONLY (zero the agent-token grad), using the video caches.
        for i in range(Self.BF * Self.AGD):
            self.grad_h[i] = Scalar[DT](0.0)
        self.dyn.set_grad_h(ghp, Self.BF)
        var gzh_t = TileTensor(gzh, row_major[Self.BF, Self.ND]())
        self.dyn.vjp["cpu", Self.BF](gzh_t, gzt_t)

        # 4. dedicated CLEAN forward on z1 (σ=1) → un-noised h_t.
        self.dyn.set_indices(
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.clean_sig.unsafe_ptr()
            ),
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.clean_step.unsafe_ptr()
            ),
            Self.BF,
        )
        self.dyn.set_agent_in(agp, Self.BF)
        # near-clean input z̃ = σ_bc·z1 + (1−σ_bc)·z0 at the highest TRAINED σ
        var sig_bc = Float64(Self.KMAX - 1) / Float64(Self.KMAX)   # 0.75
        for i in range(Self.BF * Self.ND):
            self.bc_in[i] = Scalar[DT](
                sig_bc * Float64(z1[i]) + (1.0 - sig_bc) * Float64(z0[i])
            )
        var z1_t = TileTensor(
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.bc_in.unsafe_ptr()
            ),
            row_major[Self.BF, Self.ND](),
        )
        var zh_t = TileTensor(zh, row_major[Self.BF, Self.ND]())
        self.dyn.forward["cpu", Self.BF](z1_t, output=zh_t)

        # 5. BC loss on the clean h_t → grad_h + head grads
        loss_bc = self._run_bc_loss(
            self.dyn.agent_out_ptr_cpu(), actions, rewards, bins,
            policy_weight, reward_weight,
        )

        # 6. BC vjp through the clean forward (zero flow grad), accumulating
        #    into the dynamics params; then the task-embedder grad.
        self.dyn.set_grad_h(ghp, Self.BF)
        var gzero_t = TileTensor(
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.gzero.unsafe_ptr()
            ),
            row_major[Self.BF, Self.ND](),
        )
        self.dyn.vjp["cpu", Self.BF](gzero_t, gzt_t)
        self.te.accumulate_grad["cpu", Self.B, Self.T](
            self.dyn.grad_agent_in_ptr_cpu()
        )

        return Tuple(loss_v, loss_bc)

    # ── imagination RL (Phase 4) ─────────────────────────────────────────
    def snapshot_prior(mut self) raises:
        """Freeze the current policy head as the behavioral prior π_prior (the
        PMPO reverse-KL anchor). Call once before imagination training starts."""
        polyak_module["cpu", Self.PH](self.ph, self.ph_prior, Scalar[DT](1.0))

    def imag_policy_logits_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        """Per-state policy logits [BF, PLOG] from the last `imag_train_step`
        (greedy action = argmax of the dist-0 block)."""
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.im_plog.unsafe_ptr()
        )

    def imag_train_step(
        mut self,
        ctx: UnsafePointer[Scalar[DT], MutAnyOrigin],       # [B, NCTX, ND]
        u01: UnsafePointer[Scalar[DT], MutAnyOrigin],       # [B, T] action rng
        znoise: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [B, T, ND] ODE seeds
        task_ids: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B]
        bins: UnsafePointer[Scalar[DT], MutAnyOrigin],      # [NBINS]
        gamma: Scalar[DT] = Scalar[DT](0.997),
        lam: Scalar[DT] = Scalar[DT](0.95),
        alpha: Scalar[DT] = Scalar[DT](0.5),
        beta: Scalar[DT] = Scalar[DT](0.3),
        policy_weight: Scalar[DT] = Scalar[DT](1.0),
        value_weight: Scalar[DT] = Scalar[DT](1.0),
        use_continue: Bool = False,             # discount with the continue head
        dctx: Optional[DeviceContext] = None,   # required when DYN_TARGET="gpu"
    ) raises -> Tuple[Float64, Float64]:
        """One imagination-RL step (paper §3.3). Generates an on-policy rollout
        inside the FROZEN world model, then trains ONLY the policy + value heads:
          • value head — TD-λ twohot CE vs sg(R_t^λ)   (eq. 10)
          • policy head — PMPO, sign-of-advantage + reverse-KL prior (eq. 11).
        Fills the grads of `ph` and `vh` ONLY (dyn / te / rh / ph_prior get no
        grad ⇒ frozen under a fresh heads-only optimizer). Returns
        (value_loss, policy_loss). Caller `zero_grad`s + `step`s the optimizer.
        """
        comptime assert Self.ADIM == Self.NACT, (
            "imag_train_step needs ADIM = NACT (one-hot action conditioning)"
        )
        comptime assert Self.NMTP >= 1, "need at least the dist-0 MTP block"
        var agp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.agent_in.unsafe_ptr()
        )
        var im_h_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.im_h.unsafe_ptr()
        )
        var im_act_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.im_act.unsafe_ptr()
        )
        var im_rew_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.im_rew.unsafe_ptr()
        )
        var im_val_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.im_val.unsafe_ptr()
        )
        var im_con_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.im_con.unsafe_ptr()
        )
        var im_ret_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.im_ret.unsafe_ptr()
        )

        # 1. task embeddings → agent tokens
        self.te.embed_into["cpu", Self.B, Self.T](task_ids, agp)

        # 2. imagined rollout (frozen transformer + heads, forward-only). The
        #    dynamics forward runs on DYN_TARGET (GPU = the heavy compute); the
        #    heads + everything below stay on host.
        comptime if Self.DYN_TARGET == "cpu":
            imagine_rollout[
                Self.DYN, Self.PH, Self.VH, Self.RH, Self.B, Self.T, Self.NSP,
                Self.DSP, Self.KMAX, Self.KI, Self.NCTX, Self.AGD, Self.NACT,
                Self.NBINS, Self.NMTP, "cpu",
            ](
                self.dyn, self.ph, self.vh, self.rh, ctx, agp, u01, znoise, bins,
                im_h_p, im_act_p, im_rew_p, im_val_p,
            )
        else:
            imagine_rollout[
                Self.DYN, Self.PH, Self.VH, Self.RH, Self.B, Self.T, Self.NSP,
                Self.DSP, Self.KMAX, Self.KI, Self.NCTX, Self.AGD, Self.NACT,
                Self.NBINS, Self.NMTP, "gpu",
            ](
                self.dyn, self.ph, self.vh, self.rh, ctx, agp, u01, znoise, bins,
                im_h_p, im_act_p, im_rew_p, im_val_p, dctx=dctx,
            )

        # 3. continue factor con_t. use_continue=False ⇒ constant γ (no
        #    termination). use_continue=True ⇒ con_t = γ·ĉ_t where ĉ_t is the
        #    (frozen) continue head's sigmoid read off the rollout's h_t, so the
        #    λ-return truncates at predicted terminal states. Then λ-returns.
        if use_continue:
            var imh_t = TileTensor(im_h_p, row_major[Self.BF, Self.AGD]())
            var clog_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.im_clog.unsafe_ptr()
            )
            var clog_t = TileTensor(clog_p, row_major[Self.BF, 1]())
            self.ch.forward["cpu", Self.BF](imh_t, output=clog_t)
            var chat_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.im_chat.unsafe_ptr()
            )
            continue_pred[Self.BF](clog_p, chat_p)
            for i in range(Self.BF):
                self.im_con[i] = gamma * self.im_chat[i]
        else:
            for i in range(Self.BF):
                self.im_con[i] = gamma
        lambda_returns[Self.B, Self.T](
            im_rew_p, im_val_p, im_con_p, lam, im_ret_p
        )

        # 4. advantages A_t = R_t^λ − v_t (states 0..T-2); actions on the [B,T] grid
        for b in range(Self.B):
            for t in range(Self.TM1):
                self.im_adv[b * Self.TM1 + t] = (
                    self.im_ret[b * Self.TM1 + t] - self.im_val[b * Self.T + t]
                )
            for t in range(Self.T):
                self.im_actbt[b * Self.T + t] = Scalar[DT](0.0)
            for t in range(Self.TM1):
                self.im_actbt[b * Self.T + t] = self.im_act[b * Self.TM1 + t]

        var im_h_t = TileTensor(im_h_p, row_major[Self.BF, Self.AGD]())

        # 5. value loss + backward → vh param grads
        var vlog_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.im_vlog.unsafe_ptr()
        )
        var vlog_t = TileTensor(vlog_p, row_major[Self.BF, Self.NBINS]())
        self.vh.forward["cpu", Self.BF](im_h_t, output=vlog_t)
        var vloss_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.im_vloss.unsafe_ptr()
        )
        value_td_loss_cpu[Self.B, Self.T, Self.NBINS](
            vlog_p, bins, im_ret_p, vloss_p
        )
        var vloss: Float64 = 0.0
        for i in range(Self.B * Self.TM1):
            vloss += Float64(self.im_vloss[i])
        # d_loss = value_weight (reuse the loss buffer as the cotangent)
        for i in range(Self.B * Self.TM1):
            self.im_vloss[i] = value_weight
        var gvlog_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.im_gvlog.unsafe_ptr()
        )
        value_td_loss_backward[Self.B, Self.T, Self.NBINS](
            vlog_p, bins, im_ret_p, vloss_p, gvlog_p
        )
        var gvlog_t = TileTensor(gvlog_p, row_major[Self.BF, Self.NBINS]())
        var vgi_t = TileTensor(
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.grad_h.unsafe_ptr()
            ),
            row_major[Self.BF, Self.AGD](),
        )
        self.vh.vjp["cpu", Self.BF, mode="all"](gvlog_t, vgi_t)

        # 6. policy: current + frozen-prior logits → PMPO (dist-0 block)
        var plog_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.im_plog.unsafe_ptr()
        )
        var prior_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.im_prior.unsafe_ptr()
        )
        var plog_t = TileTensor(plog_p, row_major[Self.BF, Self.PLOG]())
        var prior_t = TileTensor(prior_p, row_major[Self.BF, Self.PLOG]())
        self.ph.forward["cpu", Self.BF](im_h_t, output=plog_t)
        self.ph_prior.forward["cpu", Self.BF](im_h_t, output=prior_t)
        # extract dist-0 logits [BF, NACT]
        for s in range(Self.BF):
            for a in range(Self.NACT):
                self.im_plog0[s * Self.NACT + a] = self.im_plog[s * Self.PLOG + a]
                self.im_prior0[s * Self.NACT + a] = self.im_prior[
                    s * Self.PLOG + a
                ]
        var plog0_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.im_plog0.unsafe_ptr()
        )
        var prior0_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.im_prior0.unsafe_ptr()
        )
        var actbt_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.im_actbt.unsafe_ptr()
        )
        var adv_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.im_adv.unsafe_ptr()
        )
        var ploss = pmpo_policy_loss_cpu[Self.B, Self.T, Self.NACT](
            plog0_p, prior0_p, actbt_p, adv_p, alpha, beta
        )
        var gplog0_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.im_gplog0.unsafe_ptr()
        )
        pmpo_policy_loss_backward[Self.B, Self.T, Self.NACT](
            plog0_p, prior0_p, actbt_p, adv_p, alpha, beta, policy_weight,
            gplog0_p,
        )
        # scatter dist-0 grad into the full [BF, PLOG] grad (other blocks = 0)
        for i in range(Self.BF * Self.PLOG):
            self.im_gplog[i] = Scalar[DT](0.0)
        for s in range(Self.BF):
            for a in range(Self.NACT):
                self.im_gplog[s * Self.PLOG + a] = self.im_gplog0[
                    s * Self.NACT + a
                ]
        var gplog_t = TileTensor(
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.im_gplog.unsafe_ptr()
            ),
            row_major[Self.BF, Self.PLOG](),
        )
        var pgi_t = TileTensor(
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.grad_h_tmp.unsafe_ptr()
            ),
            row_major[Self.BF, Self.AGD](),
        )
        self.ph.vjp["cpu", Self.BF, mode="all"](gplog_t, pgi_t)

        return Tuple(vloss, ploss)
