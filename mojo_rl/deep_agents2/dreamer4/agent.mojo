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
from mojo_rl.nn2.core.target_storage import TargetStorage, assert_tag_for

from .dynamics import Dreamer4Dynamics
from .task_embedder import TaskEmbedder
from .heads import Dreamer4PolicyHead, Dreamer4RewardHead
from .shortcut_loss import dynamics_pretrain_loss
from .bc_loss import bc_mtp_loss


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

    comptime DYN = Dreamer4Dynamics[
        Self.DSP, Self.NSP, Self.D, Self.NH, Self.T, Self.NREG, Self.HID,
        Self.DEPTH, Self.KMAX, Self.USE_MAX, 0, 0, Self.NAGENT,
    ]
    comptime TE = TaskEmbedder[Self.D, Self.NTASK, Self.NAGENT]
    comptime PH = Dreamer4PolicyHead[Self.AGD, Self.HHID, Self.NACT, Self.NMTP]
    comptime RH = Dreamer4RewardHead[Self.AGD, Self.HHID, Self.NBINS, Self.NMTP]

    var dyn: Self.DYN
    var te: Self.TE
    var ph: Self.PH
    var rh: Self.RH

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
    var ts: TargetStorage

    def __init__(out self):
        self.dyn = Self.DYN()
        self.te = Self.TE()
        self.ph = Self.PH()
        self.rh = Self.RH()
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
            "Dreamer4Agent: only CPU is wired in v1 (GPU train_step deferred)"
        )
        var m = Self()
        m.dyn = Self.DYN.make[target=target, INIT=INIT](ctx)
        m.te = Self.TE.make[target=target, INIT=INIT](ctx)
        m.ph = Self.PH.make[target=target, INIT=INIT](ctx)
        m.rh = Self.RH.make[target=target, INIT=INIT](ctx)
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
        var *inputs: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
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
        mut *grad_inputs: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        raise Error("Dreamer4Agent.vjp is unused; call bc_train_step")

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["Dreamer4Agent", target](self.ts.target_tag)
        self.dyn.for_each_param[target, V](prefix + ".dyn", visitor)
        self.te.for_each_param[target, V](prefix + ".te", visitor)
        self.ph.for_each_param[target, V](prefix + ".ph", visitor)
        self.rh.for_each_param[target, V](prefix + ".rh", visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["Dreamer4Agent", target](self.ts.target_tag)
        self.dyn.zero_grad[target]()
        self.te.zero_grad[target]()
        self.ph.zero_grad[target]()
        self.rh.zero_grad[target]()

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
