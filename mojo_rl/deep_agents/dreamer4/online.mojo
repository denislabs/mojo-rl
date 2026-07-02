"""Dreamer 4 ONLINE training driver (facade-takes-tokenizer).

The offline pipeline (`examples/dreamer4/pong_reward_end2end.mojo`) trains the
tokenizer + agent on a pre-collected buffer. This driver instead collects the
data ONLINE from a live `BoxDiscreteActionEnv`, in three stages:

  Stage 0 — warmup collect : random actions fill the ring buffer to `learn_start`.
  Stage 1 — tokenizer pretrain : the Phase-A MAE loop on the warmup buffer, then
            FREEZE the tokenizer (`set_mae_p(0,0)`).
  Stage 2 — online RL loop : act with the policy (a clean-latent forward), step
            the env, append to the ring buffer; every `train_every` steps train
            the action-conditioned world model + reward/continue heads
            (`acwm_train_step` + continue BCE), and every `imag_every` steps run
            an imagination-RL update (`imag_train_step`) on the frozen WM.

`run_online_dreamer4` is a free function (not a method) so it can be parameterized
by the full agent + tokenizer + image dims without bloating `Dreamer4Agent`. The
tokenizer is passed as an argument (its dims would explode the agent's comptime
param list). DYN_TARGET must be "cpu" (the acwm/imag train steps are the CPU WM
path).

This is the driver the online CarRacing lighthouse calls; a stub-env smoke gate
lives in `tests/nn/test_dreamer4_train_online.mojo`.
"""

from std.math import sqrt, log, cos

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.core.env_traits import BoxDiscreteActionEnv
from mojo_rl.core.logger import Logger

from .agent import Dreamer4Agent
from .tokenizer import Dreamer4Tokenizer
from .frame_buffer import Dreamer4FrameBuffer
from .recon_loss import masked_recon_loss
from .perceptual_loss import masked_recon_plus_perceptual_loss
from .patchify import downscale_box, temporal_patchify
from .imag_rl_loss import continue_bce_backward
from .shortcut_loss import _mao, _ilog2
from ..dreamerv3.twohot import symexp_twohot_bins
from ...nn.models.cifar_feature_net import CifarBackbone


struct OnlineRng(Copyable, Movable):
    """Deterministic xorshift64* + Box-Muller (mirrors the end2end example)."""

    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed | 1

    def u64(mut self) -> UInt64:
        var x = self.s
        x ^= x >> 12
        x ^= x << 25
        x ^= x >> 27
        self.s = x
        return x * 0x2545F4914F6CDD1D

    def uniform(mut self) -> Float64:
        return Float64(self.u64() >> 11) * (1.0 / 9007199254740992.0)

    def gauss(mut self) -> Float64:
        var u1 = self.uniform()
        var u2 = self.uniform()
        if u1 < 1e-12:
            u1 = 1e-12
        return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2)


def _window_to_patches[
    B: Int, T: Int, IN_CH: Int, IMG: Int, TGT: Int, PATCH: Int
](
    pix: UnsafePointer[Scalar[DT], MutAnyOrigin],     # [B*T*(IN_CH*IMG*IMG)]
    frames: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B*T*TGT*TGT] scratch
    patches: UnsafePointer[Scalar[DT], MutAnyOrigin], # [B*T*NP*DP]
) raises:
    """Downscale the LATEST stacked frame of each (b,t) obs to TGT×TGT grayscale
    and patchify (mirrors end2end `_window_to_patches`)."""
    comptime IMG_DIM = IN_CH * IMG * IMG
    comptime BATCH = B * T
    for b in range(B):
        for t in range(T):
            var bt = b * T + t
            var fsrc = pix + bt * IMG_DIM + (IN_CH - 1) * IMG * IMG
            downscale_box[IMG, IMG, TGT, TGT](fsrc, frames + bt * TGT * TGT)
    temporal_patchify[BATCH, 1, TGT, TGT, PATCH](frames, patches)


def _frame_to_patches[
    IN_CH: Int, IMG: Int, TGT: Int, PATCH: Int
](
    obs: UnsafePointer[Scalar[DT], MutAnyOrigin],     # [IN_CH*IMG*IMG] single obs
    frame: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [TGT*TGT] scratch
    patches: UnsafePointer[Scalar[DT], MutAnyOrigin], # [NP*DP] out
) raises:
    """Downscale + patchify ONE obs frame (latest stacked channel)."""
    var fsrc = obs + (IN_CH - 1) * IMG * IMG
    downscale_box[IMG, IMG, TGT, TGT](fsrc, frame)
    temporal_patchify[1, 1, TGT, TGT, PATCH](frame, patches)


def _push_frame[
    IN_CH: Int, IMG: Int, TGT: Int, PATCH: Int, T: Int, NP: Int, DP: Int
](
    cur: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [IN_CH*IMG*IMG]
    mut fr1: Tensor,                                # [TGT*TGT] scratch
    mut pa1: Tensor,                                # [NP*DP] scratch
    mut win_patch: Tensor,                          # [T*NP*DP]
    mut win_act: List[Int],                         # [T]
    win_n: Int,
    a_prev: Int,
) raises -> Int:
    """Downscale+patchify `cur` into the front-aligned rolling window (slide left
    when full). Returns the new window length."""
    _frame_to_patches[IN_CH, IMG, TGT, PATCH](
        cur, _mao(fr1.data.unsafe_ptr()), _mao(pa1.data.unsafe_ptr())
    )
    var f: Int
    var new_n = win_n
    if win_n < T:
        f = win_n
        new_n = win_n + 1
    else:
        for i in range(T - 1):
            for k in range(NP * DP):
                win_patch.data[i * NP * DP + k] = win_patch.data[
                    (i + 1) * NP * DP + k
                ]
            win_act[i] = win_act[i + 1]
        f = T - 1
    for k in range(NP * DP):
        win_patch.data[f * NP * DP + k] = pa1.data[k]
    win_act[f] = a_prev
    return new_n


def _build_shortcut_schedule[
    B: Int, T: Int, B_SELF: Int, KMAX: Int, EMAX: Int, ND: Int
](
    mut rng: OnlineRng,
    sigma: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [B*T]
    sig_idx: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B*T]
    step_idx: UnsafePointer[Scalar[DT], MutAnyOrigin], # [B*T]
    z0: UnsafePointer[Scalar[DT], MutAnyOrigin],       # [B*T*ND]
):
    """The shortcut-forcing per-(b,t) signal-level / step-size schedule + the
    τ=0 Gaussian noise seed (lifted from end2end Phase B)."""
    comptime B_EMP = B - B_SELF
    for b in range(B):
        var is_self = b >= B_EMP
        for t in range(T):
            var bt = b * T + t
            var stp = EMAX
            if is_self:
                stp = Int(rng.uniform() * Float64(EMAX))
            var K = 1 << stp
            var j = Int(rng.uniform() * Float64(K))
            if j >= K:
                j = K - 1
            var scale = KMAX // K
            sigma[bt] = Scalar[DT](Float64(j) / Float64(K))
            sig_idx[bt] = Scalar[DT](Float64(j * scale))
            step_idx[bt] = Scalar[DT](Float64(stp))
    for i in range(B * T * ND):
        z0[i] = Scalar[DT](rng.gauss())


def run_online_dreamer4[
    # ── agent dims (same order as Dreamer4Agent) ──
    DSP: Int, NSP: Int, D: Int, NH: Int, T: Int, NREG: Int, HID: Int,
    DEPTH: Int, KMAX: Int, NAGENT: Int, NTASK: Int, HHID: Int, NACT: Int,
    NBINS: Int, NMTP: Int, B: Int, B_SELF: Int, USE_MAX: Bool, ADIM: Int,
    AHID: Int, K_IMAG: Int, NCTX: Int,
    # ── env / logger ──
    E: BoxDiscreteActionEnv, L: Logger,
    # ── image / tokenizer / buffer dims ──
    IN_CH: Int, IMG: Int, TGT: Int, PATCH: Int, TNP: Int, CAP: Int,
    TOK_D: Int, TOK_NH: Int, TOK_HID: Int, TOK_DEPTH: Int,
    TOK_PMIN: Float64, TOK_PMAX: Float64, TOK_SEED: UInt64,
](
    mut agent: Dreamer4Agent[
        DSP, NSP, D, NH, T, NREG, HID, DEPTH, KMAX, NAGENT, NTASK, HHID,
        NACT, NBINS, NMTP, B, B_SELF, USE_MAX, ADIM, AHID, K_IMAG, NCTX, "cpu",
    ],
    mut tok: Dreamer4Tokenizer[
        PATCH * PATCH, TOK_D, TOK_NH, T, NSP,
        TNP, DSP, TOK_HID, TOK_DEPTH,
        TOK_PMIN, TOK_PMAX, TOK_SEED,
    ],
    # Frozen perceptual backbone for the tokenizer's MSE+w·perceptual recon loss
    # (paper eq. 5). Unused when `perc_weight == 0` — pass any instance then.
    mut backbone: CifarBackbone[TGT, TGT],
    mut env: E,
    mut logger: L,
    warmup_steps: Int,
    tok_pretrain_steps: Int,
    total_env_steps: Int,
    train_every: Int,
    imag_every: Int,
    eval_every: Int,
    lr_tok: Scalar[DT] = Scalar[DT](2e-3),
    lr_agent: Scalar[DT] = Scalar[DT](1e-3),
    lr_cont: Scalar[DT] = Scalar[DT](3e-3),
    lr_imag: Scalar[DT] = Scalar[DT](1e-2),
    perc_weight: Float64 = 0.0,
    eval_max_steps: Int = 1000,
    imag_gamma: Scalar[DT] = Scalar[DT](0.997),
    seed: UInt64 = 20260701,
) raises -> Tuple[Float64, Float64, Float64, Float64, Float64]:
    """Returns (last_tok_recon, last_wm_video, last_wm_bc, last_imag_value,
    last_eval_return) for a finiteness check / logging. `last_eval_return` is the
    most recent greedy-eval episode return (0 if `eval_every <= 0`)."""
    comptime assert TNP == (TGT // PATCH) * (TGT // PATCH), (
        "TNP must equal (TGT//PATCH)^2 (the tokenizer's patch count)"
    )
    comptime BATCH = B * T
    comptime NP = TNP
    comptime DP = PATCH * PATCH
    comptime ND = NSP * DSP
    comptime ZN = BATCH * ND
    comptime AGD = NAGENT * D
    comptime IMG_DIM = IN_CH * IMG * IMG
    comptime EMAX = _ilog2(KMAX)

    comptime assert ADIM == NACT, "run_online_dreamer4 needs ADIM == NACT"

    var rng = OnlineRng(seed)
    var buf = Dreamer4FrameBuffer[IN_CH, IMG, IMG, NACT, CAP]()

    var topt = Adam(lr=lr_tok)
    var aopt = Adam(lr=lr_agent)
    var copt = Adam(lr=lr_cont)
    var iopt = Adam(lr=lr_imag)

    # ── shared scratch ──
    var frames = Tensor.alloc(BATCH * TGT * TGT)
    var patches = Tensor.alloc(BATCH * NP * DP)
    var pred = Tensor.alloc(BATCH * NP * DP)
    var gpred = Tensor.alloc(BATCH * NP * DP)
    var gin = Tensor.alloc(BATCH * NP * DP)
    var gperc = Tensor.alloc(BATCH * NP * DP)   # perceptual-grad scratch (Stage 1)
    var z1 = Tensor.alloc(ZN)
    var z0n = Tensor.alloc(ZN)
    var sigma = Tensor.alloc(BATCH)
    var sig_idx = Tensor.alloc(BATCH)
    var step_idx = Tensor.alloc(BATCH)
    var task_ids = Tensor.alloc(B)
    var act_idx = Tensor.alloc(BATCH)
    var rew = Tensor.alloc(BATCH)
    var done_b = Tensor.alloc(BATCH)
    var cont_tgt = Tensor.alloc(BATCH)
    var bins = Tensor.alloc(NBINS)
    symexp_twohot_bins[NBINS](_mao(bins.data.unsafe_ptr()), lo=Scalar[DT](-9.0))
    for b in range(B):
        task_ids.data[b] = Scalar[DT](0.0)

    var pix = Tensor.alloc(BATCH * IMG_DIM)
    var act_oh = Tensor.alloc(BATCH * NACT)

    # current observation (DT copy of the env's obs list)
    var cur = List[Scalar[DT]](length=IMG_DIM, fill=Scalar[DT](0.0))
    var ob0 = env.reset_obs_list()
    for i in range(IMG_DIM):
        cur[i] = Scalar[DT](Float64(ob0[i]))

    # ── Stage 0: warmup collect (random actions) ────────────────────────
    print("[dreamer4-online] Stage 0: warmup collect —", warmup_steps, "env steps")
    var collected = 0
    while collected < warmup_steps:
        var a = Int(rng.uniform() * Float64(NACT))
        if a >= NACT:
            a = NACT - 1
        var res = env.step_obs(a)
        var r = Scalar[DT](Float64(res[1]))
        var d = res[2]
        buf.add_step_fp32_list(cur, a, d, r)
        collected += 1
        if collected % 500 == 0:
            print("  [warmup]", collected, "/", warmup_steps, " buf=", buf.count())
        if d:
            var no = env.reset_obs_list()
            for i in range(IMG_DIM):
                cur[i] = Scalar[DT](Float64(no[i]))
        else:
            for i in range(IMG_DIM):
                cur[i] = Scalar[DT](Float64(res[0][i]))
    logger.log_scalar(String("online/warmup_frames"), Float64(buf.count()), 0)

    # ── Stage 1: tokenizer pretrain → freeze ────────────────────────────
    print("[dreamer4-online] Stage 1: tokenizer pretrain —",
          tok_pretrain_steps, "steps (perc_weight=", perc_weight, ")")
    var last_tok_loss: Float64 = 0.0
    for s in range(tok_pretrain_steps):
        buf.sample_reward_window_batch[B, T](
            pix.data.unsafe_ptr(), act_oh.data.unsafe_ptr(),
            rew.data.unsafe_ptr(), done_b.data.unsafe_ptr(),
        )
        _window_to_patches[B, T, IN_CH, IMG, TGT, PATCH](
            _mao(pix.data.unsafe_ptr()), _mao(frames.data.unsafe_ptr()),
            _mao(patches.data.unsafe_ptr()),
        )
        topt.zero_grad["cpu"](tok, None)
        tok.forward["cpu", BATCH](TensorRefs[1](patches), pred, None)
        if perc_weight > 0.0:
            # paper eq. 5: MSE + w·perceptual (frozen CIFAR backbone, BN-eval).
            var lv = masked_recon_plus_perceptual_loss[
                BATCH, 1, TGT, TGT, PATCH
            ](
                _mao(pred.data.unsafe_ptr()), _mao(patches.data.unsafe_ptr()),
                tok.mae_mask_ptr(), backbone, perc_weight,
                _mao(gpred.data.unsafe_ptr()), _mao(gperc.data.unsafe_ptr()),
            )
            last_tok_loss = lv[0] + perc_weight * lv[1]
        else:
            last_tok_loss = masked_recon_loss[NP, DP, BATCH](
                _mao(pred.data.unsafe_ptr()), _mao(patches.data.unsafe_ptr()),
                tok.mae_mask_ptr(), _mao(gpred.data.unsafe_ptr()),
            )
        tok.vjp["cpu", BATCH](
            TensorRefs[1](patches), gpred, TensorRefs[1](gin), None
        )
        topt.step["cpu"](tok, None)
        tok.advance_rng()
        if s % 50 == 0:
            print("  [tok]", s, "/", tok_pretrain_steps, " recon=", last_tok_loss)
    tok.set_mae_p(0.0, 0.0)  # FREEZE
    print("  tokenizer frozen (recon=", last_tok_loss, ")")
    logger.log_scalar(String("online/tok_recon_loss"), last_tok_loss, 0)

    # ── Stage 2: online RL loop ─────────────────────────────────────────
    # rolling window of the last ≤T encoded obs frames (front-aligned) +
    # the actions leading into each frame.
    var win_patch = Tensor.alloc(T * NP * DP)
    var win_z = Tensor.alloc(T * ND)
    var fr1 = Tensor.alloc(TGT * TGT)
    var pa1 = Tensor.alloc(NP * DP)
    var win_act = List[Int](length=T, fill=-1)
    var act_hist = List[Scalar[DT]](length=T * ADIM, fill=Scalar[DT](0.0))
    var win_n = 0
    var last_action = -1

    var ctx = Tensor.alloc(B * NCTX * ND)
    var u01 = Tensor.alloc(B * T)
    var znoise = Tensor.alloc(B * T * ND)

    var ob1 = env.reset_obs_list()
    for i in range(IMG_DIM):
        cur[i] = Scalar[DT](Float64(ob1[i]))
    var did_imag = False
    var last_v: Float64 = 0.0
    var last_p: Float64 = 0.0
    var last_video: Float64 = 0.0
    var last_bc: Float64 = 0.0
    var ep_ret: Float64 = 0.0            # training-episode return (explore=True)
    var last_eval_return: Float64 = 0.0
    # greedy-eval scratch (its own window; an eval episode interrupts the
    # in-progress training episode, so we reset both env + training window after).
    var ewin_patch = Tensor.alloc(T * NP * DP)
    var ewin_z = Tensor.alloc(T * ND)
    var efr1 = Tensor.alloc(TGT * TGT)
    var epa1 = Tensor.alloc(NP * DP)
    var ewin_act = List[Int](length=T, fill=-1)
    var eact_hist = List[Scalar[DT]](length=T * ADIM, fill=Scalar[DT](0.0))
    var ecur = List[Scalar[DT]](length=IMG_DIM, fill=Scalar[DT](0.0))
    print("[dreamer4-online] Stage 2: online RL —", total_env_steps, "env steps")
    for step in range(total_env_steps):
        if step > 0 and step % 500 == 0:
            print("  [rl]", step, "/", total_env_steps,
                  " wm_video=", last_video, " wm_bc=", last_bc,
                  " imag_v=", last_v, " eval=", last_eval_return)
        win_n = _push_frame[IN_CH, IMG, TGT, PATCH, T, NP, DP](
            _mao(cur.unsafe_ptr()), fr1, pa1, win_patch, win_act,
            win_n, last_action,
        )
        # encode the window (B'=1, T frames) → win_z [T*ND]
        # (zero the unused tail frames so the encode is deterministic)
        for i in range(win_n * NP * DP, T * NP * DP):
            win_patch.data[i] = Scalar[DT](0.0)
        tok.enc.forward["cpu", T](TensorRefs[1](win_patch), win_z, None)
        # action history one-hots leading into frames 1..win_n-1
        for i in range(T * ADIM):
            act_hist[i] = Scalar[DT](0.0)
        for fr in range(1, win_n):
            var ap = win_act[fr]
            if ap >= 0 and ap < ADIM:
                act_hist[fr * ADIM + ap] = Scalar[DT](1.0)
        var a = agent.act_from_latents(
            _mao(win_z.data.unsafe_ptr()), win_n,
            _mao(act_hist.unsafe_ptr()), 0, True, rng.uniform(),
        )

        var res = env.step_obs(a)
        var r = Scalar[DT](Float64(res[1]))
        var d = res[2]
        buf.add_step_fp32_list(cur, a, d, r)
        last_action = a
        ep_ret += Float64(r)
        if d:
            if logger.is_active():
                logger.log_scalar(String("online/train_return"), ep_ret, step)
            ep_ret = 0.0
            var no = env.reset_obs_list()
            for i in range(IMG_DIM):
                cur[i] = Scalar[DT](Float64(no[i]))
            win_n = 0
            last_action = -1
        else:
            for i in range(IMG_DIM):
                cur[i] = Scalar[DT](Float64(res[0][i]))

        # ── world-model + heads update ──
        if step % train_every == 0 and buf.count() >= BATCH:
            buf.sample_reward_window_batch[B, T](
                pix.data.unsafe_ptr(), act_oh.data.unsafe_ptr(),
                rew.data.unsafe_ptr(), done_b.data.unsafe_ptr(),
            )
            _window_to_patches[B, T, IN_CH, IMG, TGT, PATCH](
                _mao(pix.data.unsafe_ptr()), _mao(frames.data.unsafe_ptr()),
                _mao(patches.data.unsafe_ptr()),
            )
            tok.enc.forward["cpu", BATCH](TensorRefs[1](patches), z1, None)
            _build_shortcut_schedule[B, T, B_SELF, KMAX, EMAX, ND](
                rng, _mao(sigma.data.unsafe_ptr()),
                _mao(sig_idx.data.unsafe_ptr()),
                _mao(step_idx.data.unsafe_ptr()), _mao(z0n.data.unsafe_ptr()),
            )
            # dataset action ids from the sampled one-hots
            for bt in range(BATCH):
                var best = 0
                var bv = act_oh.data[bt * NACT]
                for c in range(1, NACT):
                    if act_oh.data[bt * NACT + c] > bv:
                        bv = act_oh.data[bt * NACT + c]
                        best = c
                act_idx.data[bt] = Scalar[DT](Float64(best))
            aopt.zero_grad["cpu"](agent, None)
            var losses = agent.acwm_train_step(
                _mao(z1.data.unsafe_ptr()), _mao(z0n.data.unsafe_ptr()),
                _mao(sigma.data.unsafe_ptr()), _mao(sig_idx.data.unsafe_ptr()),
                _mao(step_idx.data.unsafe_ptr()), step >= 2 * train_every,
                _mao(task_ids.data.unsafe_ptr()), _mao(act_idx.data.unsafe_ptr()),
                _mao(rew.data.unsafe_ptr()), _mao(bins.data.unsafe_ptr()),
            )
            aopt.step["cpu"](agent, None)
            last_video = losses[0]
            last_bc = losses[1]

            # continue head off the clean h_t left by acwm's clean forward
            var ht = agent.agent_out_ptr()
            var ht_t = Tensor.alloc(BATCH * AGD)
            for i in range(BATCH * AGD):
                ht_t.data[i] = ht[i]
            var clog = Tensor.alloc(BATCH)
            var gcl = Tensor.alloc(BATCH)
            var gci = Tensor.alloc(BATCH * AGD)
            for i in range(BATCH):
                cont_tgt.data[i] = Scalar[DT](1.0) - done_b.data[i]
            copt.zero_grad["cpu"](agent.ch, None)
            agent.ch.forward["cpu", BATCH](TensorRefs[1](ht_t), clog, None)
            continue_bce_backward[BATCH](
                _mao(clog.data.unsafe_ptr()), _mao(cont_tgt.data.unsafe_ptr()),
                Scalar[DT](1.0), _mao(gcl.data.unsafe_ptr()),
            )
            agent.ch.vjp["cpu", BATCH](
                TensorRefs[1](ht_t), gcl, TensorRefs[1](gci), None
            )
            copt.step["cpu"](agent.ch, None)
            if logger.is_active():
                logger.log_scalar(String("online/wm_video"), losses[0], step)
                logger.log_scalar(String("online/wm_bc"), losses[1], step)

        # ── imagination-RL update (frozen WM) ──
        if imag_every > 0 and step % imag_every == 0 and buf.count() >= BATCH:
            agent.snapshot_prior()
            buf.sample_reward_window_batch[B, T](
                pix.data.unsafe_ptr(), act_oh.data.unsafe_ptr(),
                rew.data.unsafe_ptr(), done_b.data.unsafe_ptr(),
            )
            _window_to_patches[B, T, IN_CH, IMG, TGT, PATCH](
                _mao(pix.data.unsafe_ptr()), _mao(frames.data.unsafe_ptr()),
                _mao(patches.data.unsafe_ptr()),
            )
            tok.enc.forward["cpu", BATCH](TensorRefs[1](patches), z1, None)
            for b in range(B):
                for i in range(ND):
                    ctx.data[b * NCTX * ND + i] = z1.data[(b * T + 0) * ND + i]
            for i in range(B * T):
                u01.data[i] = Scalar[DT](rng.uniform())
            for i in range(B * T * ND):
                znoise.data[i] = Scalar[DT](rng.gauss())
            iopt.zero_grad["cpu"](agent, None)
            var il = agent.imag_train_step(
                _mao(ctx.data.unsafe_ptr()), _mao(u01.data.unsafe_ptr()),
                _mao(znoise.data.unsafe_ptr()), _mao(task_ids.data.unsafe_ptr()),
                _mao(bins.data.unsafe_ptr()), use_continue=True,
                gamma=imag_gamma,
            )
            iopt.step["cpu"](agent, None)
            did_imag = True
            last_v = il[0]
            last_p = il[1]
            if logger.is_active():
                logger.log_scalar(String("online/imag_value"), il[0], step)
                logger.log_scalar(String("online/imag_policy"), il[1], step)

        # ── greedy-eval episode (measurable return; interrupts training ep) ──
        if eval_every > 0 and step % eval_every == 0 and step > 0:
            var eob = env.reset_obs_list()
            for i in range(IMG_DIM):
                ecur[i] = Scalar[DT](Float64(eob[i]))
            var ewin_n = 0
            var elast = -1
            var eret: Float64 = 0.0
            for _es in range(eval_max_steps):
                ewin_n = _push_frame[IN_CH, IMG, TGT, PATCH, T, NP, DP](
                    _mao(ecur.unsafe_ptr()), efr1, epa1, ewin_patch, ewin_act,
                    ewin_n, elast,
                )
                for i in range(ewin_n * NP * DP, T * NP * DP):
                    ewin_patch.data[i] = Scalar[DT](0.0)
                tok.enc.forward["cpu", T](
                    TensorRefs[1](ewin_patch), ewin_z, None
                )
                for i in range(T * ADIM):
                    eact_hist[i] = Scalar[DT](0.0)
                for fr in range(1, ewin_n):
                    var ap = ewin_act[fr]
                    if ap >= 0 and ap < ADIM:
                        eact_hist[fr * ADIM + ap] = Scalar[DT](1.0)
                var ea = agent.act_from_latents(
                    _mao(ewin_z.data.unsafe_ptr()), ewin_n,
                    _mao(eact_hist.unsafe_ptr()), 0, False, 0.0,
                )
                var eres = env.step_obs(ea)
                eret += Float64(eres[1])
                elast = ea
                if eres[2]:
                    break
                for i in range(IMG_DIM):
                    ecur[i] = Scalar[DT](Float64(eres[0][i]))
            last_eval_return = eret
            print("  [eval] step", step, " greedy return =", eret)
            if logger.is_active():
                logger.log_scalar(String("online/eval_return"), eret, step)
            # resume training on a fresh episode (eval consumed the env)
            var rob = env.reset_obs_list()
            for i in range(IMG_DIM):
                cur[i] = Scalar[DT](Float64(rob[i]))
            win_n = 0
            last_action = -1
            ep_ret = 0.0

    logger.flush()
    _ = did_imag
    _ = last_p
    return (last_tok_loss, last_video, last_bc, last_v, last_eval_return)
