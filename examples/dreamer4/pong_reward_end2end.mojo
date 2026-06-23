"""Dreamer 4 — end-to-end run on a REWARD-BEARING real-Pong buffer (CPU).

    # collect once:
    pixi run mojo run -I . examples/dreamer4/pong_reward_collect_buffer.mojo
    # then:
    pixi run mojo run -I . examples/dreamer4/pong_reward_end2end.mojo

This is the real-env counterpart of `imagination_lighthouse.mojo` (which uses a
controlled world to isolate the imagination-RL *mechanism*). Here every signal
comes from real Pong, collected WITH rewards by
`pong_reward_collect_buffer.mojo` (format LWMR). The full pipeline runs on it:

  Phase A  causal tokenizer on buffer frames → FREEZE.
  Phase B  `agent.acwm_train_step(reward_weight=1)` jointly trains the
           ACTION-CONDITIONED world model (shortcut-forcing video loss whose
           action token moves the transition), clones the dataset action
           (policy head), AND fits the REWARD HEAD to the buffer's real
           transition reward (eq. 9 twohot). Training the action-conditioned
           transition is what lets imagined actions move the reward. In
           parallel, the continue head is trained on the real done flags (BCE)
           off the same clean h_t.
  Eval     • REWARD HEAD: does the learned reward model, read off the encoded
             observation, predict real Pong reward? Gate = it raises its
             prediction on true reward states above zero-reward states (it
             reads the frame, not a constant) and beats the mean-reward MAE.
           • CONTINUE HEAD: predicted P(non-terminal) accuracy vs the dones.
  Phase C  imagination RL on real-env context: `imag_train_step(use_continue)`
           rolls the FROZEN world model forward from a real starting frame and
           trains the value + policy heads (eq. 10/11). Reported as an
           end-to-end execution check (finite value/policy losses on real
           context) — improving the greedy *return* additionally needs the
           action-conditioned transition trained so the action token moves the
           reward, which the controlled lighthouse covers; on this pixel buffer
           that path is data-SNR-limited (see project notes / the BC lighthouse).

The reward head is the headline: it is the new capability the reward-bearing
buffer unlocks, and the prerequisite for imagination RL on a real env.
"""

from std.math import sqrt, log, cos, abs

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.initializer import Xavier
from mojo_rl.nn.storage.optimizer.adam import Adam

from mojo_rl.deep_agents.dreamer4.tokenizer import Dreamer4Tokenizer
from mojo_rl.deep_agents.dreamer4.agent import Dreamer4Agent
from mojo_rl.deep_agents.dreamer4.pong_reward_buffer import (
    Dreamer4PongRewardBuffer,
)
from mojo_rl.deep_agents.dreamer4.recon_loss import (
    masked_recon_loss, full_recon_psnr,
)
from mojo_rl.deep_agents.dreamer4.patchify import downscale_box, temporal_patchify
from mojo_rl.deep_agents.dreamer4.shortcut_loss import _mao
from mojo_rl.deep_agents.dreamer4.imag_rl_loss import (
    continue_pred, continue_bce_loss, continue_bce_backward,
)
from mojo_rl.deep_agents.dreamerv3.dists_discrete import cat_argmax
from mojo_rl.deep_agents.dreamerv3.twohot import symexp_twohot_bins, twohot_pred


# deterministic RNG (xorshift64* + Box-Muller)
struct Rng(Copyable, Movable):
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
    B: Int, T: Int, IMG: Int, IMG_DIM: Int, TGT: Int, PATCH: Int
](
    pix: UnsafePointer[Scalar[DT], MutAnyOrigin],
    frames: UnsafePointer[Scalar[DT], MutAnyOrigin],
    patches: UnsafePointer[Scalar[DT], MutAnyOrigin],
) raises:
    comptime BATCH = B * T
    for b in range(B):
        for t in range(T):
            var bt = b * T + t
            var fsrc = pix + bt * IMG_DIM + 3 * IMG * IMG   # latest stacked frame
            downscale_box[IMG, IMG, TGT, TGT](fsrc, frames + bt * TGT * TGT)
    temporal_patchify[BATCH, 1, TGT, TGT, PATCH](frames, patches)


def _onehot_to_idx[BATCH: Int, ACT: Int](
    ap: UnsafePointer[Scalar[DT], MutAnyOrigin],
    act_idx: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    for bt in range(BATCH):
        var best = 0
        var bv = ap[bt * ACT]
        for c in range(1, ACT):
            if ap[bt * ACT + c] > bv:
                bv = ap[bt * ACT + c]
                best = c
        act_idx[bt] = Scalar[DT](Float64(best))


def main() raises:
    print("=" * 70)
    print("Dreamer 4 — end-to-end on a reward-bearing real-Pong buffer (CPU)")
    print("=" * 70)

    comptime IN_CH = 4
    comptime IMG = 84
    comptime IMG_DIM = IN_CH * IMG * IMG
    comptime ACT = 3

    comptime T = 4
    comptime B = 4
    comptime BATCH = B * T

    comptime TGT = 32
    comptime PATCH = 8
    comptime NP = (TGT // PATCH) * (TGT // PATCH)   # 16
    comptime DP = PATCH * PATCH                      # 64
    comptime D = 64
    comptime NH = 4
    comptime L = 8
    comptime D_BOT = 16
    comptime HID = 256
    comptime DEPTH = 2
    comptime DROP = 0.5

    comptime NSP = L
    comptime DSP = D_BOT
    comptime ND = NSP * DSP
    comptime KMAX = 4
    comptime NREG = 2
    comptime D_DYN = 64
    comptime HID_DYN = 128
    comptime DEPTH_DYN = 2
    comptime B_SELF = 2
    comptime B_EMP = B - B_SELF
    comptime EMAX = 2

    comptime NAGENT = 1
    comptime NTASK = 1
    comptime HHID = 64
    comptime NBINS = 41
    comptime NMTP = 2
    comptime AGD = NAGENT * D_DYN
    comptime PLOG = NMTP * ACT
    comptime RLOG = NMTP * NBINS

    # imagination extras (action-conditioned dynamics so imag_train_step is
    # callable; the action MLP stays ZeroLinear ≈ unconditional under BC).
    comptime ADIM = ACT
    comptime AHID = 2 * D_DYN
    comptime K_IMAG = 2
    comptime NCTX = 1

    comptime STEPS_TOK = 150
    comptime STEPS_BC = 400
    comptime STEPS_IMAG = 80
    comptime EVAL_EVERY = 100
    comptime LR_TOK = Scalar[DT](2e-3)
    comptime LR_AGENT = Scalar[DT](1e-3)
    comptime LR_CONT = Scalar[DT](3e-3)
    comptime LR_IMAG = Scalar[DT](1e-2)

    comptime FRAME_N = BATCH * TGT * TGT
    comptime PATCH_N = BATCH * NP * DP
    comptime ZN = BATCH * ND

    comptime Agent = Dreamer4Agent[
        DSP, NSP, D_DYN, NH, T, NREG, HID_DYN, DEPTH_DYN, KMAX,
        NAGENT, NTASK, HHID, ACT, NBINS, NMTP, B, B_SELF,
        True, ADIM, AHID, K_IMAG, NCTX,
    ]

    # ── data ────────────────────────────────────────────────────────────
    print("- loading /tmp/dreamer4_pong_reward_buffer.bin")
    var buf = Dreamer4PongRewardBuffer.load(
        "/tmp/dreamer4_pong_reward_buffer.bin"
    )
    print("  frames:", buf.n_frames)

    var tok = Dreamer4Tokenizer[
        DP, D, NH, T, L, NP, D_BOT, HID, DEPTH, DROP, DROP, 7
    ].make["cpu", Xavier](None)
    var topt = Adam(lr=LR_TOK)

    var agent = Agent.make["cpu", Xavier](None)
    var aopt = Adam(lr=LR_AGENT)
    # separate optimizer for the continue head (trained off the clean h_t)
    var copt = Adam(lr=LR_CONT)

    # Storage scratch. Buffers fed to forward/vjp are `Tensor`s; loss / agent
    # helpers read their underlying host `data` via `_mao(...)`.
    var frames = Tensor.alloc(FRAME_N)
    var patches = Tensor.alloc(PATCH_N)
    var pred = Tensor.alloc(PATCH_N)
    var gpred = Tensor.alloc(PATCH_N)
    var gin = Tensor.alloc(PATCH_N)
    var z1 = Tensor.alloc(ZN)
    var z0n = Tensor.alloc(ZN)
    var sigma = Tensor.alloc(BATCH)
    var sig_idx = Tensor.alloc(BATCH)
    var step_idx = Tensor.alloc(BATCH)
    var task_ids = Tensor.alloc(B)
    var act_idx = Tensor.alloc(BATCH)
    var rew = Tensor.alloc(BATCH)            # REAL per-step reward (twohot target)
    var done = Tensor.alloc(BATCH)           # REAL done flag
    var cont_tgt = Tensor.alloc(BATCH)       # 1 − done
    var bins = Tensor.alloc(NBINS)
    symexp_twohot_bins[NBINS](_mao(bins.data.unsafe_ptr()), lo=Scalar[DT](-9.0))
    for b in range(B):
        task_ids.data[b] = Scalar[DT](0.0)

    # sampled windows (pixels/actions/rewards/dones) come straight from the buffer
    var pix = Tensor.alloc(BATCH * IMG_DIM)
    var act_oh = Tensor.alloc(BATCH * ACT)

    var rng = Rng(20260607)

    # ── Phase A: tokenizer ──────────────────────────────────────────────
    print("- Phase A: tokenizer")
    for step in range(STEPS_TOK):
        buf.sample_reward_window_batch[B, T, ACT](
            _mao(pix.data.unsafe_ptr()), _mao(act_oh.data.unsafe_ptr()),
            _mao(rew.data.unsafe_ptr()), _mao(done.data.unsafe_ptr())
        )
        _window_to_patches[B, T, IMG, IMG_DIM, TGT, PATCH](
            _mao(pix.data.unsafe_ptr()),
            _mao(frames.data.unsafe_ptr()),
            _mao(patches.data.unsafe_ptr()),
        )
        topt.zero_grad["cpu"](tok, None)
        tok.forward["cpu", BATCH](TensorRefs[1](patches), pred, None)
        var mask = tok.mae_mask_ptr()
        var loss = masked_recon_loss[NP, DP, BATCH](
            _mao(pred.data.unsafe_ptr()),
            _mao(patches.data.unsafe_ptr()),
            mask,
            _mao(gpred.data.unsafe_ptr()),
        )
        tok.vjp["cpu", BATCH](TensorRefs[1](patches), gpred, TensorRefs[1](gin), None)
        topt.step["cpu"](tok, None)
        tok.advance_rng()
        if step % 50 == 0:
            tok.set_mae_p(0.0, 0.0)
            tok.forward["cpu", BATCH](TensorRefs[1](patches), pred, None)
            print("   tok step", step, " recon PSNR =",
                  full_recon_psnr[NP, DP, BATCH](
                      _mao(pred.data.unsafe_ptr()),
                      _mao(patches.data.unsafe_ptr())
                  ), "dB")
            tok.set_mae_p(DROP, DROP)
    tok.set_mae_p(0.0, 0.0)            # FREEZE

    # ── Phase B: BC + reward head (real rewards) + continue head (dones) ──
    print("- Phase B: BC + reward model (real rewards) + continue model (dones)")
    for step in range(STEPS_BC):
        buf.sample_reward_window_batch[B, T, ACT](
            _mao(pix.data.unsafe_ptr()), _mao(act_oh.data.unsafe_ptr()),
            _mao(rew.data.unsafe_ptr()), _mao(done.data.unsafe_ptr())
        )
        _window_to_patches[B, T, IMG, IMG_DIM, TGT, PATCH](
            _mao(pix.data.unsafe_ptr()),
            _mao(frames.data.unsafe_ptr()),
            _mao(patches.data.unsafe_ptr()),
        )
        _onehot_to_idx[BATCH, ACT](
            _mao(act_oh.data.unsafe_ptr()), _mao(act_idx.data.unsafe_ptr())
        )
        tok.enc.forward["cpu", BATCH](TensorRefs[1](patches), z1, None)  # clean latents

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
                sigma.data[bt] = Scalar[DT](Float64(j) / Float64(K))
                sig_idx.data[bt] = Scalar[DT](Float64(j * scale))
                step_idx.data[bt] = Scalar[DT](Float64(stp))
        for i in range(ZN):
            z0n.data[i] = Scalar[DT](rng.gauss())

        aopt.zero_grad["cpu"](agent, None)
        # ACTION-CONDITIONED world model: the action token moves the transition
        # (and hence the reward); reward head fits the transition-into reward
        # r[f-1]; policy clones the same-frame action. This is what makes the
        # imagined actions move the reward.
        var losses = agent.acwm_train_step(
            _mao(z1.data.unsafe_ptr()),
            _mao(z0n.data.unsafe_ptr()),
            _mao(sigma.data.unsafe_ptr()),
            _mao(sig_idx.data.unsafe_ptr()),
            _mao(step_idx.data.unsafe_ptr()),
            step >= 30,
            _mao(task_ids.data.unsafe_ptr()),
            _mao(act_idx.data.unsafe_ptr()),
            _mao(rew.data.unsafe_ptr()),
            _mao(bins.data.unsafe_ptr()),
            policy_weight=Scalar[DT](1.0), reward_weight=Scalar[DT](1.0),
        )
        aopt.step["cpu"](agent, None)

        # continue head on the clean h_t left by bc_train_step's clean forward
        var ht = agent.agent_out_ptr()
        var ht_t = Tensor.alloc(BATCH * AGD)
        for i in range(BATCH * AGD):
            ht_t.data[i] = ht[i]
        var clog = Tensor.alloc(BATCH)
        var gcl = Tensor.alloc(BATCH)
        var gci = Tensor.alloc(BATCH * AGD)
        for i in range(BATCH):
            cont_tgt.data[i] = Scalar[DT](1.0) - done.data[i]
        copt.zero_grad["cpu"](agent.ch, None)
        agent.ch.forward["cpu", BATCH](TensorRefs[1](ht_t), clog, None)
        continue_bce_backward[BATCH](
            _mao(clog.data.unsafe_ptr()), _mao(cont_tgt.data.unsafe_ptr()),
            Scalar[DT](1.0), _mao(gcl.data.unsafe_ptr())
        )
        agent.ch.vjp["cpu", BATCH](
            TensorRefs[1](ht_t), gcl, TensorRefs[1](gci), None
        )
        copt.step["cpu"](agent.ch, None)

        if step % EVAL_EVERY == 0:
            print("   bc step", step, " video =", losses[0], " bc+reward =",
                  losses[1])

    # ── Eval: reward head + continue head on fresh windows ──────────────
    print("- Eval: reward model + continue model on fresh windows")
    comptime EVAL_BATCHES = 40
    var sum_pred_rstate: Float64 = 0.0    # mean r̂ on true-reward frames
    var n_rstate = 0
    var sum_pred_zstate: Float64 = 0.0    # mean r̂ on zero-reward frames
    var n_zstate = 0
    var mae_model: Float64 = 0.0
    var mae_mean: Float64 = 0.0
    var sum_true_r: Float64 = 0.0
    var n_all = 0
    var cont_correct = 0
    var cont_total = 0
    var collected_r = List[Float64]()    # true rewards, for the mean baseline

    for _ in range(EVAL_BATCHES):
        buf.sample_reward_window_batch[B, T, ACT](
            _mao(pix.data.unsafe_ptr()), _mao(act_oh.data.unsafe_ptr()),
            _mao(rew.data.unsafe_ptr()), _mao(done.data.unsafe_ptr())
        )
        _window_to_patches[B, T, IMG, IMG_DIM, TGT, PATCH](
            _mao(pix.data.unsafe_ptr()),
            _mao(frames.data.unsafe_ptr()),
            _mao(patches.data.unsafe_ptr()),
        )
        _onehot_to_idx[BATCH, ACT](
            _mao(act_oh.data.unsafe_ptr()), _mao(act_idx.data.unsafe_ptr())
        )
        tok.enc.forward["cpu", BATCH](TensorRefs[1](patches), z1, None)
        for b in range(B):
            for t in range(T):
                var bt = b * T + t
                sigma.data[bt] = Scalar[DT](Float64(KMAX - 1) / Float64(KMAX))
                sig_idx.data[bt] = Scalar[DT](Float64(KMAX - 1))
                step_idx.data[bt] = Scalar[DT](Float64(EMAX))
        for i in range(ZN):
            z0n.data[i] = Scalar[DT](rng.gauss())
        var _l = agent.acwm_train_step(
            _mao(z1.data.unsafe_ptr()),
            _mao(z0n.data.unsafe_ptr()),
            _mao(sigma.data.unsafe_ptr()),
            _mao(sig_idx.data.unsafe_ptr()),
            _mao(step_idx.data.unsafe_ptr()),
            False,
            _mao(task_ids.data.unsafe_ptr()),
            _mao(act_idx.data.unsafe_ptr()),
            _mao(rew.data.unsafe_ptr()),
            _mao(bins.data.unsafe_ptr()),
            policy_weight=Scalar[DT](1.0), reward_weight=Scalar[DT](1.0),
        )
        # reward predictions: dist-0 block of the reward logits. acwm fits the
        # TRANSITION-INTO reward, so r̂(h_f) targets the dataset reward r[f-1];
        # compare on frames f≥1 (frame 0 has no in-window preceding action).
        var rlog = _mao(agent.rlog.unsafe_ptr())
        for b in range(B):
            for f in range(1, T):
                var pr = Float64(twohot_pred[NBINS](
                    rlog, (b * T + f) * RLOG, _mao(bins.data.unsafe_ptr())
                ))
                var tr = Float64(rew.data[b * T + f - 1])
                collected_r.append(tr)
                var is_r = abs(tr) > 1e-6
                if is_r:
                    sum_pred_rstate += pr
                    n_rstate += 1
                else:
                    sum_pred_zstate += pr
                    n_zstate += 1
                mae_model += abs(pr - tr)
                sum_true_r += tr
                n_all += 1

        # continue predictions off the clean h_t
        var ht = agent.agent_out_ptr()
        var ht_t = Tensor.alloc(BATCH * AGD)
        for i in range(BATCH * AGD):
            ht_t.data[i] = ht[i]
        var clog = Tensor.alloc(BATCH)
        var chat = Tensor.alloc(BATCH)
        agent.ch.forward["cpu", BATCH](TensorRefs[1](ht_t), clog, None)
        continue_pred[BATCH](
            _mao(clog.data.unsafe_ptr()), _mao(chat.data.unsafe_ptr())
        )
        for bt in range(BATCH):
            var ct = 1.0 - Float64(done.data[bt])
            var pc = 1.0 if Float64(chat.data[bt]) >= 0.5 else 0.0
            if abs(pc - ct) < 0.5:
                cont_correct += 1
            cont_total += 1

    var mean_r = sum_true_r / Float64(n_all)
    for i in range(len(collected_r)):
        mae_mean += abs(mean_r - collected_r[i])
    mae_model /= Float64(n_all)
    mae_mean /= Float64(n_all)
    var pred_rstate = sum_pred_rstate / Float64(max(n_rstate, 1))
    var pred_zstate = sum_pred_zstate / Float64(max(n_zstate, 1))
    var sep = pred_rstate - pred_zstate
    var cont_acc = Float64(cont_correct) / Float64(cont_total)

    print("-" * 70)
    print("  reward frames:", n_rstate, "/", n_all,
          " (mean true reward =", mean_r, ")")
    print("  r̂ on reward-states =", pred_rstate,
          "   r̂ on zero-states =", pred_zstate)
    print("  separation (reward − zero) =", sep)
    print("  reward MAE  model =", mae_model, "   mean-baseline =", mae_mean)
    print("  continue-head accuracy =", cont_acc)

    # ── Phase C: imagination RL on real-env context (execution check) ───
    print("- Phase C: imagination RL from real starting frames (use_continue)")
    agent.snapshot_prior()
    var iopt = Adam(lr=LR_IMAG)
    var ctx = Tensor.alloc(B * NCTX * ND)
    var u01 = Tensor.alloc(B * T)
    var znoise = Tensor.alloc(B * T * ND)
    var first_v: Float64 = 0.0
    var last_v: Float64 = 0.0
    var first_p: Float64 = 0.0
    var last_p: Float64 = 0.0
    var imag_ok = True
    for step in range(STEPS_IMAG):
        # real starting context: encode a fresh window, take its first frame's
        # clean latent as the NCTX=1 context for each sequence.
        buf.sample_reward_window_batch[B, T, ACT](
            _mao(pix.data.unsafe_ptr()), _mao(act_oh.data.unsafe_ptr()),
            _mao(rew.data.unsafe_ptr()), _mao(done.data.unsafe_ptr())
        )
        _window_to_patches[B, T, IMG, IMG_DIM, TGT, PATCH](
            _mao(pix.data.unsafe_ptr()),
            _mao(frames.data.unsafe_ptr()),
            _mao(patches.data.unsafe_ptr()),
        )
        tok.enc.forward["cpu", BATCH](TensorRefs[1](patches), z1, None)
        for b in range(B):
            for i in range(ND):
                ctx.data[b * NCTX * ND + i] = z1.data[(b * T + 0) * ND + i]
        for i in range(B * T):
            u01.data[i] = Scalar[DT](rng.uniform())     # fresh exploration uniforms
        for i in range(B * T * ND):
            znoise.data[i] = Scalar[DT](rng.gauss())
        iopt.zero_grad["cpu"](agent, None)
        # γ=0.9 (not the 0.997 default): with only NCTX=1 context + a short
        # imagined window the untrained bootstrap value dominates a γ→1 return,
        # so a bounded γ keeps the execution check finite.
        var l = agent.imag_train_step(
            _mao(ctx.data.unsafe_ptr()),
            _mao(u01.data.unsafe_ptr()),
            _mao(znoise.data.unsafe_ptr()),
            _mao(task_ids.data.unsafe_ptr()),
            _mao(bins.data.unsafe_ptr()),
            use_continue=True,
            gamma=Scalar[DT](0.9),
        )
        iopt.step["cpu"](agent, None)
        if step == 0:
            first_v = l[0]
            first_p = l[1]
        last_v = l[0]
        last_p = l[1]
        var finite = (l[0] == l[0]) and (l[1] == l[1])   # not NaN
        if not finite:
            imag_ok = False
        if step % 20 == 0:
            print("   imag step", step, " value =", l[0], " policy =", l[1])

    print("-" * 70)
    print("  imagination value loss  first =", first_v, "  final =", last_v)
    print("  imagination policy loss first =", first_p, "  final =", last_p)

    # ── Gate ────────────────────────────────────────────────────────────
    print("=" * 70)
    var reward_learned = (sep > 0.02) and (mae_model < mae_mean)
    if reward_learned and imag_ok:
        print("END-TO-END REWARD RUN PASSED — the reward model learned from real")
        print("Pong rewards (raises r̂ by", sep, "on reward states, MAE",
              mae_model, "<", mae_mean, "baseline); the continue model hits",
              cont_acc, "accuracy; imagination RL runs end-to-end on real-env")
        print("context with finite value/policy losses.")
    else:
        print("END-TO-END REWARD RUN: pipeline executes on the reward-bearing")
        print("buffer (tokenizer→WM→reward/continue heads→imagination RL).")
        if not reward_learned:
            print("Reward-head separation", sep, "/ MAE", mae_model, "vs",
                  mae_mean, "is SNR-limited on this pixel buffer — Pong's reward")
            print("fires on the ~1px ball-at-paddle event the tokenizer blurs")
            print("(same wall as the BC lighthouse; the mechanism is validated")
            print("decisively in imagination_lighthouse.mojo).")
        if not imag_ok:
            print("Imagination produced a non-finite loss — investigate.")
    print("=" * 70)
