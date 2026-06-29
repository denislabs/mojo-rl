"""DreamerV3Agent — facade with running-carry action selection (SAC-style).

Wraps `DreamerV3Trainer` and maintains a single-env belief carry
(`deter`, `stoch`, `last_action`) across env steps so the agent acts
on-policy during data collection (the lighthouse needs this — the
trainer's env-loop smoke used random actions).

`select_action` reuses the trainer's LIVE modules at BATCH=1 (graph/module
buffers grow monotonically, so B=1 inference shares the B=B training
instances safely — no separate inference copies, no param sync):

  token   = enc(obs)
  carry   = core.observe(belief_deter, belief_stoch, last_action, token)
            → nd (node_out_ptr) + stoch_new (posterior ST sample)
  feat    = concat([nd, stoch_new])
  m, s    = policy(feat)                          # bounded_normal raw params
  action  = tanh(m) [+ std·noise if exploring]    # NORMALIZED [-1,1]
  belief ← (nd, stoch_new) ; last_action ← action # belief/record stay [-1,1]

The agent acts in the normalized [-1,1] action space; the env-range scale
(`action_scale`) is applied by the DRIVER at env.step (the `action_scale`
field is now vestigial — kept for `make` signature compat). Recording the
normalized action keeps the WM's action input consistent between real-data
training and imagination (both [-1,1]); feeding env-scaled actions instead
saturates the WM's `ActionSquash` and de-grounds the world model.

`train_target` mirrors the trainer; `select_action` has both CPU and GPU
paths (`comptime if target=="cpu"`). The GPU path is a B=1 device-forward
hybrid (H2D obs+belief → enc/core/policy device forward → D2H posterior +
policy logits → host bounded_normal sample). Mirrors `deep_agents/sac/
agent.py` facade shape (select_action / select_greedy_action / record /
train_step).
"""

from std.math import tanh
from std.memory import alloc
from std.random import random_float64
from std.gpu.host import DeviceContext

from mojo_rl.core.env_traits import BoxDiscreteActionEnv
from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.deep_agents.dreamerv3.trainer import DreamerV3Trainer
from mojo_rl.deep_agents.dreamerv3.dists import bounded_std
from mojo_rl.deep_agents.dreamerv3.dists_discrete import (
    cat_sample, cat_argmax, UNIMIX,
)


@always_inline
def _hp(mut t: Tensor) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    """Host-pointer view of a storage Tensor's CPU `data` — for the raw-pointer
    cat_sample/cat_argmax helpers (CPU only)."""
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](t.data.unsafe_ptr())


@always_inline
def _argmax_oh(a: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int) -> Int:
    """Index of the max entry of a one-hot / logit vector (greedy discrete act)."""
    var k = 0
    var best = a[0]
    for i in range(1, n):
        if a[i] > best:
            best = a[i]
            k = i
    return k


@fieldwise_init
struct DreamerV3Agent[
    train_target: StaticString,
    OBS: Int, ACT: Int, DETER: Int, H: Int, STOCH: Int, CLASSES: Int,
    BLOCKS: Int, TOKEN: Int, DEC_U: Int, HU: Int, VU: Int, PU: Int,
    BINS: Int, B: Int, T: Int, T_IMAG: Int, CAP: Int, DISCRETE: Bool = False,
](Movable & ImplicitlyDeletable):
    comptime SC = Self.STOCH * Self.CLASSES
    comptime FEAT = Self.DETER + Self.SC
    # discrete (categorical) actor → ACT logits; continuous → 2·ACT (mean,std).
    # For DISCRETE the agent acts via one-hot actions (the WM's ActionSquash is
    # a no-op on {0,1}); the driver argmaxes `out_action` to the env index.
    comptime POUT = Self.ACT if Self.DISCRETE else 2 * Self.ACT
    comptime TrainerT = DreamerV3Trainer[
        Self.train_target, Self.OBS, Self.ACT, Self.DETER, Self.H, Self.STOCH,
        Self.CLASSES, Self.BLOCKS, Self.TOKEN, Self.DEC_U, Self.HU, Self.VU,
        Self.PU, Self.BINS, Self.B, Self.T, Self.T_IMAG, Self.CAP, Self.DISCRETE,
    ]
    comptime MINSTD = Scalar[DT](0.1)
    comptime MAXSTD = Scalar[DT](1.0)

    var trainer: Self.TrainerT
    var belief_deter: Tensor   # [DETER]
    var belief_stoch: Tensor   # [SC]
    var last_action: Tensor    # [ACT]
    var action_scale: Scalar[DT]

    @staticmethod
    def make(
        ctx: Optional[DeviceContext] = None,
        lr: Scalar[DT] = Scalar[DT](4e-5),
        learning_starts: Int = 200,
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        warmup_steps: Int = 1000,
        out_init_scale: Scalar[DT] = Scalar[DT](0.0),
        actent: Scalar[DT] = Scalar[DT](3e-4),
        slowtar: Bool = False,
        device_noise: Bool = True,
    ) raises -> Self:
        var a = Self(
            trainer=Self.TrainerT.make(
                ctx=ctx, lr=lr, learning_starts=learning_starts,
                warmup_steps=warmup_steps, out_init_scale=out_init_scale,
                actent=actent, slowtar=slowtar, device_noise=device_noise,
            ),
            belief_deter=Tensor.alloc(Self.DETER),
            belief_stoch=Tensor.alloc(Self.SC),
            last_action=Tensor.alloc(Self.ACT),
            action_scale=action_scale,
        )
        a.reset_belief()
        return a^

    def reset_belief(mut self):
        for i in range(Self.DETER):
            self.belief_deter.data[i] = 0.0
        for i in range(Self.SC):
            self.belief_stoch.data[i] = 0.0
        for i in range(Self.ACT):
            self.last_action.data[i] = 0.0

    def record(
        mut self,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        act: UnsafePointer[Scalar[DT], MutAnyOrigin],
        reward: Scalar[DT],
        done: Scalar[DT],
    ) raises:
        self.trainer.record(obs, act, reward, done)

    def record_terminal(
        mut self, obs: UnsafePointer[Scalar[DT], MutAnyOrigin]
    ) raises:
        """Store a genuine terminal observation (call right after `record` with
        done=1) so the WM continue head can learn `latent(terminal)→0`."""
        self.trainer.record_terminal(obs)

    def save(mut self, path: String) raises:
        """Write the full world model + actor-critic to one `nn-ckpt v2` file."""
        self.trainer.save_state(path)

    def load(mut self, path: String) raises:
        """Restore the full network set from a `save()` checkpoint."""
        self.trainer.load_state(path)

    def train_step(mut self, want_diag: Bool = True) raises -> Bool:
        return self.trainer.train_step(want_diag)

    # ─── Single-env training facade (discrete) ──────────────────────────────
    def _greedy_eval[
        E: BoxDiscreteActionEnv
    ](
        mut self,
        mut env: E,
        episodes: Int,
        ep_len: Int,
        obsbuf: UnsafePointer[Scalar[DT], MutAnyOrigin],
        actbuf: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises -> Scalar[DT]:
        """Mean return over `episodes` greedy (argmax) episodes. Steps `env`
        (caller resets it afterward for training continuation)."""
        var total: Scalar[DT] = 0.0
        for _e in range(episodes):
            self.reset_belief()
            var o = env.reset_obs_list()
            for _s in range(ep_len):
                for i in range(Self.OBS):
                    obsbuf[i] = o[i].cast[DT]()
                self.select_greedy_action(obsbuf, actbuf)
                var r = env.step_obs(_argmax_oh(actbuf, Self.ACT))
                total += r[1].cast[DT]()
                o = r[0].copy()
                if r[2]:
                    break
        return total / Scalar[DT](episodes)

    def train_single[
        E: BoxDiscreteActionEnv,
        L: Logger = NoOpLogger,
        USE_TRAIN_CUDA_GRAPH: Bool = False,
    ](
        mut self,
        mut env: E,
        total_steps: Int,
        *,
        learn_start: Int = 1024,
        train_every: Int = 4,
        eval_every: Int = 2500,
        eval_episodes: Int = 10,
        ep_len: Int = 500,
        print_every: Int = 2500,
        verbose: Bool = True,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
        checkpoint_path: String = String(""),
        checkpoint_every: Int = 0,
    ) raises -> Scalar[DT]:
        """Own the whole DreamerV3 single-env training loop for a DISCRETE env
        (warmup random one-hot actions → on-policy `select_action` → `record`
        (+`record_terminal` on done) → `train_step` every `train_every`), with
        periodic greedy eval, optional `RemoteLogger` metric streaming, and
        optional one-file checkpointing. Returns the final greedy eval return.

        Mirrors `SACAgent.train_single`: examples pass an env + a logger pointer
        and write no loop of their own. Eval reuses `env` (reset before+after)."""
        comptime assert Self.DISCRETE, (
            "DreamerV3Agent.train_single is the DISCRETE-action facade; build the"
            " agent with DISCRETE=True (continuous envs: use the bespoke loop)."
        )
        comptime OBSL = Self.OBS
        comptime ACTL = Self.ACT
        var obsbuf = alloc[Scalar[DT]](OBSL).as_unsafe_any_origin()
        var actbuf = alloc[Scalar[DT]](ACTL).as_unsafe_any_origin()
        var obs = env.reset_obs_list()
        self.reset_belief()
        var last_eval: Scalar[DT] = 0.0
        var ep_ret: Scalar[DT] = 0.0      # current (training) episode return
        var ep_acc: Scalar[DT] = 0.0      # Σ completed-episode returns since last log
        var ep_n: Int = 0                 # #completed episodes since last log
        var last_ep: Scalar[DT] = 0.0     # last completed episode return
        var best_ret: Scalar[DT] = 0.0    # best episode return so far

        for step in range(total_steps):
            for i in range(OBSL):
                obsbuf[i] = obs[i].cast[DT]()
            var idx: Int
            if step < learn_start:
                idx = Int(random_float64() * Float64(ACTL))
                if idx >= ACTL:
                    idx = ACTL - 1
                for a in range(ACTL):
                    actbuf[a] = Scalar[DT](1.0) if a == idx else Scalar[DT](0.0)
            else:
                self.select_action(obsbuf, actbuf, explore=True)
                idx = _argmax_oh(actbuf, ACTL)
            var res = env.step_obs(idx)
            ep_ret += res[1].cast[DT]()
            self.record(
                obsbuf, actbuf, res[1].cast[DT](),
                Scalar[DT](1.0) if res[2] else Scalar[DT](0.0),
            )
            obs = res[0].copy()
            if res[2]:
                # store the terminal (fallen) obs so the WM cont head learns it
                for i in range(OBSL):
                    obsbuf[i] = res[0][i].cast[DT]()
                self.record_terminal(obsbuf)
                obs = env.reset_obs_list()
                self.reset_belief()
                last_ep = ep_ret
                ep_acc += ep_ret
                ep_n += 1
                if ep_ret > best_ret:
                    best_ret = ep_ret
                ep_ret = Scalar[DT](0.0)
            if step >= learn_start and step % train_every == 0:
                # On the GPU device-resident path the per-train_step diagnostic
                # readout (host downloads of the imagination histories) is the
                # only remaining host cost — compute it ONLY on the train_step
                # whose metrics get logged at the upcoming eval boundary
                # (eval_every is a multiple of train_every in all examples).
                var wd = (step % eval_every == 0)
                comptime if (
                    USE_TRAIN_CUDA_GRAPH
                    and Self.train_target == "gpu"
                    and Self.DISCRETE
                ):
                    # Stage 3 P5: capture-once / replay the WM+AC device-kernel
                    # sequence on non-diag steps (want_diag steps stay eager for
                    # the metric readout). No-op capture on non-NVIDIA.
                    _ = self.trainer.train_step_captured(want_diag=wd)
                else:
                    _ = self.train_step(want_diag=wd)

            if step > 0 and step % eval_every == 0:
                var ev = self._greedy_eval[E](
                    env, eval_episodes, ep_len, obsbuf, actbuf
                )
                last_eval = ev
                var avg_ret = ep_acc / Scalar[DT](ep_n) if ep_n > 0 else last_ep
                if verbose and step % print_every == 0:
                    print(
                        "  step", step, " eval_ret=", ev, " avg_ret=", avg_ret,
                        " WM=", self.last_wm_loss(), " AC=", self.last_ac_loss(),
                        " con_m=", self.dbg_con_mean(),
                    )
                # Metric names follow the monitoring tool's KNOWN_GROUPS so they
                # land in the right panels (Episode Reward / Loss / World Model
                # Losses / KL / Critic-Policy Loss / Imagined Returns / Progress).
                comptime if L.ENABLED:
                    if logger:
                        var lg = logger.value()
                        # Episode Reward
                        lg[].log_scalar("avg_reward", Float64(avg_ret), step)
                        lg[].log_scalar("episode_reward", Float64(last_ep), step)
                        lg[].log_scalar("best_reward", Float64(best_ret), step)
                        lg[].log_scalar("eval/mean_return", Float64(ev), step)
                        # Loss + World Model Losses + KL Divergence
                        lg[].log_scalar("wm_loss", Float64(self.last_wm_loss()), step)
                        lg[].log_scalar("obs_loss", Float64(self.dbg_obs_loss()), step)
                        lg[].log_scalar("reward_loss", Float64(self.dbg_rew_loss()), step)
                        lg[].log_scalar("continue_loss", Float64(self.dbg_con_loss()), step)
                        lg[].log_scalar("dyn_kl", Float64(self.dbg_dyn_kl()), step)
                        lg[].log_scalar("rep_kl", Float64(self.dbg_rep_kl()), step)
                        # Critic / Policy Loss
                        lg[].log_scalar("value_loss", Float64(self.dbg_val_loss()), step)
                        lg[].log_scalar("policy_loss", Float64(self.dbg_pol_loss()), step)
                        # Imagined Returns + Policy Scale
                        lg[].log_scalar(
                            "imagined_reward_mean", Float64(self.dbg_rew_pred()), step
                        )
                        lg[].log_scalar("return_scale", Float64(self.dbg_rscale()), step)
                        lg[].log_scalar("pi_scale", Float64(self.dbg_rscale()), step)
                        # Training Progress
                        lg[].log_scalar(
                            "train_steps", Float64(self.train_steps_done()), step
                        )
                        lg[].flush()
                ep_acc = Scalar[DT](0.0)
                ep_n = 0
                if checkpoint_every > 0 and checkpoint_path.byte_length() > 0 and (
                    step % checkpoint_every == 0
                ):
                    self.save(checkpoint_path)
                obs = env.reset_obs_list()
                self.reset_belief()

        if checkpoint_path.byte_length() > 0:
            self.save(checkpoint_path)
        var final_ev = self._greedy_eval[E](
            env, eval_episodes, ep_len, obsbuf, actbuf
        )
        obsbuf.free()
        actbuf.free()
        return final_ev

    def can_train(self) -> Bool:
        return self.trainer.can_train()

    def last_wm_loss(self) -> Scalar[DT]:
        return self.trainer.last_wm_loss()

    def last_ac_loss(self) -> Scalar[DT]:
        return self.trainer.last_ac_loss()

    def dbg_real_rew(self) -> Scalar[DT]:
        return self.trainer.dbg_real_rew()

    def dbg_rew_pred(self) -> Scalar[DT]:
        return self.trainer.dbg_rew_pred()

    def dbg_ret_mean(self) -> Scalar[DT]:
        return self.trainer.dbg_ret_mean()

    def dbg_ret_std(self) -> Scalar[DT]:
        return self.trainer.dbg_ret_std()

    def dbg_pmean_abs(self) -> Scalar[DT]:
        return self.trainer.dbg_pmean_abs()

    def dbg_val_mean(self) -> Scalar[DT]:
        return self.trainer.dbg_val_mean()

    def dbg_pstd(self) -> Scalar[DT]:
        return self.trainer.dbg_pstd()

    def dbg_con_mean(self) -> Scalar[DT]:
        return self.trainer.dbg_con_mean()

    def dbg_con_min(self) -> Scalar[DT]:
        return self.trainer.dbg_con_min()

    def dbg_val_std(self) -> Scalar[DT]:
        return self.trainer.dbg_val_std()

    def dbg_feat_std(self) -> Scalar[DT]:
        return self.trainer.dbg_feat_std()

    def dbg_dyn_kl(self) -> Scalar[DT]:
        return self.trainer.dbg_dyn_kl()

    def dbg_rep_kl(self) -> Scalar[DT]:
        return self.trainer.dbg_rep_kl()

    def dbg_obs_loss(self) -> Scalar[DT]:
        return self.trainer.dbg_obs_loss()

    def dbg_rew_loss(self) -> Scalar[DT]:
        return self.trainer.dbg_rew_loss()

    def dbg_con_loss(self) -> Scalar[DT]:
        return self.trainer.dbg_con_loss()

    def dbg_pol_loss(self) -> Scalar[DT]:
        return self.trainer.dbg_pol_loss()

    def dbg_val_loss(self) -> Scalar[DT]:
        return self.trainer.dbg_val_loss()

    def train_steps_done(self) -> Int:
        return self.trainer.train_steps_done()

    def dbg_rscale(self) -> Scalar[DT]:
        return self.trainer.dbg_rscale()

    def select_action(
        mut self,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],       # [OBS]
        out_action: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [ACT]
        explore: Bool,
    ) raises:
        comptime D = Self.DETER
        comptime SCl = Self.SC
        comptime FEATl = Self.FEAT
        comptime ACTD = Self.ACT
        comptime POUTl = Self.POUT
        comptime TOK = Self.TOKEN
        comptime CARRY = 2 + D + SCl
        # feat = concat([nd, stoch_new]); both branches fill nd_h / sn_h (the
        # posterior carry, host) then sample identically below.
        var nd_h = Tensor.alloc(D)
        var sn_h = Tensor.alloc(SCl)
        var pol = Tensor.alloc(POUTl)
        comptime if Self.train_target == "cpu":
            # 1. encode obs → token (B=1)
            var obt = Tensor.alloc(Self.OBS)
            for k in range(Self.OBS):
                obt.data[k] = obs[k]
            var tok = Tensor.alloc(TOK)
            self.trainer.enc.forward[Self.train_target, 1](
                TensorRefs[1](obt), tok, None
            )
            # 2. observe via the core graph (B=1) → nd + posterior stoch sample
            self.trainer.core.set_input["deter", 1](self.belief_deter, None)
            self.trainer.core.set_input["stoch", 1](self.belief_stoch, None)
            self.trainer.core.set_input["action", 1](self.last_action, None)
            self.trainer.core.set_input["tokens", 1](tok, None)
            var cscr = Tensor.alloc(CARRY)
            self.trainer.core.forward[1, Self.train_target](cscr, None)
            ref nd = self.trainer.core.node_output["nd"]()
            ref stoch_new = self.trainer.core.node_output["stoch_new"]()
            for k in range(D):
                nd_h.data[k] = nd.data[k]
            for k in range(SCl):
                sn_h.data[k] = stoch_new.data[k]
            # 3. feat = concat([nd, stoch_new]) → policy(feat)
            var feat = Tensor.alloc(FEATl)
            for k in range(D):
                feat.data[k] = nd_h.data[k]
            for k in range(SCl):
                feat.data[D + k] = sn_h.data[k]
            self.trainer.policy.forward[Self.train_target, 1](
                TensorRefs[1](feat), pol, None
            )
        else:
            comptime assert Self.train_target == "gpu", (
                "select_action: train_target must be 'cpu' or 'gpu'"
            )
            # GPU B=1 inference (hybrid): device enc/core/policy forwards reusing
            # the trainer's LIVE GPU modules (buffers grow-only, so B=1 shares
            # the B=B/NS training instances). H2D obs+belief, device forwards,
            # D2H posterior + policy logits, host sample. Mirrors `_wm_gpu`'s
            # upload/download marshalling; structurally identical to the CPU
            # branch above so the sample below is shared.
            var ctx = self.trainer.ctx.value()
            # device staging Tensors (B=1 widths). `.ensure(n)` also sizes the
            # host `.data` List for the ones written host-side (obt/bd/bs/la/feat)
            # before upload — `Tensor.make["gpu"]` allocates the device buffer
            # only (empty `.data`).
            var obt = Tensor.make["gpu"](Self.OBS, self.trainer.ctx)
            obt.ensure(Self.OBS)
            var tok = Tensor.make["gpu"](TOK, self.trainer.ctx)
            var bd = Tensor.make["gpu"](D, self.trainer.ctx)
            bd.ensure(D)
            var bs = Tensor.make["gpu"](SCl, self.trainer.ctx)
            bs.ensure(SCl)
            var la = Tensor.make["gpu"](ACTD, self.trainer.ctx)
            la.ensure(ACTD)
            var feat = Tensor.make["gpu"](FEATl, self.trainer.ctx)
            feat.ensure(FEATl)
            var cscr = Tensor.make["gpu"](CARRY, self.trainer.ctx)
            # H2D obs + belief carry (belief_* / last_action are host-resident).
            for k in range(Self.OBS):
                obt.data[k] = obs[k]
            obt.upload(ctx)
            for k in range(D):
                bd.data[k] = self.belief_deter.data[k]
            bd.upload(ctx)
            for k in range(SCl):
                bs.data[k] = self.belief_stoch.data[k]
            bs.upload(ctx)
            for k in range(ACTD):
                la.data[k] = self.last_action.data[k]
            la.upload(ctx)
            # 1. encode obs → token (B=1)
            self.trainer.enc.forward[Self.train_target, 1](
                TensorRefs[1](obt), tok, self.trainer.ctx
            )
            # 2. observe via the core graph (B=1) → nd + posterior stoch sample
            self.trainer.core.set_input["deter", 1](bd, self.trainer.ctx)
            self.trainer.core.set_input["stoch", 1](bs, self.trainer.ctx)
            self.trainer.core.set_input["action", 1](la, self.trainer.ctx)
            self.trainer.core.set_input["tokens", 1](tok, self.trainer.ctx)
            self.trainer.core.forward[1, Self.train_target](cscr, self.trainer.ctx)
            ref nd = self.trainer.core.node_output["nd"]()
            ref stoch_new = self.trainer.core.node_output["stoch_new"]()
            nd.download(ctx)
            stoch_new.download(ctx)
            for k in range(D):
                nd_h.data[k] = nd.data[k]
            for k in range(SCl):
                sn_h.data[k] = stoch_new.data[k]
            # 3. feat = concat([nd, stoch_new]) on host → H2D → policy
            for k in range(D):
                feat.data[k] = nd_h.data[k]
            for k in range(SCl):
                feat.data[D + k] = sn_h.data[k]
            feat.upload(ctx)
            self.trainer.policy.forward[Self.train_target, 1](
                TensorRefs[1](feat), pol, self.trainer.ctx
            )
            pol.download(ctx)
        comptime if Self.DISCRETE:
            # ── discrete: categorical sample (explore) / argmax (greedy) →
            # one-hot out_action[ACT]. The one-hot is what the WM conditions on
            # (ActionSquash is a no-op on {0,1}); the driver argmaxes for env.
            var k: Int
            if explore:
                var u01 = Scalar[DT](random_float64())
                k = cat_sample[ACTD](_hp(pol), 0, UNIMIX, u01)
            else:
                k = cat_argmax[ACTD](_hp(pol), 0)
            for a in range(ACTD):
                out_action[a] = Scalar[DT](1.0) if a == k else Scalar[DT](0.0)
        else:
            # ── action = tanh(mean) [+ std·noise], NORMALIZED [-1,1] ──
            # The env-range scale (`action_scale`) is applied by the DRIVER at
            # env.step — NOT here — so what we output/record/feed the WM is
            # always [-1,1]. (The WM's ActionSquash then only clips rare |a|>1
            # outliers instead of saturating the whole range.)
            for a in range(ACTD):
                var mean = tanh(pol.data[a])
                var act_a = mean
                if explore:
                    var std = bounded_std(
                        pol.data[ACTD + a], Self.MINSTD, Self.MAXSTD
                    )
                    var z = Scalar[DT](random_float64() * 2.0 - 1.0)
                    act_a = mean + std * z
                if act_a > Scalar[DT](1.0):
                    act_a = Scalar[DT](1.0)
                if act_a < Scalar[DT](-1.0):
                    act_a = Scalar[DT](-1.0)
                out_action[a] = act_a
        # update belief (both paths)
        for k in range(D):
            self.belief_deter.data[k] = nd_h.data[k]
        for k in range(SCl):
            self.belief_stoch.data[k] = sn_h.data[k]
        for a in range(ACTD):
            self.last_action.data[a] = out_action[a]

    def select_greedy_action(
        mut self,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        out_action: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        self.select_action(obs, out_action, explore=False)
