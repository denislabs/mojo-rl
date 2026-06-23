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
from std.random import random_float64
from std.gpu.host import DeviceContext

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
    ) raises -> Self:
        var a = Self(
            trainer=Self.TrainerT.make(
                ctx=ctx, lr=lr, learning_starts=learning_starts,
                warmup_steps=warmup_steps, out_init_scale=out_init_scale,
                actent=actent, slowtar=slowtar,
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
    ):
        self.trainer.record(obs, act, reward, done)

    def train_step(mut self) raises -> Bool:
        return self.trainer.train_step()

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
            comptime assert not Self.DISCRETE, (
                "discrete GPU select_action not ported — use train_target='cpu'"
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
