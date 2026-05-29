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

CPU v1. `train_target` mirrors the trainer; `select_action` is `comptime
if target=="cpu"` (GPU = PR5c Step 5). Mirrors `deep_agents2/sac/agent.py`
facade shape (select_action / select_greedy_action / record / train_step).
"""

from std.memory import alloc
from std.math import tanh
from std.random import random_float64
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.dreamerv3.trainer import DreamerV3Trainer
from mojo_rl.deep_agents2.dreamerv3.dists import bounded_std


@always_inline
def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


@fieldwise_init
struct DreamerV3Agent[
    train_target: StaticString,
    OBS: Int, ACT: Int, DETER: Int, H: Int, STOCH: Int, CLASSES: Int,
    BLOCKS: Int, TOKEN: Int, DEC_U: Int, HU: Int, VU: Int, PU: Int,
    BINS: Int, B: Int, T: Int, T_IMAG: Int, CAP: Int,
](Movable & ImplicitlyDestructible):
    comptime SC = Self.STOCH * Self.CLASSES
    comptime FEAT = Self.DETER + Self.SC
    comptime TrainerT = DreamerV3Trainer[
        Self.train_target, Self.OBS, Self.ACT, Self.DETER, Self.H, Self.STOCH,
        Self.CLASSES, Self.BLOCKS, Self.TOKEN, Self.DEC_U, Self.HU, Self.VU,
        Self.PU, Self.BINS, Self.B, Self.T, Self.T_IMAG, Self.CAP,
    ]
    comptime MINSTD = Scalar[DT](0.1)
    comptime MAXSTD = Scalar[DT](1.0)

    var trainer: Self.TrainerT
    var belief_deter: UnsafePointer[Scalar[DT], MutAnyOrigin]   # [DETER]
    var belief_stoch: UnsafePointer[Scalar[DT], MutAnyOrigin]   # [SC]
    var last_action: UnsafePointer[Scalar[DT], MutAnyOrigin]    # [ACT]
    var action_scale: Scalar[DT]

    @staticmethod
    def make(
        ctx: Optional[DeviceContext] = None,
        lr: Scalar[DT] = Scalar[DT](4e-5),
        learning_starts: Int = 200,
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        warmup_steps: Int = 1000,
    ) raises -> Self:
        var a = Self(
            trainer=Self.TrainerT.make(
                ctx=ctx, lr=lr, learning_starts=learning_starts,
                warmup_steps=warmup_steps,
            ),
            belief_deter=_alloc(Self.DETER),
            belief_stoch=_alloc(Self.SC),
            last_action=_alloc(Self.ACT),
            action_scale=action_scale,
        )
        a.reset_belief()
        return a^

    def reset_belief(mut self):
        for i in range(Self.DETER):
            self.belief_deter[i] = 0.0
        for i in range(Self.SC):
            self.belief_stoch[i] = 0.0
        for i in range(Self.ACT):
            self.last_action[i] = 0.0

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

    def select_action(
        mut self,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],       # [OBS]
        out_action: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [ACT]
        explore: Bool,
    ) raises:
        comptime assert Self.train_target == "cpu", (
            "DreamerV3Agent.select_action: GPU lands in PR5c Step 5"
        )
        comptime D = Self.DETER
        comptime SCl = Self.SC
        comptime FEATl = Self.FEAT
        comptime ACTD = Self.ACT
        comptime TOK = Self.TOKEN
        comptime CARRY = 2 + D + SCl
        # 1. encode obs → token (B=1)
        var tok = _alloc(TOK)
        var tkt = TileTensor(tok, row_major[1, TOK]())
        self.trainer.enc.forward[Self.train_target, 1](
            TileTensor(obs, row_major[1, Self.OBS]()), output=tkt
        )
        # 2. observe via the core graph (B=1) → nd + posterior stoch sample
        self.trainer.core.set_input["deter", 1](
            TileTensor(self.belief_deter, row_major[1, D]())
        )
        self.trainer.core.set_input["stoch", 1](
            TileTensor(self.belief_stoch, row_major[1, SCl]())
        )
        self.trainer.core.set_input["action", 1](
            TileTensor(self.last_action, row_major[1, ACTD]())
        )
        self.trainer.core.set_input["tokens", 1](
            TileTensor(tok, row_major[1, TOK]())
        )
        var cscr = _alloc(CARRY)
        var cscrt = TileTensor(cscr, row_major[1, CARRY]())
        self.trainer.core.forward[Self.train_target, 1](cscrt)
        var nd = self.trainer.core.node_out_ptr["nd"]()
        var stoch_new = self.trainer.core.node_out_ptr["stoch_new"]()
        # 3. feat = concat([nd, stoch_new])
        var feat = _alloc(FEATl)
        for k in range(D):
            feat[k] = nd[k]
        for k in range(SCl):
            feat[D + k] = stoch_new[k]
        # 4. policy(feat) → [mean_raw, std_raw]
        var pol = _alloc(2 * ACTD)
        var polt = TileTensor(pol, row_major[1, 2 * ACTD]())
        self.trainer.policy.forward[Self.train_target, 1](
            TileTensor(feat, row_major[1, FEATl]()), output=polt
        )
        # 5. action = tanh(mean) [+ std·noise], in the NORMALIZED [-1,1]
        #    action space. The env-range scale (`action_scale`) is applied by
        #    the DRIVER at env.step — NOT here — so what we output/record/feed
        #    the WM is always [-1,1]. (The reference acts in normalized space
        #    and the env wrapper denormalizes; the WM's ActionSquash then only
        #    clips rare |a|>1 outliers instead of saturating the whole range.)
        for a in range(ACTD):
            var mean = tanh(pol[a])
            var act_a = mean
            if explore:
                var std = bounded_std(pol[ACTD + a], Self.MINSTD, Self.MAXSTD)
                var z = Scalar[DT](random_float64() * 2.0 - 1.0)
                act_a = mean + std * z
            if act_a > Scalar[DT](1.0):
                act_a = Scalar[DT](1.0)
            if act_a < Scalar[DT](-1.0):
                act_a = Scalar[DT](-1.0)
            out_action[a] = act_a
        # 6. update belief
        for k in range(D):
            self.belief_deter[k] = nd[k]
        for k in range(SCl):
            self.belief_stoch[k] = stoch_new[k]
        for a in range(ACTD):
            self.last_action[a] = out_action[a]
        tok.free(); cscr.free(); feat.free(); pol.free()

    def select_greedy_action(
        mut self,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        out_action: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        self.select_action(obs, out_action, explore=False)
