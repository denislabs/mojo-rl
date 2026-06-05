"""MuZero two-player self-play driver (CPU) — the TicTacToe lighthouse.

The two-player counterpart to `selfplay_cpu.mojo`. Same learned-model pipeline
(env → CPU MCTS over h/g/f → sequence replay → K-step BPTT unroll), but now a
zero-sum board game, which changes four things vs the single-player path:

  1. **SelfPlay MCTS backup** (`PLAYER=SelfPlay`): the value negates at each ply,
     so the tree reasons in alternating perspectives.
  2. **Canonical observation**: the env returns the board *from the player to
     move's perspective* (`get_obs_list`), so the learned value/reward are
     naturally in that player's frame — matching the n-step target convention.
  3. **`to_play` tracking**: each step records `env.current_player()`; the
     replay's n-step bootstrap sign-flips reward + bootstrap when perspectives
     differ (`zero/nstep_targets.mojo` — the legacy P0 bug, guarded here).
  4. **Legal masking at the root** + `gamma=1.0` (no discounting): MuZero only
     knows legality at the real root, the learned model plans unmasked beyond it.

Reward convention (TTT): the env returns `+1` on the move that *wins* (in the
mover's own frame) and `0` otherwise — you never lose by your own move — so the
stored per-step reward is already in `to_play[k]`'s frame, exactly what
`extract_reward_targets` (no flip) + `compute_nstep_value_targets` (flips) expect.

Periodic eval plays the agent at **greedy MCTS strength** (noise off, argmax
visits, legal-masked) vs a pluggable `CPUEvaluator` opponent, alternating who
moves first, and reports win/draw/loss from the agent's perspective. A solved
TTT agent never loses (all draws vs minimax; wins + draws, no losses vs random).
Returns the last training loss.
"""

from std.memory import alloc

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.core import TwoPlayerDiscreteEnv, Saveable
from mojo_rl.planners.tree_search import (
    GenericCPUMCTS,
    MuZeroPUCT,
    DirichletNoise,
    SelfPlay,
)

from .blocks import mz_unroll_train_step_cpu
from ..zero.mcts_adapters_mz_cpu import MZRepCPU, MZDynCPU, MZPredCPU
from ..zero.sequence_replay_mcts import MCTSSequenceReplay
from ..zero.evaluators import CPUEvaluator, RandomOpponent


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


@always_inline
def _xs(s: UInt64) -> UInt64:
    var x = s
    x ^= x << 13
    x ^= x >> 7
    x ^= x << 17
    return x


def run_muzero_selfplay_2p_cpu[
    ENV: TwoPlayerDiscreteEnv & Saveable & ImplicitlyDestructible,
    REP: Module,
    DYN: Module,
    PRED: Module,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    CAP: Int,
    B: Int,
    K: Int,
    N: Int,
    OPP: CPUEvaluator = RandomOpponent,
    BATCH_SIMS: Int = 8,
    VIRTUAL_LOSS: Int = 3,
](
    mut env: ENV,
    mut rep: REP,
    mut dyn: DYN,
    mut pred: PRED,
    mut orep: Adam,
    mut odyn: Adam,
    mut opred: Adam,
    iterations: Int,
    learning_starts: Int = 256,
    train_per_iter: Int = 1,
    gamma: Scalar[DT] = Scalar[DT](1.0),
    v_min: Scalar[DT] = Scalar[DT](-1.0),
    v_max: Scalar[DT] = Scalar[DT](1.0),
    seed: UInt64 = 0,
    max_ep_steps: Int = 9,
    value_coef: Scalar[DT] = Scalar[DT](0.25),
    eval_every: Int = 0,
    eval_games: Int = 50,
    verbose: Bool = False,
) raises -> Float64:
    # MuZeroPUCT defaults (c_base=19652, c_init=1.25); SelfPlay backup negates V.
    var mcts = GenericCPUMCTS[
        ACT, LATENT, NUM_SIMS, MAX_NODES,
        MuZeroPUCT[19652.0, 1.25], DirichletNoise[0.25, 0.25], SelfPlay,
        BATCH_SIMS, VIRTUAL_LOSS,
    ](gamma=Float64(gamma))
    var rb = MCTSSequenceReplay[OBS, ACT, CAP](seed=seed ^ UInt64(0xABCDEF))

    var rep_a = MZRepCPU[OBS, LATENT, REP](net=UnsafePointer(to=rep))
    var dyn_a = MZDynCPU[LATENT, ACT, BINS, DYN](
        net=UnsafePointer(to=dyn), v_min=v_min, v_max=v_max
    )
    var pred_a = MZPredCPU[LATENT, ACT, BINS, PRED](
        net=UnsafePointer(to=pred), v_min=v_min, v_max=v_max
    )

    # training batch slabs (time-major), allocated once
    var t_obs0 = _a(B * OBS)
    var t_act = _a(K * B)
    var t_pol = _a((K + 1) * B * ACT)
    var t_val = _a((K + 1) * B)
    var t_rew = _a(K * B)

    # episode accumulation buffers
    var e_obs = List[Scalar[DT]]()
    var e_act = List[Scalar[DT]]()
    var e_rew = List[Scalar[DT]]()
    var e_pol = List[Scalar[DT]]()
    var e_val = List[Scalar[DT]]()
    var e_tp = List[Scalar[DT]]()
    var ep_len = 0

    var rng = seed ^ UInt64(0x123456789)
    var last_loss = 0.0

    _ = env.reset()

    for it in range(iterations):
        # ── current canonical obs (player-to-move perspective) + legality ──
        var obs_raw = env.get_obs_list()
        var cur_f = List[Float64]()
        for j in range(OBS):
            cur_f.append(Float64(obs_raw[j]))
        var legal = env.legal_action_mask()
        var to_play = env.current_player()

        # ── search (root noise on for exploration) ──
        var policy = mcts.search[
            type_of(rep_a), type_of(dyn_a), type_of(pred_a)
        ](rep_a, dyn_a, pred_a, cur_f, add_noise=True, legal_mask=legal)
        var root_v = mcts.root_value()

        # ── sample a legal action ∝ visit policy ──
        rng = _xs(rng)
        var r = Float64(rng % UInt64(1_000_000)) / 1_000_000.0
        var cum = 0.0
        var action = -1
        for a in range(ACT):
            cum += policy[a]
            if r <= cum and policy[a] > 0.0:
                action = a
                break
        if action < 0:                          # numeric fallback: argmax legal
            var bv = -1.0
            for a in range(ACT):
                if policy[a] > bv:
                    bv = policy[a]
                    action = a
            if action < 0:
                action = 0

        # ── record step (o_t, a_t, π_t, v_t, to_play) ──
        for j in range(OBS):
            e_obs.append(Scalar[DT](cur_f[j]))
        e_act.append(Scalar[DT](action))
        for a in range(ACT):
            e_pol.append(Scalar[DT](policy[a]))
        e_val.append(Scalar[DT](root_v))
        e_tp.append(Scalar[DT](to_play))

        # ── env step → r_{t+1}, done (reward in mover's own frame) ──
        var stepped = env.step(env.action_from_index(action))
        var reward = Float64(stepped[1])
        var done = stepped[2]
        e_rew.append(Scalar[DT](reward))
        ep_len += 1

        if done or ep_len >= max_ep_steps:
            rb.store_episode(
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    e_obs.unsafe_ptr()
                ),
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    e_act.unsafe_ptr()
                ),
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    e_rew.unsafe_ptr()
                ),
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    e_pol.unsafe_ptr()
                ),
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    e_val.unsafe_ptr()
                ),
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    e_tp.unsafe_ptr()
                ),
                ep_len,
            )
            e_obs.clear(); e_act.clear(); e_rew.clear()
            e_pol.clear(); e_val.clear(); e_tp.clear()
            ep_len = 0
            _ = env.reset()

        # ── train ──
        if it >= learning_starts and rb.num_episodes() > 0:
            for _ in range(train_per_iter):
                rb.sample_training_batch[B, K, N](
                    gamma, t_obs0, t_act, t_pol, t_val, t_rew
                )
                last_loss = Float64(
                    mz_unroll_train_step_cpu[
                        REP, DYN, PRED, B, K, OBS, ACT, LATENT, BINS
                    ](
                        rep, dyn, pred, orep, odyn, opred,
                        t_obs0, t_act, t_pol, t_val, t_rew,
                        v_min, v_max, value_coef,
                    )
                )

        # ── periodic greedy eval vs OPP (interrupts self-play; reset after) ──
        # Inlined (not a helper) so `mcts`/adapters keep their concrete types —
        # `.search` isn't visible through a type-erased `Copyable & Movable` bound.
        if eval_every > 0 and (it + 1) % eval_every == 0:
            var wins = 0
            var draws = 0
            var losses = 0
            for g in range(eval_games):
                _ = env.reset()
                var agent_player = g % 2     # alternate first mover
                for _ply in range(max_ep_steps):
                    if env.current_player() == agent_player:
                        var oraw = env.get_obs_list()
                        var of = List[Float64]()
                        for j in range(OBS):
                            of.append(Float64(oraw[j]))
                        var lg = env.legal_action_mask()
                        var pol = mcts.search[
                            type_of(rep_a), type_of(dyn_a), type_of(pred_a)
                        ](rep_a, dyn_a, pred_a, of, add_noise=False, legal_mask=lg)
                        var best = 0
                        for a in range(1, ACT):
                            if pol[a] > pol[best]:
                                best = a
                        if env.step(env.action_from_index(best))[2]:
                            break
                    else:
                        rng = _xs(rng)
                        var oa = OPP.select_action_cpu[ENV](env, rng)
                        if env.step(env.action_from_index(oa))[2]:
                            break
                var gr = env.game_result()    # 0 ongoing,1 P0,2 P1,3 draw
                if gr == 3 or gr == 0:
                    draws += 1
                elif (gr - 1) == agent_player:
                    wins += 1
                else:
                    losses += 1
            if verbose:
                print(
                    "  [eval vs", OPP.NAME, "] step", it + 1,
                    "W", wins, "D", draws, "L", losses, "(of", eval_games, ")",
                )
            _ = env.reset()
            e_obs.clear(); e_act.clear(); e_rew.clear()
            e_pol.clear(); e_val.clear(); e_tp.clear()
            ep_len = 0

        if verbose and (it + 1) % 1000 == 0:
            print("step", it + 1, "loss", last_loss, "eps", rb.num_episodes())

    t_obs0.free(); t_act.free(); t_pol.free(); t_val.free(); t_rew.free()
    return last_loss
