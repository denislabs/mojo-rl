"""Play Connect Four against a trained Gumbel MuZero checkpoint (terminal).

Loads the spatial h/g/f trio saved by `connect_four_muzero_gumbel_spatial.mojo`
(`checkpoint_every` rolling save, or the end-of-run save) and lets you play a
game in the terminal. The agent picks its move with a full Gumbel-MCTS search
over the learned model (deterministic, `gumbel_scale=0.0`) — the deployed-agent
strength, not the bare policy head.

The network dims here MUST match the trained checkpoint (C / BINS / 6×7). If you
retrain with different `CH`/`BINS`, update them here too or the load will fail on
a shape mismatch.

Usage (after a training run has written the checkpoint):
    pixi run -e nvidia mojo run -I . examples/board_games/play_connect_four_muzero_gumbel.mojo
    pixi run -e apple  mojo run -I . examples/board_games/play_connect_four_muzero_gumbel.mojo

You enter a column 0–6 on your turn; the agent replies. X = player 0 (moves
first), O = player 1.
"""

from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import mptr
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.core.checkpoint import load_state_v2_body_gpu
from mojo_rl.deep_agents2.core.checkpoint_helpers import (
    read_file_v2, split_lines_v2, expect_v2_header,
)
from mojo_rl.deep_agents2.muzero.nets_spatial import (
    MZRepNetC4Spatial, MZDynNetC4Spatial, MZPredNetC4Spatial,
)
from mojo_rl.deep_agents2.zero.mcts_adapters_mz import (
    MZRepGPU, MZDynGPU, MZPredGPU,
)
from mojo_rl.planners.tree_search import GumbelGPUMCTS, SelfPlay
from mojo_rl.envs.board_games.connect_four.connect_four import ConnectFourEnv


def show_board(env: ConnectFourEnv[DType.float64]):
    """Text board: row 5 (top) down to row 0, X=P0, O=P1, .=empty."""
    print()
    print("   0 1 2 3 4 5 6")
    for row in range(5, -1, -1):
        var line = String(" ") + String(row) + " "
        for col in range(7):
            var v = Int(Float64(env.state[col * 6 + row]))
            if v == 1:
                line += "X "
            elif v == 2:
                line += "O "
            else:
                line += ". "
        print(line)
    print()


def main() raises:
    # ── architecture — MUST match the trained checkpoint ──
    comptime OBS = 126
    comptime ACT = 7
    comptime CH = 32
    comptime HH = 6
    comptime WW = 7
    comptime LATENT = CH * HH * WW
    comptime BINS = 51
    # search budget for play (independent of training; more = stronger).
    comptime NUM_SIMS = 200
    comptime MAX_NODES = 256
    comptime MAX_K = 4

    comptime Rep = MZRepNetC4Spatial[CH, HH, WW]
    comptime Dyn = MZDynNetC4Spatial[CH, ACT, BINS, HH, WW]
    comptime Pred = MZPredNetC4Spatial[CH, ACT, BINS, HH, WW]

    var ckpt = String("connect_four_muzero_gumbel_spatial.ckpt")

    var ctx = DeviceContext()
    var rep = Rep.make["gpu", INIT=Kaiming](ctx=ctx)
    var dyn = Dyn.make["gpu", INIT=Kaiming](ctx=ctx)
    var pred = Pred.make["gpu", INIT=Kaiming](ctx=ctx)

    # ── load the trio ──
    var content = read_file_v2(ckpt)
    var lines = split_lines_v2(content)
    expect_v2_header(lines)
    var idx = 1
    load_state_v2_body_gpu(rep, lines, idx, String("rep"), ctx)
    load_state_v2_body_gpu(dyn, lines, idx, String("dyn"), ctx)
    load_state_v2_body_gpu(pred, lines, idx, String("pred"), ctx)
    rep.set_attr["training"](Scalar[DT](0.0))
    dyn.set_attr["training"](Scalar[DT](0.0))
    pred.set_attr["training"](Scalar[DT](0.0))
    print("loaded checkpoint:", ckpt)

    # ── Gumbel planner over the learned model (deterministic play) ──
    var planner = GumbelGPUMCTS[
        1, ACT, LATENT, BINS, MAX_NODES, MAX_K, NUM_SIMS, SelfPlay
    ](ctx, gamma=1.0, v_min=-1.0, v_max=1.0, gumbel_scale=0.0)
    var rep_a = MZRepGPU[OBS, LATENT, Rep].make(rep)
    var dyn_a = MZDynGPU[LATENT, ACT, BINS, Dyn].make(dyn)
    var pred_a = MZPredGPU[LATENT, ACT, BINS, Pred].make(pred)

    var d_obs = ctx.enqueue_create_buffer[DT](OBS)
    var h_obs = ctx.enqueue_create_host_buffer[DT](OBS)
    var h_pol = ctx.enqueue_create_host_buffer[DT](ACT)
    var h_legal = ctx.enqueue_create_host_buffer[DT](ACT)
    ctx.synchronize()

    # ── choose side ──
    print("=== Connect Four vs Gumbel MuZero ===")
    print("You are X if you go first (0), O if MuZero goes first (1).")
    var human_player = -1
    while human_player < 0:
        var s = input("Play first? (0 = you first, 1 = MuZero first) > ")
        try:
            var c = Int(s)
            if c == 0 or c == 1:
                human_player = c
            else:
                print("  enter 0 or 1")
        except:
            print("  enter 0 or 1")

    var env = ConnectFourEnv[DType.float64]()
    _ = env.reset()
    var mseed = UInt32(12345)

    while env.game_result() == 0:
        show_board(env)
        var legal = env.legal_action_mask()
        if env.current_player() == human_player:
            var col = -1
            while col < 0:
                var s = input("Your move, column 0-6 > ")
                var c = -999
                try:
                    c = Int(s)
                except:
                    print("  not a number — enter 0-6")
                if c >= 0 and c < ACT:
                    if legal[c]:
                        col = c
                    else:
                        print("  column", c, "is full")
                elif c != -999:
                    print("  out of range — enter 0-6")
            _ = env.step(env.action_from_index(col))
        else:
            # MuZero move: Gumbel search over the learned model, argmax-legal.
            var obs = env.get_obs_list()
            for j in range(OBS):
                h_obs.unsafe_ptr()[j] = Scalar[DT](Float64(obs[j]))
            for a in range(ACT):
                h_legal.unsafe_ptr()[a] = (
                    Scalar[DT](1.0) if legal[a] else Scalar[DT](0.0)
                )
            ctx.enqueue_copy(d_obs, h_obs)
            ctx.enqueue_copy(planner.legal_mask_view(), h_legal)
            var obs_t = LayoutTensor[
                DT, Layout.row_major(1, OBS), MutAnyOrigin
            ](mptr(d_obs.unsafe_ptr()))
            planner.search_gpu[
                MZRepGPU[OBS, LATENT, Rep],
                MZDynGPU[LATENT, ACT, BINS, Dyn],
                MZPredGPU[LATENT, ACT, BINS, Pred],
            ](
                ctx, rep_a, dyn_a, pred_a, obs_t,
                apply_legal=True, k_actual=MAX_K, rng_seed=mseed,
            )
            mseed += UInt32(1)
            ctx.enqueue_copy(h_pol, planner.policies_view())
            ctx.synchronize()
            var best = -1
            var bv = -1.0e30
            for a in range(ACT):
                if legal[a] and Float64(h_pol.unsafe_ptr()[a]) > bv:
                    bv = Float64(h_pol.unsafe_ptr()[a])
                    best = a
            if best < 0:
                best = 0
            print("MuZero plays column", best)
            _ = env.step(env.action_from_index(best))

    show_board(env)
    var gr = env.game_result()
    var human_win = human_player + 1
    if gr == 3:
        print("Draw.")
    elif gr == human_win:
        print("You win! 🎉")
    else:
        print("MuZero wins.")
