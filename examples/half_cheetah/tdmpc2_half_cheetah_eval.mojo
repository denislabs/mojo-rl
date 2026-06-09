"""TD-MPC2 HalfCheetah — policy-vs-MPC eval on a loaded checkpoint (diagnostic).

Loads a trained checkpoint and evaluates the SAME model two ways:
  * policy:  a = π(encode(obs))                 (MPC-off acting)
  * MPC:     a = MPPI plan over the world model  (select_action_mpc)

Decisively separates a planning bug from the collection feedback loop:
  - MPC_return ≳ policy_return  → planning works; MPC's training underperformance
    was the data feedback loop / planning budget (try heavier MPC_* or warm up
    the model with MPC-off first).
  - MPC_return ≪ policy_return  → the planner's model-value ranking is off
    (planning bug to fix), independent of training dynamics.

Point CKPT at a GOOD checkpoint — e.g. tdmpc2_half_cheetah_mpcoff.ckpt from a
strong MPC-off run. Bump MPC_* toward the 512/24/64/6 reference to test budget.

Run: `pixi run -e nvidia mojo run -I . examples/half_cheetah/tdmpc2_half_cheetah_eval.mojo`
"""

from std.memory import alloc
from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.tdmpc2.agent import TDMPC2Agent
from mojo_rl.deep_agents2.tdmpc2.config import TDMPC2
from mojo_rl.envs.half_cheetah import HalfCheetah, HalfCheetahConfig

comptime CKPT = "tdmpc2_half_cheetah_mpcoff.ckpt"
comptime OBS = HalfCheetahConfig.OBS_DIM
comptime ACT = HalfCheetahConfig.ACTION_DIM
comptime ENC = 256
comptime LATENT = 512
comptime MLP = 512
comptime BINS = 101
comptime SN = 8
comptime VMIN = -10
comptime VMAX = 10
comptime B = 8          # eval-only: training batch unused
comptime H = 3
comptime CAP = 4096     # eval-only: replay unused

# MPC planning budget for the eval (match the run you're diagnosing, or bump
# toward the 512/24/64/6 reference to test whether budget is the issue).
comptime MPC_SAMPLES = 256
comptime MPC_PI_TRAJS = 12
comptime MPC_ELITES = 32
comptime MPC_ITERS = 4

comptime EP_LEN = 1_000
comptime N_EPS = 5

comptime Env = HalfCheetah[DT, TERMINATE_ON_UNHEALTHY=False]
comptime Ag = TDMPC2Agent[
    "gpu", OBS, ENC, ACT, LATENT, MLP, BINS, SN, VMIN, VMAX, B, H, CAP,
    MPC_SAMPLES, MPC_PI_TRAJS, MPC_ELITES, MPC_ITERS,
]


def _eval(mut ag: Ag, mut env: Env, use_mpc: Bool) raises -> Scalar[DT]:
    var obsbuf = alloc[Scalar[DT]](OBS)
    var actbuf = alloc[Scalar[DT]](ACT)
    var total: Scalar[DT] = 0.0
    for _ep in range(N_EPS):
        var obs = env.reset_obs_list()
        if use_mpc:
            ag.mpc_start_episode()
        for _s in range(EP_LEN):
            for i in range(OBS):
                obsbuf[i] = obs[i]
            if use_mpc:
                ag.select_action_mpc(obsbuf, actbuf, explore=False)
            else:
                ag.select_greedy_action(obsbuf, actbuf)
            var al = List[Scalar[DT]]()
            for j in range(ACT):
                al.append(actbuf[j])
            var r = env.step_continuous_vec[DT](al)
            total += r[1]
            obs = r[0].copy()
            if r[2]:
                break
    obsbuf.free(); actbuf.free()
    return total / Scalar[DT](N_EPS)


def main() raises:
    print("=" * 70)
    print("TD-MPC2 HalfCheetah — policy vs MPC eval @", CKPT)
    print("  MPC budget:", MPC_SAMPLES, "/", MPC_PI_TRAJS, "/", MPC_ELITES,
          "/", MPC_ITERS, " (samples/pi/elites/iters)")
    print("=" * 70)
    var ctx = DeviceContext()
    var env = Env()
    # Build through the Design-F preset (config.mojo) — returns exactly `Ag`.
    var ag = TDMPC2[
        "gpu", OBS, ACT, B, CAP, ENC, LATENT, MLP, BINS, SN, VMIN, VMAX, H,
        MPC_SAMPLES, MPC_PI_TRAJS, MPC_ELITES, MPC_ITERS,
    ](ctx=ctx, action_scale=Scalar[DT](1.0), learning_starts=0)
    ag.load_state(CKPT)

    var pol = _eval(ag, env, False)
    print("  policy eval return =", pol)
    var mpc = _eval(ag, env, True)
    print("  MPC    eval return =", mpc)
    print("=" * 70)
    if mpc >= pol:
        print("  MPC >= policy → planning works; MPC training gap was the data",
              "loop / budget.")
    else:
        print("  MPC < policy → planner mis-ranks vs the direct policy",
              "(planning bug to investigate).")
    print("=" * 70)
