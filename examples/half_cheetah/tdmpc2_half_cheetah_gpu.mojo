"""TD-MPC2 (deep_agents2) — HalfCheetah training (GPU, MPC-off).

The real-target harness for the deep_agents2 TD-MPC2 port. Built for the
NVIDIA run (`pixi run -e nvidia`): on CUDA the GPU path is fast (low
per-launch overhead + grouped multi-tensor Adam + big-matmul-dominated),
whereas on Apple/Metal TD-MPC2 is kernel-launch-bound and CPU is faster
(see tests/deep_agents/test_tdmpc2_perf.mojo). Acting is MPC-off
(a = π(encode(obs))); MPPI planning is a later phase.

HalfCheetah (Phyics3dEnv, MuJoCo-style):
  * 17D obs, 6D action (joint torques in [-1,1]).
  * No early termination — truncates at the episode horizon, so we record
    done=0 (the value bootstrap must continue across truncation).
  * Reward ≈ forward velocity − control cost; good policies reach a few
    thousand return / 1000-step episode.

Dims follow the reference (latent 512, mlp 512, enc 256, num_bins 101,
num_q 5, horizon 3). Drop NUM_STEPS / dims for a quick smoke.

Run:
    pixi run -e nvidia mojo run -I . examples/half_cheetah/tdmpc2_half_cheetah_gpu.mojo
"""

from std.memory import alloc
from std.random import random_float64, seed
from std.math import isfinite
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents2.tdmpc2.agent import TDMPC2Agent
from mojo_rl.envs.half_cheetah import HalfCheetah, HalfCheetahConfig

# ── target: "gpu" for the NVIDIA run; "cpu" works too (slower at this scale).
comptime TARGET = "gpu"
# ── MPC: True → act via MPPI planning (select_action_mpc, GPU only — heavy:
#    ~536 batched forwards/action at the reference 512/24 config). False →
#    MPC-off policy acting. Flip on for the full TD-MPC2 algorithm on NVIDIA.
comptime USE_MPC = False

comptime OBS = HalfCheetahConfig.OBS_DIM        # 17
comptime ACT = HalfCheetahConfig.ACTION_DIM     #  6
comptime ENC = 256
comptime LATENT = 512
comptime MLP = 512
comptime BINS = 101
comptime SN = 8
comptime VMIN = -10
comptime VMAX = 10
comptime B = 256
comptime H = 3
comptime CAP = 1_000_000

comptime LR = 3e-4
comptime ACTION_SCALE = 1.0
comptime LEARN_START = 5_000
comptime TRAIN_EVERY = 1
comptime TOTAL = 1_000_000
comptime EVAL_EVERY = 20_000
comptime DIAG_EVERY = 2_000   # flush_metrics → logger cadence
comptime CHECKPOINT_EVERY = 50_000
comptime CHECKPOINT_PATH = "tdmpc2_half_cheetah.ckpt"
comptime EVAL_EPS = 2
comptime EP_LEN = 1_000

comptime Env = HalfCheetah[DT, TERMINATE_ON_UNHEALTHY=False]
comptime Ag = TDMPC2Agent[
    TARGET, OBS, ENC, ACT, LATENT, MLP, BINS, SN, VMIN, VMAX, B, H, CAP
]


def _greedy_eval(mut ag: Ag, mut env: Env) raises -> Scalar[DT]:
    var obsbuf = alloc[Scalar[DT]](OBS)
    var actbuf = alloc[Scalar[DT]](ACT)
    var total: Scalar[DT] = 0.0
    for _ep in range(EVAL_EPS):
        var obs = env.reset_obs_list()
        comptime if USE_MPC:
            ag.mpc_start_episode()
        for _s in range(EP_LEN):
            for i in range(OBS):
                obsbuf[i] = obs[i]
            comptime if USE_MPC:
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
    return total / Scalar[DT](EVAL_EPS)


def main() raises:
    print("=" * 70)
    print("TD-MPC2 (deep_agents2) — HalfCheetah", TARGET, "(MPC-off)")
    print("  OBS=", OBS, " ACT=", ACT, " latent=", LATENT, " B=", B, " H=", H)
    print("  lr=", LR, " total=", TOTAL, " learn_start=", LEARN_START)
    print("=" * 70)
    seed(0)
    var ctx = DeviceContext()
    var env = Env()
    var ag = Ag.make(
        lr=Scalar[DT](LR), gamma=Scalar[DT](0.99), tau=Scalar[DT](0.01),
        action_scale=Scalar[DT](ACTION_SCALE), learning_starts=LEARN_START,
        ctx=ctx,
    )

    # RemoteLogger (dashboard) — URL/key from .env; no-ops if unset.
    var env_vars = load_dotenv()
    var logger = RemoteLogger(
        server_url=env_vars.get("RL_MONITOR_URL", ""),
        run_name="TD-MPC2 HalfCheetah",
        buffer_size=200,
        api_key=env_vars.get("RL_MONITOR_API_KEY", ""),
    )
    logger.set_config("algorithm", "TD-MPC2")
    logger.set_config("env", "HalfCheetah")
    var mpc_cfg = String("0")
    comptime if USE_MPC:
        mpc_cfg = String("1")
    logger.set_config("mpc", mpc_cfg)
    var logger_ptr = UnsafePointer(to=logger)

    var obs = env.reset_obs_list()
    var obsbuf = alloc[Scalar[DT]](OBS)
    var actbuf = alloc[Scalar[DT]](ACT)
    var best: Scalar[DT] = -1.0e9
    var t_start = perf_counter_ns()

    for step in range(TOTAL):
        for i in range(OBS):
            obsbuf[i] = obs[i]
        if step < LEARN_START:
            for j in range(ACT):
                actbuf[j] = Scalar[DT](random_float64() * 2.0 - 1.0)
        else:
            comptime if USE_MPC:
                ag.select_action_mpc(obsbuf, actbuf, explore=True)
            else:
                ag.select_action(obsbuf, actbuf, explore=True)
        var al = List[Scalar[DT]]()
        for j in range(ACT):
            al.append(actbuf[j])
        var res = env.step_continuous_vec[DT](al)
        # Truncation-only → record done=0 (bootstrap continues).
        ag.record(obsbuf, actbuf, res[1], Scalar[DT](0.0))
        obs = res[0].copy()
        if res[2]:
            obs = env.reset_obs_list()
            comptime if USE_MPC:
                ag.mpc_start_episode()
        if step >= LEARN_START and step % TRAIN_EVERY == 0:
            _ = ag.train_step()
        if step > 0 and step % DIAG_EVERY == 0:
            ag.flush_metrics_through_logger[RemoteLogger](logger_ptr, step)
        if step > 0 and step % CHECKPOINT_EVERY == 0:
            ag.save_state(CHECKPOINT_PATH)
        if step > 0 and step % EVAL_EVERY == 0:
            var ret = _greedy_eval(ag, env)
            if ret > best:
                best = ret
            var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
            logger.log_scalar("eval_return", Float64(ret), step)
            print(
                "  step", step, " eval_return=", ret, " best=", best,
                " wm=", ag.last_wm_loss(), " pi=", ag.last_pi_loss(),
                " (", elapsed, "s )",
            )

    ag.save_state(CHECKPOINT_PATH)
    logger.close()
    _ = logger  # lifetime extender for logger_ptr
    print("=" * 70)
    print("  FINAL best eval return =", best)
    print("  ( HalfCheetah: >3000 good, >8000 strong )")
    print("  checkpoint:", CHECKPOINT_PATH)
    print("=" * 70)
    obsbuf.free(); actbuf.free()
