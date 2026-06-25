"""DreamerV3 training on CartPole (CPU) via the `DreamerV3Agent` facade.

CartPole counterpart of `examples/humanoid/sac_humanoid_training.mojo`. Uses
the facade surface — the example writes NO training loop of its own:

  * `DreamerV3Agent[..., DISCRETE=True]` — world-model + actor-critic agent.
  * `agent.train_single[EnvT, L=RemoteLogger](env, NUM_STEPS, ...)` — owns the
    whole loop (warmup → on-policy collect → record (+terminal obs) → WM-BPTT +
    imagination AC every `train_every`), with periodic greedy eval, metric
    streaming, and one-file checkpointing.
  * `RemoteLogger` — streams `eval/mean_return` + WM/AC/con diagnostics.
  * Single-file checkpoint — `agent.save(path)` / `agent.load(path)` write/read
    ONE `nn-ckpt v2` envelope (the full world model + actor-critic).

After training, the final checkpoint is reloaded and a greedy probe confirms the
action reproduces to `|diff| < 1e-5`.

Config solves the lighthouse (mean_ret(10)=500): the termination-data fix lets
the continue head learn `latent(fall)→0`, and T_IMAG=15 + lr=1.5e-4 give the
actor enough horizon to credit the slow cart drift while staying value-stable.

Run:
    pixi run mojo run -I . examples/cartpole/cartpole_dreamerv3_training.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.dreamerv3.agent import DreamerV3Agent
from mojo_rl.envs.cartpole import CartPoleEnv


# =============================================================================
# Architecture (small RSSM; CPU-light but solves with the recipe below)
# =============================================================================
comptime EnvT = CartPoleEnv[DT]
comptime OBS = 4
comptime ACT = 2          # one-hot action dim = #actions (left/right)
comptime DETER = 128
comptime H = 32
comptime STOCH = 16
comptime CLASSES = 4
comptime BLOCKS = 4
comptime TOKEN = 32
comptime DEC_U = 32
comptime HU = 32
comptime VU = 32
comptime PU = 32
comptime BINS = 51
comptime B = 16
comptime T = 16
comptime T_IMAG = 15      # long enough to credit the slow cart drift
comptime CAP = 200_000

comptime Ag = DreamerV3Agent[
    "cpu", OBS, ACT, DETER, H, STOCH, CLASSES, BLOCKS, TOKEN, DEC_U, HU, VU,
    PU, BINS, B, T, T_IMAG, CAP, True,   # DISCRETE=True
]

comptime NUM_STEPS = 150_000
comptime LEARN_START = 1024
comptime TRAIN_EVERY = 4
comptime EVAL_EVERY = 2500
comptime EVAL_EPISODES = 10
comptime EP_LEN = 500
comptime CHECKPOINT_EVERY = 25_000
comptime CHECKPOINT_PATH = "dreamerv3_cartpole.ckpt"


def main() raises:
    seed(42)
    print("=" * 70)
    print("DreamerV3 (facade) — CartPole CPU + checkpoints + logger")
    print("=" * 70)
    print("  OBS / ACT          =", OBS, "/", ACT)
    print("  DETER/STOCH/CLASSES=", DETER, "/", STOCH, "/", CLASSES)
    print("  T / T_IMAG         =", T, "/", T_IMAG)
    print("  NUM_STEPS          =", NUM_STEPS)
    print("  CHECKPOINT_EVERY   =", CHECKPOINT_EVERY)
    print("  Checkpoint path    =", CHECKPOINT_PATH)
    print("=" * 70)

    # ─── Logger (remote) ─────────────────────────────────────────────────
    var env_vars = load_dotenv()
    var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
    var url = env_vars.get("RL_MONITOR_URL", "")
    var logger = RemoteLogger(
        server_url=url,
        run_name="DreamerV3 CartPole (CPU)",
        buffer_size=200,
        api_key=api_key,
    )
    logger.set_config("algorithm", "DreamerV3")
    logger.set_config("env", "CartPole")
    logger.set_config("t_imag", String(T_IMAG))
    var logger_ptr = UnsafePointer(to=logger)

    # ─── Agent + env ─────────────────────────────────────────────────────
    var agent = Ag.make(
        lr=Scalar[DT](1.5e-4), learning_starts=LEARN_START, warmup_steps=500,
        out_init_scale=Scalar[DT](1.0),
    )
    var env = EnvT()

    # ─── Single train() call — auto-eval + auto-log + auto-checkpoint ────
    var t_start = perf_counter_ns()
    var final_ret = agent.train_single[EnvT, L=RemoteLogger](
        env,
        NUM_STEPS,
        learn_start=LEARN_START,
        train_every=TRAIN_EVERY,
        eval_every=EVAL_EVERY,
        eval_episodes=EVAL_EPISODES,
        ep_len=EP_LEN,
        print_every=EVAL_EVERY,
        verbose=True,
        logger=logger_ptr,
        checkpoint_path=CHECKPOINT_PATH,
        checkpoint_every=CHECKPOINT_EVERY,
    )
    var elapsed_s = Float64(perf_counter_ns() - t_start) / 1e9
    logger.close()
    _ = logger  # lifetime extender for logger_ptr

    # ─── Summary ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("Training complete")
    print("  total env_steps        =", NUM_STEPS)
    print("  elapsed                =", elapsed_s, "s")
    print("  FINAL mean_ret(", EVAL_EPISODES, ")  =", final_ret)
    print("  remote points sent     =", logger.total_logged())
    if Float64(final_ret) >= 475.0:
        print("SOLVED — mean_ret >= 475.")
    elif Float64(final_ret) >= 200.0:
        print("STRONG — sustained balancing (>= 200).")
    elif Float64(final_ret) >= 50.0:
        print("LEARNING — climbing (>= 50).")
    else:
        print("EARLY — still exploring (< 50).")
    print("=" * 70)

    # ─── Save/load round-trip smoke test ─────────────────────────────────
    var probe_obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    for d in range(OBS):
        probe_obs[d] = Scalar[DT](0.05 * Float64(d) - 0.1)
    var probe_ptr = probe_obs.unsafe_ptr().as_unsafe_any_origin()
    var act_before = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    var before_ptr = act_before.unsafe_ptr().as_unsafe_any_origin()
    agent.reset_belief()
    agent.select_greedy_action(probe_ptr, before_ptr)

    agent.load(CHECKPOINT_PATH)
    var act_after = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    var after_ptr = act_after.unsafe_ptr().as_unsafe_any_origin()
    agent.reset_belief()
    agent.select_greedy_action(probe_ptr, after_ptr)

    print("Save/load round-trip on probe obs:")
    var ok = True
    for j in range(ACT):
        var diff = Float64(act_after[j] - act_before[j])
        if diff < 0:
            diff = -diff
        print("  dim", j, " before =", act_before[j], " after =", act_after[j],
              " |diff| =", diff)
        if diff > 1e-5:
            ok = False
    if ok:
        print("Round-trip OK (max |diff| < 1e-5 on every action dim).")
    else:
        print("Round-trip MISMATCH — investigate save/load semantics.")
    print("=" * 70)
    _ = probe_obs
    _ = act_before
    _ = act_after
