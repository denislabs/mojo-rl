"""Dreamer 4 OFFLINE-style CarRacing validation (the paper's setting).

Unlike the online lighthouse (which collects with its own from-scratch policy),
this drives ALL data collection — warmup AND ongoing — with a DECENT scripted
driver (`scripted_car_racing_action`, ~+100 return / 57 tiles vs random -56/1).
So the world model + BC heads train on GOOD data, giving the value/policy a real
advantage signal; the policy is trained ONLY in imagination and greedy-eval'd.

Why: Dreamer 4 is an offline method (paper + jax: pretrain WM+BC on a fixed
good dataset, then imagination-RL). Online-from-scratch on random warmup makes
the rewards near-uniform (time-penalty dominated, tiles rare) → advantage ≈ 0 →
PMPO's sign-of-advantage signal is pure noise → the policy never learns. This
run tests whether the (verified-correct) imagination-RL learns when fed good
data. SUCCESS = eval_return climbs off ~-50 toward the scripted driver's ~+100.

Run (NVIDIA): pixi run -e nvidia mojo run -I . examples/car_racing/dreamer4_car_racing_offline.mojo
"""

from std.random import seed

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Xavier
from mojo_rl.nn.core.checkpoint import load_params
from mojo_rl.nn.models.cifar_feature_net import CifarBackbone
from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.dreamer4.agent import Dreamer4Agent
from mojo_rl.deep_agents.dreamer4.tokenizer import Dreamer4Tokenizer
from mojo_rl.deep_agents.dreamer4.online import run_online_dreamer4
from mojo_rl.envs.car_racing.car_racing_mb import CarRacingMB
from mojo_rl.envs.car_racing.scripted_driver import scripted_car_racing_action


def main() raises:
    # ── image / patch dims ──
    comptime IN_CH = 4
    comptime IMG = 84                       # CarRacing PIX_RES
    comptime TGT = 64
    comptime PATCH = 8
    comptime NP = (TGT // PATCH) * (TGT // PATCH)
    comptime DP = PATCH * PATCH
    comptime CAP = 100_000

    # ── sequence / batch ──
    comptime T = 10
    comptime B = 8
    comptime B_SELF = 2

    # ── tokenizer ──
    comptime L = 16
    comptime D_BOT = 16
    comptime TOK_D = 128
    comptime TOK_NH = 4
    comptime TOK_HID = 256
    comptime TOK_DEPTH = 2
    comptime DROP = 0.5

    # ── agent / dynamics ──
    comptime NSP = L
    comptime DSP = D_BOT
    comptime ND = NSP * DSP
    comptime D_DYN = 128
    comptime NH = 4
    comptime NREG = 2
    comptime HID_DYN = 256
    comptime DEPTH_DYN = 2
    comptime KMAX = 4

    # ── heads / imagination ──
    comptime NAGENT = 1
    comptime NTASK = 1
    comptime HHID = 128
    comptime NACT = 5
    comptime NBINS = 41
    comptime NMTP = 2
    comptime ADIM = NACT
    comptime AHID = 2 * D_DYN
    comptime K_IMAG = 2
    comptime NCTX = 1

    seed(42)
    print("=" * 70)
    print("Dreamer 4 — OFFLINE CarRacing (scripted-driver dataset)")
    print("=" * 70)

    var ctx = DeviceContext()
    var env = CarRacingMB[DT, PIXEL_OBS=True, PIX_RES=IMG]()

    var agent = Dreamer4Agent[
        DSP, NSP, D_DYN, NH, T, NREG, HID_DYN, DEPTH_DYN, KMAX,
        NAGENT, NTASK, HHID, NACT, NBINS, NMTP, B, B_SELF,
        True, ADIM, AHID, K_IMAG, NCTX, "gpu",
    ].make["cpu", Xavier](Optional(ctx))

    var tok = Dreamer4Tokenizer[
        DP, TOK_D, TOK_NH, T, NSP, NP, DSP, TOK_HID, TOK_DEPTH, DROP, DROP, 7
    ].make["gpu", Xavier](Optional(ctx))

    # Perceptual backbone is unused (perc_weight=0) but the driver signature needs
    # an instance; make one (no ckpt load needed since it's never called).
    var backbone = CifarBackbone[TGT, TGT].make["gpu", Xavier](Optional(ctx))

    var env_vars = load_dotenv()
    var logger = RemoteLogger(
        server_url=env_vars.get("RL_MONITOR_URL", ""),
        run_name="Dreamer4 CarRacing (offline/scripted)",
        buffer_size=64,
        api_key=env_vars.get("RL_MONITOR_API_KEY", ""),
    )
    logger.set_config("algorithm", "Dreamer4")
    logger.set_config("env", "CarRacing")
    logger.set_config("obs", "pixel")
    logger.set_config("collect", "scripted")

    print("starting offline (scripted-collection) training...")

    # Nested capturing closure = the collection policy (Mojo passes nested
    # closures to comptime fn params; top-level named functions don't convert).
    def _collect(mut e: CarRacingMB[DT, True, IMG], step: Int) capturing raises -> Int:
        return scripted_car_racing_action(e, step)

    var summary = run_online_dreamer4[
        IN_CH=IN_CH, IMG=IMG, TGT=TGT, PATCH=PATCH, TNP=NP, CAP=CAP,
        TOK_D=TOK_D, TOK_NH=TOK_NH, TOK_HID=TOK_HID, TOK_DEPTH=TOK_DEPTH,
        TOK_PMIN=DROP, TOK_PMAX=DROP, TOK_SEED=7, DYN_TARGET="gpu",
        SCRIPTED=True, COLLECT_ACTION=_collect,
    ](
        agent, tok, backbone, env, logger,
        warmup_steps=5_000,
        tok_pretrain_steps=4_000,
        total_env_steps=200_000,
        train_every=4,
        imag_every=8,
        eval_every=10_000,
        perc_weight=0.0,
        eval_max_steps=1_000,
        num_eval_episodes=8,
        imag_gamma=Scalar[DT](0.95),
        value_bin_lo=Scalar[DT](-6.0),
        frame_repeat=4,
        diag=True,
        save_ckpt="dreamer4_carracing_offline",
        dctx=Optional(ctx),
    )
    logger.close()
    _ = logger

    print("-" * 70)
    print("done. (tok, wm_video, wm_bc, imag_value_loss, eval_return):")
    print(" ", summary[0], summary[1], summary[2], summary[3], summary[4])
    print("  last greedy-eval return =", summary[4])
