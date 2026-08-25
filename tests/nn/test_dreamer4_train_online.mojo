"""Dreamer 4 ONLINE training driver — stub-env smoke gate (CPU).

Runs `run_online_dreamer4` end to end on a tiny synthetic `BoxDiscreteActionEnv`
(pixel obs, discrete actions) with tiny dims: Stage 0 warmup-collect into the ring
buffer → Stage 1 tokenizer pretrain+freeze → Stage 2 online act/step/append +
world-model (`acwm_train_step`) + continue-head + imagination (`imag_train_step`)
updates. Asserts it completes and the summary losses (tokenizer recon, WM video,
WM bc, imagination value) are finite (no NaN).

This is a SMOKE gate on a RANDOM model — it exercises the full online wiring
(buffer, act_from_latents, tokenizer pretrain/freeze, the schedule builder, the
three train steps), not policy quality.

Run: pixi run mojo run -I . tests/nn/test_dreamer4_train_online.mojo
"""

from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Xavier
from mojo_rl.core.state import State
from mojo_rl.core.action import Action
from mojo_rl.core.env_traits import BoxDiscreteActionEnv
from mojo_rl.core.logger import NoOpLogger

from mojo_rl.deep_agents.dreamer4.agent import Dreamer4Agent
from mojo_rl.deep_agents.dreamer4.tokenizer import Dreamer4Tokenizer
from mojo_rl.deep_agents.dreamer4.online import run_online_dreamer4
from mojo_rl.nn.models.cifar_feature_net import CifarBackbone


# ── trivial State / Action ─────────────────────────────────────────────────
struct StubState(State):
    var t: Int

    def __init__(out self):
        self.t = 0

    def __init__(out self, t: Int):
        self.t = t

    def __eq__(self, other: Self) -> Bool:
        return self.t == other.t


struct StubAction(Action):
    var a: Int

    def __init__(out self):
        self.a = 0

    def __init__(out self, a: Int):
        self.a = a


# ── tiny synthetic pixel BoxDiscreteActionEnv ──────────────────────────────
struct StubPixelEnv[OBS: Int, NACT: Int](BoxDiscreteActionEnv):
    comptime dtype = DType.float32
    comptime StateType = StubState
    comptime ActionType = StubAction

    var t: Int

    def __init__(out self):
        self.t = 0

    def _obs(self) -> List[Scalar[DType.float32]]:
        var o = List[Scalar[DType.float32]](
            length=Self.OBS, fill=Scalar[DType.float32](0)
        )
        for i in range(Self.OBS):
            o[i] = Scalar[DType.float32](
                Float64(((self.t * 13 + i) % 17)) / 17.0
            )
        return o^

    def reset(mut self) -> StubState:
        self.t = 0
        return StubState(0)

    def reset_obs_list(mut self) -> List[Scalar[DType.float32]]:
        self.t = 0
        return self._obs()

    def get_obs_list(self) -> List[Scalar[DType.float32]]:
        return self._obs()

    def step_obs(
        mut self, action: Int
    ) -> Tuple[List[Scalar[DType.float32]], Scalar[DType.float32], Bool]:
        self.t += 1
        var r = Scalar[DType.float32](0.1 if (self.t % 5 == 0) else 0.0)
        var d = (self.t % 20) == 0
        return (self._obs(), r, d)

    def step(
        mut self, action: StubAction, verbose: Bool = False
    ) -> Tuple[StubState, Scalar[DType.float32], Bool]:
        var res = self.step_obs(action.a)
        return (StubState(self.t), res[1], res[2])

    def get_state(mut self) -> StubState:
        return StubState(self.t)

    def close(mut self):
        pass

    def num_actions(self) -> Int:
        return Self.NACT

    def action_dim(self) -> Int:
        return 1

    def action_from_index(self, action_idx: Int) -> StubAction:
        return StubAction(action_idx)

    def obs_dim(self) -> Int:
        return Self.OBS


def main() raises:
    print("Dreamer4 online training driver smoke (CPU)")
    comptime IN_CH = 4
    comptime IMG = 16
    comptime TGT = 8
    comptime PATCH = 4
    comptime NP = (TGT // PATCH) * (TGT // PATCH)   # 4
    comptime DP = PATCH * PATCH                      # 16
    comptime OBS = IN_CH * IMG * IMG

    comptime T = 3
    comptime B = 2
    comptime NSP = 4
    comptime DSP = 4                                 # ND = 16 = L·D_BOT
    comptime D_DYN = 32
    comptime NH = 2
    comptime NREG = 1
    comptime HID_DYN = 64
    comptime DEPTH = 1
    comptime KMAX = 4
    comptime NAGENT = 1
    comptime NTASK = 1
    comptime HHID = 32
    comptime NACT = 3
    comptime NBINS = 5
    comptime NMTP = 2
    comptime B_SELF = 1
    comptime ADIM = NACT
    comptime AHID = 2 * D_DYN
    comptime K_IMAG = 2
    comptime NCTX = 1

    comptime TOK_D = 32
    comptime TOK_NH = 2
    comptime TOK_HID = 64
    comptime TOK_DEPTH = 1
    comptime CAP = 256

    var env = StubPixelEnv[OBS, NACT]()
    var logger = NoOpLogger()

    var agent = Dreamer4Agent[
        DSP, NSP, D_DYN, NH, T, NREG, HID_DYN, DEPTH, KMAX,
        NAGENT, NTASK, HHID, NACT, NBINS, NMTP, B, B_SELF,
        True, ADIM, AHID, K_IMAG, NCTX,
    ].make["cpu", Xavier](None)

    var tok = Dreamer4Tokenizer[
        DP, TOK_D, TOK_NH, T, NSP, NP, DSP, TOK_HID, TOK_DEPTH, 0.5, 0.5, 7
    ].make["cpu", Xavier](None)

    # Random backbone; perc_weight=0 (default) so it is never used (its purpose is
    # only exercised via the perceptual gate). Passing it satisfies the signature.
    var backbone = CifarBackbone[TGT, TGT].make["cpu", Xavier](None)

    var summary = run_online_dreamer4[
        IN_CH=IN_CH, IMG=IMG, TGT=TGT, PATCH=PATCH, TNP=NP, CAP=CAP,
        TOK_D=TOK_D, TOK_NH=TOK_NH, TOK_HID=TOK_HID, TOK_DEPTH=TOK_DEPTH,
        TOK_PMIN=0.5, TOK_PMAX=0.5, TOK_SEED=7,
    ](
        agent, tok, backbone, env, logger,
        warmup_steps=30, tok_pretrain_steps=10, total_env_steps=60,
        train_every=4, imag_every=20, eval_every=25, frame_repeat=2, diag=True,
        save_ckpt="d4_online_smoke",   # exercises the ckpt save path (*.ckpt is
                                       # gitignored); files land in cwd.
    )
    print("  summary (tok, wm_video, wm_bc, imag_value, eval_return) =",
          summary[0], summary[1], summary[2], summary[3], summary[4])
    var ok = (
        (summary[0] == summary[0]) and (summary[1] == summary[1])
        and (summary[2] == summary[2]) and (summary[3] == summary[3])
        and (summary[4] == summary[4])
    )
    print("  online driver ran end-to-end, losses finite:",
          "OK" if ok else "FAIL")
    assert_true(ok, "online driver smoke: finite summary losses")
    print("DREAMER4 TRAIN ONLINE SMOKE OK")
