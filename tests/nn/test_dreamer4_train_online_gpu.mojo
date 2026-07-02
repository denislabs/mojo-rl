"""Dreamer 4 ONLINE driver — GPU-dynamics stub-env smoke gate.

Same as `test_dreamer4_train_online.mojo` but with `DYN_TARGET="gpu"`: the agent's
dynamics transformer is device-resident, the WM train step runs on GPU
(`acwm_train_step_gpu`), and the driver steps the device dynamics + host heads as
separate submodules. Heads + tokenizer + backbone stay on host. Asserts the full
three-stage run completes with a finite 5-tuple summary.

SMOKE gate on a RANDOM model — exercises the GPU wiring once, not policy quality.
Uses SMALL step counts (Metal can be flaky under sustained load on some setups).

Run: pixi run -e apple  mojo run -I . tests/nn/test_dreamer4_train_online_gpu.mojo
     pixi run -e nvidia mojo run -I . tests/nn/test_dreamer4_train_online_gpu.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

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
            o[i] = Scalar[DType.float32](Float64(((self.t * 13 + i) % 17)) / 17.0)
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

    def get_state(self) -> StubState:
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
    print("Dreamer4 online training driver smoke (GPU dynamics)")
    comptime IN_CH = 4
    comptime IMG = 16
    comptime TGT = 8
    comptime PATCH = 4
    comptime NP = (TGT // PATCH) * (TGT // PATCH)
    comptime DP = PATCH * PATCH
    comptime OBS = IN_CH * IMG * IMG

    comptime T = 3
    comptime B = 2
    comptime NSP = 4
    comptime DSP = 4
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

    var c = DeviceContext()
    var env = StubPixelEnv[OBS, NACT]()
    var logger = NoOpLogger()

    # DYN_TARGET="gpu": dynamics device-resident (make needs the ctx); heads on host.
    var agent = Dreamer4Agent[
        DSP, NSP, D_DYN, NH, T, NREG, HID_DYN, DEPTH, KMAX,
        NAGENT, NTASK, HHID, NACT, NBINS, NMTP, B, B_SELF,
        True, ADIM, AHID, K_IMAG, NCTX, "gpu",
    ].make["cpu", Xavier](Optional(c))

    var tok = Dreamer4Tokenizer[
        DP, TOK_D, TOK_NH, T, NSP, NP, DSP, TOK_HID, TOK_DEPTH, 0.5, 0.5, 7
    ].make["cpu", Xavier](None)

    var backbone = CifarBackbone[TGT, TGT].make["cpu", Xavier](None)

    var summary = run_online_dreamer4[
        IN_CH=IN_CH, IMG=IMG, TGT=TGT, PATCH=PATCH, TNP=NP, CAP=CAP,
        TOK_D=TOK_D, TOK_NH=TOK_NH, TOK_HID=TOK_HID, TOK_DEPTH=TOK_DEPTH,
        TOK_PMIN=0.5, TOK_PMAX=0.5, TOK_SEED=7, DYN_TARGET="gpu",
    ](
        agent, tok, backbone, env, logger,
        warmup_steps=20, tok_pretrain_steps=5, total_env_steps=20,
        train_every=4, imag_every=8, eval_every=10,
        dctx=Optional(c),
    )
    print("  summary (tok, wm_video, wm_bc, imag_value, eval_return) =",
          summary[0], summary[1], summary[2], summary[3], summary[4])
    var ok = (
        (summary[0] == summary[0]) and (summary[1] == summary[1])
        and (summary[2] == summary[2]) and (summary[3] == summary[3])
        and (summary[4] == summary[4])
    )
    print("  GPU-dynamics online driver ran end-to-end, losses finite:",
          "OK" if ok else "FAIL")
    assert_true(ok, "GPU online driver smoke: finite summary losses")
    print("DREAMER4 TRAIN ONLINE GPU SMOKE OK")
