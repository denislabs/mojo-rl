"""DreamerV3 trainer-blocks CPU smoke gate (storage migration).

Constructs DreamerState + WMStep + ParamSyncStep + ACStep + every graph/net +
one DreamerOpt per trained module at TINY dims, fills DreamerState.mb_* with
deterministic synthetic replay, then runs a few iterations of:

    wmstep.step["cpu", T_IMAG](...)
    paramsync.step["cpu"](core, imagine)
    acstep.step["cpu", ...](...)

and asserts last_wm_loss / last_ac_loss are FINITE and NONZERO.

Run: rm -f mojo_rl.mojoc && pixi run -e apple mojo run -I . \
       tests/nn/test_dreamerv3_blocks_cpu_smoke.mojo
"""

from std.testing import assert_true
from std.math import isfinite

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.optimizer.dreamer_opt import DreamerOpt
from mojo_rl.deep_agents.dreamerv3.twohot import symexp_twohot_bins
from mojo_rl.deep_agents.dreamerv3.normalize import PercentileNormalize
from mojo_rl.deep_agents.dreamerv3.blocks import (
    DreamerState, WMStep, ParamSyncStep, ACStep,
)
from mojo_rl.deep_agents.dreamerv3.wm import (
    WMCoreGraph, WMImagineGraph, DecLossGraph, RewLossGraph, ConLossGraph,
)
from mojo_rl.deep_agents.dreamerv3.nets import (
    DreamerEncoder, DreamerValue, DreamerPolicyHead,
)
from mojo_rl.nn.primitives.ops.swish_op import SwishOp


comptime OBS = 3
comptime ACT = 1
comptime DETER = 8
comptime H = 8
comptime STOCH = 3
comptime CLASSES = 4
comptime SC = STOCH * CLASSES
comptime BLOCKS = 2
comptime TOKEN = 6
comptime DEC_U = 8
comptime HU = 8
comptime VU = 8
comptime PU = 8
comptime BINS = 7
comptime B = 4
comptime T = 3
comptime T_IMAG = 3


def _abs(x: Scalar[DT]) -> Scalar[DT]:
    return x if x >= Scalar[DT](0) else -x


def main() raises:
    print("DreamerV3 trainer-blocks CPU smoke (storage)")
    comptime target = "cpu"

    # ── modules / graphs ──
    var enc = DreamerEncoder[OBS, TOKEN, SwishOp].make[target, Deterministic](None)
    var core = WMCoreGraph[
        DETER, H, STOCH, CLASSES, BLOCKS, ACT, TOKEN, SwishOp
    ].make[target, Deterministic](None)
    var dec = DecLossGraph[SC, DETER, OBS, DEC_U, SwishOp].make[
        target, Deterministic
    ](None)
    var rew = RewLossGraph[DETER, SC, HU, BINS, SwishOp].make[
        target, Deterministic
    ](None)
    var con = ConLossGraph[DETER, SC, HU, SwishOp].make[target, Deterministic](None)
    var imagine = WMImagineGraph[
        DETER, H, STOCH, CLASSES, BLOCKS, ACT, SwishOp
    ].make[target, Deterministic](None)
    var value = DreamerValue[DETER + SC, VU, BINS, SwishOp].make[
        target, Deterministic
    ](None)
    var slowvalue = DreamerValue[DETER + SC, VU, BINS, SwishOp].make[
        target, Deterministic
    ](None)
    var policy = DreamerPolicyHead[DETER + SC, PU, ACT, False, SwishOp].make[
        target, Deterministic
    ](None)

    # ── one DreamerOpt per trained module ──
    var oe = DreamerOpt(lr=Scalar[DT](1e-4))
    var ocore = DreamerOpt(lr=Scalar[DT](1e-4))
    var odec = DreamerOpt(lr=Scalar[DT](1e-4))
    var orew = DreamerOpt(lr=Scalar[DT](1e-4))
    var ocon = DreamerOpt(lr=Scalar[DT](1e-4))
    var oval = DreamerOpt(lr=Scalar[DT](1e-4))
    var opol = DreamerOpt(lr=Scalar[DT](1e-4))

    # ── blocks ──
    var wmstep = WMStep[
        OBS, ACT, DETER, H, STOCH, CLASSES, BLOCKS, TOKEN, DEC_U, HU, BINS, B, T
    ].make[target](None)
    var paramsync = ParamSyncStep[
        DETER, H, STOCH, CLASSES, BLOCKS, ACT, TOKEN
    ].make[target](None)
    var acstep = ACStep[
        OBS, ACT, DETER, H, STOCH, CLASSES, BLOCKS, TOKEN, HU, VU, PU, BINS,
        B, T, T_IMAG, False,
    ].make[target](None)

    var st = DreamerState[OBS, ACT, DETER, SC, TOKEN, B, T, T_IMAG].make[target](None)

    # ── twohot bins + retnorm ──
    var binsT = Tensor.alloc(BINS)
    symexp_twohot_bins[BINS](binsT.data.unsafe_ptr(), Scalar[DT](-9.0))
    var bins = rebind[Pointer[Scalar[DT], MutAnyOrigin]](
        binsT.data.unsafe_ptr()
    )
    var retnorm = PercentileNormalize.make(String("perc"))

    # ── deterministic synthetic replay ──
    for i in range(B * (T + 1) * OBS):
        st.mb_obs.data[i] = Scalar[DT]((i % 7) - 3) * 0.13
    for i in range(B * T * ACT):
        st.mb_act.data[i] = Scalar[DT]((i % 5) - 2) * 0.25
    for i in range(B * T):
        st.mb_rew.data[i] = Scalar[DT]((i % 4) - 1) * 0.4
        st.mb_dne.data[i] = Scalar[DT](1.0) if (i % 11 == 10) else Scalar[DT](0.0)
    for i in range(T_IMAG * T * B * ACT):
        st.noise.data[i] = Scalar[DT](((i * 3) % 9) - 4) * 0.2

    var ok = True
    for it in range(3):
        wmstep.step[target, T_IMAG](
            st, enc, core, dec, rew, con, oe, ocore, odec, orew, ocon
        )
        paramsync.step[target](core, imagine)
        acstep.step[target](
            st, imagine, value, slowvalue, policy, rew, con,
            oval, opol, retnorm, bins,
        )
        var wm = st.last_wm_loss
        var ac = st.last_ac_loss
        print(
            "  iter", it, "wm_loss =", wm, " ac_loss =", ac,
            " ret_mean =", st.dbg_ret_mean,
        )
        if not isfinite(Float64(wm)):
            ok = False
        if not isfinite(Float64(ac)):
            ok = False
        if _abs(wm) < Scalar[DT](1e-12):
            ok = False
        if _abs(ac) < Scalar[DT](1e-12):
            ok = False

    assert_true(ok, "DreamerV3 blocks CPU smoke: losses finite + nonzero")
    print("DREAMERV3 BLOCKS CPU SMOKE OK")
