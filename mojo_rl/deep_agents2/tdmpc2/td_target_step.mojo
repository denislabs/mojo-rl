"""TD-MPC2 TD-target step — stop-grad value targets for the WM value loss.

Reference `_td_target` + the `encode(obs[1:])` in `_update`
(`references/tdmpc2-main/tdmpc2/tdmpc2.py:242-264`):

    next_z = encode(obs[t+1])                      # stop-grad
    a ~ π(next_z)                                  # stochastic sample
    Q = min over 2 random TARGET-Q heads, two-hot decoded to scalar
    td[t] = reward[t] + γ·(1 − done[t])·Q

Forward-only (no autograd): calls the encoder, policy, RSample, the two
target-Q heads, and TwoHotDecode directly. The two target-Q heads are the
random pair the trainer selects each step (reference resamples per call);
this block receives them by ref. Output `td [H, B]` feeds `WMStep` as the
stop-grad `td` input.

Termination/episodic is deferred (port plan); `done` here is the
truncation/terminal mask used only to drop the bootstrap, matching the
non-episodic HalfCheetah setting.

CPU only (P4 = GPU).
"""

from std.memory import alloc
from std.math import min
from layout import TileTensor, row_major
from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.deep_agents2.primitives.rsample import RSample

from .nets import TDMPC2Encoder, TDMPC2Policy, TDMPC2QNet
from .losses import TwoHotDecode


@always_inline
def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


struct TDTargetStep[
    OBS: Int,
    ENC: Int,
    ACT: Int,
    LATENT: Int,
    MLP: Int,
    BINS: Int,
    SN: Int,
    VMIN: Int,
    VMAX: Int,
    B: Int,
    H: Int,
](Movable & ImplicitlyDestructible):
    comptime EncT = TDMPC2Encoder[Self.OBS, Self.ENC, Self.LATENT, Self.SN]
    comptime PolicyT = TDMPC2Policy[Self.LATENT, Self.ACT, Self.MLP]
    comptime QNetT = TDMPC2QNet[Self.LATENT, Self.ACT, Self.MLP, Self.BINS]

    var rsample: RSample[Self.ACT]
    var decode: TwoHotDecode[Self.BINS, Self.VMIN, Self.VMAX]

    def __init__(out self):
        self.rsample = RSample[Self.ACT]()
        self.decode = TwoHotDecode[Self.BINS, Self.VMIN, Self.VMAX]()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu", (
            "TDTargetStep: only the CPU path is implemented (GPU is P4)"
        )
        var s = Self()
        s.rsample = RSample[Self.ACT].make[target, INIT=Zero](ctx=ctx)
        s.decode = TwoHotDecode[
            Self.BINS, Self.VMIN, Self.VMAX
        ].make[target, INIT=Zero](ctx=ctx)
        return s^

    def step[target: StaticString](
        mut self,
        mut enc: Self.EncT,
        mut policy: Self.PolicyT,
        mut qt: List[Self.QNetT],
        qi: Int,
        qj: Int,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],     # [(H+1),B,OBS]
        reward: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [H,B]
        done: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [H,B]
        td_out: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [H,B]
        gamma: Scalar[DT],
    ) raises:
        comptime assert target == "cpu", "TDTargetStep.step: CPU only (P4=GPU)"
        comptime LAT = Self.LATENT
        comptime A = Self.ACT
        comptime ZA = LAT + A

        var nz = _alloc(Self.B * LAT)
        var pio = _alloc(Self.B * 2 * A)
        var alp = _alloc(Self.B * (A + 1))
        var za = _alloc(Self.B * ZA)
        var qlog1 = _alloc(Self.B * Self.BINS)
        var qlog2 = _alloc(Self.B * Self.BINS)
        var qa = _alloc(Self.B)
        var qb = _alloc(Self.B)

        for t in range(Self.H):
            # next_z = encode(obs[t+1])  (stop-grad)
            var nz_t = TileTensor(nz, row_major[Self.B, LAT]())
            enc.forward[target, Self.B](
                TileTensor(
                    obs + (t + 1) * Self.B * Self.OBS,
                    row_major[Self.B, Self.OBS](),
                ),
                output=nz_t,
            )
            # a ~ π(next_z)
            var pio_t = TileTensor(pio, row_major[Self.B, 2 * A]())
            policy.forward[target, Self.B](nz_t, output=pio_t)
            var alp_t = TileTensor(alp, row_major[Self.B, A + 1]())
            self.rsample.forward[target, Self.B](pio_t, output=alp_t)
            # za = [next_z | action]
            for b in range(Self.B):
                for k in range(LAT):
                    za[b * ZA + k] = nz[b * LAT + k]
                for k in range(A):
                    za[b * ZA + LAT + k] = alp[b * (A + 1) + k]
            var za_t = TileTensor(za, row_major[Self.B, ZA]())
            # Q = min of 2 target heads, two-hot decoded.
            var ql1_t = TileTensor(qlog1, row_major[Self.B, Self.BINS]())
            var qa_t = TileTensor(qa, row_major[Self.B, 1]())
            qt[qi].forward[target, Self.B](za_t, output=ql1_t)
            self.decode.forward[target, Self.B](ql1_t, output=qa_t)
            var ql2_t = TileTensor(qlog2, row_major[Self.B, Self.BINS]())
            var qb_t = TileTensor(qb, row_major[Self.B, 1]())
            qt[qj].forward[target, Self.B](za_t, output=ql2_t)
            self.decode.forward[target, Self.B](ql2_t, output=qb_t)
            for b in range(Self.B):
                var qmin = min(qa[b], qb[b])
                var d = done[t * Self.B + b]
                td_out[t * Self.B + b] = reward[t * Self.B + b] + gamma * (
                    Scalar[DT](1.0) - d
                ) * qmin

        nz.free(); pio.free(); alp.free(); za.free()
        qlog1.free(); qlog2.free(); qa.free(); qb.free()
