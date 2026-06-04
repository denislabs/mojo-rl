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
from layout import Layout, LayoutTensor, TileTensor, row_major
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn2.constants import DT, TPB
from mojo_rl.nn2.initializer import Zero
from mojo_rl.deep_agents2.primitives.rsample import RSample

from .nets import TDMPC2Encoder, TDMPC2Policy, TDMPC2QNet
from .losses import TwoHotDecode


@always_inline
def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


@always_inline
def _dp(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


@always_inline
def _lt[N: Int](
    p: UnsafePointer[Scalar[DT], MutAnyOrigin]
) -> LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin]:
    return LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](p)


def _upload(
    ctx: DeviceContext, src: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int
) raises -> DeviceBuffer[DT]:
    var d = ctx.enqueue_create_buffer[DT](n)
    var h = ctx.enqueue_create_host_buffer[DT](n)
    ctx.synchronize()
    for i in range(n):
        h.unsafe_ptr()[i] = src[i]
    ctx.enqueue_copy(d, h)
    ctx.synchronize()
    return d^


def _build_za_k[B_: Int, LAT_: Int, A_: Int, ALP_: Int](
    nz: LayoutTensor[DT, Layout.row_major(B_ * LAT_), MutAnyOrigin],
    alp: LayoutTensor[DT, Layout.row_major(B_ * ALP_), MutAnyOrigin],
    za: LayoutTensor[DT, Layout.row_major(B_ * (LAT_ + A_)), MutAnyOrigin],
):
    """za[b] = [nz[b] | action[b]] where action = alp[b, :A_]."""
    var i = Int(global_idx.x)
    var ZA = LAT_ + A_
    if i < B_ * ZA:
        var b = i // ZA
        var k = i % ZA
        if k < LAT_:
            za[i] = rebind[Scalar[DT]](nz[b * LAT_ + k])
        else:
            za[i] = rebind[Scalar[DT]](alp[b * ALP_ + (k - LAT_)])


def _td_combine_k[B_: Int](
    rew: LayoutTensor[DT, Layout.row_major(B_), MutAnyOrigin],
    done: LayoutTensor[DT, Layout.row_major(B_), MutAnyOrigin],
    qa: LayoutTensor[DT, Layout.row_major(B_), MutAnyOrigin],
    qb: LayoutTensor[DT, Layout.row_major(B_), MutAnyOrigin],
    td_out: LayoutTensor[DT, Layout.row_major(B_), MutAnyOrigin],
    gamma: Scalar[DT],
):
    """td[b] = rew[b] + gamma·(1−done[b])·min(qa,qb)."""
    var b = Int(global_idx.x)
    if b < B_:
        var a = rebind[Scalar[DT]](qa[b])
        var bb = rebind[Scalar[DT]](qb[b])
        var qm = a if a < bb else bb
        td_out[b] = rebind[Scalar[DT]](rew[b]) + gamma * (
            Scalar[DT](1.0) - rebind[Scalar[DT]](done[b])
        ) * qm


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
    QP: Float64 = 0.0,
](Movable & ImplicitlyDestructible):
    comptime EncT = TDMPC2Encoder[Self.OBS, Self.ENC, Self.LATENT, Self.SN]
    comptime PolicyT = TDMPC2Policy[Self.LATENT, Self.ACT, Self.MLP]
    comptime QNetT = TDMPC2QNet[Self.LATENT, Self.ACT, Self.MLP, Self.BINS, Self.QP]

    var rsample: RSample[Self.ACT]
    var decode: TwoHotDecode[Self.BINS, Self.VMIN, Self.VMAX]

    def __init__(out self):
        self.rsample = RSample[Self.ACT]()
        self.decode = TwoHotDecode[Self.BINS, Self.VMIN, Self.VMAX]()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "TDTargetStep: target must be 'cpu' or 'gpu'"
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
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime if target == "cpu":
            self._td_cpu[target](
                enc, policy, qt, qi, qj, obs, reward, done, td_out, gamma
            )
        else:
            self._td_gpu[target](
                enc, policy, qt, qi, qj, obs, reward, done, td_out, gamma,
                ctx.value(),
            )

    def _td_cpu[target: StaticString](
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

    def _td_gpu[target: StaticString](
        mut self,
        mut enc: Self.EncT,
        mut policy: Self.PolicyT,
        mut qt: List[Self.QNetT],
        qi: Int,
        qj: Int,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],     # host [(H+1),B,OBS]
        reward: UnsafePointer[Scalar[DT], MutAnyOrigin],  # host [H,B]
        done: UnsafePointer[Scalar[DT], MutAnyOrigin],    # host [H,B]
        td_out: UnsafePointer[Scalar[DT], MutAnyOrigin],  # host [H,B] (written)
        gamma: Scalar[DT],
        ctx: DeviceContext,
    ) raises:
        comptime LAT = Self.LATENT
        comptime A = Self.ACT
        comptime ZA = LAT + A
        comptime ALP = A + 1
        comptime BB = Self.B

        var d_obs = _upload(ctx, obs, (Self.H + 1) * BB * Self.OBS)
        var d_rew = _upload(ctx, reward, Self.H * BB)
        var d_done = _upload(ctx, done, Self.H * BB)
        var d_td = ctx.enqueue_create_buffer[DT](Self.H * BB)

        var d_nz = ctx.enqueue_create_buffer[DT](BB * LAT)
        var d_pio = ctx.enqueue_create_buffer[DT](BB * 2 * A)
        var d_alp = ctx.enqueue_create_buffer[DT](BB * ALP)
        var d_za = ctx.enqueue_create_buffer[DT](BB * ZA)
        var d_q1 = ctx.enqueue_create_buffer[DT](BB * Self.BINS)
        var d_q2 = ctx.enqueue_create_buffer[DT](BB * Self.BINS)
        var d_qa = ctx.enqueue_create_buffer[DT](BB)
        var d_qb = ctx.enqueue_create_buffer[DT](BB)

        comptime nb_za = (BB * ZA + TPB - 1) // TPB
        comptime nb_b = (BB + TPB - 1) // TPB
        comptime za_k = _build_za_k[BB, LAT, A, ALP]
        comptime td_k = _td_combine_k[BB]

        for t in range(Self.H):
            var nz_t = TileTensor(_dp(d_nz), row_major[BB, LAT]())
            enc.forward[target, BB](
                TileTensor(
                    _dp(d_obs) + (t + 1) * BB * Self.OBS,
                    row_major[BB, Self.OBS](),
                ),
                output=nz_t,
            )
            var pio_t = TileTensor(_dp(d_pio), row_major[BB, 2 * A]())
            policy.forward[target, BB](nz_t, output=pio_t)
            var alp_t = TileTensor(_dp(d_alp), row_major[BB, ALP]())
            self.rsample.forward[target, BB](pio_t, output=alp_t)
            ctx.enqueue_function[za_k](
                _lt[BB * LAT](_dp(d_nz)), _lt[BB * ALP](_dp(d_alp)),
                _lt[BB * ZA](_dp(d_za)), grid_dim=nb_za, block_dim=TPB,
            )
            var za_t = TileTensor(_dp(d_za), row_major[BB, ZA]())
            var q1_t = TileTensor(_dp(d_q1), row_major[BB, Self.BINS]())
            var qa_t = TileTensor(_dp(d_qa), row_major[BB, 1]())
            qt[qi].forward[target, BB](za_t, output=q1_t)
            self.decode.forward[target, BB](q1_t, output=qa_t)
            var q2_t = TileTensor(_dp(d_q2), row_major[BB, Self.BINS]())
            var qb_t = TileTensor(_dp(d_qb), row_major[BB, 1]())
            qt[qj].forward[target, BB](za_t, output=q2_t)
            self.decode.forward[target, BB](q2_t, output=qb_t)
            ctx.enqueue_function[td_k](
                _lt[BB](_dp(d_rew) + t * BB),
                _lt[BB](_dp(d_done) + t * BB),
                _lt[BB](_dp(d_qa)), _lt[BB](_dp(d_qb)),
                _lt[BB](_dp(d_td) + t * BB),
                gamma, grid_dim=nb_b, block_dim=TPB,
            )

        var h = ctx.enqueue_create_host_buffer[DT](Self.H * BB)
        ctx.enqueue_copy(h, d_td)
        ctx.synchronize()
        for i in range(Self.H * BB):
            td_out[i] = h.unsafe_ptr()[i]
