"""TD-MPC2 multi-task TD-target step (CPU + GPU) — item C, §14.3.

Clone of `td_target_step.mojo` with the task embedding concatenated into the
encoder input (`[obs|tem]`), the policy input (`[next_z|tem]`), and the Q input
(`za = [next_z|action|tem]`). Forward-only / stop-grad, so it contributes NO
gradient to the embedding table (matches the reference: targets are detached).
See `td_target_step.mojo` for the bootstrap math.
"""

from std.memory import alloc
from std.math import min
from layout import Layout, LayoutTensor, TileTensor, row_major
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn2.constants import DT, TPB
from mojo_rl.nn2.initializer import Zero
from mojo_rl.deep_agents2.primitives.rsample import RSample

from .nets_mt import TDMPC2EncoderMT, TDMPC2PolicyMT, TDMPC2QNetMT
from .losses import TwoHotDecode
from .td_target_step import _dp, _lt, _upload, _td_combine_k, _alloc
from .task_embedding import TaskEmbedding


def _cat2_k[B_: Int, A_: Int, BD_: Int](
    a: LayoutTensor[DT, Layout.row_major(B_ * A_), MutAnyOrigin],
    b: LayoutTensor[DT, Layout.row_major(B_ * BD_), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(B_ * (A_ + BD_)), MutAnyOrigin],
):
    """dst[row] = [a[row] | b[row]] (both per-row, widths A_ and BD_)."""
    var i = Int(global_idx.x)
    var W = A_ + BD_
    if i < B_ * W:
        var row = i // W
        var c = i % W
        if c < A_:
            dst[i] = rebind[Scalar[DT]](a[row * A_ + c])
        else:
            dst[i] = rebind[Scalar[DT]](b[row * BD_ + (c - A_)])


def _build_za3_k[B_: Int, LAT_: Int, A_: Int, ALP_: Int, EMB_: Int](
    nz: LayoutTensor[DT, Layout.row_major(B_ * LAT_), MutAnyOrigin],
    alp: LayoutTensor[DT, Layout.row_major(B_ * ALP_), MutAnyOrigin],
    tem: LayoutTensor[DT, Layout.row_major(B_ * EMB_), MutAnyOrigin],
    za: LayoutTensor[DT, Layout.row_major(B_ * (LAT_ + A_ + EMB_)), MutAnyOrigin],
):
    """za[row] = [nz | action(alp[:A_]) | tem]."""
    var i = Int(global_idx.x)
    var W = LAT_ + A_ + EMB_
    if i < B_ * W:
        var row = i // W
        var c = i % W
        if c < LAT_:
            za[i] = rebind[Scalar[DT]](nz[row * LAT_ + c])
        elif c < LAT_ + A_:
            za[i] = rebind[Scalar[DT]](alp[row * ALP_ + (c - LAT_)])
        else:
            za[i] = rebind[Scalar[DT]](tem[row * EMB_ + (c - LAT_ - A_)])


struct TDTargetStepMT[
    MAX_OBS: Int,
    ENC: Int,
    MAX_ACT: Int,
    LATENT: Int,
    MLP: Int,
    BINS: Int,
    SN: Int,
    VMIN: Int,
    VMAX: Int,
    B: Int,
    H: Int,
    NUM_TASKS: Int,
    TASK_EMB: Int,
    QP: Float64 = 0.0,
](Movable & ImplicitlyDestructible):
    comptime AOBS = Self.MAX_OBS + Self.TASK_EMB
    comptime PIN = Self.LATENT + Self.TASK_EMB          # policy input width
    comptime ZA = Self.LATENT + Self.MAX_ACT + Self.TASK_EMB
    comptime EncT = TDMPC2EncoderMT[
        Self.MAX_OBS, Self.ENC, Self.LATENT, Self.SN, Self.TASK_EMB
    ]
    comptime PolicyT = TDMPC2PolicyMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.TASK_EMB
    ]
    comptime QNetT = TDMPC2QNetMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.BINS, Self.TASK_EMB, Self.QP
    ]
    comptime EmbT = TaskEmbedding[Self.NUM_TASKS, Self.TASK_EMB]

    var rsample: RSample[Self.MAX_ACT]
    var decode: TwoHotDecode[Self.BINS, Self.VMIN, Self.VMAX]

    def __init__(out self):
        self.rsample = RSample[Self.MAX_ACT]()
        self.decode = TwoHotDecode[Self.BINS, Self.VMIN, Self.VMAX]()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "TDTargetStepMT: target must be 'cpu' or 'gpu'"
        )
        var s = Self()
        s.rsample = RSample[Self.MAX_ACT].make[target, INIT=Zero](ctx=ctx)
        s.decode = TwoHotDecode[
            Self.BINS, Self.VMIN, Self.VMAX
        ].make[target, INIT=Zero](ctx=ctx)
        return s^

    def step[target: StaticString](
        mut self,
        mut enc: Self.EncT,
        mut policy: Self.PolicyT,
        mut qt: List[Self.QNetT],
        mut task_emb: Self.EmbT,
        qi: Int,
        qj: Int,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],       # [(H+1),B,MAX_OBS]
        reward: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [H,B]
        done: UnsafePointer[Scalar[DT], MutAnyOrigin],      # [H,B]
        td_out: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [H,B]
        task_ids: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B] per window
        gamma: Scalar[DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime if target == "cpu":
            self._td_cpu[target](
                enc, policy, qt, task_emb, qi, qj, obs, reward, done, td_out,
                task_ids, gamma,
            )
        else:
            self._td_gpu[target](
                enc, policy, qt, task_emb, qi, qj, obs, reward, done, td_out,
                task_ids, gamma, ctx.value(),
            )

    def _td_cpu[target: StaticString](
        mut self,
        mut enc: Self.EncT,
        mut policy: Self.PolicyT,
        mut qt: List[Self.QNetT],
        mut task_emb: Self.EmbT,
        qi: Int,
        qj: Int,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        reward: UnsafePointer[Scalar[DT], MutAnyOrigin],
        done: UnsafePointer[Scalar[DT], MutAnyOrigin],
        td_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        task_ids: UnsafePointer[Scalar[DT], MutAnyOrigin],
        gamma: Scalar[DT],
    ) raises:
        comptime LAT = Self.LATENT
        comptime A = Self.MAX_ACT
        comptime MO = Self.MAX_OBS
        comptime EMB = Self.TASK_EMB
        comptime AOBS = Self.AOBS
        comptime PIN = Self.PIN
        comptime ZA = Self.ZA

        var tem = _alloc(Self.B * EMB)
        task_emb.gather[target, Self.B](task_ids, tem)

        var ein = _alloc(Self.B * AOBS)
        var nz = _alloc(Self.B * LAT)
        var pin = _alloc(Self.B * PIN)
        var pio = _alloc(Self.B * 2 * A)
        var alp = _alloc(Self.B * (A + 1))
        var za = _alloc(Self.B * ZA)
        var qlog1 = _alloc(Self.B * Self.BINS)
        var qlog2 = _alloc(Self.B * Self.BINS)
        var qa = _alloc(Self.B)
        var qb = _alloc(Self.B)

        for t in range(Self.H):
            # ein = [obs[t+1] | tem]
            var obs_src = obs + (t + 1) * Self.B * MO
            for b in range(Self.B):
                for i in range(MO):
                    ein[b * AOBS + i] = obs_src[b * MO + i]
                for e in range(EMB):
                    ein[b * AOBS + MO + e] = tem[b * EMB + e]
            var nz_t = TileTensor(nz, row_major[Self.B, LAT]())
            enc.forward[target, Self.B](
                TileTensor(ein, row_major[Self.B, AOBS]()), output=nz_t,
            )
            # pin = [nz | tem]
            for b in range(Self.B):
                for k in range(LAT):
                    pin[b * PIN + k] = nz[b * LAT + k]
                for e in range(EMB):
                    pin[b * PIN + LAT + e] = tem[b * EMB + e]
            var pio_t = TileTensor(pio, row_major[Self.B, 2 * A]())
            policy.forward[target, Self.B](
                TileTensor(pin, row_major[Self.B, PIN]()), output=pio_t,
            )
            var alp_t = TileTensor(alp, row_major[Self.B, A + 1]())
            self.rsample.forward[target, Self.B](pio_t, output=alp_t)
            # za = [nz | action | tem]
            for b in range(Self.B):
                for k in range(LAT):
                    za[b * ZA + k] = nz[b * LAT + k]
                for k in range(A):
                    za[b * ZA + LAT + k] = alp[b * (A + 1) + k]
                for e in range(EMB):
                    za[b * ZA + LAT + A + e] = tem[b * EMB + e]
            var za_t = TileTensor(za, row_major[Self.B, ZA]())
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

        ein.free(); nz.free(); pin.free(); pio.free(); alp.free(); za.free()
        qlog1.free(); qlog2.free(); qa.free(); qb.free(); tem.free()

    def _td_gpu[target: StaticString](
        mut self,
        mut enc: Self.EncT,
        mut policy: Self.PolicyT,
        mut qt: List[Self.QNetT],
        mut task_emb: Self.EmbT,
        qi: Int,
        qj: Int,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        reward: UnsafePointer[Scalar[DT], MutAnyOrigin],
        done: UnsafePointer[Scalar[DT], MutAnyOrigin],
        td_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        task_ids: UnsafePointer[Scalar[DT], MutAnyOrigin],
        gamma: Scalar[DT],
        ctx: DeviceContext,
    ) raises:
        comptime LAT = Self.LATENT
        comptime A = Self.MAX_ACT
        comptime MO = Self.MAX_OBS
        comptime EMB = Self.TASK_EMB
        comptime AOBS = Self.AOBS
        comptime PIN = Self.PIN
        comptime ZA = Self.ZA
        comptime ALP = A + 1
        comptime BB = Self.B

        var d_obs = _upload(ctx, obs, (Self.H + 1) * BB * MO)
        var d_rew = _upload(ctx, reward, Self.H * BB)
        var d_done = _upload(ctx, done, Self.H * BB)
        var d_tids = _upload(ctx, task_ids, BB)
        var d_td = ctx.enqueue_create_buffer[DT](Self.H * BB)

        var d_tem = ctx.enqueue_create_buffer[DT](BB * EMB)
        task_emb.gather[target, BB](_dp(d_tids), _dp(d_tem), ctx=ctx)

        var d_ein = ctx.enqueue_create_buffer[DT](BB * AOBS)
        var d_nz = ctx.enqueue_create_buffer[DT](BB * LAT)
        var d_pin = ctx.enqueue_create_buffer[DT](BB * PIN)
        var d_pio = ctx.enqueue_create_buffer[DT](BB * 2 * A)
        var d_alp = ctx.enqueue_create_buffer[DT](BB * ALP)
        var d_za = ctx.enqueue_create_buffer[DT](BB * ZA)
        var d_q1 = ctx.enqueue_create_buffer[DT](BB * Self.BINS)
        var d_q2 = ctx.enqueue_create_buffer[DT](BB * Self.BINS)
        var d_qa = ctx.enqueue_create_buffer[DT](BB)
        var d_qb = ctx.enqueue_create_buffer[DT](BB)

        comptime nb_ein = (BB * AOBS + TPB - 1) // TPB
        comptime nb_pin = (BB * PIN + TPB - 1) // TPB
        comptime nb_za = (BB * ZA + TPB - 1) // TPB
        comptime nb_b = (BB + TPB - 1) // TPB
        comptime ein_k = _cat2_k[BB, MO, EMB]
        comptime pin_k = _cat2_k[BB, LAT, EMB]
        comptime za_k = _build_za3_k[BB, LAT, A, ALP, EMB]
        comptime td_k = _td_combine_k[BB]

        for t in range(Self.H):
            ctx.enqueue_function[ein_k](
                _lt[BB * MO](_dp(d_obs) + (t + 1) * BB * MO),
                _lt[BB * EMB](_dp(d_tem)),
                _lt[BB * AOBS](_dp(d_ein)),
                grid_dim=nb_ein, block_dim=TPB,
            )
            var nz_t = TileTensor(_dp(d_nz), row_major[BB, LAT]())
            enc.forward[target, BB](
                TileTensor(_dp(d_ein), row_major[BB, AOBS]()), output=nz_t,
            )
            ctx.enqueue_function[pin_k](
                _lt[BB * LAT](_dp(d_nz)), _lt[BB * EMB](_dp(d_tem)),
                _lt[BB * PIN](_dp(d_pin)), grid_dim=nb_pin, block_dim=TPB,
            )
            var pio_t = TileTensor(_dp(d_pio), row_major[BB, 2 * A]())
            policy.forward[target, BB](
                TileTensor(_dp(d_pin), row_major[BB, PIN]()), output=pio_t,
            )
            var alp_t = TileTensor(_dp(d_alp), row_major[BB, ALP]())
            self.rsample.forward[target, BB](pio_t, output=alp_t)
            ctx.enqueue_function[za_k](
                _lt[BB * LAT](_dp(d_nz)), _lt[BB * ALP](_dp(d_alp)),
                _lt[BB * EMB](_dp(d_tem)), _lt[BB * ZA](_dp(d_za)),
                grid_dim=nb_za, block_dim=TPB,
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
