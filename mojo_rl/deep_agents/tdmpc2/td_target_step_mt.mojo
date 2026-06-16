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
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.initializer import Zero
from mojo_rl.deep_agents.primitives.rsample import RSample

from .nets_mt import TDMPC2EncoderMT, TDMPC2PolicyMT, TDMPC2QNetMT
from .losses import TwoHotDecode
from .td_target_step import _dp, _lt, _td_combine_k, _alloc
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

    # Persistent GPU scratch (allocated once in `make`, reused every step —
    # per-step `enqueue_create_buffer` explodes disk on NVIDIA).
    var d_obs: Optional[DeviceBuffer[DT]]
    var d_rew: Optional[DeviceBuffer[DT]]
    var d_done: Optional[DeviceBuffer[DT]]
    var d_tids: Optional[DeviceBuffer[DT]]
    var d_td: Optional[DeviceBuffer[DT]]
    var d_tem: Optional[DeviceBuffer[DT]]
    var d_ein: Optional[DeviceBuffer[DT]]
    var d_nz: Optional[DeviceBuffer[DT]]
    var d_pin: Optional[DeviceBuffer[DT]]
    var d_pio: Optional[DeviceBuffer[DT]]
    var d_alp: Optional[DeviceBuffer[DT]]
    var d_za: Optional[DeviceBuffer[DT]]
    var d_q1: Optional[DeviceBuffer[DT]]
    var d_q2: Optional[DeviceBuffer[DT]]
    var d_qa: Optional[DeviceBuffer[DT]]
    var d_qb: Optional[DeviceBuffer[DT]]
    var h_obs: Optional[HostBuffer[DT]]
    var h_rew: Optional[HostBuffer[DT]]
    var h_done: Optional[HostBuffer[DT]]
    var h_tids: Optional[HostBuffer[DT]]
    var h_td: Optional[HostBuffer[DT]]

    def __init__(out self):
        self.rsample = RSample[Self.MAX_ACT]()
        self.decode = TwoHotDecode[Self.BINS, Self.VMIN, Self.VMAX]()
        self.d_obs = None; self.d_rew = None; self.d_done = None
        self.d_tids = None; self.d_td = None; self.d_tem = None
        self.d_ein = None; self.d_nz = None; self.d_pin = None
        self.d_pio = None; self.d_alp = None; self.d_za = None
        self.d_q1 = None; self.d_q2 = None; self.d_qa = None
        self.d_qb = None
        self.h_obs = None; self.h_rew = None; self.h_done = None
        self.h_tids = None; self.h_td = None

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
        comptime if target == "gpu":
            var c = ctx.value()
            comptime LAT = Self.LATENT
            comptime A = Self.MAX_ACT
            comptime MO = Self.MAX_OBS
            comptime EMB = Self.TASK_EMB
            comptime AOBS = Self.AOBS
            comptime PIN = Self.PIN
            comptime ZA = Self.ZA
            comptime ALP = A + 1
            comptime BB = Self.B
            s.d_obs = c.enqueue_create_buffer[DT]((Self.H + 1) * BB * MO)
            s.d_rew = c.enqueue_create_buffer[DT](Self.H * BB)
            s.d_done = c.enqueue_create_buffer[DT](Self.H * BB)
            s.d_tids = c.enqueue_create_buffer[DT](BB)
            s.d_td = c.enqueue_create_buffer[DT](Self.H * BB)
            s.d_tem = c.enqueue_create_buffer[DT](BB * EMB)
            s.d_ein = c.enqueue_create_buffer[DT](BB * AOBS)
            s.d_nz = c.enqueue_create_buffer[DT](BB * LAT)
            s.d_pin = c.enqueue_create_buffer[DT](BB * PIN)
            s.d_pio = c.enqueue_create_buffer[DT](BB * 2 * A)
            s.d_alp = c.enqueue_create_buffer[DT](BB * ALP)
            s.d_za = c.enqueue_create_buffer[DT](BB * ZA)
            s.d_q1 = c.enqueue_create_buffer[DT](BB * Self.BINS)
            s.d_q2 = c.enqueue_create_buffer[DT](BB * Self.BINS)
            s.d_qa = c.enqueue_create_buffer[DT](BB)
            s.d_qb = c.enqueue_create_buffer[DT](BB)
            s.h_obs = c.enqueue_create_host_buffer[DT]((Self.H + 1) * BB * MO)
            s.h_rew = c.enqueue_create_host_buffer[DT](Self.H * BB)
            s.h_done = c.enqueue_create_host_buffer[DT](Self.H * BB)
            s.h_tids = c.enqueue_create_host_buffer[DT](BB)
            s.h_td = c.enqueue_create_host_buffer[DT](Self.H * BB)
            c.synchronize()
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

        # Reuse persistent scratch (allocated once in `make`). Upload host
        # inputs through the cached pinned host buffers.
        var d_obs = self.d_obs.value()
        var d_rew = self.d_rew.value()
        var d_done = self.d_done.value()
        var d_tids = self.d_tids.value()
        var d_td = self.d_td.value()
        var d_tem = self.d_tem.value()
        var d_ein = self.d_ein.value()
        var d_nz = self.d_nz.value()
        var d_pin = self.d_pin.value()
        var d_pio = self.d_pio.value()
        var d_alp = self.d_alp.value()
        var d_za = self.d_za.value()
        var d_q1 = self.d_q1.value()
        var d_q2 = self.d_q2.value()
        var d_qa = self.d_qa.value()
        var d_qb = self.d_qb.value()
        var h_obs = self.h_obs.value()
        var h_rew = self.h_rew.value()
        var h_done = self.h_done.value()
        var h_tids = self.h_tids.value()

        var n_obs = (Self.H + 1) * BB * MO
        for i in range(n_obs):
            h_obs.unsafe_ptr()[i] = obs[i]
        for i in range(Self.H * BB):
            h_rew.unsafe_ptr()[i] = reward[i]
            h_done.unsafe_ptr()[i] = done[i]
        for i in range(BB):
            h_tids.unsafe_ptr()[i] = task_ids[i]
        ctx.enqueue_copy(d_obs, h_obs)
        ctx.enqueue_copy(d_rew, h_rew)
        ctx.enqueue_copy(d_done, h_done)
        ctx.enqueue_copy(d_tids, h_tids)

        task_emb.gather[target, BB](_dp(d_tids), _dp(d_tem), ctx=ctx)

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

        var h_td = self.h_td.value()
        ctx.enqueue_copy(h_td, d_td)
        ctx.synchronize()
        for i in range(Self.H * BB):
            td_out[i] = h_td.unsafe_ptr()[i]
