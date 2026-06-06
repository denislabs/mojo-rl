"""Dreamer4Dynamics — interactive dynamics model (model.py:Dynamics).

The dynamics transformer runs the *same* block-causal backbone as the
tokenizer, but over an interleaved per-frame token sequence

    [ action | signal | step | spatial(n_spatial) | register(n_register) ]

(the agent tokens of §3.3 are added in Phase 3; here `n_agent = 0`). With no
latents and no agent tokens, the `wm_agent_isolated` space mask reduces to
**full mixing** (`"wm_agent"`) and `latents_only_time=False` makes the time
layer attend over *all* S tokens — which `TimeAttentionLatents[D,NH,T,S, L=S]`
already gives (every token treated as a latent ⇒ no zeroing). So the backbone
is `Dreamer4Stack[..., L=S, MODE="wm_agent"]` with **no new block code**.

The module splits into a bespoke **front-end** (assemble the S-token grid from
the spatial projection, the learned action base, the step/signal embedding
tables, and the learned register tokens) and a pure-combinator **tail**
(`SinusoidalPosAddBT → Dreamer4Stack → Slice(spatial) → Tokenwise[ZeroLinear]`).
The tail owns the whole transformer + the zero-init `flow_x_head` (ZeroLinear
⇒ x-prediction starts at 0, the shortcut-forcing fixed point). The front-end
hand-rolls the concat/gather and its vjp.

Driven directly by the shortcut-forcing loss (not a Trainer). The per-sample
signal/step indices are *host-known control inputs*, so they are pushed via
`set_indices(sig, step, BATCH)` before each forward rather than threaded as
heterogeneous tensor inputs:
    set_indices(sig, step)            sig/step : [B·T] fp buffers (exact ints)
    forward(input=z̃)                  z̃        : [B·T, n_spatial·d_spatial]
    output = x̂1 (x-prediction)         [B·T, n_spatial·d_spatial]

ACTION CONDITIONING: this first cut is **unconditional** (the reference's
`actions is None` path — a single learned base token), which still yields a
valid video world model and isolates the shortcut-forcing novelty. The
continuous-action MLP encoder (`ActionEncoder`) is a follow-up; it only
changes the action token, not the loss or backbone.

PHASE 2: CPU forward + vjp (2.2). GPU forward + vjp (2.4) — the tail
(Sequential) is already GPU-capable; the bespoke front-end gets device
kernels for token assembly + its vjp (grad-input gather + four param-grad
kernels). The signal/step embedding-table grads use index-masked batch
reductions (one thread per (vocab_row, channel), looping the small batch) so
no atomics are needed — the signal/step vocabs are tiny (KMAX+1, log2 KMAX+1).
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn2.constants import DT, TPB
from mojo_rl.nn2.core import (
    Initializer, AMPPolicy, NoAMP, Param, ParamVisitor,
    for_each_param_auto, zero_grad_auto,
)
from mojo_rl.nn2.core.module import Module, typed_view, typed_view_mut
from mojo_rl.nn2.core.target_storage import (
    TargetStorage, assert_tag_for, ensure_cpu_buffer,
)
from mojo_rl.nn2.combinators import Sequential, Tokenwise
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.slice import Slice
from mojo_rl.nn2.primitives.zero_linear import ZeroLinear
from mojo_rl.nn2.primitives.sinusoidal_pos_bt import SinusoidalPosAddBT
from .blocks import Dreamer4Stack
from .shortcut_loss import ShortcutDynamics


def _ilog2(n: Int) -> Int:
    var k = 0
    var v = n
    while v > 1:
        v //= 2
        k += 1
    return k


def _dev_tile[
    BATCH: Int, N: Int
](buf: DeviceBuffer[DT]) -> TileTensor[
    DT, type_of(row_major[BATCH, N]()), MutAnyOrigin
]:
    return TileTensor(
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](buf.unsafe_ptr()),
        row_major[BATCH, N](),
    )


def _init_param_gpu[
    INIT: Initializer
](
    ctx: DeviceContext, dev: DeviceBuffer[DT], size: Int, rows: Int, cols: Int,
) raises:
    """Host-init a param then upload (Xavier etc. use the host RNG)."""
    var host = ctx.enqueue_create_host_buffer[DT](size)
    ctx.synchronize()
    INIT.init_weight(host.unsafe_ptr(), size, rows, cols)
    ctx.enqueue_copy(dev, host)
    ctx.synchronize()


# ── GPU front-end kernels ───────────────────────────────────────────────
# Assemble the (B·T, S·D) token grid: [action | signal | step | spatial | reg].
def _dyn_assemble_kernel[
    BT: Int, S: Int, D: Int, NSP: Int, NREG: Int, NSIG: Int, NSTEP: Int
](
    proj_out: LayoutTensor[DT, Layout.row_major(BT * NSP * D), MutAnyOrigin],
    action_base: LayoutTensor[DT, Layout.row_major(D), MutAnyOrigin],
    signal_table: LayoutTensor[DT, Layout.row_major(NSIG * D), MutAnyOrigin],
    step_table: LayoutTensor[DT, Layout.row_major(NSTEP * D), MutAnyOrigin],
    register: LayoutTensor[DT, Layout.row_major(NREG * D), MutAnyOrigin],
    sig: LayoutTensor[DT, Layout.row_major(BT), MutAnyOrigin],
    step: LayoutTensor[DT, Layout.row_major(BT), MutAnyOrigin],
    grid: LayoutTensor[DT, Layout.row_major(BT * S * D), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= BT * S * D:
        return
    comptime SD = S * D
    var bt = idx // SD
    var col = idx % SD
    var token = col // D
    var d = col % D
    if token == 0:
        grid.ptr[idx] = rebind[Scalar[DT]](action_base.ptr[d])
    elif token == 1:
        var si = Int(rebind[Scalar[DT]](sig.ptr[bt]) + Scalar[DT](0.5))
        grid.ptr[idx] = rebind[Scalar[DT]](signal_table.ptr[si * D + d])
    elif token == 2:
        var pi = Int(rebind[Scalar[DT]](step.ptr[bt]) + Scalar[DT](0.5))
        grid.ptr[idx] = rebind[Scalar[DT]](step_table.ptr[pi * D + d])
    elif token < 3 + NSP:
        grid.ptr[idx] = rebind[Scalar[DT]](
            proj_out.ptr[bt * (NSP * D) + (token - 3) * D + d]
        )
    else:
        grid.ptr[idx] = rebind[Scalar[DT]](
            register.ptr[(token - (3 + NSP)) * D + d]
        )


# grad_grid → grad_proj_out (spatial tokens at cols [3D, (3+NSP)D)).
def _dyn_grad_proj_kernel[BT: Int, S: Int, D: Int, NSP: Int](
    ggrid: LayoutTensor[DT, Layout.row_major(BT * S * D), MutAnyOrigin],
    gproj: LayoutTensor[DT, Layout.row_major(BT * NSP * D), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= BT * NSP * D:
        return
    var bt = idx // (NSP * D)
    var k = idx % (NSP * D)
    gproj.ptr[idx] = rebind[Scalar[DT]](ggrid.ptr[bt * (S * D) + 3 * D + k])


# action_base grad: gbase[d] += Σ_bt ggrid[bt, d].
def _dyn_grad_base_kernel[BT: Int, S: Int, D: Int](
    ggrid: LayoutTensor[DT, Layout.row_major(BT * S * D), MutAnyOrigin],
    gbase: LayoutTensor[DT, Layout.row_major(D), MutAnyOrigin],
):
    var d = Int(global_idx.x)
    if d >= D:
        return
    var acc = Scalar[DT](0.0)
    for bt in range(BT):
        acc += rebind[Scalar[DT]](ggrid.ptr[bt * (S * D) + d])
    gbase.ptr[d] = rebind[Scalar[DT]](gbase.ptr[d]) + acc


# register grad: greg[k] += Σ_bt ggrid[bt, REG_OFF + k].
def _dyn_grad_reg_kernel[BT: Int, S: Int, D: Int, NSP: Int, NREG: Int](
    ggrid: LayoutTensor[DT, Layout.row_major(BT * S * D), MutAnyOrigin],
    greg: LayoutTensor[DT, Layout.row_major(NREG * D), MutAnyOrigin],
):
    var k = Int(global_idx.x)
    if k >= NREG * D:
        return
    comptime REG_OFF = (3 + NSP) * D
    var acc = Scalar[DT](0.0)
    for bt in range(BT):
        acc += rebind[Scalar[DT]](ggrid.ptr[bt * (S * D) + REG_OFF + k])
    greg.ptr[k] = rebind[Scalar[DT]](greg.ptr[k]) + acc


# signal-table grad (index-masked batch reduction; signal token at col [D,2D)).
def _dyn_grad_sig_kernel[BT: Int, S: Int, D: Int, NSIG: Int](
    ggrid: LayoutTensor[DT, Layout.row_major(BT * S * D), MutAnyOrigin],
    sig: LayoutTensor[DT, Layout.row_major(BT), MutAnyOrigin],
    gsig: LayoutTensor[DT, Layout.row_major(NSIG * D), MutAnyOrigin],
):
    var e = Int(global_idx.x)
    if e >= NSIG * D:
        return
    var v = e // D
    var d = e % D
    var acc = Scalar[DT](0.0)
    for bt in range(BT):
        var si = Int(rebind[Scalar[DT]](sig.ptr[bt]) + Scalar[DT](0.5))
        if si == v:
            acc += rebind[Scalar[DT]](ggrid.ptr[bt * (S * D) + D + d])
    gsig.ptr[e] = rebind[Scalar[DT]](gsig.ptr[e]) + acc


# step-table grad (index-masked; step token at col [2D,3D)).
def _dyn_grad_step_kernel[BT: Int, S: Int, D: Int, NSTEP: Int](
    ggrid: LayoutTensor[DT, Layout.row_major(BT * S * D), MutAnyOrigin],
    step: LayoutTensor[DT, Layout.row_major(BT), MutAnyOrigin],
    gstep: LayoutTensor[DT, Layout.row_major(NSTEP * D), MutAnyOrigin],
):
    var e = Int(global_idx.x)
    if e >= NSTEP * D:
        return
    var v = e // D
    var d = e % D
    var acc = Scalar[DT](0.0)
    for bt in range(BT):
        var pi = Int(rebind[Scalar[DT]](step.ptr[bt]) + Scalar[DT](0.5))
        if pi == v:
            acc += rebind[Scalar[DT]](ggrid.ptr[bt * (S * D) + 2 * D + d])
    gstep.ptr[e] = rebind[Scalar[DT]](gstep.ptr[e]) + acc


struct Dreamer4Dynamics[
    DSP: Int,      # d_spatial (packed bottleneck width per spatial token)
    NSP: Int,      # n_spatial
    D: Int,        # d_model
    NH: Int,       # n_heads
    T: Int,        # frames per window
    NREG: Int,     # n_register
    HID: Int,      # SwiGLU hidden
    DEPTH: Int,    # transformer depth
    KMAX: Int,     # k_max (max integration steps; power of two)
    USE_MAX: Bool = True,
](ShortcutDynamics):
    comptime ARITY: Int = 1
    comptime S: Int = 3 + Self.NSP + Self.NREG          # tokens per frame
    comptime NSIG: Int = Self.KMAX + 1                  # signal vocab
    comptime NSTEP: Int = _ilog2(Self.KMAX) + 1         # step vocab (num bins)
    comptime SD: Int = Self.S * Self.D
    comptime SPAT_OFF: Int = 3 * Self.D                 # spatial token col start
    comptime REG_OFF: Int = (3 + Self.NSP) * Self.D     # register col start
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.NSP * Self.DSP)
    comptime OUT_DIM = Self.NSP * Self.DSP

    comptime PROJ = Tokenwise[Self.NSP, Linear[Self.DSP, Self.D]]
    comptime TAIL = Sequential[
        SinusoidalPosAddBT[Self.T, Self.S, Self.D],
        Dreamer4Stack[
            Self.D, Self.NH, Self.T, Self.S, Self.S, Self.HID, Self.DEPTH,
            "wm_agent", Self.USE_MAX,
        ],
        Slice[Self.SD, Self.SPAT_OFF, Self.REG_OFF],     # spatial tokens out
        Tokenwise[Self.NSP, ZeroLinear[Self.D, Self.DSP]],  # flow_x_head (zero)
    ]

    var proj: Self.PROJ
    var tail: Self.TAIL
    var action_base: Param["action_base", False, Self.D]
    var signal_table: Param["signal_table", True, Self.NSIG * Self.D]
    var step_table: Param["step_table", True, Self.NSTEP * Self.D]
    var register: Param["register", False, Self.NREG * Self.D]

    # CPU scratch
    var grid: List[Scalar[DT]]            # [BATCH, S*D]
    var grad_grid: List[Scalar[DT]]
    var proj_out: List[Scalar[DT]]        # [BATCH, NSP*D]
    var grad_proj_out: List[Scalar[DT]]
    var cache_sig: List[Int]              # [BATCH] signal index per sample
    var cache_step: List[Int]             # [BATCH] step index per sample
    # GPU scratch
    var grid_dev: Optional[DeviceBuffer[DT]]
    var ggrid_dev: Optional[DeviceBuffer[DT]]
    var po_dev: Optional[DeviceBuffer[DT]]
    var gpo_dev: Optional[DeviceBuffer[DT]]
    var sig_dev: Optional[DeviceBuffer[DT]]      # uploaded indices (device)
    var step_dev: Optional[DeviceBuffer[DT]]
    var sig_hbuf: Optional[HostBuffer[DT]]       # host staging for upload
    var step_hbuf: Optional[HostBuffer[DT]]
    var scratch_batch: Int
    var ts: TargetStorage

    def __init__(out self):
        self.proj = Self.PROJ()
        self.tail = Self.TAIL()
        self.action_base = Param["action_base", False, Self.D]()
        self.signal_table = Param["signal_table", True, Self.NSIG * Self.D]()
        self.step_table = Param["step_table", True, Self.NSTEP * Self.D]()
        self.register = Param["register", False, Self.NREG * Self.D]()
        self.grid = List[Scalar[DT]]()
        self.grad_grid = List[Scalar[DT]]()
        self.proj_out = List[Scalar[DT]]()
        self.grad_proj_out = List[Scalar[DT]]()
        self.cache_sig = List[Int]()
        self.cache_step = List[Int]()
        self.grid_dev = None
        self.ggrid_dev = None
        self.po_dev = None
        self.gpo_dev = None
        self.sig_dev = None
        self.step_dev = None
        self.sig_hbuf = None
        self.step_hbuf = None
        self.scratch_batch = 0
        self.ts = TargetStorage.make_uninit()

    def _ensure_scratch_gpu(mut self, batch: Int) raises:
        if self.scratch_batch < batch:
            var ctx = self.ts.ctx.value()
            self.grid_dev = ctx.enqueue_create_buffer[DT](batch * Self.SD)
            self.ggrid_dev = ctx.enqueue_create_buffer[DT](batch * Self.SD)
            self.po_dev = ctx.enqueue_create_buffer[DT](batch * Self.NSP * Self.D)
            self.gpo_dev = ctx.enqueue_create_buffer[DT](batch * Self.NSP * Self.D)
            self.sig_dev = ctx.enqueue_create_buffer[DT](batch)
            self.step_dev = ctx.enqueue_create_buffer[DT](batch)
            self.sig_hbuf = ctx.enqueue_create_host_buffer[DT](batch)
            self.step_hbuf = ctx.enqueue_create_host_buffer[DT](batch)
            ctx.synchronize()
            self.scratch_batch = batch

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "Dreamer4Dynamics: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        m.proj = Self.PROJ.make[target=target, INIT=INIT](ctx)
        m.tail = Self.TAIL.make[target=target, INIT=INIT](ctx)
        comptime NS = Self.NSIG * Self.D
        comptime NT = Self.NSTEP * Self.D
        comptime NR = Self.NREG * Self.D
        comptime if target == "cpu":
            m.action_base = Param["action_base", False, Self.D].make_cpu()
            m.signal_table = Param["signal_table", True, NS].make_cpu()
            m.step_table = Param["step_table", True, NT].make_cpu()
            m.register = Param["register", False, NR].make_cpu()
            # Init conditioning params with the graph initializer (the
            # reference uses normal(std=0.02) for base/register; INIT is close
            # enough — the zero-init flow head is the only init that matters
            # for convergence).
            INIT.init_weight(
                m.action_base.value_unsafe_ptr_cpu(), Self.D, 1, Self.D
            )
            INIT.init_weight(
                m.signal_table.value_unsafe_ptr_cpu(), NS, Self.NSIG, Self.D
            )
            INIT.init_weight(
                m.step_table.value_unsafe_ptr_cpu(), NT, Self.NSTEP, Self.D
            )
            INIT.init_weight(
                m.register.value_unsafe_ptr_cpu(), NR, Self.NREG, Self.D
            )
            m.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("Dreamer4Dynamics.make[gpu]: ctx required")
            var c = ctx.value()
            m.action_base = Param["action_base", False, Self.D].make_gpu(c)
            m.signal_table = Param["signal_table", True, NS].make_gpu(c)
            m.step_table = Param["step_table", True, NT].make_gpu(c)
            m.register = Param["register", False, NR].make_gpu(c)
            # init on host then upload (Xavier uses host RNG)
            _init_param_gpu[INIT](c, m.action_base.value_dev.value(), Self.D, 1, Self.D)
            _init_param_gpu[INIT](c, m.signal_table.value_dev.value(), NS, Self.NSIG, Self.D)
            _init_param_gpu[INIT](c, m.step_table.value_dev.value(), NT, Self.NSTEP, Self.D)
            _init_param_gpu[INIT](c, m.register.value_dev.value(), NR, Self.NREG, Self.D)
            m.ts = TargetStorage.make_gpu(c)
        return m^

    @staticmethod
    def display_label() -> String:
        return String("Dreamer4Dynamics")

    def set_indices(
        mut self,
        sig: UnsafePointer[Scalar[DT], MutAnyOrigin],
        step: UnsafePointer[Scalar[DT], MutAnyOrigin],
        batch: Int,
    ):
        """Push the per-sample signal/step indices (fp buffers holding exact
        non-negative ints) used by the next forward/vjp. sig ∈ [0, KMAX],
        step ∈ [0, NSTEP)."""
        if len(self.cache_sig) < batch:
            self.cache_sig.resize(batch, 0)
            self.cache_step.resize(batch, 0)
        for bt in range(batch):
            self.cache_sig[bt] = Int(Float64(sig[bt]) + 0.5)
            self.cache_step[bt] = Int(Float64(step[bt]) + 0.5)

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        var *inputs: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["Dreamer4Dynamics", target](self.ts.target_tag)
        var packed = typed_view[BATCH, Self.NSP * Self.DSP](inputs[0])
        var out = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            ensure_cpu_buffer(self.grid, BATCH * Self.SD)
            ensure_cpu_buffer(self.proj_out, BATCH * Self.NSP * Self.D)
            var po = TileTensor(
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    self.proj_out.unsafe_ptr()
                ),
                row_major[BATCH, Self.NSP * Self.D](),
            )
            self.proj.forward[target, BATCH, POLICY=POLICY](packed, output=po)

            var grid = TileTensor(
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    self.grid.unsafe_ptr()
                ),
                row_major[BATCH, Self.SD](),
            )
            var ab = TileTensor(self.action_base.value, row_major[Self.D]())
            var sigt = TileTensor(
                self.signal_table.value, row_major[Self.NSIG * Self.D]()
            )
            var stpt = TileTensor(
                self.step_table.value, row_major[Self.NSTEP * Self.D]()
            )
            var reg = TileTensor(
                self.register.value, row_major[Self.NREG * Self.D]()
            )
            for bt in range(BATCH):
                var si = self.cache_sig[bt]   # set via set_indices()
                var pi = self.cache_step[bt]
                for d in range(Self.D):
                    grid[bt, d] = ab[d]                          # action
                    grid[bt, Self.D + d] = sigt[si * Self.D + d]  # signal
                    grid[bt, 2 * Self.D + d] = stpt[pi * Self.D + d]  # step
                for k in range(Self.NSP * Self.D):
                    grid[bt, Self.SPAT_OFF + k] = po[bt, k]       # spatial
                for k in range(Self.NREG * Self.D):
                    grid[bt, Self.REG_OFF + k] = reg[k]           # register

            self.tail.forward[target, BATCH, POLICY=POLICY](grid, output=out)
        else:
            self._ensure_scratch_gpu(BATCH)
            var ctx = self.ts.ctx.value()
            # upload the cached indices (set via set_indices) → device
            var sh = self.sig_hbuf.value()
            var th = self.step_hbuf.value()
            for bt in range(BATCH):
                sh.unsafe_ptr()[bt] = Scalar[DT](Float64(self.cache_sig[bt]))
                th.unsafe_ptr()[bt] = Scalar[DT](Float64(self.cache_step[bt]))
            ctx.enqueue_copy(self.sig_dev.value(), sh)
            ctx.enqueue_copy(self.step_dev.value(), th)

            var po = _dev_tile[BATCH, Self.NSP * Self.D](self.po_dev.value())
            self.proj.forward[target, BATCH, POLICY=POLICY](packed, output=po)

            comptime AN = BATCH * Self.SD
            comptime PN = BATCH * Self.NSP * Self.D
            var proj_lt = LayoutTensor[DT, Layout.row_major(PN), MutAnyOrigin](
                self.po_dev.value()
            )
            var ab_lt = LayoutTensor[DT, Layout.row_major(Self.D), MutAnyOrigin](
                self.action_base.value_dev.value()
            )
            var sg_lt = LayoutTensor[
                DT, Layout.row_major(Self.NSIG * Self.D), MutAnyOrigin
            ](self.signal_table.value_dev.value())
            var st_lt = LayoutTensor[
                DT, Layout.row_major(Self.NSTEP * Self.D), MutAnyOrigin
            ](self.step_table.value_dev.value())
            var rg_lt = LayoutTensor[
                DT, Layout.row_major(Self.NREG * Self.D), MutAnyOrigin
            ](self.register.value_dev.value())
            var si_lt = LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin](
                self.sig_dev.value()
            )
            var sp_lt = LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin](
                self.step_dev.value()
            )
            var grid_lt = LayoutTensor[DT, Layout.row_major(AN), MutAnyOrigin](
                self.grid_dev.value()
            )
            comptime ak = _dyn_assemble_kernel[
                BATCH, Self.S, Self.D, Self.NSP, Self.NREG, Self.NSIG, Self.NSTEP
            ]
            ctx.enqueue_function[ak](
                proj_lt, ab_lt, sg_lt, st_lt, rg_lt, si_lt, sp_lt, grid_lt,
                grid_dim=(AN + TPB - 1) // TPB, block_dim=TPB,
            )
            var grid_t = _dev_tile[BATCH, Self.SD](self.grid_dev.value())
            self.tail.forward[target, BATCH, POLICY=POLICY](grid_t, output=out)

    def vjp[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut *grad_inputs: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["Dreamer4Dynamics", target](self.ts.target_tag)
        var go = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var gpacked = typed_view_mut[BATCH, Self.NSP * Self.DSP](grad_inputs[0])

        comptime if target == "cpu":
            ensure_cpu_buffer(self.grad_grid, BATCH * Self.SD)
            ensure_cpu_buffer(self.grad_proj_out, BATCH * Self.NSP * Self.D)
            var ggrid = TileTensor(
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    self.grad_grid.unsafe_ptr()
                ),
                row_major[BATCH, Self.SD](),
            )
            self.tail.vjp[target, BATCH, POLICY=POLICY, mode=mode](go, ggrid)

            # spatial token grad → proj input grad
            var gpo = TileTensor(
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    self.grad_proj_out.unsafe_ptr()
                ),
                row_major[BATCH, Self.NSP * Self.D](),
            )
            for bt in range(BATCH):
                for k in range(Self.NSP * Self.D):
                    gpo[bt, k] = ggrid[bt, Self.SPAT_OFF + k]
            self.proj.vjp[target, BATCH, POLICY=POLICY, mode=mode](gpo, gpacked)

            comptime if mode == "all":
                var gab = TileTensor(self.action_base.grad, row_major[Self.D]())
                var gsig = TileTensor(
                    self.signal_table.grad, row_major[Self.NSIG * Self.D]()
                )
                var gstp = TileTensor(
                    self.step_table.grad, row_major[Self.NSTEP * Self.D]()
                )
                var greg = TileTensor(
                    self.register.grad, row_major[Self.NREG * Self.D]()
                )
                for bt in range(BATCH):
                    var si = self.cache_sig[bt]
                    var pi = self.cache_step[bt]
                    for d in range(Self.D):
                        gab[d] += ggrid[bt, d]
                        gsig[si * Self.D + d] += ggrid[bt, Self.D + d]
                        gstp[pi * Self.D + d] += ggrid[bt, 2 * Self.D + d]
                    for k in range(Self.NREG * Self.D):
                        greg[k] += ggrid[bt, Self.REG_OFF + k]
        else:
            self._ensure_scratch_gpu(BATCH)
            var ctx = self.ts.ctx.value()
            comptime AN = BATCH * Self.SD
            comptime PN = BATCH * Self.NSP * Self.D
            var ggrid_t = _dev_tile[BATCH, Self.SD](self.ggrid_dev.value())
            self.tail.vjp[target, BATCH, POLICY=POLICY, mode=mode](go, ggrid_t)

            var ggrid_lt = LayoutTensor[DT, Layout.row_major(AN), MutAnyOrigin](
                self.ggrid_dev.value()
            )
            var gpo_lt = LayoutTensor[DT, Layout.row_major(PN), MutAnyOrigin](
                self.gpo_dev.value()
            )
            comptime gpk = _dyn_grad_proj_kernel[BATCH, Self.S, Self.D, Self.NSP]
            ctx.enqueue_function[gpk](
                ggrid_lt, gpo_lt, grid_dim=(PN + TPB - 1) // TPB, block_dim=TPB,
            )
            var gpo_t = _dev_tile[BATCH, Self.NSP * Self.D](self.gpo_dev.value())
            self.proj.vjp[target, BATCH, POLICY=POLICY, mode=mode](gpo_t, gpacked)

            comptime if mode == "all":
                var gab_lt = LayoutTensor[
                    DT, Layout.row_major(Self.D), MutAnyOrigin
                ](self.action_base.grad_dev.value())
                var greg_lt = LayoutTensor[
                    DT, Layout.row_major(Self.NREG * Self.D), MutAnyOrigin
                ](self.register.grad_dev.value())
                var gsig_lt = LayoutTensor[
                    DT, Layout.row_major(Self.NSIG * Self.D), MutAnyOrigin
                ](self.signal_table.grad_dev.value())
                var gstp_lt = LayoutTensor[
                    DT, Layout.row_major(Self.NSTEP * Self.D), MutAnyOrigin
                ](self.step_table.grad_dev.value())
                var si_lt = LayoutTensor[
                    DT, Layout.row_major(BATCH), MutAnyOrigin
                ](self.sig_dev.value())
                var sp_lt = LayoutTensor[
                    DT, Layout.row_major(BATCH), MutAnyOrigin
                ](self.step_dev.value())
                comptime bk = _dyn_grad_base_kernel[BATCH, Self.S, Self.D]
                ctx.enqueue_function[bk](
                    ggrid_lt, gab_lt,
                    grid_dim=(Self.D + TPB - 1) // TPB, block_dim=TPB,
                )
                comptime rk = _dyn_grad_reg_kernel[
                    BATCH, Self.S, Self.D, Self.NSP, Self.NREG
                ]
                ctx.enqueue_function[rk](
                    ggrid_lt, greg_lt,
                    grid_dim=(Self.NREG * Self.D + TPB - 1) // TPB,
                    block_dim=TPB,
                )
                comptime sk = _dyn_grad_sig_kernel[
                    BATCH, Self.S, Self.D, Self.NSIG
                ]
                ctx.enqueue_function[sk](
                    ggrid_lt, si_lt, gsig_lt,
                    grid_dim=(Self.NSIG * Self.D + TPB - 1) // TPB,
                    block_dim=TPB,
                )
                comptime tk = _dyn_grad_step_kernel[
                    BATCH, Self.S, Self.D, Self.NSTEP
                ]
                ctx.enqueue_function[tk](
                    ggrid_lt, sp_lt, gstp_lt,
                    grid_dim=(Self.NSTEP * Self.D + TPB - 1) // TPB,
                    block_dim=TPB,
                )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["Dreamer4Dynamics", target](self.ts.target_tag)
        # raw conditioning params (action_base, signal/step tables, register)
        for_each_param_auto[Self, V, target](self, prefix, visitor)
        # child modules
        self.proj.for_each_param[target, V](prefix + ".proj", visitor)
        self.tail.for_each_param[target, V](prefix + ".tail", visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["Dreamer4Dynamics", target](self.ts.target_tag)
        zero_grad_auto[Self, target](self)
        self.proj.zero_grad[target]()
        self.tail.zero_grad[target]()
