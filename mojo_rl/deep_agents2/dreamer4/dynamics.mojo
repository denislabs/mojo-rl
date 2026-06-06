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

PHASE 2: CPU forward + vjp. GPU follows in Phase 2.4.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
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
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu", (
            "Dreamer4Dynamics: only 'cpu' target is implemented (Phase 2.2);"
            " GPU lands in Phase 2.4"
        )
        var m = Self()
        m.proj = Self.PROJ.make[target=target, INIT=INIT](ctx)
        m.tail = Self.TAIL.make[target=target, INIT=INIT](ctx)
        m.action_base = Param["action_base", False, Self.D].make_cpu()
        m.signal_table = Param[
            "signal_table", True, Self.NSIG * Self.D
        ].make_cpu()
        m.step_table = Param["step_table", True, Self.NSTEP * Self.D].make_cpu()
        m.register = Param["register", False, Self.NREG * Self.D].make_cpu()
        # Init conditioning params with the graph initializer (the reference
        # uses normal(std=0.02) for base/register; INIT is close enough — the
        # zero-init flow head is the only init that matters for convergence).
        INIT.init_weight(m.action_base.value_unsafe_ptr_cpu(), Self.D, 1, Self.D)
        INIT.init_weight(
            m.signal_table.value_unsafe_ptr_cpu(),
            Self.NSIG * Self.D, Self.NSIG, Self.D,
        )
        INIT.init_weight(
            m.step_table.value_unsafe_ptr_cpu(),
            Self.NSTEP * Self.D, Self.NSTEP, Self.D,
        )
        INIT.init_weight(
            m.register.value_unsafe_ptr_cpu(),
            Self.NREG * Self.D, Self.NREG, Self.D,
        )
        m.ts = TargetStorage.make_cpu()
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
        comptime assert target == "cpu", "Dreamer4Dynamics: CPU only (Phase 2.2)"
        var packed = typed_view[BATCH, Self.NSP * Self.DSP](inputs[0])
        var out = typed_view_mut[BATCH, Self.OUT_DIM](output)

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
        comptime assert target == "cpu", "Dreamer4Dynamics: CPU only (Phase 2.2)"
        var go = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var gpacked = typed_view_mut[BATCH, Self.NSP * Self.DSP](grad_inputs[0])

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
