"""TD-MPC2 multi-task learned task-embedding table (item C, §14.3) — storage.

A standalone `[NUM_TASKS, TASK_EMB]` learned lookup table with its own grad +
hand-rolled Adam moments, stored as owning storage `Tensor`s (host `.data` /
optional device `.dev`). It is NOT an nn `Module` in any graph: the embedding's
gradient arrives at THREE disjoint sites (the WM ComputeGraph's `task_emb` input
slot, the encoder's input-grad tail, and the policy ComputeGraph's `task_emb`
slot) and must scatter-add by `task_id` into ONE table before a single Adam step.
A standalone table with an explicit `gather`/`accumulate`/`step` API is the
natural meeting point.

Per the multi-task contract, one replay window = one env = one task, so the
caller passes per-row task ids (a host `Tensor` whose `.data` holds DT-encoded
ids; on GPU the same `Tensor`'s `.dev` is read by the kernels). `gather` reads
rows; `accumulate` scatter-adds row grads (atomic-free: one thread per table
element loops rows). All inputs/outputs are storage `Tensor`s — no raw
`UnsafePointer` / `mptr` / `MutUntrackedOrigin` fields remain; the hand-rolled
Adam math is identical to the legacy version.

Built LAST in the agent (RNG discipline): a non-zero init draws from the global
RNG, which would shift the rollout stream — irrelevant to the single-task agent
(which never constructs this) but kept last so multi-task runs stay reproducible.
"""

from std.math import sqrt
from std.random import random_float64
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.storage.core.tensor import Tensor


# ── GPU kernels (operate over storage Tensor `.lt` views) ────────────────
def _gather_k[
    NTASKS: Int, NROWS: Int, EMB: Int
](
    task_ids: LayoutTensor[DT, Layout.row_major(NROWS), MutAnyOrigin],
    param: LayoutTensor[DT, Layout.row_major(NTASKS * EMB), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(NROWS * EMB), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < NROWS * EMB:
        var row = i // EMB
        var e = i % EMB
        var t = Int(rebind[Scalar[DT]](task_ids[row]))
        dst[i] = rebind[Scalar[DT]](param[t * EMB + e])


def _accum_k[
    NTASKS: Int, EMB: Int, NROWS: Int, ROWW: Int, COFF: Int
](
    task_ids: LayoutTensor[DT, Layout.row_major(NROWS), MutAnyOrigin],
    grad_wide: LayoutTensor[DT, Layout.row_major(NROWS * ROWW), MutAnyOrigin],
    grad_tab: LayoutTensor[DT, Layout.row_major(NTASKS * EMB), MutAnyOrigin],
):
    """One thread per table element (task,e); loops rows summing matching
    contributions → atomic-free scatter-add. NROWS is small relative to the
    nets, so the per-thread row loop is cheap."""
    var idx = Int(global_idx.x)
    if idx < NTASKS * EMB:
        var task = idx // EMB
        var e = idx % EMB
        var s: Scalar[DT] = 0.0
        for row in range(NROWS):
            if Int(rebind[Scalar[DT]](task_ids[row])) == task:
                s += rebind[Scalar[DT]](grad_wide[row * ROWW + COFF + e])
        grad_tab[idx] = rebind[Scalar[DT]](grad_tab[idx]) + s


def _adam_k[
    N: Int
](
    param: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    grad: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    m: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    v: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    lr: Scalar[DT],
    beta1: Scalar[DT],
    beta2: Scalar[DT],
    eps: Scalar[DT],
    bc1: Scalar[DT],
    bc2: Scalar[DT],
):
    var i = Int(global_idx.x)
    if i < N:
        var one: Scalar[DT] = 1.0
        var g = rebind[Scalar[DT]](grad[i])
        var m_new = beta1 * rebind[Scalar[DT]](m[i]) + (one - beta1) * g
        var v_new = beta2 * rebind[Scalar[DT]](v[i]) + (one - beta2) * g * g
        m[i] = m_new
        v[i] = v_new
        param[i] = rebind[Scalar[DT]](param[i]) - lr * (m_new / bc1) / (
            sqrt(v_new / bc2) + eps
        )


def _zero_k[N: Int](
    ptr: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin]
):
    var i = Int(global_idx.x)
    if i < N:
        ptr[i] = Scalar[DT](0.0)


struct TaskEmbedding[NUM_TASKS: Int, TASK_EMB: Int](
    Movable & ImplicitlyDeletable
):
    comptime N = Self.NUM_TASKS * Self.TASK_EMB

    # Owning storage Tensors (host `.data` always present; `.dev` only on GPU).
    var param: Tensor
    var grad: Tensor
    var m: Tensor
    var v: Tensor
    var ctx: Optional[DeviceContext]
    var is_gpu: Bool

    # Adam state (host-driven bias correction; tiny table, capture not needed).
    var lr: Scalar[DT]
    var beta1: Scalar[DT]
    var beta2: Scalar[DT]
    var eps: Scalar[DT]
    var b1pow: Scalar[DT]
    var b2pow: Scalar[DT]

    def __init__(out self):
        self.param = Tensor()
        self.grad = Tensor()
        self.m = Tensor()
        self.v = Tensor()
        self.ctx = None
        self.is_gpu = False
        self.lr = Scalar[DT](3e-4)
        self.beta1 = Scalar[DT](0.9)
        self.beta2 = Scalar[DT](0.999)
        self.eps = Scalar[DT](1e-8)
        self.b1pow = Scalar[DT](1.0)
        self.b2pow = Scalar[DT](1.0)

    @staticmethod
    def make[
        target: StaticString
    ](
        ctx: Optional[DeviceContext] = None,
        lr: Scalar[DT] = Scalar[DT](3e-4),
        zero_init: Bool = False,
    ) raises -> Self:
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "TaskEmbedding: target must be 'cpu' or 'gpu'"
        var s = Self()
        s.lr = lr
        comptime N = Self.N
        # Host-resident slabs first (`.data` always sized for both targets — the
        # GPU path uploads from them, and checkpoint/sync read them back). Param
        # init: small uniform ~U(-r, r), r = 1/sqrt(EMB), zeros if requested.
        # Drawn from the global RNG → call make() last.
        s.param = Tensor.alloc(N)
        s.grad = Tensor.alloc(N)
        s.m = Tensor.alloc(N)
        s.v = Tensor.alloc(N)
        var r = Scalar[DT](1.0) / Scalar[DT](
            sqrt(Float64(Self.TASK_EMB)) if Self.TASK_EMB > 0 else 1.0
        )
        for i in range(N):
            if zero_init:
                s.param.data[i] = Scalar[DT](0.0)
            else:
                s.param.data[i] = Scalar[DT](random_float64() * 2.0 - 1.0) * r
            s.grad.data[i] = Scalar[DT](0.0)
            s.m.data[i] = Scalar[DT](0.0)
            s.v.data[i] = Scalar[DT](0.0)
        comptime if target == "gpu":
            s.is_gpu = True
            s.ctx = ctx
            var c = ctx.value()
            # `n` is set by alloc; upload (re)allocates each device buffer + fills
            # it from the host `.data` slab.
            s.param.upload(c)
            s.grad.upload(c)
            s.m.upload(c)
            s.v.upload(c)
        return s^

    def zero_grad[target: StaticString](mut self) raises:
        comptime N = Self.N
        comptime if target == "cpu":
            for i in range(N):
                self.grad.data[i] = Scalar[DT](0.0)
        else:
            var c = self.ctx.value()
            comptime nb = (N + TPB - 1) // TPB
            c.enqueue_function[_zero_k[N]](
                self.grad.lt["gpu", Layout.row_major(N)](),
                grid_dim=nb, block_dim=TPB,
            )

    def gather[
        target: StaticString, NROWS: Int
    ](
        mut self,
        mut task_ids: Tensor,  # [NROWS] DT-encoded ids
        mut dst: Tensor,       # [NROWS, TASK_EMB]
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """`dst[row] = param[task_ids[row]]`. CPU reads `.data`; GPU reads `.dev`
        views."""
        comptime EMB = Self.TASK_EMB
        comptime if target == "cpu":
            for row in range(NROWS):
                var t = Int(task_ids.data[row])
                for e in range(EMB):
                    dst.data[row * EMB + e] = self.param.data[t * EMB + e]
        else:
            var c = ctx.value()
            comptime nb = (NROWS * EMB + TPB - 1) // TPB
            c.enqueue_function[_gather_k[Self.NUM_TASKS, NROWS, EMB]](
                task_ids.lt["gpu", Layout.row_major(NROWS)](),
                self.param.lt["gpu", Layout.row_major(Self.N)](),
                dst.lt["gpu", Layout.row_major(NROWS * EMB)](),
                grid_dim=nb, block_dim=TPB,
            )

    def accumulate[
        target: StaticString, NROWS: Int, ROWW: Int, COFF: Int
    ](
        mut self,
        mut task_ids: Tensor,    # [NROWS]
        mut grad_wide: Tensor,   # [NROWS, ROWW]
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """`grad_table[task_ids[row]] += grad_wide[row, COFF : COFF+TASK_EMB]`.
        `ROWW`/`COFF` let one impl serve both the graph slot grad (ROWW=TASK_EMB,
        COFF=0) and the encoder input-grad tail (ROWW=MAX_OBS+TASK_EMB,
        COFF=MAX_OBS)."""
        comptime EMB = Self.TASK_EMB
        comptime if target == "cpu":
            for row in range(NROWS):
                var t = Int(task_ids.data[row])
                for e in range(EMB):
                    self.grad.data[t * EMB + e] += grad_wide.data[
                        row * ROWW + COFF + e
                    ]
        else:
            var c = ctx.value()
            comptime nt = (Self.N + TPB - 1) // TPB
            c.enqueue_function[
                _accum_k[Self.NUM_TASKS, EMB, NROWS, ROWW, COFF]
            ](
                task_ids.lt["gpu", Layout.row_major(NROWS)](),
                grad_wide.lt["gpu", Layout.row_major(NROWS * ROWW)](),
                self.grad.lt["gpu", Layout.row_major(Self.N)](),
                grid_dim=nt, block_dim=TPB,
            )

    def step[target: StaticString](mut self) raises:
        comptime N = Self.N
        self.b1pow = self.b1pow * self.beta1
        self.b2pow = self.b2pow * self.beta2
        var bc1 = Scalar[DT](1.0) - self.b1pow
        var bc2 = Scalar[DT](1.0) - self.b2pow
        comptime if target == "cpu":
            for i in range(N):
                var g = self.grad.data[i]
                var m_new = (
                    self.beta1 * self.m.data[i]
                    + (Scalar[DT](1.0) - self.beta1) * g
                )
                var v_new = (
                    self.beta2 * self.v.data[i]
                    + (Scalar[DT](1.0) - self.beta2) * g * g
                )
                self.m.data[i] = m_new
                self.v.data[i] = v_new
                self.param.data[i] = self.param.data[i] - self.lr * (
                    m_new / bc1
                ) / (sqrt(v_new / bc2) + self.eps)
        else:
            var c = self.ctx.value()
            comptime nb = (N + TPB - 1) // TPB
            c.enqueue_function[_adam_k[N]](
                self.param.lt["gpu", Layout.row_major(N)](),
                self.grad.lt["gpu", Layout.row_major(N)](),
                self.m.lt["gpu", Layout.row_major(N)](),
                self.v.lt["gpu", Layout.row_major(N)](),
                self.lr,
                self.beta1,
                self.beta2,
                self.eps,
                bc1,
                bc2,
                grid_dim=nb,
                block_dim=TPB,
            )

    # ── Checkpoint (CPU-resident serialization; GPU syncs around it) ──────
    def sync_to_host(mut self) raises:
        """D2H param/m/v into the host `.data` slabs (GPU only)."""
        if not self.is_gpu:
            return
        var c = self.ctx.value()
        self.param.download(c)
        self.m.download(c)
        self.v.download(c)

    def upload_from_host(mut self) raises:
        if not self.is_gpu:
            return
        var c = self.ctx.value()
        self.param.upload(c)
        self.m.upload(c)
        self.v.upload(c)

    def save_body(mut self, mut out: String, name: String) raises:
        """Append param + m + v + Adam scalars as a small text body. On GPU,
        call `sync_to_host` first."""
        self.sync_to_host()
        out += name + "#size=" + String(Self.N) + "\n"
        for i in range(Self.N):
            out += String(self.param.data[i]) + "\n"
        for i in range(Self.N):
            out += String(self.m.data[i]) + "\n"
        for i in range(Self.N):
            out += String(self.v.data[i]) + "\n"
        # Adam bias-correction powers as raw lines (read positionally on load).
        out += String(self.b1pow) + "\n"
        out += String(self.b2pow) + "\n"

    def load_body(
        mut self, lines: List[String], mut idx: Int, name: String
    ) raises:
        var header = lines[idx]
        var expect = name + "#size=" + String(Self.N)
        if header != expect:
            raise Error(
                "TaskEmbedding.load_body: header mismatch, expected `"
                + expect
                + "`, got `"
                + header
                + "`"
            )
        idx += 1
        for i in range(Self.N):
            self.param.data[i] = Scalar[DT](atof(lines[idx]))
            idx += 1
        for i in range(Self.N):
            self.m.data[i] = Scalar[DT](atof(lines[idx]))
            idx += 1
        for i in range(Self.N):
            self.v.data[i] = Scalar[DT](atof(lines[idx]))
            idx += 1
        self.b1pow = Scalar[DT](atof(lines[idx]))
        idx += 1
        self.b2pow = Scalar[DT](atof(lines[idx]))
        idx += 1
        self.upload_from_host()
