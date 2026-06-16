"""TD-MPC2 multi-task learned task-embedding table (item C, §14.3).

A standalone `[NUM_TASKS, TASK_EMB]` learned lookup table with its own grad +
hand-rolled Adam moments (CPU `UnsafePointer` / GPU `DeviceBuffer`). It is NOT
an nn `Module` in any graph: the embedding's gradient arrives at THREE disjoint
sites (the WM ComputeGraph's `task_emb` input slot, the encoder's input-grad
tail, and the policy ComputeGraph's `task_emb` slot) and must scatter-add by
`task_id` into ONE table before a single Adam step. A standalone table with an
explicit `gather`/`accumulate`/`step` API is the natural meeting point.

Per the multi-task contract, one replay window = one env = one task, so the
caller passes per-row task ids (DT-encoded). `gather` reads rows; `accumulate`
scatter-adds row grads (atomic-free: one thread per table element loops rows).

Built LAST in the agent (RNG discipline): a non-zero init draws from the global
RNG, which would shift the rollout stream — irrelevant to the single-task agent
(which never constructs this) but kept last so multi-task runs stay reproducible.
"""

from std.math import sqrt
from std.memory import alloc
from std.random import random_float64
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.core.module import mptr


@always_inline
def _dp(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return mptr(b.unsafe_ptr())


# ── GPU kernels ────────────────────────────────────────────────────────
def _gather_k[NROWS: Int, EMB: Int](
    task_ids: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [NROWS]
    param: UnsafePointer[Scalar[DT], MutAnyOrigin],      # [NUM_TASKS, EMB]
    dst: UnsafePointer[Scalar[DT], MutAnyOrigin],        # [NROWS, EMB]
):
    var i = Int(global_idx.x)
    if i < NROWS * EMB:
        var row = i // EMB
        var e = i % EMB
        var t = Int(task_ids[row])
        dst[i] = param[t * EMB + e]


def _accum_k[NTASKS: Int, EMB: Int, NROWS: Int, ROWW: Int, COFF: Int](
    task_ids: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [NROWS]
    grad_wide: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [NROWS, ROWW]
    grad_tab: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [NTASKS, EMB] (+=)
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
            if Int(task_ids[row]) == task:
                s += grad_wide[row * ROWW + COFF + e]
        grad_tab[idx] = grad_tab[idx] + s


def _adam_k[N: Int](
    param: UnsafePointer[Scalar[DT], MutAnyOrigin],
    grad: UnsafePointer[Scalar[DT], MutAnyOrigin],
    m: UnsafePointer[Scalar[DT], MutAnyOrigin],
    v: UnsafePointer[Scalar[DT], MutAnyOrigin],
    lr: Scalar[DT], beta1: Scalar[DT], beta2: Scalar[DT], eps: Scalar[DT],
    bc1: Scalar[DT], bc2: Scalar[DT],
):
    var i = Int(global_idx.x)
    if i < N:
        var one: Scalar[DT] = 1.0
        var g = grad[i]
        var m_new = beta1 * m[i] + (one - beta1) * g
        var v_new = beta2 * v[i] + (one - beta2) * g * g
        m[i] = m_new
        v[i] = v_new
        param[i] = param[i] - lr * (m_new / bc1) / (sqrt(v_new / bc2) + eps)


def _zero_k[N: Int](ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]):
    var i = Int(global_idx.x)
    if i < N:
        ptr[i] = 0.0


struct TaskEmbedding[NUM_TASKS: Int, TASK_EMB: Int](
    Movable & ImplicitlyDestructible
):
    comptime N = Self.NUM_TASKS * Self.TASK_EMB

    # CPU storage (null on GPU).
    var param: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var grad: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var m: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var v: UnsafePointer[Scalar[DT], MutAnyOrigin]
    # GPU storage (None on CPU).
    var d_param: Optional[DeviceBuffer[DT]]
    var d_grad: Optional[DeviceBuffer[DT]]
    var d_m: Optional[DeviceBuffer[DT]]
    var d_v: Optional[DeviceBuffer[DT]]
    var ctx: Optional[DeviceContext]
    var is_gpu: Bool
    # Number of host param/m/v elements actually allocated (0 = only the 1-elem
    # placeholder from __init__; set to N once host buffers are sized). Used in
    # place of a null-pointer check (UnsafePointer is non-nullable in Mojo).
    var host_n: Int

    # Adam state (host-driven bias correction; tiny table, capture not needed).
    var lr: Scalar[DT]
    var beta1: Scalar[DT]
    var beta2: Scalar[DT]
    var eps: Scalar[DT]
    var b1pow: Scalar[DT]
    var b2pow: Scalar[DT]

    def __init__(out self):
        self.param = alloc[Scalar[DT]](1)
        self.grad = alloc[Scalar[DT]](1)
        self.m = alloc[Scalar[DT]](1)
        self.v = alloc[Scalar[DT]](1)
        self.d_param = None; self.d_grad = None; self.d_m = None; self.d_v = None
        self.ctx = None
        self.is_gpu = False
        self.host_n = 0
        self.lr = Scalar[DT](3e-4)
        self.beta1 = Scalar[DT](0.9)
        self.beta2 = Scalar[DT](0.999)
        self.eps = Scalar[DT](1e-8)
        self.b1pow = Scalar[DT](1.0)
        self.b2pow = Scalar[DT](1.0)

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
        lr: Scalar[DT] = Scalar[DT](3e-4),
        zero_init: Bool = False,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "TaskEmbedding: target must be 'cpu' or 'gpu'"
        )
        var s = Self()
        s.lr = lr
        comptime N = Self.N
        # Host-init the param table (small uniform ~U(-r, r), r = 1/sqrt(EMB)),
        # zeros if requested. Drawn from the global RNG → call make() last.
        var host = alloc[Scalar[DT]](N)
        var r = Scalar[DT](1.0) / Scalar[DT](
            sqrt(Float64(Self.TASK_EMB)) if Self.TASK_EMB > 0 else 1.0
        )
        for i in range(N):
            if zero_init:
                host[i] = Scalar[DT](0.0)
            else:
                host[i] = Scalar[DT](random_float64() * 2.0 - 1.0) * r
        comptime if target == "cpu":
            s.is_gpu = False
            s.param = alloc[Scalar[DT]](N)
            s.grad = alloc[Scalar[DT]](N)
            s.m = alloc[Scalar[DT]](N)
            s.v = alloc[Scalar[DT]](N)
            s.host_n = N
            for i in range(N):
                s.param[i] = host[i]
                s.grad[i] = Scalar[DT](0.0)
                s.m[i] = Scalar[DT](0.0)
                s.v[i] = Scalar[DT](0.0)
            host.free()
        else:
            var c = ctx.value()
            s.is_gpu = True
            s.ctx = ctx
            var dp = c.enqueue_create_buffer[DT](N)
            var dg = c.enqueue_create_buffer[DT](N)
            var dm = c.enqueue_create_buffer[DT](N)
            var dv = c.enqueue_create_buffer[DT](N)
            var hb = c.enqueue_create_host_buffer[DT](N)
            c.synchronize()
            for i in range(N):
                hb.unsafe_ptr()[i] = host[i]
            c.enqueue_copy(dp, hb)
            dg.enqueue_fill(0.0)
            dm.enqueue_fill(0.0)
            dv.enqueue_fill(0.0)
            c.synchronize()
            s.d_param = dp^; s.d_grad = dg^; s.d_m = dm^; s.d_v = dv^
            host.free()
        return s^

    @always_inline
    def _pp(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return _dp(self.d_param.value()) if self.is_gpu else self.param

    @always_inline
    def _gp(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return _dp(self.d_grad.value()) if self.is_gpu else self.grad

    def zero_grad[target: StaticString](mut self) raises:
        comptime N = Self.N
        comptime if target == "cpu":
            for i in range(N):
                self.grad[i] = Scalar[DT](0.0)
        else:
            var c = self.ctx.value()
            comptime nb = (N + TPB - 1) // TPB
            c.enqueue_function[_zero_k[N]](
                self._gp(), grid_dim=nb, block_dim=TPB
            )

    def gather[target: StaticString, NROWS: Int](
        self,
        task_ids: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [NROWS]
        dst: UnsafePointer[Scalar[DT], MutAnyOrigin],        # [NROWS, TASK_EMB]
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """`dst[row] = param[task_ids[row]]`. CPU pointers / GPU device pointers."""
        comptime EMB = Self.TASK_EMB
        comptime if target == "cpu":
            for row in range(NROWS):
                var t = Int(task_ids[row])
                for e in range(EMB):
                    dst[row * EMB + e] = self.param[t * EMB + e]
        else:
            var c = ctx.value()
            comptime nb = (NROWS * EMB + TPB - 1) // TPB
            c.enqueue_function[_gather_k[NROWS, EMB]](
                task_ids, self._pp(), dst, grid_dim=nb, block_dim=TPB
            )

    def accumulate[
        target: StaticString, NROWS: Int, ROWW: Int, COFF: Int
    ](
        mut self,
        task_ids: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [NROWS]
        grad_wide: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [NROWS, ROWW]
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """`grad_table[task_ids[row]] += grad_wide[row, COFF : COFF+TASK_EMB]`.
        `ROWW`/`COFF` let one impl serve both the graph slot grad (ROWW=TASK_EMB,
        COFF=0) and the encoder input-grad tail (ROWW=MAX_OBS+TASK_EMB,
        COFF=MAX_OBS)."""
        comptime EMB = Self.TASK_EMB
        comptime if target == "cpu":
            for row in range(NROWS):
                var t = Int(task_ids[row])
                for e in range(EMB):
                    self.grad[t * EMB + e] += grad_wide[row * ROWW + COFF + e]
        else:
            var c = ctx.value()
            comptime nt = (Self.N + TPB - 1) // TPB
            c.enqueue_function[
                _accum_k[Self.NUM_TASKS, EMB, NROWS, ROWW, COFF]
            ](
                task_ids, grad_wide, self._gp(), grid_dim=nt, block_dim=TPB
            )

    def step[target: StaticString](mut self) raises:
        comptime N = Self.N
        self.b1pow = self.b1pow * self.beta1
        self.b2pow = self.b2pow * self.beta2
        var bc1 = Scalar[DT](1.0) - self.b1pow
        var bc2 = Scalar[DT](1.0) - self.b2pow
        comptime if target == "cpu":
            for i in range(N):
                var g = self.grad[i]
                var m_new = self.beta1 * self.m[i] + (
                    Scalar[DT](1.0) - self.beta1
                ) * g
                var v_new = self.beta2 * self.v[i] + (
                    Scalar[DT](1.0) - self.beta2
                ) * g * g
                self.m[i] = m_new
                self.v[i] = v_new
                self.param[i] = self.param[i] - self.lr * (m_new / bc1) / (
                    sqrt(v_new / bc2) + self.eps
                )
        else:
            var c = self.ctx.value()
            comptime nb = (N + TPB - 1) // TPB
            c.enqueue_function[_adam_k[N]](
                self._pp(), self._gp(),
                _dp(self.d_m.value()), _dp(self.d_v.value()),
                self.lr, self.beta1, self.beta2, self.eps, bc1, bc2,
                grid_dim=nb, block_dim=TPB,
            )

    # ── Checkpoint (CPU-resident serialization; GPU syncs around it) ──────
    def sync_to_host(mut self) raises:
        """D2H param/m/v into freshly-allocated host buffers (GPU only)."""
        if not self.is_gpu:
            return
        var c = self.ctx.value()
        comptime N = Self.N
        if self.host_n < N:
            self.param = alloc[Scalar[DT]](N)
            self.m = alloc[Scalar[DT]](N)
            self.v = alloc[Scalar[DT]](N)
            self.host_n = N
        var hp = c.enqueue_create_host_buffer[DT](N)
        var hm = c.enqueue_create_host_buffer[DT](N)
        var hv = c.enqueue_create_host_buffer[DT](N)
        c.enqueue_copy(hp, self.d_param.value())
        c.enqueue_copy(hm, self.d_m.value())
        c.enqueue_copy(hv, self.d_v.value())
        c.synchronize()
        for i in range(N):
            self.param[i] = hp.unsafe_ptr()[i]
            self.m[i] = hm.unsafe_ptr()[i]
            self.v[i] = hv.unsafe_ptr()[i]

    def upload_from_host(mut self) raises:
        if not self.is_gpu:
            return
        var c = self.ctx.value()
        comptime N = Self.N
        var hp = c.enqueue_create_host_buffer[DT](N)
        var hm = c.enqueue_create_host_buffer[DT](N)
        var hv = c.enqueue_create_host_buffer[DT](N)
        c.synchronize()
        for i in range(N):
            hp.unsafe_ptr()[i] = self.param[i]
            hm.unsafe_ptr()[i] = self.m[i]
            hv.unsafe_ptr()[i] = self.v[i]
        c.enqueue_copy(self.d_param.value(), hp)
        c.enqueue_copy(self.d_m.value(), hm)
        c.enqueue_copy(self.d_v.value(), hv)
        c.synchronize()

    def save_body(mut self, mut out: String, name: String) raises:
        """Append param + m + v + Adam scalars as a small text body. On GPU,
        call `sync_to_host` first."""
        self.sync_to_host()
        out += name + "#size=" + String(Self.N) + "\n"
        for i in range(Self.N):
            out += String(self.param[i]) + "\n"
        for i in range(Self.N):
            out += String(self.m[i]) + "\n"
        for i in range(Self.N):
            out += String(self.v[i]) + "\n"
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
                + expect + "`, got `" + header + "`"
            )
        idx += 1
        if self.host_n < Self.N:
            self.param = alloc[Scalar[DT]](Self.N)
            self.m = alloc[Scalar[DT]](Self.N)
            self.v = alloc[Scalar[DT]](Self.N)
            self.host_n = Self.N
        for i in range(Self.N):
            self.param[i] = Scalar[DT](atof(lines[idx])); idx += 1
        for i in range(Self.N):
            self.m[i] = Scalar[DT](atof(lines[idx])); idx += 1
        for i in range(Self.N):
            self.v[i] = Scalar[DT](atof(lines[idx])); idx += 1
        self.b1pow = Scalar[DT](atof(lines[idx])); idx += 1
        self.b2pow = Scalar[DT](atof(lines[idx])); idx += 1
        self.upload_from_host()
