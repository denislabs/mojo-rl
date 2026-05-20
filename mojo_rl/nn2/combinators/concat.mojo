"""Concat[*BRANCHES] — variadic column-concat. Phase 8.4.

Variadic generalization of the Phase 5 `Parallel[A, B]` combinator: N
branches sharing the same input dimension, outputs concatenated
side-by-side into a single packed tile.

    output[b, j_block_0]            = BRANCHES[0](input)[b, j_block_0]
    output[b, sum_{<i}(OUT) + j]    = BRANCHES[i](input)[b, j]

Constraints (comptime-checked):
    - N >= 1
    - All BRANCHES share IN_DIM
    - OUT_DIM = Σ BRANCHES[i].OUT_DIM

Backward:
    grad_input = Σ BRANCHES[i].backward(grad_output_slice_i)

Internal scratch (per-branch, lazy-grown):
    out_slabs[i] : BATCH × BRANCHES[i].OUT_DIM
                   forward output of branch i; also reused on backward as
                   the grad_output slice fed to BRANCHES[i].backward.
    gi_temp      : BATCH × IN_DIM
                   shared accumulator for backward — each branch.backward
                   writes here, and we sum-into-output. First branch
                   overwrites grad_input; subsequent branches += into it.

Phase 8.4 ships CPU only. GPU paths raise — no validating user yet.
"""

from std.memory import alloc
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout, row_major

from ..constants import DT, CPU_SIMD_W
from ..core import (
    Module, ParamVisitor, Initializer,
    AMPPolicy, NoAMP,
    TARGET_UNINIT, TARGET_CPU, TARGET_GPU, target_tag_for,
)


# ──────────────────────────────────────────────────────────────────────────
# Comptime helpers for variadic sum and cumulative offsets.
# ──────────────────────────────────────────────────────────────────────────


def _total_out_dim[*BRANCHES: Module]() -> Int:
    var s: Int = 0
    comptime for i in range(BRANCHES.size):
        s += BRANCHES[i].OUT_DIM
    return s


def _cumulative_offset[index: Int, *BRANCHES: Module]() -> Int:
    """Σ_{j<index} BRANCHES[j].OUT_DIM."""
    var s: Int = 0
    comptime for j in range(index):
        s += BRANCHES[j].OUT_DIM
    return s


# ──────────────────────────────────────────────────────────────────────────
# Concat — variadic column-concat.
# ──────────────────────────────────────────────────────────────────────────


struct Concat[*BRANCHES: Module](Module):
    comptime N = Self.BRANCHES.size
    comptime IN_DIM = Self.BRANCHES[0].IN_DIM
    comptime OUT_DIM = _total_out_dim[*Self.BRANCHES]()

    var branches: Tuple[*Self.BRANCHES]
    var ctx: Optional[DeviceContext]

    # Per-branch output scratch slabs (N entries).
    var out_slabs_cpu: List[UnsafePointer[Scalar[DT], MutAnyOrigin]]
    var out_slab_caps: List[Int]
    # Shared per-branch grad_input temp.
    var gi_temp_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var gi_temp_cap: Int

    var _target_tag: Int8
    var _inference: Bool

    # ------------------------------------------------------------------
    # Defaultable + ctor.
    # ------------------------------------------------------------------

    def __init__(out self):
        comptime assert Self.N >= 1, "Concat requires at least one branch"
        comptime for i in range(Self.N):
            comptime assert (
                Self.BRANCHES[i].IN_DIM == Self.BRANCHES[0].IN_DIM
            ), "Concat: all BRANCHES must share IN_DIM"
        self.branches = Tuple[*Self.BRANCHES]()
        self.ctx = None
        self.out_slabs_cpu = List[UnsafePointer[Scalar[DT], MutAnyOrigin]]()
        self.out_slab_caps = List[Int]()
        self.gi_temp_cpu = alloc[Scalar[DT]](1)
        self.gi_temp_cap = 0
        self._target_tag = TARGET_UNINIT
        self._inference = False

    def __init__(out self, var *branches: *Self.BRANCHES):
        """CPU variadic constructor — accepts pre-built CPU branches."""
        comptime assert Self.N >= 1, "Concat requires at least one branch"
        comptime for i in range(Self.N):
            comptime assert (
                Self.BRANCHES[i].IN_DIM == Self.BRANCHES[0].IN_DIM
            ), "Concat: all BRANCHES must share IN_DIM"
        self.branches = Tuple(*branches^)
        self.ctx = None
        self.out_slabs_cpu = List[UnsafePointer[Scalar[DT], MutAnyOrigin]]()
        self.out_slab_caps = List[Int]()
        for _ in range(Self.N):
            self.out_slabs_cpu.append(alloc[Scalar[DT]](1))
            self.out_slab_caps.append(0)
        self.gi_temp_cpu = alloc[Scalar[DT]](1)
        self.gi_temp_cap = 0
        self._target_tag = TARGET_CPU
        self._inference = False

    # ------------------------------------------------------------------
    # make[target, INIT] — recursive build.
    # ------------------------------------------------------------------

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "Concat.make[target='gpu', INIT] requires a DeviceContext"
        )
        var c = Self()
        comptime for i in range(Self.N):
            c.branches[i] = Self.BRANCHES[i].make[target, INIT]()
        for _ in range(Self.N):
            c.out_slabs_cpu.append(alloc[Scalar[DT]](1))
            c.out_slab_caps.append(0)
        c._target_tag = TARGET_CPU
        return c^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "Concat.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        # GPU make stamps the tag so trait dispatch works, but every
        # method raises until kernels land (no validating user yet).
        var c = Self()
        comptime for i in range(Self.N):
            c.branches[i] = Self.BRANCHES[i].make[target, INIT](ctx)
        c.ctx = ctx
        c._target_tag = TARGET_GPU
        return c^

    def __del__(deinit self):
        for p in self.out_slabs_cpu:
            p.free()
        self.gi_temp_cpu.free()

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "Concat: method called with [target='"
                + String(target)
                + "'] but module was make'd for a different target (tag="
                + String(Int(self._target_tag)) + ")"
            )

    def _ensure_slab_cpu[i: Int](mut self, needed: Int):
        if self.out_slab_caps[i] < needed:
            self.out_slabs_cpu[i].free()
            self.out_slabs_cpu[i] = alloc[Scalar[DT]](needed)
            self.out_slab_caps[i] = needed

    def _ensure_gi_temp_cpu(mut self, needed: Int):
        if self.gi_temp_cap < needed:
            self.gi_temp_cpu.free()
            self.gi_temp_cpu = alloc[Scalar[DT]](needed)
            self.gi_temp_cap = needed

    # ------------------------------------------------------------------
    # Forward.
    # ------------------------------------------------------------------

    def forward[
        target: StaticString,
        BATCH: Int,
        LIN: TensorLayout,
        LOUT: TensorLayout,
        OIN: MutOrigin,
        OOUT: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        input: TileTensor[DT, LIN, OIN],
        mut output: TileTensor[DT, LOUT, OOUT],
    ) raises:
        comptime assert input.flat_rank == 2, "input rank-2"
        comptime assert output.flat_rank == 2, "output rank-2"
        self._assert_tag[target]()

        comptime if target == "cpu":
            _concat_forward_cpu[target, BATCH, POLICY=POLICY](self, input, output)
        else:
            raise Error("Concat: GPU forward not yet implemented (Phase 8.4 CPU only)")

    # ------------------------------------------------------------------
    # Backward.
    # ------------------------------------------------------------------

    def backward[
        target: StaticString,
        BATCH: Int,
        LGO: TensorLayout,
        LGI: TensorLayout,
        OGO: MutOrigin,
        OGI: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        grad_output: TileTensor[DT, LGO, OGO],
        mut grad_input: TileTensor[DT, LGI, OGI],
    ) raises:
        comptime assert grad_output.flat_rank == 2, "grad_output rank-2"
        comptime assert grad_input.flat_rank == 2, "grad_input rank-2"
        self._assert_tag[target]()

        comptime if target == "cpu":
            _concat_backward_cpu[target, BATCH, POLICY=POLICY, use_backward_input=False](
                self, grad_output, grad_input
            )
        else:
            raise Error("Concat: GPU backward not yet implemented (Phase 8.4 CPU only)")

    def backward_input[
        target: StaticString,
        BATCH: Int,
        LGO: TensorLayout,
        LGI: TensorLayout,
        OGO: MutOrigin,
        OGI: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        grad_output: TileTensor[DT, LGO, OGO],
        mut grad_input: TileTensor[DT, LGI, OGI],
    ) raises:
        comptime assert grad_output.flat_rank == 2, "grad_output rank-2"
        comptime assert grad_input.flat_rank == 2, "grad_input rank-2"
        self._assert_tag[target]()

        comptime if target == "cpu":
            _concat_backward_cpu[target, BATCH, POLICY=POLICY, use_backward_input=True](
                self, grad_output, grad_input
            )
        else:
            raise Error("Concat: GPU backward_input not yet implemented (Phase 8.4 CPU only)")

    # ------------------------------------------------------------------
    # for_each_param — recurse with indexed prefix (mirrors Sequential).
    # ------------------------------------------------------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V,) raises:
        self._assert_tag[target]()
        var sep = "." if prefix.byte_length() > 0 else ""
        comptime for i in range(Self.N):
            self.branches[i].for_each_param[target](
                prefix + sep + String(i), visitor
            )

    def set_inference(mut self, value: Bool):
        self._inference = value
        comptime for i in range(Self.N):
            self.branches[i].set_inference(value)


# ──────────────────────────────────────────────────────────────────────────
# Free-function bodies (mirror Sequential's pattern to keep Concat's
# per-method body short and avoid the inline-call-explosion trap).
# ──────────────────────────────────────────────────────────────────────────


def _concat_forward_cpu[
    target: StaticString,
    BATCH: Int,
    LIN: TensorLayout,
    LOUT: TensorLayout,
    OIN: MutOrigin,
    OOUT: MutOrigin,
    POLICY: AMPPolicy,
    *BRANCHES: Module,
](
    mut c: Concat[*BRANCHES],
    input: TileTensor[DT, LIN, OIN],
    mut output: TileTensor[DT, LOUT, OOUT],
) raises:
    comptime assert input.flat_rank == 2, "input rank-2"
    comptime assert output.flat_rank == 2, "output rank-2"
    comptime N = BRANCHES.size

    # Grow per-branch output slabs to BATCH * BRANCHES[i].OUT_DIM.
    comptime for i in range(N):
        c._ensure_slab_cpu[i](BATCH * BRANCHES[i].OUT_DIM)

    # Forward each branch into its slab.
    comptime for i in range(N):
        var slab_ptr = c.out_slabs_cpu[i]
        var slab_tt = TileTensor(slab_ptr, row_major[BATCH, BRANCHES[i].OUT_DIM]())
        c.branches[i].forward[target, BATCH, POLICY=POLICY](input, slab_tt)

    # Concat slabs into packed output. We do the copy row-by-row,
    # branch-by-branch — at typical RL output sizes (≤30 dims per branch)
    # this is negligible.
    comptime for i in range(N):
        comptime off = _cumulative_offset[i, *BRANCHES]()
        comptime out_i = BRANCHES[i].OUT_DIM
        var slab_ptr = c.out_slabs_cpu[i]
        for b in range(BATCH):
            for j in range(out_i):
                output[b, off + j] = slab_ptr[b * out_i + j]


def _concat_backward_cpu[
    target: StaticString,
    BATCH: Int,
    LGO: TensorLayout,
    LGI: TensorLayout,
    OGO: MutOrigin,
    OGI: MutOrigin,
    POLICY: AMPPolicy,
    use_backward_input: Bool,
    *BRANCHES: Module,
](
    mut c: Concat[*BRANCHES],
    grad_output: TileTensor[DT, LGO, OGO],
    mut grad_input: TileTensor[DT, LGI, OGI],
) raises:
    comptime assert grad_output.flat_rank == 2, "grad_output rank-2"
    comptime assert grad_input.flat_rank == 2, "grad_input rank-2"
    comptime N = BRANCHES.size
    comptime IN_DIM = BRANCHES[0].IN_DIM

    comptime for i in range(N):
        c._ensure_slab_cpu[i](BATCH * BRANCHES[i].OUT_DIM)
    c._ensure_gi_temp_cpu(BATCH * IN_DIM)

    # Zero the caller's grad_input (we'll accumulate into it).
    var grad_input_w = rebind[TileTensor[DT, LGI, MutAnyOrigin]](grad_input)
    var gi_p = grad_input_w.ptr
    var zero_v = SIMD[DT, CPU_SIMD_W](0)
    comptime N_TOTAL = BATCH * IN_DIM
    var k0 = 0
    while k0 + CPU_SIMD_W <= N_TOTAL:
        gi_p.store(k0, zero_v)
        k0 += CPU_SIMD_W
    while k0 < N_TOTAL:
        gi_p[k0] = Scalar[DT](0)
        k0 += 1

    comptime for i in range(N):
        comptime off = _cumulative_offset[i, *BRANCHES]()
        comptime out_i = BRANCHES[i].OUT_DIM
        var slab_ptr = c.out_slabs_cpu[i]
        # Split grad_output slice [:, off:off+out_i] → slab[i] (reused as
        # the grad-output scratch fed to BRANCHES[i].backward).
        for b in range(BATCH):
            for j in range(out_i):
                slab_ptr[b * out_i + j] = grad_output[b, off + j]
        var go_tt = TileTensor(slab_ptr, row_major[BATCH, out_i]())
        # Branch backward into gi_temp.
        var gi_temp = TileTensor(c.gi_temp_cpu, row_major[BATCH, IN_DIM]())
        comptime if use_backward_input:
            c.branches[i].backward_input[target, BATCH, POLICY=POLICY](
                go_tt, gi_temp
            )
        else:
            c.branches[i].backward[target, BATCH, POLICY=POLICY](go_tt, gi_temp)
        # Accumulate gi_temp into grad_input via SIMD add.
        var ap = c.gi_temp_cpu
        var k = 0
        while k + CPU_SIMD_W <= N_TOTAL:
            gi_p.store(
                k,
                gi_p.load[width=CPU_SIMD_W](k) + ap.load[width=CPU_SIMD_W](k),
            )
            k += CPU_SIMD_W
        while k < N_TOTAL:
            gi_p[k] = gi_p[k] + ap[k]
            k += 1
