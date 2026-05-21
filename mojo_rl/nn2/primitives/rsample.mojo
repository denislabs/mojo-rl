"""RSample — retrofit (Phase E). Thin Module wrapper around the
canonical `squashed_gaussian_forward` / `squashed_gaussian_backward`
free functions from `nn2/loss/squashed_gaussian.mojo`.

Topology unchanged from v1:
    input  [BATCH, 2*ACT]   packed [mu | log_std]
    output [BATCH, ACT+1]   packed [action | log_prob]

But the math body is now ~30 lines instead of v1's ~120 — everything
substantive lives in the canonical pair, this struct just splits/joins
the packed tiles and owns the z-cache.

`action_scale` stays a public mut field.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from ..constants import DT
from ..core import Initializer, AMPPolicy, NoAMP, ParamVisitor
from ..core.module import Module
from ..core.target_storage import (
    TargetStorage, assert_tag_for, ensure_cpu_buffer,
)
from ..random.box_muller import box_muller_normal
from ..loss.squashed_gaussian import (
    squashed_gaussian_forward,
    squashed_gaussian_backward,
)


struct RSample[ACT: Int](Module):
    comptime IN_DIM = 2 * Self.ACT
    comptime OUT_DIM = Self.ACT + 1

    var action_scale: Scalar[DT]

    # Backward caches (CPU). z_cache: fresh noise drawn each forward.
    # in_cache: raw input (mu/log_std) — caller may overwrite between
    # forward and backward, so we copy it.
    var z_cache: List[Scalar[DT]]
    var in_cache: List[Scalar[DT]]
    var cache_n_batch: Int

    # GPU placeholders — Phase E still ships CPU only.
    var z_cache_dev: Optional[DeviceBuffer[DT]]
    var in_cache_dev: Optional[DeviceBuffer[DT]]

    var ts: TargetStorage

    def __init__(out self):
        self.action_scale = Scalar[DT](1.0)
        self.z_cache = List[Scalar[DT]]()
        self.in_cache = List[Scalar[DT]]()
        self.cache_n_batch = 0
        self.z_cache_dev = None
        self.in_cache_dev = None
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "RSample.make[target='gpu', INIT] requires a DeviceContext"
        )
        var r = Self()
        r.ts = TargetStorage.make_cpu()
        return r^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "RSample.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        var r = Self()
        r.ts = TargetStorage.make_gpu(ctx)
        return r^

    def _ensure_cache_cpu(mut self, batch: Int):
        if self.cache_n_batch < batch:
            self.z_cache.resize(batch * Self.ACT, Scalar[DT](0.0))
            self.in_cache.resize(batch * (2 * Self.ACT), Scalar[DT](0.0))
            self.cache_n_batch = batch

    # ----- Forward ---------------------------------------------------------

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        input: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        comptime assert input.flat_rank == 2, "input must be rank-2 [BATCH, 2*ACT]"
        comptime assert output.flat_rank == 2, "output must be rank-2 [BATCH, ACT+1]"
        comptime assert Self.ACT >= 1, "RSample[ACT]: ACT >= 1"
        assert_tag_for["RSample", target](self.ts.target_tag)

        comptime if target == "cpu":
            self._ensure_cache_cpu(BATCH)
            # Cache input + draw fresh z.
            for b in range(BATCH):
                for j in range(2 * Self.ACT):
                    self.in_cache[b * (2 * Self.ACT) + j] = input[b, j]
            box_muller_normal(self.z_cache.unsafe_ptr(), BATCH * Self.ACT)

            # Build local TileTensors for action and log_prob slices of
            # the packed output, plus a TileTensor view over the z cache.
            # We can't view output directly because action is [BATCH, ACT]
            # but output is [BATCH, ACT+1] — log_prob lives at column ACT.
            var z_t = TileTensor(self.z_cache, row_major[BATCH, Self.ACT]())

            # Use scratch buffers for action + log_prob (mojo tiles can't
            # alias non-contiguous slices). Then copy into the packed output.
            # For BATCH × ACT this is negligible.
            var act_buf = List[Scalar[DT]](length=BATCH * Self.ACT, fill=0.0)
            var lp_buf = List[Scalar[DT]](length=BATCH, fill=0.0)
            var act_t = TileTensor(act_buf, row_major[BATCH, Self.ACT]())
            var lp_t  = TileTensor(lp_buf,  row_major[BATCH]())

            squashed_gaussian_forward[Self.ACT, BATCH](
                input, z_t, self.action_scale, act_t, lp_t,
            )

            # Pack into output [BATCH, ACT+1]:
            #   output[b, j] = action[b, j]   for j in [0, ACT)
            #   output[b, ACT] = log_prob[b]
            for b in range(BATCH):
                for j in range(Self.ACT):
                    output[b, j] = act_buf[b * Self.ACT + j]
                output[b, Self.ACT] = lp_buf[b]
        else:
            raise Error("RSample[ACT]: GPU path not yet implemented")

    # ----- Backward --------------------------------------------------------

    def backward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        mut grad_input: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        comptime assert grad_output.flat_rank == 2, "grad_output rank-2"
        comptime assert grad_input.flat_rank == 2, "grad_input rank-2"
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["RSample", target](self.ts.target_tag)

        comptime if target == "cpu":
            # Unpack grad_output [BATCH, ACT+1] → grad_action [BATCH, ACT]
            # + grad_log_prob [BATCH]. Scratch buffers (small, neg cost).
            var ga_buf = List[Scalar[DT]](length=BATCH * Self.ACT, fill=0.0)
            var glp_buf = List[Scalar[DT]](length=BATCH, fill=0.0)
            for b in range(BATCH):
                for j in range(Self.ACT):
                    ga_buf[b * Self.ACT + j] = grad_output[b, j]
                glp_buf[b] = grad_output[b, Self.ACT]
            var ga_t  = TileTensor(ga_buf,  row_major[BATCH, Self.ACT]())
            var glp_t = TileTensor(glp_buf, row_major[BATCH]())
            var z_t   = TileTensor(self.z_cache, row_major[BATCH, Self.ACT]())
            var ic_t  = TileTensor(
                self.in_cache, row_major[BATCH, 2 * Self.ACT](),
            )

            squashed_gaussian_backward[Self.ACT, BATCH](
                ic_t, z_t, ga_t, glp_t, self.action_scale, grad_input,
            )
        else:
            raise Error("RSample[ACT]: GPU backward not yet implemented")
