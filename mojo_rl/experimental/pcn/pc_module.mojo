"""PCModule — PCN composite on nn2's storage layer (Phase A spike).

Re-architecture of the PCN composite onto nn2's Tensor core. The weight
slab — previously a caller-owned raw `params` buffer juggled by
`PCTrainer` — now lives in a single owned `Param`, which makes the network
walkable by nn2's `Adam`/`AdamW` (optimizer) and, later, the v2 checkpoint
envelope. **PCN keeps its own settling loop** (the local error-minimization
that IS the method); only where the weights live changes.

The struct is a *minimal* `Module` purely so it satisfies the `M: Module`
bound on `Adam.make` / `Adam.step`:
  - `forward` / `vjp` raise — PCN does not do backprop; learning runs
    through `pc_module_train_one_batch` (settling + `weight_grad`).
  - `for_each_param` / `zero_grad` / `for_each_state` are INHERITED from
    the trait defaults (reflection over `IsParam` fields), which discover
    the `weights` `Param` automatically.

Storage mapping (this phase): weights → `Param`. Working buffers (latents,
μ/ε, a_below, z_below, dx) stay caller-allocated in the trainer fn — moving
them onto the holder as `Scratch`/`Cache` is Phase B's storage propagation.

Sign convention: `PCBlock.weight_grad` already bakes the −sign so `grads`
holds the standard +∂E/∂W (descent gradient) consumed directly by
`params -= lr·grad`. That is exactly Adam's convention, so the grad buffer
feeds Adam with **no negation** — confirmed in `pc_block.mojo` (the "−sign
expected by Optimizer.step").
"""

from layout import Layout, LayoutTensor, TileTensor
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core import Initializer, AMPPolicy, NoAMP
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.core.param import Param
from mojo_rl.nn2.core.tensor_pack import TensorPack
from mojo_rl.nn2.core.target_storage import TargetStorage

from .predictive_model import PCBlockTrait
from .pc_sequential import PCSequential
from .pc_initializer import PCInitializer, PCXavier


struct PCModule[*BLOCKS: PCBlockTrait](Module):
    """Stateful PCN composite — owns the weight slab as one `Param`.

    Compose exactly like `PCSequential`:
        comptime Net = PCModule[
            PCBlock[4, 8, PCIdentity],
            PCBlock[8, 2, PCIdentity],
        ]
        var net = Net.make_pcn[PCXavier]()
        var opt = Adam.make["cpu", Net](net)
    """

    comptime NET = PCSequential[*Self.BLOCKS]

    # ── Module trait surface ─────────────────────────────────────────────
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.NET.IN_DIM)
    comptime OUT_DIM: Int = Self.NET.OUT_DIM

    # The whole concatenated parameter slab (per-block W|b, same layout as
    # the legacy `params` buffer). `APPLY_DECAY=False`: PC weight decay, if
    # any, is handled in the PC energy, not via the optimizer.
    var weights: Param["pc_params", False, Self.NET.PARAM_SIZE]
    var ts: TargetStorage

    def __init__(out self):
        self.weights = Param["pc_params", False, Self.NET.PARAM_SIZE]()
        self.ts = TargetStorage.make_uninit()

    # ── Real PCN constructor (CPU) ───────────────────────────────────────

    @staticmethod
    def make_pcn[INIT: PCInitializer = PCXavier]() raises -> Self:
        """Allocate CPU weight storage and run per-block PCN init via the
        composite's `init_params_pc` (each block type's own layout: linear /
        conv W via INIT, norm γ=1, etc.). Legacy-`nn`-free. The
        nn2-`Initializer`-based `make` below is unusable for PCN (different
        init contract), so this is the constructor real code calls."""
        var net = Self()
        net.weights = Param["pc_params", False, Self.NET.PARAM_SIZE].make_cpu()
        net.ts = TargetStorage.make_cpu()

        var params = LayoutTensor[
            DT, Layout.row_major(Self.NET.PARAM_SIZE), MutAnyOrigin
        ](net.weights.value_unsafe_ptr_cpu())
        Self.NET.pc_init_params[INIT, DT](params)
        return net^

    @staticmethod
    def make_pcn_gpu[
        INIT: PCInitializer = PCXavier
    ](ctx: DeviceContext) raises -> Self:
        """GPU counterpart of `make_pcn`: allocate device weight storage,
        run the (host) per-block init, and upload. Weights live in
        `weights.val.dev`; nn2 `Adam.make['gpu']` and `compute_grads_only_gpu`
        read them on-device."""
        var net = Self()
        net.weights = Param["pc_params", False, Self.NET.PARAM_SIZE].make_gpu(
            ctx
        )
        net.ts = TargetStorage.make_gpu(ctx)
        # Init on host, then upload to the device value buffer.
        var host = List[Scalar[DT]](
            length=Self.NET.PARAM_SIZE, fill=Scalar[DT](0)
        )
        var host_view = LayoutTensor[
            DT, Layout.row_major(Self.NET.PARAM_SIZE), MutAnyOrigin
        ](host.unsafe_ptr())
        Self.NET.pc_init_params[INIT, DT](host_view)
        ctx.enqueue_copy(net.weights.val.dev.value(), host.unsafe_ptr())
        ctx.synchronize()
        _ = host^
        return net^

    # ── Module factory — required by the trait; unusable for PCN ─────────

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        raise Error(
            "PCModule.make: PCN initializes per-block (fan_in/fan_out), not"
            " via the nn2 Initializer contract. Use make_pcn[PCInitializer]()."
        )

    # ── forward / vjp — PCN does not backprop; these only satisfy Module ─

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorPack[Self.ARITY],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        raise Error(
            "PCModule.forward: PCN uses the settling loop"
            " (pc_module_train_one_batch / forward_eval), not Module.forward."
        )

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
        grad_inputs: TensorPack[Self.ARITY],
    ) raises:
        raise Error(
            "PCModule.vjp: PCN uses local error-minimization (weight_grad),"
            " not vector-Jacobian backprop."
        )

    @staticmethod
    def display_label() -> String:
        return String("PCModule")
