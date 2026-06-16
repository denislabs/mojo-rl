"""DuelingHeadC51[NA, N_ATOMS] — Distributional Dueling DQN aggregation.

The Rainbow-style fusion of `DuelingHead` and `C51`. Input layout:

    [B, (1 + NA) · N_ATOMS]
        |  V (atoms)         |  A_0 (atoms)  |  A_1  | ... | A_{NA-1}  |
        |   N_ATOMS columns  |  N_ATOMS cols | ...                     |

Output: `[B, NA · N_ATOMS]`, with per-atom Wang et al. 2016 dueling
aggregation:

    Q[b, a, k] = V[b, k] + A[b, a, k] − (1/NA) · Σ_a' A[b, a', k]

Each atom k gets its own V-stream and its own mean-subtracted A-stream.
After atom-wise softmax (done by the C51 loss/target paths), the
resulting per-action distribution implicitly fuses the value baseline
with the advantage shape, which Rainbow reports as a strong gain over
plain Dueling at the same parameter count.

Backward:
    grad_in[b, V_k]     = Σ_a grad_out[b, a, k]
    grad_in[b, A_a_k]   = grad_out[b, a, k] − (1/NA) · Σ_a grad_out[b, a, k]

Use as: `Sequential[backbone..., Linear[H, (1 + NA)·N_ATOMS], DuelingHeadC51[NA, N_ATOMS]]`.
Pure architectural swap — no trainer or block change.

CPU + GPU.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor

from ..constants import DT, TPB
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module, typed_view, typed_view_mut
from ..core.tensor_pack import TensorPack
from ..core.target_storage import TargetStorage, assert_tag_for


def _dueling_c51_combine_kernel[
    BATCH: Int, NA: Int, N_ATOMS: Int,
](
    raw_in: LayoutTensor[
        DT, Layout.row_major(BATCH, (1 + NA) * N_ATOMS), MutAnyOrigin,
    ],
    q_out: LayoutTensor[
        DT, Layout.row_major(BATCH, NA * N_ATOMS), MutAnyOrigin,
    ],
):
    """One thread per (batch, atom). For each atom k:
        mean_A_k = (1/NA) · Σ_a A[b, a, k]
        Q[b, a, k] = V[b, k] + A[b, a, k] − mean_A_k  ∀ a
    """
    var lin = Int(global_idx.x)
    var total = BATCH * N_ATOMS
    if lin < total:
        var b = lin // N_ATOMS
        var k = lin % N_ATOMS
        var v_k = rebind[Scalar[DT]](raw_in[b, k])
        var sum_a: Scalar[DT] = 0.0
        for a in range(NA):
            sum_a = sum_a + rebind[Scalar[DT]](
                raw_in[b, N_ATOMS + a * N_ATOMS + k]
            )
        var mean_a = sum_a * (Scalar[DT](1.0) / Scalar[DT](NA))
        for a in range(NA):
            var adv = rebind[Scalar[DT]](
                raw_in[b, N_ATOMS + a * N_ATOMS + k]
            )
            q_out[b, a * N_ATOMS + k] = v_k + (adv - mean_a)


def _dueling_c51_grad_kernel[
    BATCH: Int, NA: Int, N_ATOMS: Int,
](
    grad_out: LayoutTensor[
        DT, Layout.row_major(BATCH, NA * N_ATOMS), MutAnyOrigin,
    ],
    grad_in: LayoutTensor[
        DT, Layout.row_major(BATCH, (1 + NA) * N_ATOMS), MutAnyOrigin,
    ],
):
    """One thread per (batch, atom). For each atom k:
        dV[b, k]      = Σ_a grad_out[b, a, k]
        dA[b, a, k]   = grad_out[b, a, k] − (1/NA) · dV[b, k]
    """
    var lin = Int(global_idx.x)
    var total = BATCH * N_ATOMS
    if lin < total:
        var b = lin // N_ATOMS
        var k = lin % N_ATOMS
        var sum_dq: Scalar[DT] = 0.0
        for a in range(NA):
            sum_dq = sum_dq + rebind[Scalar[DT]](
                grad_out[b, a * N_ATOMS + k]
            )
        grad_in[b, k] = sum_dq
        var inv = Scalar[DT](1.0) / Scalar[DT](NA)
        for a in range(NA):
            grad_in[b, N_ATOMS + a * N_ATOMS + k] = (
                rebind[Scalar[DT]](grad_out[b, a * N_ATOMS + k])
                - inv * sum_dq
            )


struct DuelingHeadC51[NA: Int, N_ATOMS: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=(1 + Self.NA) * Self.N_ATOMS)
    comptime OUT_DIM: Int = Self.NA * Self.N_ATOMS

    var ts: TargetStorage

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "DuelingHeadC51: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.NA > 0, "DuelingHeadC51: NA must be > 0"
        comptime assert Self.N_ATOMS > 0, "DuelingHeadC51: N_ATOMS must be > 0"
        var h = Self()
        comptime if target == "cpu":
            h.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("DuelingHeadC51.make[target='gpu']: ctx required")
            h.ts = TargetStorage.make_gpu(ctx.value())
        return h^

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
        assert_tag_for["DuelingHeadC51", target](self.ts.target_tag)
        var input = inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            var inv = Scalar[DT](1.0) / Scalar[DT](Self.NA)
            for b in range(BATCH):
                for k in range(Self.N_ATOMS):
                    var v_k = input[b, k]
                    var sum_a: Scalar[DT] = 0.0
                    for a in range(Self.NA):
                        sum_a = sum_a + input[
                            b, Self.N_ATOMS + a * Self.N_ATOMS + k
                        ]
                    var mean_a = sum_a * inv
                    for a in range(Self.NA):
                        var adv = input[
                            b, Self.N_ATOMS + a * Self.N_ATOMS + k
                        ]
                        output_v[b, a * Self.N_ATOMS + k] = v_k + (adv - mean_a)
        else:
            var in_p = input.ptr
            var out_p = output_v.ptr
            var in_lt = LayoutTensor[
                DT,
                Layout.row_major(BATCH, (1 + Self.NA) * Self.N_ATOMS),
                MutAnyOrigin,
            ](in_p)
            var out_lt = LayoutTensor[
                DT,
                Layout.row_major(BATCH, Self.NA * Self.N_ATOMS),
                MutAnyOrigin,
            ](out_p)
            comptime total = BATCH * Self.N_ATOMS
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _dueling_c51_combine_kernel[
                BATCH, Self.NA, Self.N_ATOMS,
            ]
            self.ts.ctx.value().enqueue_function[kernel](
                in_lt, out_lt, grid_dim=n_blocks, block_dim=TPB,
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
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["DuelingHeadC51", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = grad_inputs.tile[0, BATCH, Self.IN_DIMS[0]]()

        comptime if target == "cpu":
            var inv = Scalar[DT](1.0) / Scalar[DT](Self.NA)
            for b in range(BATCH):
                for k in range(Self.N_ATOMS):
                    var sum_dq: Scalar[DT] = 0.0
                    for a in range(Self.NA):
                        sum_dq = sum_dq + grad_output_v[
                            b, a * Self.N_ATOMS + k
                        ]
                    grad_input_v[b, k] = sum_dq
                    for a in range(Self.NA):
                        grad_input_v[
                            b, Self.N_ATOMS + a * Self.N_ATOMS + k
                        ] = (
                            grad_output_v[b, a * Self.N_ATOMS + k]
                            - inv * sum_dq
                        )
        else:
            var go_p = grad_output_v.ptr
            var gi_p = grad_input_v.ptr
            var go_lt = LayoutTensor[
                DT,
                Layout.row_major(BATCH, Self.NA * Self.N_ATOMS),
                MutAnyOrigin,
            ](go_p)
            var gi_lt = LayoutTensor[
                DT,
                Layout.row_major(BATCH, (1 + Self.NA) * Self.N_ATOMS),
                MutAnyOrigin,
            ](gi_p)
            comptime total = BATCH * Self.N_ATOMS
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime kernel = _dueling_c51_grad_kernel[
                BATCH, Self.NA, Self.N_ATOMS,
            ]
            self.ts.ctx.value().enqueue_function[kernel](
                go_lt, gi_lt, grid_dim=n_blocks, block_dim=TPB,
            )
