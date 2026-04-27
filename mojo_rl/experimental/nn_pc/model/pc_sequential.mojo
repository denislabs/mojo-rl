"""PCSequential — variadic stack of PCLinear levels.

Layers are listed in feedforward / nn.Sequential order. The LAST layer is
treated as the readout (must use `ACT=PCIdentity`); all preceding layers are
hidden levels.

For arch [3072, 1000, 500, 10] + readout [10, 10]:
    PCSequential[
        PCLinear[3072, 1000],          # level 0   — predicts x^(0)=input from x^(1)
        PCLinear[1000, 500],           # level 1   — predicts x^(1) from x^(2)
        PCLinear[500, 10],             # level 2   — predicts x^(2) from x^(3)
        PCLinear[10, 10, PCIdentity],  # readout   — produces y_hat from x^(3)
    ]

PCN structure (paper notation):
  - N_LINEARS = N (here 4)
  - N_LATENTS = N - 1 (= L in the paper) (here 3)
  - x^(0)  = input data (NOT stored — passed in by trainer)
  - x^(l)  for l = 1..L = hidden latents, stored in trainer's latent buffer
           dim of x^(l) = OUT_DIM of layer l-1 = IN_DIM of layer l
  - y_hat  = output of readout, dim = readout's IN_DIM (= num classes)

Composition constraint (enforced by Mojo type system at use sites):
  layer[i].OUT_DIM == layer[i+1].IN_DIM for i = 0..N-2.
"""

from layout import Layout, LayoutTensor
from mojo_rl.nn.initializer import Initializer

from ..predictive_model import PCLayer


@fieldwise_init
struct PCSequential[*LAYERS: PCLayer]:
    """Variadic stack of PCN levels."""

    comptime layer_types = Self.LAYERS
    comptime N_LINEARS: Int = Self.layer_types.size
    comptime N_LATENTS: Int = Self.N_LINEARS - 1  # paper's L

    # Architecture-level dims for the trainer to view.
    comptime IN_DIM: Int = Self.layer_types[0].IN_DIM
    comptime OUT_DIM: Int = Self.layer_types[Self.N_LINEARS - 1].IN_DIM
    comptime TOP_LATENT_DIM: Int = Self.layer_types[Self.N_LINEARS - 1].OUT_DIM

    # ───── Param-buffer layout ─────────────────────────────────────────────

    @staticmethod
    def _sum_param_size() -> Int:
        var total = 0
        comptime for i in range(Self.N_LINEARS):
            total += Self.layer_types[i].PARAM_SIZE
        return total

    comptime PARAM_SIZE: Int = Self._sum_param_size()

    @staticmethod
    def _param_offset[idx: Int]() -> Int:
        var total = 0
        comptime for j in range(idx):
            total += Self.layer_types[j].PARAM_SIZE
        return total

    # ───── Latent-buffer layout (per sample) ──────────────────────────────
    # Latents stored: x^(1), x^(2), ..., x^(L). x^(0)=input is passed in,
    # not stored. Latent at index i (0-indexed) corresponds to x^(i+1) and
    # has dim = OUT_DIM of layer i.

    @staticmethod
    def _sum_latent_size() -> Int:
        var total = 0
        comptime for i in range(Self.N_LATENTS):
            total += Self.layer_types[i].OUT_DIM
        return total

    comptime LATENT_SIZE_PER_SAMPLE: Int = Self._sum_latent_size()

    @staticmethod
    def _latent_offset[i: Int]() -> Int:
        """Offset (per sample) of latent at index i (= x^(i+1))."""
        var total = 0
        comptime for j in range(i):
            total += Self.layer_types[j].OUT_DIM
        return total

    # ───── Layer-scratch layout (per sample) ──────────────────────────────
    # The trainer needs four buffers each of size sum(IN_DIM_i) per sample:
    # `a`, `x_hat`, `eps`, `gm` — one slot per layer (incl. readout).

    @staticmethod
    def _sum_in_dim() -> Int:
        var total = 0
        comptime for i in range(Self.N_LINEARS):
            total += Self.layer_types[i].IN_DIM
        return total

    comptime LAYER_SCRATCH_PER_SAMPLE: Int = Self._sum_in_dim()

    @staticmethod
    def _layer_scratch_offset[idx: Int]() -> Int:
        """Offset (per sample) of layer idx's slot inside one layer-scratch buffer."""
        var total = 0
        comptime for j in range(idx):
            total += Self.layer_types[j].IN_DIM
        return total

    # ───── Initialization ─────────────────────────────────────────────────

    @staticmethod
    def initialize_params[
        INIT: Initializer, dtype: DType = DType.float32
    ](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Init each layer's W slice with INIT, using its own (in_dim, out_dim)."""
        comptime for i in range(Self.N_LINEARS):
            var li_p = LayoutTensor[
                dtype,
                Layout.row_major(Self.layer_types[i].PARAM_SIZE),
                MutAnyOrigin,
            ](params.ptr + Self._param_offset[i]())
            Self.layer_types[i].initialize_params[INIT, dtype](li_p)
