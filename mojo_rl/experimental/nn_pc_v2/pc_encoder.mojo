"""PCEncoder — 2-layer MLP amortized posterior for hybrid PC (Tschantz 2023).

Encoder maps `[prev_z, action, obs]` → initial guess for the latent `z_t`.
Pairs with `PCTrainer.compute_grads_from_latents`: encoder produces the warm
start, K SGLD refinement steps then settle to the local energy minimum.

    z_pre  = W1 · input + b1
    z_hid  = tanh(z_pre)
    output = W2 · z_hid + b2

Matches the hand-rolled encoder used in the world-model amortized PC tests
(`test_pendulum_amortized_pc.mojo`, `test_mountain_car_amortized_pc.mojo`).
Hidden activation is tanh (no parametrization yet — extend if a real use case
needs ReLU/GELU).

Layout (single flat param tensor):

    [ W1 (IN×HID) | b1 (HID) | W2 (HID×OUT) | b2 (OUT) ]

CPU only for now. The PC inference loop is the GPU bottleneck; encoder
forward/backward is small (a few hundred params for typical world-model
sizes) and runs on host without dominating wall time.
"""

from layout import Layout, LayoutTensor
from std.math import sqrt, tanh
from std.random.philox import Random as PhiloxRandom


struct PCEncoder[in_dim: Int, hidden_dim: Int, out_dim: Int]:
    """2-layer MLP encoder with tanh hidden activation.

    Caller owns all buffers (params, scratch, output) so they can be allocated
    once and reused across many batches.
    """

    comptime IN_DIM: Int = Self.in_dim
    comptime HIDDEN_DIM: Int = Self.hidden_dim
    comptime OUT_DIM: Int = Self.out_dim

    comptime W1_SIZE: Int = Self.in_dim * Self.hidden_dim
    comptime B1_SIZE: Int = Self.hidden_dim
    comptime W2_SIZE: Int = Self.hidden_dim * Self.out_dim
    comptime B2_SIZE: Int = Self.out_dim

    comptime W1_OFFSET: Int = 0
    comptime B1_OFFSET: Int = Self.W1_SIZE
    comptime W2_OFFSET: Int = Self.W1_SIZE + Self.B1_SIZE
    comptime B2_OFFSET: Int = Self.W1_SIZE + Self.B1_SIZE + Self.W2_SIZE

    comptime PARAM_SIZE: Int = (
        Self.W1_SIZE + Self.B1_SIZE + Self.W2_SIZE + Self.B2_SIZE
    )

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    # =========================================================================
    # Initialization — Xavier-uniform on W1, W2; zeros on biases
    # =========================================================================

    @staticmethod
    def xavier_init[
        dtype: DType = DType.float32
    ](
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        seed: UInt64,
    ):
        """Xavier-uniform init for W1, W2; zero biases.

        W1 uses `seed`, W2 uses `seed + 1` (independent RNG streams).
        """
        # W1: bound = sqrt(6 / (fan_in + fan_out))
        var rng1 = PhiloxRandom(seed=seed, offset=UInt64(0))
        var bound1 = sqrt(
            Float64(6.0) / Float64(Self.in_dim + Self.hidden_dim)
        )
        for i in range(Self.W1_SIZE):
            var u = Float64(rng1.step_uniform()[0])
            params.ptr[Self.W1_OFFSET + i] = Scalar[dtype](
                (u * 2.0 - 1.0) * bound1
            )
        # b1 = 0
        for i in range(Self.B1_SIZE):
            params.ptr[Self.B1_OFFSET + i] = Scalar[dtype](0.0)
        # W2: bound = sqrt(6 / (hidden_dim + out_dim))
        var rng2 = PhiloxRandom(seed=seed + UInt64(1), offset=UInt64(0))
        var bound2 = sqrt(
            Float64(6.0) / Float64(Self.hidden_dim + Self.out_dim)
        )
        for i in range(Self.W2_SIZE):
            var u = Float64(rng2.step_uniform()[0])
            params.ptr[Self.W2_OFFSET + i] = Scalar[dtype](
                (u * 2.0 - 1.0) * bound2
            )
        # b2 = 0
        for i in range(Self.B2_SIZE):
            params.ptr[Self.B2_OFFSET + i] = Scalar[dtype](0.0)

    # =========================================================================
    # Forward:  output = W2 · tanh(W1 · input + b1) + b2
    # =========================================================================

    @staticmethod
    def forward[
        BATCH: Int, dtype: DType = DType.float32
    ](
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        enc_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        mut hidden_pre: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN_DIM), MutAnyOrigin
        ],
        mut hidden_act: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN_DIM), MutAnyOrigin
        ],
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
    ):
        """Forward pass; caches `hidden_pre`, `hidden_act` for backward."""
        for b in range(BATCH):
            # h_pre = W1·input + b1 ;  h_act = tanh(h_pre)
            for j in range(Self.hidden_dim):
                var s = Float64(params.ptr[Self.B1_OFFSET + j])
                for i in range(Self.in_dim):
                    s += Float64(enc_input.ptr[b * Self.in_dim + i]) * Float64(
                        params.ptr[
                            Self.W1_OFFSET + i * Self.hidden_dim + j
                        ]
                    )
                hidden_pre.ptr[b * Self.hidden_dim + j] = Scalar[dtype](s)
                hidden_act.ptr[b * Self.hidden_dim + j] = Scalar[dtype](
                    tanh(s)
                )
            # output = W2·h_act + b2
            for j in range(Self.out_dim):
                var s = Float64(params.ptr[Self.B2_OFFSET + j])
                for i in range(Self.hidden_dim):
                    s += Float64(
                        hidden_act.ptr[b * Self.hidden_dim + i]
                    ) * Float64(
                        params.ptr[
                            Self.W2_OFFSET + i * Self.out_dim + j
                        ]
                    )
                output.ptr[b * Self.out_dim + j] = Scalar[dtype](s)

    # =========================================================================
    # Backward:  given dL/d(output), accumulate into grads (zero-initialized).
    # Standard MLP chain rule.
    # =========================================================================

    @staticmethod
    def backward[
        BATCH: Int, dtype: DType = DType.float32
    ](
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        enc_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        hidden_act: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN_DIM), MutAnyOrigin
        ],
        dz_out: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Writes gradients into `grads` (overwrites; caller need not zero)."""
        for i in range(Self.PARAM_SIZE):
            grads.ptr[i] = Scalar[dtype](0.0)

        # Per-batch dh buffers — small, allocate as Lists.
        var dh_act_list = List[Float64](capacity=Self.hidden_dim)
        var dh_pre_list = List[Float64](capacity=Self.hidden_dim)
        for _ in range(Self.hidden_dim):
            dh_act_list.append(0.0)
            dh_pre_list.append(0.0)

        for b in range(BATCH):
            # Reset dh_act
            for j in range(Self.hidden_dim):
                dh_act_list[j] = 0.0

            # dW2[i, j] += h_act[b, i] * dz[b, j]; db2[j] += dz[b, j]
            # dh_act[i] += W2[i, j] * dz[b, j]
            for j in range(Self.out_dim):
                var dz_bj = Float64(dz_out.ptr[b * Self.out_dim + j])
                grads.ptr[Self.B2_OFFSET + j] = Scalar[dtype](
                    Float64(grads.ptr[Self.B2_OFFSET + j]) + dz_bj
                )
                for i in range(Self.hidden_dim):
                    var idx = Self.W2_OFFSET + i * Self.out_dim + j
                    grads.ptr[idx] = Scalar[dtype](
                        Float64(grads.ptr[idx])
                        + Float64(hidden_act.ptr[b * Self.hidden_dim + i])
                        * dz_bj
                    )
                    dh_act_list[i] += (
                        Float64(params.ptr[idx]) * dz_bj
                    )

            # dh_pre = dh_act * (1 - tanh²(h_pre)) = dh_act * (1 - h_act²)
            for j in range(Self.hidden_dim):
                var ha = Float64(hidden_act.ptr[b * Self.hidden_dim + j])
                dh_pre_list[j] = dh_act_list[j] * (1.0 - ha * ha)

            # dW1[i, j] += input[b, i] * dh_pre[j]; db1[j] += dh_pre[j]
            for j in range(Self.hidden_dim):
                grads.ptr[Self.B1_OFFSET + j] = Scalar[dtype](
                    Float64(grads.ptr[Self.B1_OFFSET + j]) + dh_pre_list[j]
                )
                for i in range(Self.in_dim):
                    var idx = Self.W1_OFFSET + i * Self.hidden_dim + j
                    grads.ptr[idx] = Scalar[dtype](
                        Float64(grads.ptr[idx])
                        + Float64(enc_input.ptr[b * Self.in_dim + i])
                        * dh_pre_list[j]
                    )
