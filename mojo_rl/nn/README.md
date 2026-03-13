# nn/ - Deep Learning Framework

Trait-based neural network framework with compile-time dimensions, automatic differentiation, and CPU/GPU support.

## Architecture

The framework follows a **stateless design**: models describe computation graphs without storing weights. All mutable state (params, grads, optimizer moments) is managed externally via `LayoutTensor` views for zero-copy composition.

## Module Structure

```
nn/
├── constants.mojo          # Global constants (dtype=float32, TILE=16, TPB=256)
├── composites.mojo         # Pre-built architectures (ResBlock, ResNet, LeNet, FFN)
├── model/                  # Neural network layers (Model trait)
│   ├── model.mojo          # Model trait definition
│   ├── linear.mojo         # Linear[in_dim, out_dim]: y = x @ W + b
│   ├── linear_relu.mojo    # LinearReLU[in_dim, out_dim]: fused Linear + ReLU
│   ├── linear_tanh.mojo    # LinearTanh[in_dim, out_dim]: fused Linear + Tanh
│   ├── relu.mojo           # ReLU[dim]
│   ├── tanh.mojo           # Tanh[dim]
│   ├── sigmoid.mojo        # Sigmoid[dim]
│   ├── softmax.mojo        # Softmax[dim] (numerically stable)
│   ├── mish.mojo           # Mish[dim]: x * tanh(softplus(x))
│   ├── layer_norm.mojo     # LayerNorm[dim, EPSILON]
│   ├── simnorm.mojo        # SimNorm[dim, simplex_dim] (simplicial normalization)
│   ├── dropout.mojo        # Dropout[dim, p, SEED, training]
│   ├── normed_linear.mojo  # NormedLinear: Linear -> LayerNorm -> Mish (TDMPC2 block)
│   ├── stochastic_actor.mojo # StochasticActor[in_dim, action_dim] (Gaussian policy)
│   └── sequential.mojo     # Sequential[*LAYERS]: variadic N-layer composition
├── optimizer/              # Optimizers (Optimizer trait)
│   ├── optimizer.mojo      # Optimizer trait definition
│   ├── sgd.mojo            # SGD[LR]
│   ├── adam.mojo           # Adam[LR, BETA1, BETA2, EPS]
│   ├── adamw.mojo          # AdamW[LR, BETA1, BETA2, EPS, WEIGHT_DECAY]
│   ├── rmsprop.mojo        # RMSprop[LR, ALPHA, EPS]
│   └── muon.mojo           # Muon optimizer
├── loss/                   # Loss functions (LossFunction trait)
│   ├── loss.mojo           # LossFunction trait definition
│   ├── mse.mojo            # MSELoss
│   ├── huber.mojo          # HuberLoss[delta] (robust to outliers)
│   ├── cross_entropy.mojo  # CrossEntropyLoss
│   ├── soft_cross_entropy.mojo # SoftCrossEntropyLoss (for probability distributions)
│   └── two_hot.mojo        # Two-hot encoding for C51 distributional RL
├── initializer/            # Weight initializers (Initializer trait)
│   └── initializers.mojo   # Xavier, Kaiming, LeCun, Zeros, Ones, Constant, Uniform, Normal
├── training/               # Training infrastructure
│   ├── trainer.mojo        # Trainer[MODEL, OPT, LOSS]: CPU/GPU training loops
│   ├── network.mojo        # Network[MODEL, OPT]: stateless forward/backward
│   ├── network_state.mojo  # NetworkState: CPU params/grads/optimizer state
│   ├── gpu_network_state.mojo # GPUNetworkState: device memory management
│   └── network_pair.mojo   # NetworkPair / GPUNetworkPair: (online, target) pairs
├── checkpoint/             # Model serialization
│   ├── checkpoint.mojo     # Text-based checkpoint I/O
│   └── binary_checkpoint.mojo # Binary checkpoint format
├── autodiff/               # Automatic differentiation framework
│   ├── op.mojo             # DiffOp trait + OpID enum
│   ├── chain.mojo          # AutoDiffChain[*OPS]: variadic composition
│   ├── fusion.mojo         # FusionAnalyzer: compile-time pattern detection
│   ├── auto_fused.mojo     # AutoFused[*OPS]: greedy compile-time fusion
│   ├── primitives/         # Fine-grained DiffOp implementations
│   │   ├── matmul.mojo     # MatMul[in_dim, out_dim]
│   │   ├── bias.mojo       # BiasAdd[dim]
│   │   ├── activations.mojo # ReLUOp, TanhOp, SigmoidOp, MishOp
│   │   ├── softmax.mojo    # SoftmaxOp[dim]
│   │   ├── layer_norm.mojo # LayerNormOp[dim]
│   │   ├── rms_norm.mojo   # RMSNormOp[dim]
│   │   ├── scale.mojo      # Scale[dim, scalar]
│   │   ├── elem_mul.mojo   # ElemMul[dim]
│   │   ├── reduce.mojo     # ReduceSum, ReduceMean
│   │   ├── dropout.mojo    # DropoutOp[dim, p, SEED]
│   │   ├── reshape.mojo    # Flatten[in_h, in_w, in_c]
│   │   ├── embedding.mojo  # Embedding[vocab_size, embed_dim]
│   │   ├── conv2d.mojo     # Conv2D
│   │   ├── pool.mojo       # MaxPool2D, AvgPool2D
│   │   └── attention.mojo  # ScaledDotProductAttention
│   ├── fused/              # Pre-computed fused kernels
│   │   ├── matmul_bias.mojo      # FusedMatMulBias
│   │   ├── matmul_bias_relu.mojo # FusedMatMulBiasReLU
│   │   ├── matmul_bias_tanh.mojo # FusedMatMulBiasTanh
│   │   ├── matmul_bias_act.mojo  # FusedMatMulBiasActivation[ACT]
│   │   └── activation.mojo       # Activation trait + implementations
│   └── combinators/        # Structural composition patterns
│       ├── residual.mojo   # Residual[Inner]: skip connection y = Inner(x) + x
│       ├── parallel.mojo   # Parallel[*BRANCHES]: multi-branch concat
│       └── repeat.mojo     # Repeat[n, Inner]: weight-shared repetition
├── replay/                 # Experience replay buffers
│   └── replay_buffer.mojo  # ReplayBuffer, PrioritizedReplayBuffer
├── gpu/                    # GPU utilities
│   ├── elementwise.mojo    # gpu_add, gpu_mul, gpu_relu, gpu_tanh, gpu_sigmoid
│   ├── matmul.mojo         # Tiled matmul + MMA tensor core dispatch
│   ├── matmul_ops.mojo     # Fused matmul+bias+activation kernels
│   ├── matmul_v2.mojo      # Alternative matmul implementations
│   ├── matmul_apple.mojo   # Apple Silicon Metal-specific optimizations
│   └── random.mojo         # Gaussian noise (Box-Muller)
└── tests/                  # Test suite (9 files covering all autodiff phases)
```

## Key Traits

| Trait | Purpose | Key Comptime Constants |
|-------|---------|----------------------|
| **Model** | Stateless neural network layer | `IN_DIM`, `OUT_DIM`, `PARAM_SIZE`, `CACHE_SIZE` |
| **Optimizer** | Parameter update rule | `STATE_PER_PARAM` (1 for SGD, 2 for Adam) |
| **LossFunction** | Loss computation | `forward()`, `backward()` |
| **Initializer** | Weight initialization | `init[SIZE, FAN_IN, FAN_OUT]()` |
| **DiffOp** | Differentiable operation (autodiff) | `OP_ID`, `IN_DIM`, `OUT_DIM`, `PARAM_SIZE`, `CACHE_SIZE` |

## Autodiff Convenience Aliases

```mojo
comptime Dense[i, o] = AutoDiffChain[MatMul[i, o], BiasAdd[o]]
comptime DenseReLU[i, o] = AutoDiffChain[MatMul[i, o], BiasAdd[o], ReLUOp[o]]
comptime DenseTanh[i, o] = AutoDiffChain[MatMul[i, o], BiasAdd[o], TanhOp[o]]
```

## Usage

```mojo
from mojo_rl.nn import Sequential, Linear, ReLU, Adam, MSELoss, Kaiming, Trainer

# Define model at compile time: 4 -> 64 (ReLU) -> 64 (ReLU) -> 2
comptime MLP = Sequential[Linear[4, 64], ReLU[64], Linear[64, 64], ReLU[64], Linear[64, 2]]

var trainer = Trainer[MLP, Adam, MSELoss, Kaiming](
    MLP(), Adam(lr=0.001), MSELoss(), Kaiming(), epochs=100,
)
var result = trainer.train[BATCH](input_data, target_data)
```
