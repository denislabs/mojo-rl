"""Deep RL neural network package.

Provides neural network building blocks using compile-time dimensions
with InlineArray for maximum performance.

Modules:
    tensor: Tensor operations (matmul, activations, etc.)
    linear: Linear (fully connected) layer
    mlp: Multi-layer perceptron architectures
    adam: Adam optimizer
    actor_critic: Actor and Critic networks for continuous control
"""

from .tensor import (
    # Matrix operations
    matmul,
    matmul_add_bias,
    transpose,
    # Activations
    relu,
    tanh_activation,
    sigmoid,
    # Activation gradients
    relu_grad,
    tanh_grad,
    sigmoid_grad,
    # Element-wise operations
    elementwise_mul,
    elementwise_sub,
    scale,
    # Initialization
    zeros,
    xavier_init,
    random_uniform,
    # Reductions
    sum_all,
    mean_all,
    sum_axis0,
    # Utility
    print_matrix,
    copy_array,
)

from .linear import Linear

from .mlp import MLP2, MLP3

from .adam import AdamState, LinearAdam

from .actor_critic import Actor, Critic, StochasticActor

from .replay_buffer import ReplayBuffer
