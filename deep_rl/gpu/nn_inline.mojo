"""Inline neural network operations for use inside GPU kernels.

These are @always_inline functions designed to be called within fused GPU kernels
without any kernel launch overhead. They operate on scalars and InlineArrays.

This is different from elementwise.mojo which provides kernel-based operations
that launch separate GPU kernels for tensor operations.

Patterns inspired by Modular's MAX nn/activations.mojo:
- Use @always_inline for zero overhead
- Use .select() for branchless operations where possible
- Numerically stable implementations (e.g., softmax subtracts max)

Usage inside a GPU kernel:
    from deep_rl.gpu import relu_inline, softmax2_inline

    fn my_kernel[...](...):
        # Scalar ReLU
        var h = relu_inline(pre_activation)

        # Softmax for 2 classes
        var probs = softmax2_inline(logit0, logit1)
        var prob0 = probs[0]
        var prob1 = probs[1]
"""

from math import exp, log, tanh, sqrt

# =============================================================================
# Constants
# =============================================================================

comptime dtype = DType.float32


# =============================================================================
# Scalar Activation Functions
# =============================================================================


@always_inline
fn relu_inline(x: Scalar[dtype]) -> Scalar[dtype]:
    """ReLU activation: max(x, 0).

    Args:
        x: Input scalar.

    Returns:
        max(x, 0).
    """
    return x if x > Scalar[dtype](0) else Scalar[dtype](0)


@always_inline
fn relu_inline_generic[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    """Generic ReLU activation for any dtype.

    Args:
        x: Input scalar.

    Returns:
        max(x, 0).
    """
    return x if x > Scalar[dt](0) else Scalar[dt](0)


@always_inline
fn leaky_relu_inline(
    x: Scalar[dtype], negative_slope: Scalar[dtype] = 0.01
) -> Scalar[dtype]:
    """Leaky ReLU activation.

    Args:
        x: Input scalar.
        negative_slope: Slope for negative values (default 0.01).

    Returns:
        x if x > 0, else negative_slope * x.
    """
    return x if x > Scalar[dtype](0) else negative_slope * x


@always_inline
fn tanh_inline(x: Scalar[dtype]) -> Scalar[dtype]:
    """Tanh activation.

    Args:
        x: Input scalar.

    Returns:
        tanh(x).
    """
    return tanh(x)


@always_inline
fn sigmoid_inline(x: Scalar[dtype]) -> Scalar[dtype]:
    """Sigmoid activation: 1 / (1 + exp(-x)).

    Args:
        x: Input scalar.

    Returns:
        sigmoid(x).
    """
    return Scalar[dtype](1) / (Scalar[dtype](1) + exp(-x))


# =============================================================================
# Softmax Functions (Numerically Stable)
# =============================================================================


@always_inline
fn softmax2_inline(
    logit0: Scalar[dtype], logit1: Scalar[dtype]
) -> InlineArray[Scalar[dtype], 2]:
    """Numerically stable softmax for 2 classes.

    Subtracts max logit before exponentiating to prevent overflow.

    Args:
        logit0: Logit for class 0.
        logit1: Logit for class 1.

    Returns:
        InlineArray with [prob0, prob1].
    """
    var max_logit = logit0 if logit0 > logit1 else logit1
    var exp0 = exp(logit0 - max_logit)
    var exp1 = exp(logit1 - max_logit)
    var sum_exp = exp0 + exp1

    var result = InlineArray[Scalar[dtype], 2](fill=Scalar[dtype](0))
    result[0] = exp0 / sum_exp
    result[1] = exp1 / sum_exp
    return result


@always_inline
fn softmax3_inline(
    logit0: Scalar[dtype], logit1: Scalar[dtype], logit2: Scalar[dtype]
) -> InlineArray[Scalar[dtype], 3]:
    """Numerically stable softmax for 3 classes.

    Subtracts max logit before exponentiating to prevent overflow.

    Args:
        logit0: Logit for class 0.
        logit1: Logit for class 1.
        logit2: Logit for class 2.

    Returns:
        InlineArray with [prob0, prob1, prob2].
    """
    var max_logit = logit0
    if logit1 > max_logit:
        max_logit = logit1
    if logit2 > max_logit:
        max_logit = logit2

    var exp0 = exp(logit0 - max_logit)
    var exp1 = exp(logit1 - max_logit)
    var exp2 = exp(logit2 - max_logit)
    var sum_exp = exp0 + exp1 + exp2

    var result = InlineArray[Scalar[dtype], 3](fill=Scalar[dtype](0))
    result[0] = exp0 / sum_exp
    result[1] = exp1 / sum_exp
    result[2] = exp2 / sum_exp
    return result


@always_inline
fn softmax_inline[
    NUM_CLASSES: Int
](logits: InlineArray[Scalar[dtype], NUM_CLASSES]) -> InlineArray[
    Scalar[dtype], NUM_CLASSES
]:
    """Numerically stable softmax for N classes.

    Subtracts max logit before exponentiating to prevent overflow.

    Args:
        logits: InlineArray of logits.

    Returns:
        InlineArray of probabilities.
    """
    # Find max
    var max_logit = logits[0]
    for i in range(1, NUM_CLASSES):
        if logits[i] > max_logit:
            max_logit = logits[i]

    # Compute exp and sum
    var exps = InlineArray[Scalar[dtype], NUM_CLASSES](fill=Scalar[dtype](0))
    var sum_exp: Scalar[dtype] = 0
    for i in range(NUM_CLASSES):
        exps[i] = exp(logits[i] - max_logit)
        sum_exp += exps[i]

    # Normalize
    var result = InlineArray[Scalar[dtype], NUM_CLASSES](fill=Scalar[dtype](0))
    for i in range(NUM_CLASSES):
        result[i] = exps[i] / sum_exp
    return result


# =============================================================================
# Log Softmax Functions (Numerically Stable)
# =============================================================================


@always_inline
fn log_softmax2_inline(
    logit0: Scalar[dtype], logit1: Scalar[dtype]
) -> InlineArray[Scalar[dtype], 2]:
    """Numerically stable log softmax for 2 classes.

    Uses log-sum-exp trick: log(softmax(x_i)) = x_i - max(x) - log(sum(exp(x - max(x))))

    Args:
        logit0: Logit for class 0.
        logit1: Logit for class 1.

    Returns:
        InlineArray with [log_prob0, log_prob1].
    """
    var max_logit = logit0 if logit0 > logit1 else logit1
    var exp0 = exp(logit0 - max_logit)
    var exp1 = exp(logit1 - max_logit)
    var log_sum_exp = log(exp0 + exp1)

    var result = InlineArray[Scalar[dtype], 2](fill=Scalar[dtype](0))
    result[0] = logit0 - max_logit - log_sum_exp
    result[1] = logit1 - max_logit - log_sum_exp
    return result


@always_inline
fn log_softmax_inline[
    NUM_CLASSES: Int
](logits: InlineArray[Scalar[dtype], NUM_CLASSES]) -> InlineArray[
    Scalar[dtype], NUM_CLASSES
]:
    """Numerically stable log softmax for N classes.

    Uses log-sum-exp trick: log(softmax(x_i)) = x_i - max(x) - log(sum(exp(x - max(x))))

    Args:
        logits: InlineArray of logits.

    Returns:
        InlineArray of log probabilities.
    """
    # Find max
    var max_logit = logits[0]
    for i in range(1, NUM_CLASSES):
        if logits[i] > max_logit:
            max_logit = logits[i]

    # Compute log-sum-exp
    var sum_exp: Scalar[dtype] = 0
    for i in range(NUM_CLASSES):
        sum_exp += exp(logits[i] - max_logit)
    var log_sum_exp = log(sum_exp)

    # Compute log probabilities
    var result = InlineArray[Scalar[dtype], NUM_CLASSES](fill=Scalar[dtype](0))
    for i in range(NUM_CLASSES):
        result[i] = logits[i] - max_logit - log_sum_exp
    return result


# =============================================================================
# Sampling Utilities
# =============================================================================


@always_inline
fn sample_from_probs2(
    prob0: Scalar[dtype], uniform_sample: Scalar[dtype]
) -> Int:
    """Sample action from 2-class probability distribution.

    Args:
        prob0: Probability of class 0.
        uniform_sample: Uniform random sample in [0, 1).

    Returns:
        0 if uniform_sample < prob0, else 1.
    """
    return 0 if uniform_sample < prob0 else 1


@always_inline
fn sample_from_probs[
    NUM_CLASSES: Int
](probs: InlineArray[Scalar[dtype], NUM_CLASSES], uniform_sample: Scalar[dtype]) -> Int:
    """Sample action from N-class probability distribution.

    Args:
        probs: Probability distribution (must sum to 1).
        uniform_sample: Uniform random sample in [0, 1).

    Returns:
        Sampled class index.
    """
    var cumsum: Scalar[dtype] = 0
    for i in range(NUM_CLASSES - 1):
        cumsum += probs[i]
        if uniform_sample < cumsum:
            return i
    return NUM_CLASSES - 1


# =============================================================================
# Gradient Helpers (for backprop inside kernels)
# =============================================================================


@always_inline
fn relu_grad_inline(x: Scalar[dtype]) -> Scalar[dtype]:
    """ReLU gradient: 1 if x > 0, else 0.

    Args:
        x: Pre-activation value.

    Returns:
        1 if x > 0, else 0.
    """
    return Scalar[dtype](1) if x > Scalar[dtype](0) else Scalar[dtype](0)


@always_inline
fn tanh_grad_inline(y: Scalar[dtype]) -> Scalar[dtype]:
    """Tanh gradient given output: 1 - y^2.

    Args:
        y: Tanh output (not input).

    Returns:
        1 - y^2.
    """
    return Scalar[dtype](1) - y * y


@always_inline
fn sigmoid_grad_inline(y: Scalar[dtype]) -> Scalar[dtype]:
    """Sigmoid gradient given output: y * (1 - y).

    Args:
        y: Sigmoid output (not input).

    Returns:
        y * (1 - y).
    """
    return y * (Scalar[dtype](1) - y)
