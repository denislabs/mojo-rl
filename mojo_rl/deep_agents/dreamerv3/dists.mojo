"""Continuous-policy distribution helpers — DreamerV3 `bounded_normal`.

Ports `embodied/jax/heads.py:Head.bounded_normal` + `outs.py:Normal`:

  mean = tanh(mean_raw)
  std  = (maxstd - minstd)·sigmoid(std_raw + 2) + minstd
  logp(a)   = −0.5·((a-mean)/std)² − log(std) − 0.5·log(2π)   (per dim)
  entropy() = 0.5·log(2π) + log(std) + 0.5                    (per dim)

The policy head wraps a shaped (ACT,) space in `Agg(., 1, sum)`, so the
imag-loss caller sums logp/entropy over the action dim. These helpers are
per-dim; the caller does the sum. Forward only (PR5a).
"""

from std.math import log, tanh, exp, pi

from mojo_rl.nn.constants import DT


@always_inline
def _sigmoid(x: Scalar[DT]) -> Scalar[DT]:
    return Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-x))


@always_inline
def bounded_mean(mean_raw: Scalar[DT]) -> Scalar[DT]:
    return tanh(mean_raw)


@always_inline
def bounded_std(
    std_raw: Scalar[DT], minstd: Scalar[DT], maxstd: Scalar[DT]
) -> Scalar[DT]:
    return (maxstd - minstd) * _sigmoid(std_raw + Scalar[DT](2.0)) + minstd


@always_inline
def normal_logp(
    event: Scalar[DT], mean: Scalar[DT], std: Scalar[DT]
) -> Scalar[DT]:
    comptime LOG2PI = Scalar[DT](1.8378770664093453)  # log(2π)
    var z = (event - mean) / std
    return -Scalar[DT](0.5) * z * z - log(std) - Scalar[DT](0.5) * LOG2PI


@always_inline
def normal_entropy(std: Scalar[DT]) -> Scalar[DT]:
    comptime LOG2PI = Scalar[DT](1.8378770664093453)
    return Scalar[DT](0.5) * LOG2PI + log(std) + Scalar[DT](0.5)
