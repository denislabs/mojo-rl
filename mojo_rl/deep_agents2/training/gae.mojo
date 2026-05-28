"""Generalized Advantage Estimation (GAE) helper functions.

Walks backward through a rollout, computes:

    delta_t      = r_t + γ · V(s_{t+1}) · (1 − terminated_t) − V(s_t)
    advantage_t  = delta_t + γ · λ · (1 − terminated_t) · advantage_{t+1}
    return_t     = advantage_t + V(s_t)

`terminated` distinguishes a *real* episode terminal (V(s_{t+1}) = 0) from
a *truncation* (V(s_{t+1}) bootstrapped). This is the Gymnasium
`terminated` vs `truncated` semantic — a critical fix vs the naive
`done = terminated OR truncated` which crashes on Pendulum (no real
terminal, just time-limit at step 200). See
`feedback_ppo_pendulum_timelimit_gae` for the diagnostic story.

For envs with no real terminals (Pendulum, etc.), pass an all-zero
`terminated` buffer — `compute_gae` will then bootstrap every step.

Inputs (caller-allocated, length N_STEPS unless noted):
    rewards          [N_STEPS]
    values           [N_STEPS]      V(s_t) — actor-predicted at rollout time
    terminated       [N_STEPS]      1.0 if real terminal at t, else 0.0
    next_value       scalar         V(s_{N_STEPS}) — bootstrap at rollout end

Outputs (caller-allocated, length N_STEPS):
    advantages       [N_STEPS]      GAE advantage per step (NOT normalized;
                                    caller does mean/std normalization per
                                    minibatch in production PPO)
    returns          [N_STEPS]      advantage + value (target for V regression)

The `compute_gae` function is free, not a struct — there's no
amortizable state, and rollout buffers are caller-owned (PPO needs to
shuffle indices over them, etc).

`normalize_in_place` is the standard mean/std advantage normalizer,
provided as a separate helper.
"""

from std.math import sqrt as fsqrt

from mojo_rl.nn2.constants import DT


def compute_gae(
    n_steps: Int,
    rewards: UnsafePointer[Scalar[DT], MutAnyOrigin],
    values: UnsafePointer[Scalar[DT], MutAnyOrigin],
    terminated: UnsafePointer[Scalar[DT], MutAnyOrigin],
    next_value: Scalar[DT],
    gamma: Scalar[DT],
    gae_lambda: Scalar[DT],
    advantages: UnsafePointer[Scalar[DT], MutAnyOrigin],
    returns: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    """Standard GAE backward pass over a contiguous rollout.

    `n_steps` is a *runtime* arg, not comptime — at e.g. ROLLOUT_LEN=2048
    Mojo nightly unrolls the comptime-templated form fully, exploding
    compile time (multi-hour). Runtime loop bound keeps the loop body as
    a single function instantiation.
    """
    var last_gae: Scalar[DT] = 0.0
    for t in range(n_steps - 1, -1, -1):
        var nonterm = Scalar[DT](1.0) - terminated[t]
        var nv: Scalar[DT]
        if t == n_steps - 1:
            nv = next_value
        else:
            nv = values[t + 1]
        var delta = rewards[t] + gamma * nv * nonterm - values[t]
        last_gae = delta + gamma * gae_lambda * nonterm * last_gae
        advantages[t] = last_gae
        returns[t] = last_gae + values[t]


def normalize_in_place(
    n: Int,
    buf: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    """Subtract mean, divide by std + 1e-8. In-place on `buf[0:n]`.

    `n` is runtime (see compute_gae comment for the comptime-unroll trap)."""
    var s: Scalar[DT] = 0.0
    for t in range(n):
        s = s + buf[t]
    var mean = s / Scalar[DT](n)
    var sq: Scalar[DT] = 0.0
    for t in range(n):
        var d = buf[t] - mean
        sq = sq + d * d
    var std = fsqrt(sq / Scalar[DT](n))
    for t in range(n):
        buf[t] = (buf[t] - mean) / (std + Scalar[DT](1e-8))
