"""ScalarAdam — single-scalar Adam optimizer.

Phase 9B. Bespoke (NOT `Optimizer`-conforming, since the `Optimizer` trait
walks Module-tree params via `for_each_param`). Used for SAC's auto-tuned
entropy temperature `log_alpha`: a single trainable scalar with its own
moments and bias-correction.

Hyperparameters mirror Adam (β₁=0.9, β₂=0.999, ε=1e-8).

CPU path (`step`): pure host scalars; bias correction was originally an
O(t) `for _ in range(t-1)` β^t loop, now an incremental running product
`beta*_pow_t` (same left-to-right fp32 multiply sequence → bit-identical,
O(1) per step). The host `step` is the SAC CPU bit-identity path
(−169.04118) and must stay byte-stable.

GPU path (Slice 4b — CUDA-graph capture): the scalar state lives in a
device buffer `state_dev` ([log_alpha, m, v, β₁ᵗ, β₂ᵗ, α]); `step_device`
enqueues a 1-thread kernel that reads the device-resident grad (SAC's
reduced `lp_mean`), runs the same Adam math, and writes `α = exp(log_α)`
back into `state_dev[ALPHA]`. That α slot is what the actor-loss and
target-y `Scale` nodes read via `multiplier_ptr` (no host scalar baked
per step → capturable). The host fields are NOT updated on the GPU path;
`read_alpha` D2Hs the live α at flush cadence for logging.
"""

from std.math import sqrt as fsqrt, exp as fexp
from std.gpu import thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer

from ..constants import DT


# state_dev layout — single [6] device buffer.
comptime _SA_LOG_ALPHA = 0
comptime _SA_M = 1
comptime _SA_V = 2
comptime _SA_B1POW = 3
comptime _SA_B2POW = 4
comptime _SA_ALPHA = 5
comptime _SA_STATE_N = 6


def _scalar_adam_step_kernel(
    state: UnsafePointer[Scalar[DT], MutAnyOrigin],
    lp_mean: UnsafePointer[Scalar[DT], MutAnyOrigin],
    target_entropy: Scalar[DT],
    lr: Scalar[DT],
    beta1: Scalar[DT],
    beta2: Scalar[DT],
    eps: Scalar[DT],
):
    """1-thread on-device ScalarAdam step (launch grid=1, block=1).

    Reads the entropy grad on-device — `grad = -(lp_mean[0] + H_target)`
    — exactly the host `AlphaUpdateStep` expression, then runs the same
    incremental-bias-correction Adam update on `state` and writes
    `state[ALPHA] = exp(state[LOG_ALPHA])`. lr/β/ε/H_target are constants
    (capture-safe baked scalar args — they never change between replays)."""
    if Int(thread_idx.x) != 0:
        return
    var one: Scalar[DT] = 1.0
    var grad = -(lp_mean[0] + target_entropy)
    var m = beta1 * state[_SA_M] + (one - beta1) * grad
    var v = beta2 * state[_SA_V] + (one - beta2) * grad * grad
    # β₁ᵗ / β₂ᵗ as running products — same fp32 sequence the host loop
    # produced (β₁·β₁·…·β₁, t times), so the device update tracks the host
    # math to ULP modulo the lp_mean reduction-order delta.
    var b1pow = state[_SA_B1POW] * beta1
    var b2pow = state[_SA_B2POW] * beta2
    var m_hat = m / (one - b1pow)
    var v_hat = v / (one - b2pow)
    var la = state[_SA_LOG_ALPHA] - lr * m_hat / (fsqrt(v_hat) + eps)
    state[_SA_M] = m
    state[_SA_V] = v
    state[_SA_B1POW] = b1pow
    state[_SA_B2POW] = b2pow
    state[_SA_LOG_ALPHA] = la
    state[_SA_ALPHA] = fexp(la)


struct ScalarAdam(Movable & ImplicitlyDestructible):
    var value: Scalar[DT]
    var m: Scalar[DT]
    var v: Scalar[DT]
    var t: Int
    var lr: Scalar[DT]
    var beta1: Scalar[DT]
    var beta2: Scalar[DT]
    var eps: Scalar[DT]
    # Incremental bias-correction products (β₁ᵗ, β₂ᵗ). Maintained on the
    # CPU `step` path in lock-step with `t` so bias correction is O(1).
    var beta1_pow_t: Scalar[DT]
    var beta2_pow_t: Scalar[DT]

    # Slice 4b — device-resident state for CUDA-graph capture. None on CPU.
    var state_dev: Optional[DeviceBuffer[DT]]
    var _ctx: Optional[DeviceContext]

    def __init__(
        out self,
        value: Scalar[DT],
        m: Scalar[DT],
        v: Scalar[DT],
        t: Int,
        lr: Scalar[DT],
        beta1: Scalar[DT],
        beta2: Scalar[DT],
        eps: Scalar[DT],
    ):
        self.value = value
        self.m = m
        self.v = v
        self.t = t
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.beta1_pow_t = Scalar[DT](1.0)
        self.beta2_pow_t = Scalar[DT](1.0)
        self.state_dev = None
        self._ctx = None

    @staticmethod
    def new(initial: Scalar[DT], lr: Scalar[DT]) -> Self:
        return Self(
            value=initial, m=0.0, v=0.0, t=0,
            lr=lr, beta1=0.9, beta2=0.999, eps=1e-8,
        )

    @staticmethod
    def new_device(
        ctx: DeviceContext, initial: Scalar[DT], lr: Scalar[DT],
    ) raises -> Self:
        """GPU factory — allocates `state_dev` = [log_α, m, v, β₁ᵗ, β₂ᵗ, α]
        initialised to [initial, 0, 0, 1, 1, exp(initial)]. The host fields
        carry the same seed so a host-side `step` (used by no one on GPU)
        would still be consistent; the device buffer is the source of
        truth on GPU."""
        var s = Self.new(initial, lr)
        var buf = ctx.enqueue_create_buffer[DT](_SA_STATE_N)
        var host = ctx.enqueue_create_host_buffer[DT](_SA_STATE_N)
        ctx.synchronize()
        var hp = host.unsafe_ptr()
        hp[_SA_LOG_ALPHA] = initial
        hp[_SA_M] = Scalar[DT](0.0)
        hp[_SA_V] = Scalar[DT](0.0)
        hp[_SA_B1POW] = Scalar[DT](1.0)
        hp[_SA_B2POW] = Scalar[DT](1.0)
        hp[_SA_ALPHA] = fexp(initial)
        ctx.enqueue_copy(buf, host)
        s.state_dev = buf^
        s._ctx = ctx
        return s^

    def step(mut self, grad: Scalar[DT]):
        """CPU step — host scalars only. Incremental β^t (bit-identical to
        the original O(t) loop). This is the SAC CPU bit-identity path."""
        self.t += 1
        var one: Scalar[DT] = 1.0
        self.m = self.beta1 * self.m + (one - self.beta1) * grad
        self.v = self.beta2 * self.v + (one - self.beta2) * grad * grad
        self.beta1_pow_t *= self.beta1
        self.beta2_pow_t *= self.beta2
        var m_hat = self.m / (one - self.beta1_pow_t)
        var v_hat = self.v / (one - self.beta2_pow_t)
        self.value = self.value - self.lr * m_hat / (fsqrt(v_hat) + self.eps)

    def step_device(
        mut self,
        ctx: DeviceContext,
        lp_mean_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        target_entropy: Scalar[DT],
    ) raises:
        """GPU step — enqueues the 1-thread kernel reading the device
        `lp_mean` grad. No D2H, no host scalar update; `state_dev[ALPHA]`
        (the `Scale` nodes' multiplier source) is refreshed in place."""
        comptime kernel = _scalar_adam_step_kernel
        ctx.enqueue_function[kernel](
            self.state_dev.value().unsafe_ptr(),
            lp_mean_ptr,
            target_entropy,
            self.lr,
            self.beta1,
            self.beta2,
            self.eps,
            grid_dim=1,
            block_dim=1,
        )

    def alpha_dev_ptr(mut self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        """Pointer to `state_dev[ALPHA]` — the live α the `Scale` nodes
        read via `multiplier_ptr`. Stable for the buffer's lifetime
        (one-time wiring at trainer make)."""
        return self.state_dev.value().unsafe_ptr() + _SA_ALPHA

    def read_alpha(mut self) raises -> Scalar[DT]:
        """D2H the live device α (flush cadence only — NOT per step)."""
        var ctx = self._ctx.value()
        var h = ctx.enqueue_create_host_buffer[DT](_SA_STATE_N)
        ctx.enqueue_copy(h, self.state_dev.value())
        ctx.synchronize()
        return h.unsafe_ptr()[_SA_ALPHA]
