# +--------------------------------------------------------------------------+ #
# | CPU/GPU parity statistics for gates that compare two precisions
# +--------------------------------------------------------------------------+ #
"""How to decide whether a GPU tensor agrees with its CPU original.

Shared BECAUSE it drifted: `test_resnet18_gpu.mojo` and
`test_act_gpu_vs_cpu.mojo` each carried a private copy of the same constants
and the same helper, so a correction had to be made twice and was, twice, made
once. One definition, imported by both.

## The thing that makes this hard

NVIDIA runs fp32 matmuls on TF32 tensor cores: a 10-bit mantissa, ~1e-3
relative per matmul, compounding with depth. Apple has no TF32 and sits at
~1e-7. So a tolerance calibrated on Metal cannot serve CUDA — the mistake
recorded in `feedback_fd_gradcheck_tf32`, and made twice more in this file's
history.

But the precision gap is the easy half. The hard half is that **ReLU is
continuous and its derivative is not**, and that splits the problem in two:

  * A FORWARD value is a continuous function of the inputs. A pre-activation
    perturbed across zero changes the output by about the perturbation, so
    forward agreement degrades smoothly with precision and every element stays
    close. Measured on a 5090, ResNet18's layer4 output: worst element
    **4.3e-3** of the tensor's own max, over 6144 values — no outliers at all.

  * A GRADIENT is not. A pre-activation within TF32 noise of zero has its SIGN
    decided by rounding, so one side propagates the gradient through that unit
    and the other zeroes it. The resulting difference is the size of whatever
    gradient was flowing — up to the tensor's own maximum — at a handful of
    elements. Measured on the same 5090 run:

        grad_input   worst 0.103 x max|a|    1 of      36,864 elements
        conv grads   worst 0.119 x max|a|    8 of  11,171,712
        BN affine    worst 0.110 x max|a|    2 of       9,600

    while the L2 norms agreed to 0.24% - 0.62%.

**No per-element tolerance can decide that case.** The outlier's size is set by
the gradient magnitude at the flipped unit, not by the precision, so raising
RTOL until it passes buys nothing: the next seed puts a bigger gradient at the
boundary. Two rounds of this gate were lost to exactly that, the second time
with RTOL "measured" on Apple hardware that has no TF32 to measure.

Confirmed on both: `test-act-gpu` 4/4 on an RTX 5090 (645 s) and 4/4 on an
M1 Pro (250 s), on the split verdicts below.

## So the verdicts differ by what is being compared

`ok_continuous` — forward activations, outputs, weights. Every element must be
within tolerance; a single one outside is a real disagreement, because nothing
here is discontinuous. RTOL is TIGHT (5x headroom over the measured 4.3e-3).

`ok_gradient` — anything downstream of a `d/dx`. Three arms:

  1. the L2 norm over every element — the arm with teeth, since structurally
     wrong gradients cannot match in norm to a fraction of a percent;
  2. the FRACTION past tolerance, not the max — a boundary flip is rare by
     construction (it needs a pre-activation inside a 1e-3 band), so the count
     is what separates "a few flips" from "systematically wrong". The bound is
     5x the worst fraction observed on CUDA (2.1e-4, on the smallest tensor);
  3. an outlier CAP: no single element may differ by more than the tensor's own
     maximum. A ReLU flip cannot exceed that — the gradient it gates is part of
     the same tensor — but a sign error, a bad index or an uninitialised read
     can, and that is what this arm is for.

⚠ The gradient policy is deliberately blind to ONE element of a large tensor
being silently zeroed: that is under the fraction bound and under the cap. It
is the price of the discontinuity, and the norm arm plus the per-layer coverage
in the gates is what stands in for it. Do not "fix" it by lowering the
fraction bound to 0 — that is where this started.
"""

from mojo_rl.nn.constants import DT


comptime PARITY_ATOL: Float64 = 1e-5
"""Floor, so an all-but-zero tensor does not divide by nothing."""

comptime PARITY_RTOL_CONTINUOUS: Float64 = 2e-2
"""Forward/weight tolerance, as a fraction of the tensor's max. 5x the 4.3e-3
worst element measured across ResNet18's output on a 5090."""

comptime PARITY_RTOL_GRAD: Float64 = 1e-1
"""Gradient tolerance. Only decides what COUNTS as an outlier — `ok_gradient`
gates on how many there are, not on the worst one."""

comptime PARITY_NORM_RTOL: Float64 = 1e-2
"""L2-norm agreement. 1.6x the worst observed on CUDA (6.2e-3) and 5 orders
above Apple's (2.7e-8)."""

comptime PARITY_GRAD_OVER_FRAC: Float64 = 1e-3
"""Fraction of elements allowed past `PARITY_RTOL_GRAD`. 5x the worst observed
(2 of 9,600 = 2.1e-4); CUDA's other gradient tensors ran 7e-7 to 2.7e-5."""

comptime PARITY_OUTLIER_CAP: Float64 = 1.0
"""No element may differ by more than the tensor's own max. A ReLU sign flip
cannot; an index or sign bug can."""


@fieldwise_init
struct Parity(ImplicitlyCopyable):
    """The comparison of two same-length tensors, with the verdict left open.

    `worst` is reported as a MULTIPLE of the tolerance so it is readable next
    to a pass/fail; `worst_abs` and `scale` are kept because the cap arm needs
    the raw numbers and a reader needs to see what the ratio was against.
    """

    var worst: Float64
    """`max|a-b|` divided by the tolerance — 1.0 is exactly at the line."""
    var worst_abs: Float64
    """`max|a-b|` itself."""
    var scale: Float64
    """`max|a|` — the tensor's own magnitude, which is what the error scales
    with. NOT `max|a-b|`'s own element; two different elements."""
    var nrel: Float64
    """`|‖a‖ - ‖b‖| / ‖a‖` over every element."""
    var n_over: Int
    var n: Int

    def ok_continuous(self) -> Bool:
        """No element outside tolerance, and the norms agree. For anything
        whose value is a continuous function of the inputs."""
        return self.nrel < PARITY_NORM_RTOL and self.n_over == 0

    def ok_gradient(self) -> Bool:
        """Norms agree, few enough outliers, and none of them enormous. See
        the module docstring for why the max cannot be the gate here."""
        return (
            self.nrel < PARITY_NORM_RTOL
            and Float64(self.n_over)
            <= PARITY_GRAD_OVER_FRAC * Float64(self.n)
            and self.worst_abs <= PARITY_OUTLIER_CAP * self.scale
        )

    def detail(self) -> String:
        """One line carrying every number the verdict used, so a failure says
        which arm tripped without a second run."""
        return (
            "worst " + String(self.worst) + "x tol ("
            + String(self.worst_abs) + " on scale " + String(self.scale)
            + ")  norm-rel " + String(self.nrel) + "  over-tol "
            + String(self.n_over) + "/" + String(self.n)
        )


def parity(
    ref a: List[Scalar[DT]], ref b: List[Scalar[DT]], rtol: Float64
) raises -> Parity:
    """Compare `b` against reference `a`, tolerance scaled by `max|a|`.

    ⚠ NOT per-element relative error (unbounded at a true zero, and both
    quantities produce true zeros for structural reasons — see the header) and
    NOT `max|a-b|` against `max|a|` as a single ratio, which hides how many
    elements are involved.
    """
    if len(a) != len(b):
        raise Error(
            "parity: length mismatch " + String(len(a)) + " vs "
            + String(len(b))
        )
    if len(a) == 0:
        raise Error("parity: empty tensor — the comparison would be vacuous")

    var scale = Float64(0.0)
    for i in range(len(a)):
        scale = max(scale, abs(Float64(a[i])))
    var tol = PARITY_ATOL + rtol * scale

    var worst_abs = Float64(0.0)
    var n_over = 0
    var sa = Float64(0.0)
    var sb = Float64(0.0)
    for i in range(len(a)):
        var x = Float64(a[i])
        var y = Float64(b[i])
        var d = abs(x - y)
        worst_abs = max(worst_abs, d)
        if d > tol:
            n_over += 1
        sa += x * x
        sb += y * y
    var na = sa ** 0.5
    var nb = sb ** 0.5
    return Parity(
        worst_abs / tol,
        worst_abs,
        scale,
        abs(na - nb) / (na + 1e-30),
        n_over,
        len(a),
    )


def parity_continuous(
    ref a: List[Scalar[DT]], ref b: List[Scalar[DT]]
) raises -> Parity:
    """Forward values, outputs, weights — pair with `ok_continuous`."""
    return parity(a, b, PARITY_RTOL_CONTINUOUS)


def parity_gradient(
    ref a: List[Scalar[DT]], ref b: List[Scalar[DT]]
) raises -> Parity:
    """Anything downstream of a derivative — pair with `ok_gradient`."""
    return parity(a, b, PARITY_RTOL_GRAD)
