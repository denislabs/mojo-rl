"""The `z` sampler — latent task vectors on the radius-sqrt(d) sphere.

`z` is the "remote control": one vector picks one stationary policy out of the
whole family `pi_z`. Two things about it decide whether FB trains at all.

**1. The mixture.** Roughly half the draws are uniform on the sphere, half come
from `B(s)` at states taken from the batch. A purely uniform `z` addresses a
latent space that corresponds to nothing reachable — the policy is asked to
optimise rewards no state distribution can produce — while a purely `B(s)`-based
one never explores outside what `B` already encodes and collapses onto it. The
default `UNIFORM_FRAC = 0.5` is the published FB / Meta Motivo setting.

**2. The norm.** `z` lives on the sphere of radius `sqrt(d)`, and this is the
single most dangerous invariant in the whole component. `docs/BFM_ZERO_SHOT_RL.md`
§11 ranks it first among the silent failures: a `z` with the wrong norm crashes
nothing, raises nothing, and trains to a policy that emits plausible garbage.
The countermeasure is structural rather than documentary —

    ⚠⚠ **Every path out of this module renormalises, inside the function that
    produces `z`. Never at the call site.**

`sample_z`, `z_from_b` and `z_from_reward` all end in `_project_to_sphere`.
Inference must do the same: `z_from_reward` is the zero-shot path
(`z = E_rho[B(s)·r(s)]`), and it renormalises for exactly the same reason the
training sampler does. A call site that "already knows" its `z` is normalised
is one refactor away from being wrong, with no symptom.

`_project_to_sphere` is the only place the radius is computed, so `d` and the
radius cannot drift apart.

Degenerate rows are handled rather than propagated: a zero vector has no
direction to project, so it is replaced by a fixed unit axis scaled to the
sphere. Rescaling a ~0 vector by `sqrt(d)/~0` would otherwise amplify float
noise into a full-magnitude `z` pointing in a direction decided by rounding.
"""

from std.math import cos as fcos, log as flog, sqrt
from std.random import random_float64

from mojo_rl.nn.constants import DT


def _project_to_sphere[D: Int](mut z: List[Scalar[DT]], row: Int):
    """Rescale `z[row*D : (row+1)*D]` onto the radius-sqrt(D) sphere.

    The ONE place the radius is defined. See the module docstring on why no
    caller is allowed to do this itself.
    """
    var radius = sqrt(Float64(D))
    var acc = Float64(0)
    for k in range(D):
        var v = Float64(z[row * D + k])
        acc += v * v
    var n = sqrt(acc)
    if n < 1e-12:
        # No direction to preserve. Scaling by radius/n here would turn
        # rounding noise into a unit-length z aimed nowhere in particular.
        for k in range(D):
            z[row * D + k] = Scalar[DT](0)
        z[row * D] = Scalar[DT](radius)
        return
    var s = radius / n
    for k in range(D):
        z[row * D + k] = Scalar[DT](Float64(z[row * D + k]) * s)


def _fill_gaussian_row[D: Int](mut z: List[Scalar[DT]], row: Int):
    """One row of iid N(0, 1) — the direction is then uniform on the sphere.

    Uses the host RNG directly rather than `box_muller_normal`, which fills a
    whole buffer: the mixture draws a per-ROW decision, so rows are produced
    one at a time.
    """
    for k in range(D):
        var u1 = random_float64()
        if u1 < 1e-10:
            u1 = 1e-10
        var u2 = random_float64()
        z[row * D + k] = Scalar[DT](
            sqrt(-2.0 * flog(u1)) * fcos(6.283185307179586 * u2)
        )


def sample_z_uniform[D: Int](batch: Int) raises -> List[Scalar[DT]]:
    """`batch` rows, each uniform on the radius-sqrt(D) sphere."""
    if batch <= 0:
        raise Error("sample_z_uniform: batch must be > 0")
    var z = List[Scalar[DT]](length=batch * D, fill=Scalar[DT](0))
    for r in range(batch):
        _fill_gaussian_row[D](z, r)
        _project_to_sphere[D](z, r)
    return z^


def z_from_b[
    D: Int
](ref b_states: List[Scalar[DT]], n_rows: Int) raises -> List[Scalar[DT]]:
    """One `z` per row of `B(s)`, projected onto the sphere.

    `b_states` is `[n_rows, D]` — the backward network's output on a batch of
    states. Meta Motivo's `norm_z: true` is this projection.
    """
    if n_rows <= 0:
        raise Error("z_from_b: n_rows must be > 0")
    if len(b_states) < n_rows * D:
        raise Error(
            "z_from_b: b_states holds " + String(len(b_states))
            + " elements, need " + String(n_rows * D)
        )
    var z = List[Scalar[DT]](length=n_rows * D, fill=Scalar[DT](0))
    for r in range(n_rows):
        for k in range(D):
            z[r * D + k] = b_states[r * D + k]
        _project_to_sphere[D](z, r)
    return z^


def sample_z[
    D: Int
](
    batch: Int,
    ref b_states: List[Scalar[DT]],
    n_b_rows: Int,
    uniform_frac: Float64 = 0.5,
) raises -> List[Scalar[DT]]:
    """The training mixture: `uniform_frac` on the sphere, the rest from `B(s)`.

    `b_states` is `[n_b_rows, D]`, the backward network's output on states drawn
    from the batch; rows are picked from it uniformly WITH replacement, so
    `n_b_rows` need not equal `batch`.

    Passing `n_b_rows == 0` degrades to a fully uniform draw — legitimate only
    at the very start of training, before `B` has been evaluated once. It is
    not silent: the caller asked for a mixture and got one component, so
    anything downstream comparing the two halves would see nothing to compare.
    """
    if batch <= 0:
        raise Error("sample_z: batch must be > 0")
    if uniform_frac < 0.0 or uniform_frac > 1.0:
        raise Error(
            "sample_z: uniform_frac must be in [0, 1], got "
            + String(uniform_frac)
        )
    var z = List[Scalar[DT]](length=batch * D, fill=Scalar[DT](0))
    var have_b = n_b_rows > 0 and len(b_states) >= n_b_rows * D
    for r in range(batch):
        if (not have_b) or random_float64() < uniform_frac:
            _fill_gaussian_row[D](z, r)
        else:
            var src = Int(random_float64() * Float64(n_b_rows))
            if src >= n_b_rows:
                src = n_b_rows - 1
            for k in range(D):
                z[r * D + k] = b_states[src * D + k]
        # ⚠ Unconditional, and inside the producer. Both branches renormalise:
        # the Gaussian one because its radius is chi-distributed, the B(s) one
        # because B carries whatever scale its last layer happened to learn.
        _project_to_sphere[D](z, r)
    return z^


def z_from_reward[
    D: Int
](
    ref b_states: List[Scalar[DT]],
    ref rewards: List[Scalar[DT]],
    n_rows: Int,
) raises -> List[Scalar[DT]]:
    """Zero-shot inference: `z = E_rho[B(s)·r(s)]`, projected onto the sphere.

    This is the whole of "component 3" — the step that turns a reward function
    into a policy without any training. `b_states` is `[n_rows, D]` and
    `rewards` is `[n_rows]`, both evaluated on states drawn from the dataset.

    The projection matters as much here as in training, and for a subtler
    reason: the expectation's SCALE is arbitrary (doubling every reward doubles
    `z`), while the policy family is indexed by direction on the sphere. Without
    it, `pi_z` is being queried at a point the training distribution never
    reached, and the resulting behaviour is not the reward's optimum but an
    extrapolation.
    """
    if n_rows <= 0:
        raise Error("z_from_reward: n_rows must be > 0")
    if len(b_states) < n_rows * D:
        raise Error("z_from_reward: b_states too short")
    if len(rewards) < n_rows:
        raise Error("z_from_reward: rewards too short")
    var z = List[Scalar[DT]](length=D, fill=Scalar[DT](0))
    for r in range(n_rows):
        var w = Float64(rewards[r])
        for k in range(D):
            z[k] = Scalar[DT](
                Float64(z[k]) + w * Float64(b_states[r * D + k])
            )
    var inv = 1.0 / Float64(n_rows)
    for k in range(D):
        z[k] = Scalar[DT](Float64(z[k]) * inv)
    _project_to_sphere[D](z, 0)
    return z^
