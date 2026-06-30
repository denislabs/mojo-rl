"""TD3TargetYBlock isolation gate — smoothed target-y vs hand oracle (CPU).

Builds a tiny actor (Sequential[Linear, Tanh]) + 2 critics (Linear[SA, 1]) and
the storage `TD3TargetYBlock`, runs `step` with FIXED (zero) smoothing noise —
forced deterministic by setting `noise_std=0` so `sigma = 0` and the box-muller
draw is multiplied to exactly 0 regardless of RNG. With zero noise:

    noise_clip(0) = 0
    a_smoothed    = clamp(actor(s') + 0, ±scale)   (= actor(s'), Tanh ∈ [-1,1])
    y[b]          = r[b] + (1-term[b])·γ·min(Q1, Q2)(s', a_smoothed)

The oracle recomputes a_smoothed + both critic forwards INDEPENDENTLY of the
graph (separate net instances re-forwarded on host), isolating the smoothing /
concat / min / Bellman wiring from the RNG. Assert match < 1e-4.

Run: pixi run mojo run -I . tests/deep_agents/test_storage_td3_target_y.mojo
"""

from std.math import isnan, isinf
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import Tanh
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.core.initializer import Xavier
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.deep_agents.training.trainer_block import TrainerState
from mojo_rl.deep_agents.td3.target_y_block import TD3TargetYBlock


comptime OBS = 3
comptime ACT = 1
comptime SA = OBS + ACT
comptime H = 16
comptime BATCH = 16
comptime GAMMA = Scalar[DT](0.99)
comptime ASCALE = Scalar[DT](2.0)
# Actor: Sequential[Linear, Tanh]; critic: Linear[SA, 1].
comptime ACTOR = Sequential[Linear[OBS, ACT], Tanh[ACT]]
comptime CRITIC = Linear[SA, 1]
comptime BLK = TD3TargetYBlock[ACTOR, CRITIC, BATCH, OBS, ACT]


def main() raises:
    print("=" * 60)
    print("TD3TargetYBlock isolation gate (fixed zero noise, CPU)")
    print("=" * 60)

    # noise_std=0 → deterministic zero smoothing noise.
    var blk = BLK.make["cpu"](
        action_scale=ASCALE, gamma=GAMMA,
        noise_std=Scalar[DT](0.0), noise_clip=Scalar[DT](0.5),
    )
    var actor = ACTOR.make["cpu", Xavier]()
    var t1 = CRITIC.make["cpu", Xavier]()
    var t2 = CRITIC.make["cpu", Xavier]()
    var st = TrainerState[OBS, ACT, BATCH].make["cpu"]()
    for i in range(BATCH * OBS):
        st.mb_sp.data[i] = Scalar[DT]((i % 7) - 3) * 0.2
    for b in range(BATCH):
        st.mb_r.data[b] = Scalar[DT]((b % 4) - 2) * 0.3
        st.mb_d.data[b] = Scalar[DT](1.0) if (b % 8 == 7) else Scalar[DT](0.0)

    blk.step["cpu"](st, actor, t1, t2)

    # ── Independent oracle: re-forward the SAME nets on host. ──
    # a' = clamp(actor(s'), ±scale)  (noise = 0)
    var sp = Tensor.alloc(BATCH * OBS)
    for i in range(BATCH * OBS):
        sp.data[i] = st.mb_sp.data[i]
    var a_act = Tensor.alloc(BATCH * ACT)
    actor.forward["cpu", BATCH](TensorRefs[ACTOR.ARITY](sp), a_act)
    # clamp to ±scale (Tanh output ∈ [-1,1], scale=2 → no-op, but apply anyway).
    for k in range(BATCH * ACT):
        var v = a_act.data[k]
        if v > ASCALE:
            v = ASCALE
        elif v < -ASCALE:
            v = -ASCALE
        a_act.data[k] = v
    # sa = concat(s', a')  → [BATCH, SA]
    var sa = Tensor.alloc(BATCH * SA)
    for b in range(BATCH):
        for d in range(OBS):
            sa.data[b * SA + d] = sp.data[b * OBS + d]
        for j in range(ACT):
            sa.data[b * SA + OBS + j] = a_act.data[b * ACT + j]
    var q1 = Tensor.alloc(BATCH)
    var q2 = Tensor.alloc(BATCH)
    t1.forward["cpu", BATCH](TensorRefs[CRITIC.ARITY](sa), q1)
    t2.forward["cpu", BATCH](TensorRefs[CRITIC.ARITY](sa), q2)

    var ok = True
    var max_err = Scalar[DT](0.0)
    for b in range(BATCH):
        var mq = q1.data[b] if q1.data[b] < q2.data[b] else q2.data[b]
        var y_ref = (
            st.mb_r.data[b]
            + (Scalar[DT](1.0) - st.mb_d.data[b]) * GAMMA * mq
        )
        if isnan(st.mb_y.data[b]) or isinf(st.mb_y.data[b]):
            ok = False
        var err = abs(st.mb_y.data[b] - y_ref)
        if err > max_err:
            max_err = err
        if err > Scalar[DT](1e-4):
            ok = False

    print("  max |mb_y - oracle| :", max_err)
    assert_true(
        ok, "mb_y == r + (1-term)·γ·min(Q1,Q2)(s', clamp(actor(s'),±scale))"
    )
    print("TD3 TARGET-Y ISOLATION OK")
