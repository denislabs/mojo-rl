"""Dreamer4Dynamics agent tokens — isolation + h_t gradient flow (Phase 3.3).

    pixi run mojo run -I . tests/nn2/test_dreamer4_dynamics_agent.mojo

With NAGENT>0 the dynamics inserts agent tokens (paper §3.3) under the
`wm_agent_bc` mask: agent tokens read the whole frame, but NOTHING attends back
to them. Three checks:

  A. FLOW ISOLATION — the x-prediction (flow) output is INVARIANT to the agent
     token input. Changing the agent input cannot change the world-model's
     predictions (no token attends to agent tokens). This is the crucial
     no-contamination property.
  B. h_t RESPONDS — the agent output h_t DOES change with the agent input
     (the agent token attends to itself) AND with the frame input (it reads
     the world) — so it carries task-conditioned, world-aware information.
  C. GRADIENT FLOW — pushing a grad of h_t via `set_grad_h` and training the
     transformer fits h_t to a target (loss decreases), and the grad wrt the
     agent input (`grad_agent_in`, which feeds the TaskEmbedder) is non-zero.
"""

from std.memory import alloc
from std.math import sin, abs
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.nn2.optimizer import Adam
from mojo_rl.deep_agents2.dreamer4.dynamics import Dreamer4Dynamics


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


comptime DSP = 4
comptime NSP = 4
comptime D = 8
comptime NH = 2
comptime T = 2
comptime NREG = 2
comptime HID = 16
comptime DEPTH = 2
comptime KMAX = 4
comptime NAGENT = 1
comptime B = 2
comptime BF = B * T
comptime ND = NSP * DSP
comptime N = BF * ND
comptime AGD = NAGENT * D          # agent-token total width per sample


comptime Dyn = Dreamer4Dynamics[
    DSP, NSP, D, NH, T, NREG, HID, DEPTH, KMAX, True, 0, 0, NAGENT
]


def _fwd(
    mut dyn: Dyn,
    z: UnsafePointer[Scalar[DT], MutAnyOrigin],
    agent_in: UnsafePointer[Scalar[DT], MutAnyOrigin],
    sig: UnsafePointer[Scalar[DT], MutAnyOrigin],
    stp: UnsafePointer[Scalar[DT], MutAnyOrigin],
    flow_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
) raises:
    dyn.set_indices(sig, stp, BF)
    dyn.set_agent_in(agent_in, BF)
    var zt = TileTensor(z, row_major[BF, ND]())
    var ot = TileTensor(flow_out, row_major[BF, ND]())
    dyn.forward["cpu", BF](zt, output=ot)


def _maxdiff(
    a: UnsafePointer[Scalar[DT], MutAnyOrigin],
    b: UnsafePointer[Scalar[DT], MutAnyOrigin],
    n: Int,
) -> Float64:
    var m: Float64 = 0.0
    for i in range(n):
        var d = abs(Float64(a[i]) - Float64(b[i]))
        if d > m:
            m = d
    return m


def main() raises:
    print("=" * 70)
    print("Dreamer4Dynamics — agent tokens (isolation + h_t grad)  CPU")
    print("=" * 70)

    var dyn = Dyn.make[target="cpu", INIT=Xavier]()

    var z = _alloc(N)
    var z2 = _alloc(N)
    var agA = _alloc(BF * AGD)
    var agB = _alloc(BF * AGD)
    var sig = _alloc(BF)
    var stp = _alloc(BF)
    var flowA = _alloc(N)
    var flowB = _alloc(N)
    var hA = _alloc(BF * AGD)
    var hB = _alloc(BF * AGD)

    for i in range(N):
        z[i] = Scalar[DT](0.5 + 0.4 * sin(0.3 + 0.5 * Float64(i)))
        z2[i] = Scalar[DT](0.2 * sin(1.7 + 0.9 * Float64(i)))
    for bt in range(BF):
        sig[bt] = 2.0
        stp[bt] = 1.0
    for i in range(BF * AGD):
        agA[i] = Scalar[DT](0.7 * sin(0.2 + 0.8 * Float64(i)))
        agB[i] = Scalar[DT](-0.6 * sin(1.1 + 0.5 * Float64(i)))

    # ── A. flow isolation: agent input must NOT affect the flow output ──
    _fwd(dyn, z, agA, sig, stp, flowA)
    for i in range(BF * AGD):
        hA[i] = dyn.agent_out_ptr_cpu()[i]
    _fwd(dyn, z, agB, sig, stp, flowB)
    for i in range(BF * AGD):
        hB[i] = dyn.agent_out_ptr_cpu()[i]

    var flow_diff = _maxdiff(flowA, flowB, N)
    var h_diff_input = _maxdiff(hA, hB, BF * AGD)
    print("   flow max|Δ| over agent input =", flow_diff, " (must be 0)")
    print("   h_t  max|Δ| over agent input =", h_diff_input, " (must be >0)")
    assert_true(flow_diff == 0.0, "flow output must be isolated from agent input")
    assert_true(h_diff_input > 1e-6, "h_t must respond to the agent input")

    # ── B. h_t responds to the frame (agent reads the world) ────────────
    var hz1 = _alloc(BF * AGD)
    var hz2 = _alloc(BF * AGD)
    _fwd(dyn, z, agA, sig, stp, flowA)
    for i in range(BF * AGD):
        hz1[i] = dyn.agent_out_ptr_cpu()[i]
    _fwd(dyn, z2, agA, sig, stp, flowA)
    for i in range(BF * AGD):
        hz2[i] = dyn.agent_out_ptr_cpu()[i]
    var h_diff_frame = _maxdiff(hz1, hz2, BF * AGD)
    print("   h_t  max|Δ| over frame input =", h_diff_frame, " (must be >0)")
    assert_true(h_diff_frame > 1e-6, "h_t must read the world (frame)")

    # ── C. gradient flow: fit h_t to a target by training the transformer ─
    var optim = Adam.make["cpu", M=type_of(dyn)](dyn)
    optim.lr = Scalar[DT](3e-3)

    var target = _alloc(BF * AGD)
    for i in range(BF * AGD):
        target[i] = Scalar[DT](0.3 * sin(0.5 + 0.4 * Float64(i)))
    var grad_h = _alloc(BF * AGD)
    var grad_flow = _alloc(N)
    for i in range(N):
        grad_flow[i] = Scalar[DT](0.0)        # pure h_t loss → no flow grad
    var grad_in = _alloc(N)
    var gflow_t = TileTensor(grad_flow, row_major[BF, ND]())
    var gin_t = TileTensor(grad_in, row_major[BF, ND]())

    var first: Float64 = 0.0
    var last: Float64 = 0.0
    var max_gain: Float64 = 0.0
    for step in range(200):
        optim.zero_grad["cpu"](dyn)
        _fwd(dyn, z, agA, sig, stp, flowA)
        var loss: Float64 = 0.0
        for i in range(BF * AGD):
            var h = dyn.agent_out_ptr_cpu()[i]
            var diff = h - target[i]
            grad_h[i] = diff
            loss += 0.5 * Float64(diff) * Float64(diff)
        dyn.set_grad_h(grad_h, BF)
        dyn.vjp["cpu", BF](gflow_t, gin_t)
        if step == 0:
            # grad wrt agent input (feeds the TaskEmbedder) must be populated
            for i in range(BF * AGD):
                var g = abs(Float64(dyn.grad_agent_in_ptr_cpu()[i]))
                if g > max_gain:
                    max_gain = g
        optim.step["cpu"](dyn)
        if step == 0:
            first = loss
        last = loss
        if step % 40 == 0:
            print("   step", step, " h_t loss =", loss)
    print("   first =", first, "  last =", last)
    print("   grad_agent_in max|·| (step 0) =", max_gain, " (must be >0)")

    assert_true(last < 0.2 * first, "h_t loss must decrease (grad flows)")
    assert_true(max_gain > 1e-7, "grad wrt agent input must be non-zero")

    print("=" * 70)
    print("ALL PASSED — Dreamer4Dynamics agent tokens (CPU)")
    print("=" * 70)
