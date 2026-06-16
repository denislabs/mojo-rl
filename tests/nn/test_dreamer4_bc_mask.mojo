"""Dreamer 4 BC agent-isolation mask (Phase 3.1).

    pixi run mojo run -I . tests/nn/test_dreamer4_bc_mask.mojo

Host-side check of `build_modality_mask` on the dynamics token layout

    [ action | signal | step | spatial×NSP | register×NREG | agent×NAGENT ]

(modality ids 0,1,2,3,4,5; agent = highest id = 5). Verifies the three
world-model mask variants against the paper §3.3 / reference `_build_allow`
semantics:

  - "wm_agent"          : full mixing — allow[i,j] = True everywhere.
  - "wm_agent_isolated" : agent q → only agent keys; non-agent q → all
                          NON-agent keys (inert pretraining).
  - "wm_agent_bc"       : agent q → ALL keys; non-agent q → all NON-agent
                          keys (paper BC: agent reads the world, nothing reads
                          back). The KEY property for Phase 3: a masked entry
                          is NEG (∞-suppressed), an allowed entry is 0.0.
"""

from std.testing import assert_true, assert_equal

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.masked_attention import build_modality_mask, MASK_NEG


comptime NSP = 4
comptime NREG = 2
comptime NAGENT = 1
# action(0) signal(1) step(2) spatial×NSP(3) register×NREG(4) agent×NAGENT(5)
comptime S = 3 + NSP + NREG + NAGENT


def _ids() -> List[Int]:
    var ids = List[Int]()
    ids.append(0)                       # action
    ids.append(1)                       # signal
    ids.append(2)                       # step
    for _ in range(NSP):
        ids.append(3)                   # spatial
    for _ in range(NREG):
        ids.append(4)                   # register
    for _ in range(NAGENT):
        ids.append(5)                   # agent (highest ⇒ AGENT modality)
    return ids^


def _allow(m: List[Scalar[DT]], i: Int, j: Int) -> Bool:
    # allowed ⇒ 0.0, disallowed ⇒ NEG.
    return m[i * S + j] == Scalar[DT](0.0)


def _is_agent(j: Int) -> Bool:
    return j >= S - NAGENT


def main() raises:
    print("=" * 70)
    print("Dreamer 4 BC agent-isolation mask")
    print("=" * 70)
    var ids = _ids()
    assert_equal(len(ids), S, "layout size")

    # ── "wm_agent": full mixing ─────────────────────────────────────────
    var full = build_modality_mask["wm_agent"](ids.copy(), n_latents=0)
    assert_equal(len(full), S * S, "mask size")
    for i in range(S):
        for j in range(S):
            assert_true(_allow(full, i, j), "wm_agent must allow all")
    print("   wm_agent: full mixing OK")

    # ── "wm_agent_isolated": inert pretraining ──────────────────────────
    var iso = build_modality_mask["wm_agent_isolated"](ids.copy(), n_latents=0)
    for i in range(S):
        for j in range(S):
            var got = _allow(iso, i, j)
            var want: Bool
            if _is_agent(i):
                want = _is_agent(j)         # agent q → only agent keys
            else:
                want = not _is_agent(j)     # non-agent q → all non-agent keys
            assert_true(got == want, "wm_agent_isolated mismatch")
    print("   wm_agent_isolated: agent→agent, non-agent→non-agent OK")

    # ── "wm_agent_bc": paper BC ─────────────────────────────────────────
    var bc = build_modality_mask["wm_agent_bc"](ids.copy(), n_latents=0)
    var n_agent_reads = 0
    for i in range(S):
        for j in range(S):
            var got = _allow(bc, i, j)
            var want: Bool
            if _is_agent(i):
                want = True                 # agent q → ALL keys
            else:
                want = not _is_agent(j)     # non-agent q → all non-agent keys
            assert_true(got == want, "wm_agent_bc mismatch")
        if _is_agent(i):
            for j in range(S):
                if _allow(bc, i, j):
                    n_agent_reads += 1
    # The single agent row reads all S keys.
    assert_equal(n_agent_reads, NAGENT * S, "agent reads full world")

    # No non-agent token may read the agent column (no WM contamination).
    for i in range(S - NAGENT):
        for j in range(S - NAGENT, S):
            assert_true(not _allow(bc, i, j), "non-agent must NOT read agent")
            assert_true(bc[i * S + j] == Scalar[DT](MASK_NEG), "NEG fill")
    print("   wm_agent_bc: agent reads all, nothing reads agent OK")

    print("=" * 70)
    print("ALL PASSED — Dreamer 4 BC agent-isolation mask")
    print("=" * 70)
