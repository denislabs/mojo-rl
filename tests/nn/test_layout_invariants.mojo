"""Compile-time layout invariant tests.

Verifies that PARAM_SIZE, CACHE_SIZE, and WORKSPACE_SIZE_PER_SAMPLE are
consistent with internal offset calculations across all model composition
primitives. These are comptime asserts — if any invariant breaks, the file
will fail to compile.

Run: cd mojo-rl && pixi run mojo run -I . tests/nn/test_layout_invariants.mojo
"""

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import (
    Sequential,
    Linear,
    LinearReLU,
    LinearTanh,
    RSample,
    Min,
    Slice,
    Negate,
    Parallel,
    SkipConcat,
    DualPath,
    SplitApply,
    FanOut,
    Residual,
    Repeat,
)
from mojo_rl.nn.autodiff import CompositeParams
from mojo_rl.deep_agents.core.strategies.actor_loss import (
    AutodiffMaxEntLoss,
    AutodiffDPGLoss,
    AutodiffTD3Loss,
    DPGLoss,
    MaxEntLoss,
)
from mojo_rl.deep_agents.core.workspace import OffPolicyTrainWS


# =============================================================================
# Helper: round up to next multiple of 4 (same as _seq_align4 / _align4)
# =============================================================================
@always_inline
def _align4(x: Int) -> Int:
    return (x + 3) & ~3


# =============================================================================
# 1. Sequential layout invariants
# =============================================================================

def test_sequential():
    # --- Small model (odd PARAM_SIZE to stress alignment) ---
    comptime SmallSeq = Sequential[Linear[3, 5], Linear[5, 2]]
    # Linear[3,5]: PS = 3*5+5 = 20, Linear[5,2]: PS = 5*2+2 = 12
    comptime assert SmallSeq.PARAM_SIZE == _align4(20) + 12
    comptime assert SmallSeq.PARAM_SIZE == SmallSeq._param_offset[1]() + 12

    # --- Three-layer with activation (3-op fusion) ---
    comptime ThreeLayer = Sequential[LinearReLU[7, 16], LinearReLU[16, 16], Linear[16, 3]]
    # LinearReLU[7,16]: PS = 128, LinearReLU[16,16]: PS = 272, Linear[16,3]: PS = 51
    comptime assert ThreeLayer.PARAM_SIZE == _align4(128) + _align4(272) + 51
    comptime assert ThreeLayer.PARAM_SIZE == ThreeLayer._param_offset[2]() + 51

    # Cache: simple sum, no padding
    comptime assert ThreeLayer.CACHE_SIZE == ThreeLayer._cache_offset[2]() + Linear[16, 3].CACHE_SIZE

    # Workspace: _total_inter + _sum_ws
    comptime assert ThreeLayer.WORKSPACE_SIZE_PER_SAMPLE == ThreeLayer._total_inter() + ThreeLayer._sum_ws()
    comptime assert ThreeLayer._total_inter() == 16 + 16
    comptime assert ThreeLayer._ws_layer_offset[2]() + Linear[16, 3].WORKSPACE_SIZE_PER_SAMPLE == ThreeLayer.WORKSPACE_SIZE_PER_SAMPLE

    print("  PASS: Sequential")


# =============================================================================
# 2. Parallel layout invariants
# =============================================================================

def test_parallel():
    comptime Par = Parallel[Linear[8, 3], LinearTanh[8, 3]]
    # Linear[8,3]: PS = 27, LinearTanh[8,3]: PS = 27 — no alignment between branches
    comptime assert Par.PARAM_SIZE == 27 + 27
    comptime assert Par.PARAM_SIZE == Par._param_offset[1]() + 27
    comptime assert Par.CACHE_SIZE == Linear[8, 3].CACHE_SIZE + LinearTanh[8, 3].CACHE_SIZE

    print("  PASS: Parallel")


# =============================================================================
# 3. DualPath layout invariants
# =============================================================================

def test_dual_path():
    comptime DP = DualPath[Linear[5, 3], Linear[5, 3]]
    # Linear[5,3]: PS = 18 → _align4(18) = 20
    comptime assert DP.PARAM_SIZE == _align4(18) + 18
    comptime assert DP.CACHE_SIZE == 2 * Linear[5, 3].CACHE_SIZE

    # With already-aligned sizes
    comptime DPEven = DualPath[Linear[3, 2], Linear[3, 2]]
    # Linear[3,2]: PS = 8, already 4-aligned
    comptime assert DPEven.PARAM_SIZE == _align4(8) + 8

    print("  PASS: DualPath")


# =============================================================================
# 4. SplitApply layout invariants
# =============================================================================

def test_split_apply():
    comptime SA = SplitApply[Linear[4, 1], Linear[3, 1], 4]
    comptime assert SA.PARAM_SIZE == _align4(Linear[4, 1].PARAM_SIZE) + Linear[3, 1].PARAM_SIZE
    comptime assert SA.CACHE_SIZE == Linear[4, 1].CACHE_SIZE + Linear[3, 1].CACHE_SIZE

    print("  PASS: SplitApply")


# =============================================================================
# 5. SkipConcat layout invariants
# =============================================================================

def test_skip_concat():
    comptime SC = SkipConcat[Linear[8, 4]]
    comptime assert SC.PARAM_SIZE == Linear[8, 4].PARAM_SIZE
    comptime assert SC.CACHE_SIZE == Linear[8, 4].CACHE_SIZE
    comptime assert SC.IN_DIM == 8
    comptime assert SC.OUT_DIM == 8 + 4  # IN_DIM + Inner.OUT_DIM

    print("  PASS: SkipConcat")


# =============================================================================
# 6. FanOut layout invariants
# =============================================================================

def test_fan_out():
    comptime FO = FanOut[Linear[4, 3], 3]
    # Linear[4,3]: PS = 15 → _align4(15) = 16
    comptime assert FO.PARAM_SIZE == 2 * _align4(15) + 15

    print("  PASS: FanOut")


# =============================================================================
# 7. Residual layout invariants
# =============================================================================

def test_residual():
    comptime Res = Residual[Linear[4, 4]]
    comptime assert Res.PARAM_SIZE == Linear[4, 4].PARAM_SIZE
    comptime assert Res.CACHE_SIZE == Linear[4, 4].CACHE_SIZE

    print("  PASS: Residual")


# =============================================================================
# 8. CompositeParams layout invariants
# =============================================================================

def test_composite_params():
    comptime CP2 = CompositeParams[Linear[3, 5], Linear[5, 2]]
    comptime assert CP2.TOTAL_SIZE == _align4(20) + 12
    comptime assert CP2.TOTAL_SIZE == CP2.offset[1]() + 12

    comptime CP3 = CompositeParams[LinearReLU[8, 64], Linear[10, 1], Linear[10, 1]]
    comptime assert CP3.TOTAL_SIZE == (
        _align4(LinearReLU[8, 64].PARAM_SIZE)
        + _align4(Linear[10, 1].PARAM_SIZE)
        + Linear[10, 1].PARAM_SIZE
    )
    comptime assert CP3.TOTAL_SIZE == CP3.offset[2]() + Linear[10, 1].PARAM_SIZE

    print("  PASS: CompositeParams")


# =============================================================================
# 9. Full SAC graph: ws_size vs actual GPU layout
# =============================================================================

def _check_sac[OBS: Int, ACT: Int, H: Int, BS: Int]():
    """Verify SAC graph layout invariants for given dimensions."""

    comptime ActorModel = Sequential[
        LinearReLU[OBS, H],
        LinearReLU[H, H],
        Parallel[Linear[H, ACT], LinearTanh[H, ACT]],
    ]
    comptime CriticModel = Sequential[
        LinearReLU[OBS + ACT, H],
        LinearReLU[H, H],
        Linear[H, 1],
    ]
    comptime ActorRSample = Sequential[ActorModel, RSample[ACT]]
    comptime ActorSkip = SkipConcat[ActorRSample]
    comptime TwinCriticMin = Sequential[DualPath[CriticModel, CriticModel], Min[1]]
    comptime LogProbPass = Slice[1, 0, 1]
    comptime SACOutput = SplitApply[TwinCriticMin, LogProbPass, OBS + ACT]
    comptime SACGraph = Sequential[ActorSkip, SACOutput]

    # Shape checks
    comptime assert SACGraph.IN_DIM == OBS
    comptime assert SACGraph.OUT_DIM == 2

    # PARAM_SIZE: must match nested alignment formula
    comptime APS = ActorModel.PARAM_SIZE
    comptime CPS = CriticModel.PARAM_SIZE
    comptime EXPECTED_PS = _align4(APS) + _align4(_align4(_align4(CPS) + CPS))
    comptime assert SACGraph.PARAM_SIZE == EXPECTED_PS

    # GPU layout actual size
    comptime TOTAL_PS = SACGraph.PARAM_SIZE
    comptime ACTUAL_NEEDED = (
        2 * TOTAL_PS
        + BS * 2
        + max(1, BS * SACGraph.CACHE_SIZE)
        + max(1, BS * SACGraph.WORKSPACE_SIZE_PER_SAMPLE)
        + BS * 2
        + BS * OBS
        + BS
    )

    # ws_size must cover actual GPU layout
    comptime WS_EST = AutodiffMaxEntLoss[].ws_size[
        BS, ACT, ActorModel, CriticModel,
    ]()
    comptime assert WS_EST >= ACTUAL_NEEDED

    # WORKSPACE cross-check: exact formula vs graph computation
    comptime FORMULA_WS = (
        3 * OBS + 7 * ACT + ActorModel.OUT_DIM
        + ActorModel.WORKSPACE_SIZE_PER_SAMPLE
        + 6 * CriticModel.OUT_DIM
        + 2 * CriticModel.WORKSPACE_SIZE_PER_SAMPLE + 4
    )
    comptime assert SACGraph.WORKSPACE_SIZE_PER_SAMPLE == FORMULA_WS


def test_sac_layouts():
    _check_sac[8, 2, 128, 128]()      # Swimmer
    _check_sac[4, 1, 256, 256]()      # InvertedPendulum (H=256)
    _check_sac[4, 1, 64, 64]()       # InvertedPendulum (actual: H=64, BS=64)
    _check_sac[17, 6, 256, 256]()     # HalfCheetah
    _check_sac[11, 3, 256, 256]()     # Hopper
    _check_sac[27, 8, 256, 256]()     # Ant
    _check_sac[376, 17, 256, 256]()   # Humanoid
    _check_sac[2, 1, 8, 4]()          # Tiny (edge case)
    _check_sac[17, 6, 64, 32]()       # Odd PS (Linear[256,6]=1542)

    print("  PASS: SAC graph layouts (8 dimension combos)")


# =============================================================================
# 10. AutodiffDPGLoss (DDPG) graph layout
# =============================================================================

def _check_ddpg[OBS: Int, ACT: Int, H: Int, BS: Int]():
    """Verify DDPG graph layout invariants."""

    comptime ActorModel = Sequential[
        LinearReLU[OBS, H],
        LinearReLU[H, H],
        LinearTanh[H, ACT],
    ]
    comptime CriticModel = Sequential[
        LinearReLU[OBS + ACT, H],
        LinearReLU[H, H],
        Linear[H, 1],
    ]

    # Build the same graph as AutodiffDPGLoss.update_actor_gpu
    comptime ActorSkip = SkipConcat[ActorModel]
    comptime DDPGGraph = Sequential[ActorSkip, CriticModel, Negate[1]]

    comptime APS = ActorModel.PARAM_SIZE
    comptime CPS = CriticModel.PARAM_SIZE

    # PARAM_SIZE: Sequential[SkipConcat, Critic, Negate]
    # = _align4(APS) + _align4(CPS) + 0
    comptime EXPECTED_PS = _align4(APS) + _align4(CPS)
    comptime assert DDPGGraph.PARAM_SIZE == EXPECTED_PS

    # Concat kernel offset must use _align4(APS), not APS
    comptime CRITIC_OFF = _align4(APS)
    comptime assert CRITIC_OFF + CPS <= DDPGGraph.PARAM_SIZE

    # WORKSPACE cross-check: exact formula vs graph computation
    # DDPGGraph = Sequential[SkipConcat[Actor], Critic, Negate[1]]
    # WS = (OBS+ACTIONS+CRITIC_OUT) + (ACTIONS+ACTOR_WS) + CRITIC_WS
    comptime EXPECTED_WS = (
        OBS + 2 * ACT + CriticModel.OUT_DIM
        + ActorModel.WORKSPACE_SIZE_PER_SAMPLE
        + CriticModel.WORKSPACE_SIZE_PER_SAMPLE
    )
    comptime assert DDPGGraph.WORKSPACE_SIZE_PER_SAMPLE == EXPECTED_WS

    # CACHE cross-check
    comptime assert DDPGGraph.CACHE_SIZE == ActorModel.CACHE_SIZE + CriticModel.CACHE_SIZE

    # GPU layout actual size
    comptime TOTAL_PS = DDPGGraph.PARAM_SIZE
    comptime ACTUAL_NEEDED = (
        2 * TOTAL_PS
        + BS
        + max(1, BS * DDPGGraph.CACHE_SIZE)
        + max(1, BS * DDPGGraph.WORKSPACE_SIZE_PER_SAMPLE)
        + BS
        + BS * OBS
    )

    # ws_size must cover actual GPU layout
    comptime WS_EST = AutodiffDPGLoss.ws_size[
        BS, ACT, ActorModel, CriticModel,
    ]()
    comptime assert WS_EST >= ACTUAL_NEEDED


def test_ddpg_layouts():
    _check_ddpg[17, 6, 256, 256]()    # HalfCheetah
    _check_ddpg[4, 1, 256, 256]()     # InvertedPendulum (ACT=1 → odd PS!)
    _check_ddpg[11, 3, 256, 256]()    # Hopper (ACT=3 → odd PS!)
    _check_ddpg[8, 2, 128, 128]()     # Swimmer
    _check_ddpg[27, 8, 256, 256]()    # Ant

    print("  PASS: DDPG graph layouts (5 dimension combos)")


# =============================================================================
# 11. AutodiffTD3Loss graph layout
# =============================================================================

def _check_td3[OBS: Int, ACT: Int, H: Int, BS: Int]():
    """Verify TD3 graph layout invariants."""

    comptime ActorModel = Sequential[
        LinearReLU[OBS, H],
        LinearReLU[H, H],
        LinearTanh[H, ACT],
    ]
    comptime CriticModel = Sequential[
        LinearReLU[OBS + ACT, H],
        LinearReLU[H, H],
        Linear[H, 1],
    ]

    # Build the same graph as AutodiffTD3Loss.update_actor_gpu
    comptime ActorSkip = SkipConcat[ActorModel]
    comptime TwinCriticMin = Sequential[DualPath[CriticModel, CriticModel], Min[1]]
    comptime TD3Graph = Sequential[ActorSkip, TwinCriticMin, Negate[1]]

    comptime APS = ActorModel.PARAM_SIZE
    comptime CPS = CriticModel.PARAM_SIZE

    # Concat kernel offsets must use alignment
    comptime CRITIC1_OFF = _align4(APS)
    comptime CRITIC2_OFF = _align4(APS) + _align4(CPS)
    comptime assert CRITIC1_OFF + CPS <= TD3Graph.PARAM_SIZE
    comptime assert CRITIC2_OFF + CPS <= TD3Graph.PARAM_SIZE

    # WORKSPACE cross-check: exact formula vs graph computation
    # TD3Graph = Sequential[SkipConcat[Actor], Sequential[DualPath[C,C], Min[1]], Negate[1]]
    # WS = intermediates + SkipConcat.WS + TwinCriticMin.WS
    comptime EXPECTED_WS = (
        # Sequential intermediates: ActorSkip.OUT + TwinCriticMin.OUT
        (OBS + ACT) + 1
        # SkipConcat[Actor]: inner_out + inner_ws
        + ACT + ActorModel.WORKSPACE_SIZE_PER_SAMPLE
        # TwinCriticMin = Sequential[DualPath, Min]:
        #   inter: DualPath.OUT = 2*CO
        #   DualPath.WS = (2*CO + OBS+ACT) + 2*CRITIC_WS
        #   Min.WS = CO (AutoDiffChain cache)
        + 2 * CriticModel.OUT_DIM
        + (2 * CriticModel.OUT_DIM + OBS + ACT + 2 * CriticModel.WORKSPACE_SIZE_PER_SAMPLE)
        + CriticModel.OUT_DIM
    )
    comptime assert TD3Graph.WORKSPACE_SIZE_PER_SAMPLE == EXPECTED_WS

    # CACHE cross-check
    comptime assert TD3Graph.CACHE_SIZE == (
        ActorModel.CACHE_SIZE + 2 * CriticModel.CACHE_SIZE + CriticModel.OUT_DIM
    )

    # GPU layout actual size
    comptime TOTAL_PS = TD3Graph.PARAM_SIZE
    comptime ACTUAL_NEEDED = (
        2 * TOTAL_PS
        + BS
        + max(1, BS * TD3Graph.CACHE_SIZE)
        + max(1, BS * TD3Graph.WORKSPACE_SIZE_PER_SAMPLE)
        + BS
        + BS * OBS
    )

    # ws_size must cover actual GPU layout
    comptime WS_EST = AutodiffTD3Loss.ws_size[
        BS, ACT, ActorModel, CriticModel,
    ]()
    comptime assert WS_EST >= ACTUAL_NEEDED


def test_td3_layouts():
    _check_td3[17, 6, 256, 256]()     # HalfCheetah
    _check_td3[4, 1, 256, 256]()      # InvertedPendulum (ACT=1 → odd PS!)
    _check_td3[11, 3, 256, 256]()     # Hopper (ACT=3 → odd PS!)
    _check_td3[8, 2, 128, 128]()      # Swimmer
    _check_td3[27, 8, 256, 256]()     # Ant

    print("  PASS: TD3 graph layouts (5 dimension combos)")


# =============================================================================
# 12. Alignment stress test: non-4-aligned PARAM_SIZE
# =============================================================================

def test_alignment_edge_cases():
    # Linear[H, ACT] with ACT producing non-4-aligned PS
    # Linear[256, 1]: PS = 257 (not aligned)
    # Linear[256, 3]: PS = 771 (not aligned)
    # Linear[256, 5]: PS = 1285 (not aligned)
    comptime assert _align4(257) == 260
    comptime assert _align4(771) == 772
    comptime assert _align4(1285) == 1288

    # Verify Sequential alignment padding is accounted for
    comptime OddSeq = Sequential[Linear[4, 3], Linear[3, 2]]
    # Linear[4,3]: PS = 15, _align4 = 16
    # Linear[3,2]: PS = 8
    comptime assert OddSeq.PARAM_SIZE == 16 + 8  # 24
    comptime assert OddSeq._param_offset[1]() == 16  # critic starts at aligned offset

    # Verify DualPath with non-aligned inner
    comptime OddDP = DualPath[Linear[4, 3], Linear[4, 3]]
    # Linear[4,3]: PS = 15
    comptime assert OddDP.PARAM_SIZE == _align4(15) + 15  # 16 + 15 = 31

    print("  PASS: Alignment edge cases")


# =============================================================================
# 13. Nested composition stress tests
# =============================================================================

def test_nested():
    # Sequential[DualPath[Sequential[...], Sequential[...]], Min]
    comptime InnerSeq = Sequential[LinearReLU[4, 8], Linear[8, 1]]
    comptime DeepNest = Sequential[
        DualPath[InnerSeq, InnerSeq],
        Min[1],
    ]
    comptime assert DeepNest.PARAM_SIZE == (
        _align4(_align4(InnerSeq.PARAM_SIZE) + InnerSeq.PARAM_SIZE) + 0
    )

    # Triple Sequential nesting
    comptime Inner1 = Sequential[LinearReLU[4, 8], Linear[8, 4]]
    comptime Inner2 = Sequential[LinearReLU[4, 8], Linear[8, 2]]
    comptime TripleNest = Sequential[Inner1, Inner2]
    comptime assert TripleNest.PARAM_SIZE == _align4(Inner1.PARAM_SIZE) + Inner2.PARAM_SIZE

    print("  PASS: Nested composition")


# =============================================================================
# 14. OffPolicyTrainWS offset chain
# =============================================================================

def test_offpolicy_workspace():
    """Verify the OffPolicyTrainWS offset chain is contiguous and TOTAL_SIZE
    equals the sum of all regions."""

    # Typical SAC config: OBS=17, ACT=6, H=256, BS=256, 2 critics
    comptime ActorModel = Sequential[
        LinearReLU[17, 256],
        LinearReLU[256, 256],
        Parallel[Linear[256, 6], LinearTanh[256, 6]],
    ]
    comptime CriticModel = Sequential[
        LinearReLU[23, 256],
        LinearReLU[256, 256],
        Linear[256, 1],
    ]
    comptime BS = 256
    comptime OBS = 17
    comptime ACT = 6
    comptime NC = 2  # NUM_CRITICS

    comptime WS = OffPolicyTrainWS[
        BS, OBS, ACT,
        ActorModel.OUT_DIM,
        CriticModel.IN_DIM,
        CriticModel.OUT_DIM,
        CriticModel.CACHE_SIZE,
        ActorModel.CACHE_SIZE,
        CriticModel.WORKSPACE_SIZE_PER_SAMPLE,
        ActorModel.WORKSPACE_SIZE_PER_SAMPLE,
        NC,
        1,  # STRAT_WS placeholder (just needs > 0)
        1,  # TARGET_STRAT_WS placeholder
    ]

    # Region 1: Target computation — offset chain must be monotonically increasing
    comptime assert WS._O_NEXT_ACT == 0
    comptime assert WS._O_NEXT_LP > WS._O_NEXT_ACT
    comptime assert WS._O_NEXT_CI > WS._O_NEXT_LP
    comptime assert WS._O_NEXT_Q > WS._O_NEXT_CI
    comptime assert WS._O_TARGETS > WS._O_NEXT_Q

    # Region 2: Critic update
    comptime assert WS._O_CI > WS._O_TARGETS
    comptime assert WS._O_Q_OUTS > WS._O_CI
    comptime assert WS._O_Q_CACHES > WS._O_Q_OUTS
    comptime assert WS._O_CRITIC_WS_START > WS._O_Q_CACHES
    comptime assert WS._O_Q_GRAD > WS._O_CRITIC_WS_START
    comptime assert WS._O_D_CI > WS._O_Q_GRAD

    # Region 3: Actor workspace
    comptime assert WS._O_ACTOR_WS > WS._O_D_CI

    # Region 4: Strategy workspaces
    comptime assert WS._O_STRAT_WS > WS._O_ACTOR_WS
    comptime assert WS._O_TARGET_STRAT_WS > WS._O_STRAT_WS
    comptime assert WS.TOTAL_SIZE > WS._O_TARGET_STRAT_WS

    # Verify exact region sizes (spot-check critical ones)
    comptime assert WS._O_NEXT_LP - WS._O_NEXT_ACT == BS * ACT
    comptime assert WS._O_NEXT_CI - WS._O_NEXT_LP == BS
    comptime assert WS._O_NEXT_Q - WS._O_NEXT_CI == BS * CriticModel.IN_DIM
    comptime assert WS._O_Q_OUTS - WS._O_CI == BS * CriticModel.IN_DIM
    comptime assert WS._O_Q_CACHES - WS._O_Q_OUTS == NC * BS * CriticModel.OUT_DIM
    comptime assert WS._O_D_CI - WS._O_Q_GRAD == BS * CriticModel.OUT_DIM

    print("  PASS: OffPolicyTrainWS offset chain")


# =============================================================================
# 15. Manual loss strategy layouts (DPGLoss, MaxEntLoss)
# =============================================================================

def test_manual_loss_layouts():
    """Verify manual (non-autodiff) loss ws_size covers their GPU layouts."""

    comptime BS = 128
    comptime ACT = 6
    comptime H = 256

    # DDPG-style actor/critic
    comptime DDPGActor = Sequential[
        LinearReLU[17, H], LinearReLU[H, H], LinearTanh[H, ACT],
    ]
    comptime DDPGCritic = Sequential[
        LinearReLU[17 + ACT, H], LinearReLU[H, H], Linear[H, 1],
    ]

    # DPGLoss GPU layout: 9 buffers
    comptime DPG_WS = DPGLoss.ws_size[BS, ACT, DDPGActor, DDPGCritic]()
    comptime DPG_ACTUAL = (
        BS * ACT                        # actor_act
        + BS * DDPGActor.CACHE_SIZE     # actor_cache
        + BS * DDPGCritic.IN_DIM        # new_ci
        + BS * DDPGCritic.OUT_DIM       # new_q
        + BS * DDPGCritic.CACHE_SIZE    # new_q_cache
        + BS * DDPGCritic.OUT_DIM       # dq
        + BS * DDPGCritic.IN_DIM        # d_ci
        + BS * ACT                      # d_act
        + BS * DDPGActor.IN_DIM         # d_obs
    )
    comptime assert DPG_WS == DPG_ACTUAL

    # SAC-style actor/critic
    comptime SACActor = Sequential[
        LinearReLU[17, H], LinearReLU[H, H],
        Parallel[Linear[H, ACT], LinearTanh[H, ACT]],
    ]
    comptime SACCritic = Sequential[
        LinearReLU[17 + ACT, H], LinearReLU[H, H], Linear[H, 1],
    ]

    # MaxEntLoss GPU layout: 22 buffers
    comptime ME_WS = MaxEntLoss[].ws_size[BS, ACT, SACActor, SACCritic]()
    comptime ME_ACTUAL = (
        BS * SACActor.OUT_DIM          # raw_out
        + BS * SACActor.CACHE_SIZE     # actor_cache
        + BS * ACT                     # mean
        + BS * ACT                     # log_std
        + BS * ACT                     # noise
        + BS * ACT                     # act
        + BS                           # log_probs
        + BS * ACT                     # z_cache
        + BS * SACCritic.IN_DIM        # critic_input
        + BS * SACCritic.OUT_DIM       # Q1
        + BS * SACCritic.CACHE_SIZE    # Q1 cache
        + BS * SACCritic.OUT_DIM       # Q2
        + BS * SACCritic.CACHE_SIZE    # Q2 cache
        + BS * SACCritic.OUT_DIM       # dq1
        + BS * SACCritic.OUT_DIM       # dq2
        + BS * SACCritic.IN_DIM        # d_ci1
        + BS * SACCritic.IN_DIM        # d_ci2
        + BS * ACT                     # d_act
        + BS * ACT                     # grad_mean
        + BS * ACT                     # grad_log_std
        + BS * SACActor.OUT_DIM        # actor_grad
        + BS * SACActor.IN_DIM         # d_obs
    )
    comptime assert ME_WS == ME_ACTUAL

    print("  PASS: Manual loss layouts (DPGLoss, MaxEntLoss)")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("TEST: Layout Invariants (compile-time)")
    print("=" * 70)
    print()

    test_sequential()
    test_parallel()
    test_dual_path()
    test_split_apply()
    test_skip_concat()
    test_fan_out()
    test_residual()
    test_composite_params()
    test_sac_layouts()
    test_ddpg_layouts()
    test_td3_layouts()
    test_alignment_edge_cases()
    test_nested()
    test_offpolicy_workspace()
    test_manual_loss_layouts()

    print()
    print("  ALL PASSED: Layout invariants verified")
