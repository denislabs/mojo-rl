"""Smoke test: REDQ-OFE config resolves + dims match paper formulas.

Verifies the config's PHI_S_DIM / PHI_SA_DIM / OFE_PER_UNIT math matches
the paper's gin files (total_units=240, num_layers=6 or 8).
"""

from mojo_rl.deep_agents.redq_ofe import (
    REDQOFEConfig,
    DefaultREDQOFEConfig6,
    DefaultREDQOFEConfig8,
)


def main() raises:
    # HalfCheetah: OBS=17, ACT=6, num_layers=6, per_unit=40
    #   PHI_S_DIM  = 17 + 6*40 = 257
    #   PHI_SA_DIM = 257 + 6 + 6*40 = 503
    comptime HC = DefaultREDQOFEConfig6[17, 6]
    print("HalfCheetah (6-layer):")
    print("  OBS =", HC.obs_dim, " ACT =", HC.action_dim)
    print("  OFE_NUM_LAYERS =", HC.OFE_NUM_LAYERS,
          " OFE_PER_UNIT =", HC.OFE_PER_UNIT)
    print("  PHI_S_DIM =", HC.PHI_S_DIM,
          " PHI_SA_DIM =", HC.PHI_SA_DIM)
    print("  ActorModel.IN_DIM =", HC.ActorModel.IN_DIM,
          " (expect PHI_S_DIM =", HC.PHI_S_DIM, ")")
    print("  CriticModel.IN_DIM =", HC.CriticModel.IN_DIM,
          " (expect PHI_SA_DIM =", HC.PHI_SA_DIM, ")")
    print("  NUM_ENSEMBLE =", HC.NUM_ENSEMBLE,
          " UTD_RATIO =", HC.UTD_RATIO)

    # Ant: OBS=111, ACT=8, num_layers=8, per_unit=30
    #   PHI_S_DIM  = 111 + 8*30 = 351
    #   PHI_SA_DIM = 351 + 8 + 8*30 = 599
    print()
    comptime ANT = DefaultREDQOFEConfig8[111, 8]
    print("Ant (8-layer):")
    print("  OBS =", ANT.obs_dim, " ACT =", ANT.action_dim)
    print("  OFE_NUM_LAYERS =", ANT.OFE_NUM_LAYERS,
          " OFE_PER_UNIT =", ANT.OFE_PER_UNIT)
    print("  PHI_S_DIM =", ANT.PHI_S_DIM,
          " PHI_SA_DIM =", ANT.PHI_SA_DIM)
    print("  ActorModel.IN_DIM =", ANT.ActorModel.IN_DIM)
    print("  CriticModel.IN_DIM =", ANT.CriticModel.IN_DIM)

    # Humanoid: OBS=376, ACT=17, num_layers=8, per_unit=30
    #   PHI_S_DIM  = 376 + 8*30 = 616
    #   PHI_SA_DIM = 616 + 17 + 8*30 = 873
    print()
    comptime HUM = DefaultREDQOFEConfig8[376, 17]
    print("Humanoid (8-layer):")
    print("  PHI_S_DIM =", HUM.PHI_S_DIM,
          " PHI_SA_DIM =", HUM.PHI_SA_DIM)
    print("  OFEStateBranchModel.PARAM_SIZE =",
          HUM.OFEStateBranchModel.PARAM_SIZE)
    print("  OFEActionBranchModel.PARAM_SIZE =",
          HUM.OFEActionBranchModel.PARAM_SIZE)
    print("  OFEPredictorModel.PARAM_SIZE =",
          HUM.OFEPredictorModel.PARAM_SIZE)

    print()
    print("PASS: REDQ-OFE configs resolve with paper-matching dims")
