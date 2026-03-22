"""Test MuZero configs compilation and dimensions."""

from mojo_rl.deep_agents.muzero.configs import (
    MuZeroMLPConfig,
    MuZeroCNNConfig,
    MuZeroResNetConfig,
    MuZeroLargeConfig,
    AlphaZeroConfig,
    EfficientZeroConfig,
)


fn main():
    print("=== MuZero Configs Test ===")

    # MLP Config
    comptime C1 = MuZeroMLPConfig[4, 2, LATENT=64, HIDDEN=64, BINS=21]
    print("MLP: Rep", C1.RepModel.IN_DIM, "->", C1.RepModel.OUT_DIM,
          "| Dyn", C1.DynModel.IN_DIM, "->", C1.DynModel.OUT_DIM,
          "| Pred", C1.PredModel.IN_DIM, "->", C1.PredModel.OUT_DIM)

    # CNN Config
    comptime C2 = MuZeroCNNConfig[3]
    print("CNN: Rep", C2.RepModel.IN_DIM, "->", C2.RepModel.OUT_DIM,
          "| obs_dim:", C2.obs_dim)

    # ResNet Config
    comptime C3 = MuZeroResNetConfig[17, 6, LATENT=128, HIDDEN=128]
    print("ResNet: Rep PARAMS:", C3.RepModel.PARAM_SIZE,
          "| Dyn PARAMS:", C3.DynModel.PARAM_SIZE)

    # Large Config
    comptime C4 = MuZeroLargeConfig[8, 4]
    print("Large: latent:", C4.latent_dim, "bins:", C4.num_bins,
          "sims:", C4.num_simulations)

    # AlphaZero Config
    comptime C5 = AlphaZeroConfig[64, 4]  # Chess-like: 64 obs, 4 actions
    print("AlphaZero: Search.USE_LEARNED_DYNAMICS:", C5.Search.USE_LEARNED_DYNAMICS,
          "| Search.NEEDS_GAME_STATE:", C5.Search.NEEDS_GAME_STATE,
          "| Encoding.IS_DISTRIBUTIONAL:", C5.Encoding.IS_DISTRIBUTIONAL,
          "| PUCT.C_INIT:", C5.PUCT.C_INIT,
          "| Backup.BACKUP_TYPE:", C5.Backup.BACKUP_TYPE,
          "| Players.IS_SELF_PLAY:", C5.Players.IS_SELF_PLAY)

    # EfficientZero Config
    comptime C6 = EfficientZeroConfig[4, 2]
    print("EfficientZero: Backup.LAMBDA:", C6.Backup.LAMBDA,
          "| USE_REANALYZE:", C6.USE_REANALYZE)

    print("PASS: All configs compile (including strategies)")
    print("=== Done ===")
