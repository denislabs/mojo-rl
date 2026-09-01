# +--------------------------------------------------------------------------+ #
# | ACT — dimensions and hyperparameters
# +--------------------------------------------------------------------------+ #
"""Compile-time shape/hyperparameter aliases for ACT on the SO-ARM101.

The reference (`references/act-main/imitate_episodes.py:52`,
`detr/main.py:get_args_parser`) hardcodes ALOHA's bimanual 14-DoF state and a
480x640 4-camera rig. Everything that was a constant there is a parameter here,
because the SO-101 is 6-DoF with two cameras.

Reference values are quoted beside each alias so a divergence is visible
without opening the Python.
"""


# ── SO-ARM101 / dataset shape ────────────────────────────────────────────
# LeRobot v3 `DenisLabs/record-test_20260825_094319`, robot_type `so_follower`.
# `observation.state` and `action` are both 6-vectors in lerobot units:
# 5 body joints in DEGREES + gripper in 0..100 (see mojo_rl/robot/so101/arm.mojo).
comptime SO101_QPOS: Int = 6
comptime SO101_ADIM: Int = 6
comptime SO101_N_CAM: Int = 2  # observation.images.{front,side}
comptime SO101_FPS: Int = 30

# Working resolution. The converter resamples; the paper's rig is 480x640,
# which a ResNet18 takes to 15x20 = 300 tokens per camera (602 memory tokens
# at 2 cameras). 240x320 -> 8x10 = 80 per camera (162 memory tokens) is the
# CPU-gate and first-training default.
comptime SO101_IMG_H: Int = 240
comptime SO101_IMG_W: Int = 320


# ── ACT hyperparameters ──────────────────────────────────────────────────
# `imitate_episodes.py:53` (enc_layers 4, dec_layers 7, nheads 8) and
# `detr/main.py` (hidden_dim 512 / dim_feedforward 3200 as passed by ACT's
# own README command line; the argparse DEFAULTS of 256/2048 are overridden).
comptime ACT_CHUNK: Int = 100  # k — `--chunk_size 100`
comptime ACT_HIDDEN: Int = 512  # `--hidden_dim 512`
comptime ACT_FF: Int = 3200  # `--dim_feedforward 3200`
comptime ACT_HEADS: Int = 8  # `nheads = 8`
comptime ACT_ENC_LAYERS: Int = 4  # `enc_layers = 4` (BOTH the CVAE encoder
#                                    and the transformer encoder; two separate
#                                    stacks with the same depth)
comptime ACT_LATENT: Int = 32  # `detr_vae.py:69  self.latent_dim = 32`
comptime ACT_DROPOUT: Float64 = 0.1  # `--dropout 0.1`

# ⚠ dec_layers. `detr_vae.py:139` reads `self.transformer(...)[0]`, and
# `build_transformer` passes `return_intermediate_dec=True`, so the decoder
# returns a LAYER-indexed stack and `[0]` selects the FIRST decoder layer's
# output. Layers 2..7 of the reference receive no gradient and cannot affect
# the prediction — the official `dec_layers=7` model is output-equivalent to
# `dec_layers=1`. We default to 1 (bit-equivalent, 7x cheaper) and expose the
# corrected variant (LeRobot's reading: take the LAST layer) via
# `ACT_USE_LAST_HS = True` with `ACT_DEC_LAYERS = 7`.
comptime ACT_DEC_LAYERS: Int = 1
comptime ACT_USE_LAST_HS: Bool = False

comptime ACT_KL_WEIGHT: Float64 = 10.0  # `--kl_weight 10`
comptime ACT_LR: Float64 = 1e-5  # `--lr 1e-5`
comptime ACT_WEIGHT_DECAY: Float64 = 1e-4  # `detr/main.py --weight_decay`
comptime ACT_BATCH: Int = 8  # `--batch_size 8`
comptime ACT_EPOCHS: Int = 2000  # `--num_epochs 2000` (README: real-world
#                                   data wants 5000+, or 3-4x past plateau)
comptime ACT_CLIP_MAX_NORM: Float64 = 0.1  # `detr/main.py --clip_max_norm`
# ⚠ the reference PARSES clip_max_norm and never applies it (`imitate_episodes.
# py:forward_pass` has no clip). We do clip; recorded as a deviation, since an
# unclipped from-scratch ResNet on 4 episodes is a divergence waiting to happen.
# ⚠ DEVIATION: the reference gives backbone params `lr_backbone = 1e-5` in a
# second AdamW param group. Our `Adam.step` takes a `Module` and has no name
# filter; and with a FROM-SCRATCH backbone a lower backbone lr would freeze the
# vision tower rather than gently fine-tune it. One lr for everything.

# ── the configuration the SO-101 examples actually run ───────────────────
# SEPARATE FROM THE PAPER CONSTANTS ABOVE, and shared BECAUSE it drifted: the
# training example and the open-loop evaluation each carried their own copy,
# the training run moved to K=60/dim=256 and the evaluation stayed at the CPU
# smoke's K=20/dim=64. A checkpoint written by the run the trainer tells you to
# evaluate could not be loaded by the evaluator. Parameter shapes are the one
# thing two programs must agree on exactly, so they read it from one place.
#
# These are NOT `ACT_*` above. Deviations and why, in `docs/ACT_PORT.md`:
#   RUN_K = 60 is the paper's TWO-SECOND horizon at 30 fps, not its frame
#   count (ALOHA is 50 Hz x 100 steps). RUN_DIM/RUN_FF are below the paper's
#   512/3200 because compile time bounds the graph type, not accuracy.

comptime RUN_K: Int = 60
comptime RUN_DIM: Int = 256
comptime RUN_HEADS: Int = 8
comptime RUN_FF: Int = 1024
comptime RUN_LATENT: Int = ACT_LATENT
comptime RUN_ENC_LAYERS: Int = 4
comptime RUN_DEC_LAYERS: Int = ACT_DEC_LAYERS
comptime RUN_LR: Float64 = 1e-5
"""Both references: 1e-5, for the model AND the backbone (`act-main`'s
`detr/main.py`, LeRobot's `optimizer_lr` / `optimizer_lr_backbone`).

This was 1e-4, justified as "the backbone is random at step 0 and has to learn
vision from scratch". That stopped being true the moment `ACT_PRETRAINED`
started loading ImageNet weights, and the justification outlived the condition
by one run: with a pretrained backbone the first 50-episode run reached the
old floor 12x sooner but only 3.3% lower, which is what a rate that washes the
pretraining out in the first few hundred steps would look like.

⚠ Our single learning rate remains a deviation in FORM — the references put the
backbone in its own parameter group — but no longer in VALUE, since both set
that group to 1e-5 as well. `Adam` here has no name filter; a filtered
`ParamVisitor` is what a second group would need."""


comptime ACT_TEMPORAL_ENSEMBLE_M: Float64 = 0.01
"""`k = 0.01` in `imitate_episodes.py:253` — `w_i = exp(-k*i)`, i indexing
oldest-first. Called `m` in Algorithm 2 of the paper."""


# ── normalization ────────────────────────────────────────────────────────
# `policy.py:22` — torchvision ImageNet statistics, applied AFTER /255.
comptime IMAGENET_MEAN_R: Float64 = 0.485
comptime IMAGENET_MEAN_G: Float64 = 0.456
comptime IMAGENET_MEAN_B: Float64 = 0.406
comptime IMAGENET_STD_R: Float64 = 0.229
comptime IMAGENET_STD_G: Float64 = 0.224
comptime IMAGENET_STD_B: Float64 = 0.225

comptime NORM_STD_FLOOR: Float64 = 1e-2
"""`utils.py:96` — `torch.clip(std, 1e-2, inf)`. A joint that never moves in
the demonstrations would otherwise divide by ~0 and produce inf features."""

comptime TRAIN_SPLIT_RATIO: Float64 = 0.8
"""`utils.py:112`. With 5 episodes: 4 train, 1 validation."""


# ── the pretrained backbone, resolved in ONE place ───────────────────────

comptime ACT_PRETRAINED_DEFAULT = "hub"
"""What `ACT_PRETRAINED` means when it is not set.

⚠ **THE DEFAULT IS THE PRETRAINED BACKBONE, NOT A RANDOM ONE.** It used to be
random, and the failure mode was silent: a run launched without the variable
trained a from-scratch ResNet18 on ~12,000 frames and produced a curve that
looked like a bad hyperparameter rather than a missing download. The paper's
backbone is `resnet18(weights=IMAGENET1K_V1)`; that should be what you get by
default, and opting OUT should be the deliberate act.

The fetch costs 47 MB once and is cached, so the default is not expensive —
it is just no longer forgettable."""


def act_pretrained_spec() raises -> String:
    """`ACT_PRETRAINED`, resolved. Empty means "use a random backbone".

    ⚠ ONE RULE, ONE PLACE. `act_so101_train_cpu.mojo` and `..._gpu.mojo` both
    read this variable and both print what they got; a default spelled out at
    two call sites is this repo's most frequent defect shape, and the two
    disagreeing would mean the CPU and GPU runs silently trained different
    models.

        unset                   -> "hub"  (ImageNet, no PyTorch)
        "random" | "none" | "0" -> ""     (random backbone, deliberately)
        anything else           -> itself, for `load_backbone_auto`
    """
    from std.os import getenv

    var v = getenv("ACT_PRETRAINED")
    if v.byte_length() == 0:
        return String(ACT_PRETRAINED_DEFAULT)
    if v == "random" or v == "none" or v == "0":
        return String("")
    return v^
