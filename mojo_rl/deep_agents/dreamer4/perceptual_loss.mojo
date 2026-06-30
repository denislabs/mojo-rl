"""Perceptual feature loss for the Dreamer 4 tokenizer (LPIPS-style).

The paper's tokenizer objective is `L = L_MSE + 0.2·L_LPIPS` (eq. 5). True LPIPS
needs a pretrained ImageNet AlexNet/VGG + calibrated linear layers, which we do
not have and cannot `pip install` in Mojo. Instead we use a frozen ResNet-20
feature backbone (`CifarBackbone`, CIFAR-trained) as the feature extractor and a
calibration-free **feature-MSE** perceptual term over a single deep feature map
(Johnson et al. perceptual loss). This keeps the LPIPS *structure* — frozen deep
features, squared difference in feature space — while being fully reproducible
inside the repo. Multi-layer + channel-normalization are an easy extension.

    pred/target patches  ── temporal_unpatchify ──▶  images [BT, C_IMG, H, W]
    ── replicate to 3ch ──▶  backbone (frozen, BN-eval)  ──▶  features
    L = mean‖feat_pred − feat_tgt‖²

The gradient flows back to the tokenizer's patch-space output: feature cotangent
→ `backbone.vjp` (input grad) → collapse 3ch → `temporal_patchify` (the adjoint
of the bijective unpatchify). Backbone params are frozen (we `zero_grad` it each
call and never step it); only the input gradient is used.

CPU-only (the tokenizer recon loss runs host-side). Returns the scalar loss and
fills `grad_pred_patches`.
"""

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.models.cifar_feature_net import CifarBackbone
from .patchify import temporal_patchify, temporal_unpatchify
from .shortcut_loss import _mao


def perceptual_feature_loss[
    BT: Int, C_IMG: Int, H: Int, W: Int, PATCH: Int
](
    pred_patches: UnsafePointer[Scalar[DT], MutAnyOrigin],
    target_patches: UnsafePointer[Scalar[DT], MutAnyOrigin],
    mut backbone: CifarBackbone[H, W],
    grad_pred_patches: UnsafePointer[Scalar[DT], MutAnyOrigin],
) raises -> Float64:
    comptime assert C_IMG == 1 or C_IMG == 3, (
        "perceptual_feature_loss: C_IMG must be 1 (grayscale) or 3 (RGB)"
    )
    comptime HW = H * W
    comptime IMG1 = C_IMG * HW          # unpatchified image, per frame
    comptime IMG3 = 3 * HW              # backbone input (3-channel), per frame
    comptime NFEAT = CifarBackbone[H, W].OUT_DIM
    comptime N = BT * NFEAT

    # Run the frozen backbone in BN-EVAL mode: it normalizes with its trained
    # running stats (constants), so the perceptual features are deterministic and
    # batch-independent — and a trained checkpoint's BN stats actually matter.
    # BatchNorm2D's CPU eval-mode vjp (gi = γ·inv_std_running·dy) backs this.
    # Weights stay frozen (we zero grads each call and never step the backbone).
    backbone.set_attr["training"](Scalar[DT](0.0))
    backbone.zero_grad["cpu"](None)

    # 1) patches → images.
    var img_pred = Tensor.alloc(BT * IMG1)
    var img_tgt = Tensor.alloc(BT * IMG1)
    temporal_unpatchify[BT, C_IMG, H, W, PATCH](
        pred_patches, _mao(img_pred.data.unsafe_ptr())
    )
    temporal_unpatchify[BT, C_IMG, H, W, PATCH](
        target_patches, _mao(img_tgt.data.unsafe_ptr())
    )

    # 2) replicate grayscale → 3 channels (identity if already RGB).
    var img3_pred = Tensor.alloc(BT * IMG3)
    var img3_tgt = Tensor.alloc(BT * IMG3)
    comptime if C_IMG == 3:
        for i in range(BT * IMG3):
            img3_pred.data[i] = img_pred.data[i]
            img3_tgt.data[i] = img_tgt.data[i]
    else:
        for bt in range(BT):
            for k in range(3):
                for i in range(HW):
                    img3_pred.data[bt * IMG3 + k * HW + i] = img_pred.data[
                        bt * HW + i
                    ]
                    img3_tgt.data[bt * IMG3 + k * HW + i] = img_tgt.data[
                        bt * HW + i
                    ]

    # 3) frozen backbone features. Each forward overwrites the per-layer cache,
    # so run TARGET first and PRED last → the cache the vjp (step 5) consumes
    # corresponds to `img3_pred`.
    var feat_pred = Tensor.alloc(N)
    var feat_tgt = Tensor.alloc(N)
    backbone.forward["cpu", BT](TensorRefs[1](img3_tgt), feat_tgt, None)
    backbone.forward["cpu", BT](TensorRefs[1](img3_pred), feat_pred, None)

    # 4) feature-MSE + cotangent.
    var grad_feat = Tensor.alloc(N)
    var loss: Float64 = 0.0
    var inv_n = 1.0 / Float64(N)
    for i in range(N):
        var diff = Float64(feat_pred.data[i]) - Float64(feat_tgt.data[i])
        loss += diff * diff
        grad_feat.data[i] = Scalar[DT](2.0 * diff * inv_n)
    loss *= inv_n

    # 5) backbone vjp → input (image) gradient.
    var grad_img3 = Tensor.alloc(BT * IMG3)
    backbone.vjp["cpu", BT](
        TensorRefs[1](img3_pred), grad_feat, TensorRefs[1](grad_img3), None
    )

    # 6) collapse 3ch grad → C_IMG (adjoint of step 2: sum the replicas).
    var grad_img = Tensor.alloc(BT * IMG1)
    comptime if C_IMG == 3:
        for i in range(BT * IMG3):
            grad_img.data[i] = grad_img3.data[i]
    else:
        for bt in range(BT):
            for i in range(HW):
                var s = (
                    Float64(grad_img3.data[bt * IMG3 + 0 * HW + i])
                    + Float64(grad_img3.data[bt * IMG3 + 1 * HW + i])
                    + Float64(grad_img3.data[bt * IMG3 + 2 * HW + i])
                )
                grad_img.data[bt * HW + i] = Scalar[DT](s)

    # 7) image grad → patch grad (adjoint of unpatchify == patchify).
    temporal_patchify[BT, C_IMG, H, W, PATCH](
        _mao(grad_img.data.unsafe_ptr()), grad_pred_patches
    )
    return loss
