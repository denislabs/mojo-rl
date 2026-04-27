# `nn_pc/` — Predictive Coding Networks (Rao-Ballard family)

**Status (2026-04-27): closed POC.** Implementation is correct and validated; the paper that motivated the work has a misleading headline claim. Kept here as a reference implementation, not a production component.

## What this is

A Mojo implementation of supervised Predictive Coding Networks following the **Rao-Ballard / Whittington-Bogacz** lineage:

- **Latent state** lives between layers and is updated by iterative inference (Algorithm 2 in arxiv 2506.06332).
- **Weights** are updated by local Hebbian-like rules using gain-modulated prediction errors.
- **No backprop, no autograd.** Per batch the algorithm alternates: T_infer iterations of latent updates with weights frozen, then T_learn iterations of weight updates with latents frozen.

This is *not* the same algorithm as more recent "PCN-flavored" approaches (e.g., variational predictive modules in deep RL, world models with predictive objectives) — see "Other PCN flavors" below.

## Why it lives in `experimental/`

The originally-targeted result (arxiv 2506.06332, Monadillo) — "99.92% top-1 on CIFAR-10, beating ViT-H/14" — turns out to be a **label-leakage artifact** in the test protocol (`test_pcn` feeds true labels into inference dynamics; their paper's §5.3 disclosure is technically correct but the leaderboard comparison in §5.5 doesn't flag the protocol mismatch). The honest CIFAR-10 number for this architecture is **~12-19% top-1**, independently corroborated by GitHub-issue replications.

A backprop MLP on similar capacity (3.58M params) reaches **~52% top-1** on the same task. So PCN-MLP is roughly 3× worse than BP-MLP at this scale — it's a slower, weaker training algorithm dressed up with a biological-plausibility story.

That said, the engineering is clean and reusable for other PCN research (different inference rules, convolutional layers, biologically-plausible learning experiments). Hence: kept, but quarantined under `experimental/`.

## Architecture

```
predictive_model.mojo   PCActivation trait + PCReLU, PCIdentity
                        PCLayer trait
model/
  pc_linear.mojo        PCLinear[in, out, ACT=PCReLU]
                        CPU + tiled-2x2 GPU (Apple) + MMA GPU (NVIDIA)
  pc_sequential.mojo    PCSequential[*LAYERS: PCLayer]
trainer/
  pc_trainer.mojo       PCTrainer[*LAYERS: PCLayer, dtype]
                        - train_one_batch (CPU)
                        - train_one_batch_gpu
                        - inference_gpu  (free PCN inference, eps_L=0)
                        - supervised_inference_gpu  (paper protocol, label-driven)
```

### Convention

`PCLinear[in_dim, out_dim, ACT=PCReLU]` matches `nn.Linear[in, out]`:
- W stored `[in_dim, out_dim]`.
- `in_dim` = predicted dim (lower in PCN hierarchy / "input" side in feedforward).
- `out_dim` = latent-above dim ("output" side in feedforward).
- Predict goes top-down via W^T: `x_hat = ACT(x_above @ W^T)`.

The activation is bundled into the layer (comptime param). PCReLU on hidden levels, PCIdentity on the readout. The last `PCLinear` in a `PCSequential` IS the readout — it must use `PCIdentity`.

### Sign conventions worth remembering

- Non-readout: `eps_l = x_above - x_hat` ; weight grad `W_l += +eta/B * (h_l^T @ x_above)`.
- Readout: `eps_sup = y_hat - y_target` (OPPOSITE sign) ; weight grad `W_R += -eta/B * (h_R^T @ x_above)`.
- The trainer's `weight_grad_step` takes a signed `scale = ±eta/B` and the sign is set per layer.
- Top latent x^(L) has no self-prediction-error; it's updated by `eps_L = h_R @ W_R` (supervised pull-back).

## Validation

| Test | What it checks | Result |
|---|---|---|
| [`tests/nn_pc/test_pc_smoke.mojo`](../../../tests/nn_pc/test_pc_smoke.mojo) | 3-layer convergence on synthetic data | sup_loss 0.22 → 0.03 over 8 batches |
| [`tests/nn_pc/test_pc_step_cpu.mojo`](../../../tests/nn_pc/test_pc_step_cpu.mojo) | Hand-verified one inference + one learning step | matches Float64 hand computation to ~1e-8 |
| [`tests/nn_pc/test_pc_cpu_vs_gpu.mojo`](../../../tests/nn_pc/test_pc_cpu_vs_gpu.mojo) | CPU vs GPU equivalence | bitwise (Apple, max err 0.0); 1e-3 (NVIDIA — MMA reorders sums) |
| [`tests/nn_pc/test_pc_mnist.mojo`](../../../tests/nn_pc/test_pc_mnist.mojo) | MNIST 1 epoch end-to-end | 47% test / 59% train (free inference) |
| [`tests/nn_pc/test_pc_cifar10.mojo`](../../../tests/nn_pc/test_pc_cifar10.mojo) | Small-budget CIFAR (paper arch, 20 batches) | algorithm runs at 3.58M-param scale |
| [`tests/nn_pc/test_pc_cifar10_paper.mojo`](../../../tests/nn_pc/test_pc_cifar10_paper.mojo) | Full paper hyperparams + diagnostics, both protocols | see "Final benchmark" below |
| [`tests/nn_pc/test_bp_mlp_cifar10_baseline.mojo`](../../../tests/nn_pc/test_bp_mlp_cifar10_baseline.mojo) | Backprop MLP on equivalent capacity | 51.79% top-1 |

## Final benchmark — CIFAR-10, ~3.58M params

| Setup | Top-1 | Top-3 | Notes |
|---|---|---|---|
| **PCN paper headline** (supervised inference) | 99.92% | 99.99% | Label-leak protocol; not comparable to standard test acc |
| **PCN free inference** (honest) | 11.6 – 19% | ~33% | Our run + GitHub-issue replications agree |
| **BP MLP, paper's exact 4-layer arch** | ~10% | ~30% | Degenerate — the 10-dim ReLU bottleneck dies at init |
| **BP MLP, 3-layer (no bottleneck)** | **51.79%** | **80.58%** | Adam, 30 epochs, ~4 s on NVIDIA |

**Honest interpretation**: at this capacity the PCN-MLP training algorithm is ~3× weaker than backprop. The paper's chosen architecture is also pathological — even backprop can't train it because the bottleneck has no top-down driving force.

## When to use this code

- Reproducing or extending PCN experiments (different inference rules, latent-state initialization schemes, convolutional PCN, etc.)
- Reference implementation for biologically-plausible learning research
- Pedagogical demo of energy-based latent inference + Hebbian learning

## When **not** to use this code

- Chasing benchmark accuracy on standard ML tasks — backprop wins handily on equivalent capacity.
- Building production agents — the algorithm is slower per batch (T_infer + T_learn inner loops) and gives weaker representations on what we tested.

## GPU performance notes

- CPU: naive (one thread per output element).
- GPU (Apple): register-tiled 2×2 (32×32 block, 16-elem K tile) — modeled after `mojo_rl/nn/autodiff/primitives/matmul.mojo`. Bitwise-matches CPU.
- GPU (NVIDIA): MMA m16n8k8 tensor cores, ~150× over Apple naive at MNIST scale, ~86× at CIFAR scale.
- Further wins would need fusion (combine `predict + activation + sub + gain_mod` per layer) or CUDA Graph capture. Not pursued — algorithm validation was the goal.

## Other PCN flavors (if you want to keep exploring)

- **Variational predictive modules in meta-RL** — Kuo, Hou, Dabney, Walker, NeurIPS 2025 (arxiv 2510.22039). VAE-style encoder + decoders + ELBO regularization, integrated with RL² for partial-observability tasks. A much more careful paper. Almost zero overlap with this code — would need RNN encoder, VAE primitives, and meta-RL integration as new work.
- **Hybrid predictive coding** — Tschantz et al. 2023 (PLOS Comp Bio). Amortized feedforward init for latents to skip the slow iterative settling. Could be a small addition on top of this framework.
- **iPC / stable PCN** — Salvatori et al. 2024 (ICLR). Claims better convergence properties.

## Mojo gotchas to remember

- `MODEL: PCSequential` doesn't work as a struct-parameter constraint (PCSequential is parametric). Workaround used here: `*LAYERS: PCLayer` variadic + `comptime MODEL = PCSequential[*Self.LAYERS]`.
- LayoutTensor indexing returns `SIMD[dtype, k]` not Scalar — arithmetic needs `rebind[Scalar[dtype]](tensor[i, j])`.
- Struct-parameter access inside body must use `Self.in_dim`, not bare `in_dim`. Same for `Self.ACT`.
- Multi-step iterative numerics: NVIDIA MMA reorders reductions vs CPU sequential; 1e-3 tolerance not bitwise.
