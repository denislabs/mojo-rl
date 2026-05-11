"""pc_inference — DEPRECATED entry point; inference is now a method on PCTrainer.

The local-rule x-update logic now lives at:
    PCTrainer._inference_step  (Phase C: dx update)
    PCTrainer._forward_eps     (Phases A+B: predict + ε compute)
    PCTrainer._total_energy
    PCTrainer._readout_loss

This file is kept as a stub so the package layout in `docs/PCN_REDESIGN.md`
matches the file tree, but contains no public API. To call inference,
construct a PCTrainer and call its static methods.

Reason for inlining: `PCSequential` is a parametric struct, so `NET: PCSequential`
cannot be used as a parameter constraint. The variadic-pack workaround
(`*BLOCKS: PCBlockTrait`) propagates cleanly only inside a single struct
that uses `comptime NET = PCSequential[*Self.BLOCKS]`. Cross-function
variadic pack passing adds friction without benefit at this scope.
"""
