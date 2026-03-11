1. Clean up nn/ — unify the two systems

  The autodiff layers (Dense, DenseReLU, DenseTanh, etc.) already implement Model and add automatic differentiation on top. The
  manual layers (Linear, LinearReLU, etc.) are older duplicates with hand-written backward passes. You could:

  - Keep autodiff as the canonical implementation — it's more general and already conforms to Model
  - Alias or deprecate the manual layers — e.g. comptime Linear = Dense or phase them out
  - Move Sequential, Parallel, Residual to nn/model/ since they're general-purpose combinators, not autodiff-specific

  2. Then the SAC actor becomes clean immediately

  comptime ActorModel = Sequential[
      DenseReLU[OBS, HIDDEN],
      DenseReLU[HIDDEN, HIDDEN],
      Parallel[
          Dense[HIDDEN, ACTIONS],  # mean head
          Dense[HIDDEN, ACTIONS],  # log_std head
      ],
  ]

  No new code needed — just using what already exists in autodiff.

  3. Optional: GaussianHead to eliminate duplicated clamping

  A thin Model wrapping the tanh-scaling + clamping of log_std that's currently copy-pasted 3 times in sac.mojo. But this is
  secondary to the cleanup.

  Want me to start with an inventory of what overlaps between nn/model/ and nn/autodiff/ to plan the consolidation?