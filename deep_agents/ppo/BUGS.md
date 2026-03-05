  Potential Issues Found (not definitive bugs)

  1. RunningMeanStd.normalize subtracts mean ((r - mean) / std instead of just r / std). This differs from CleanRL's reward normalization which only divides by std. As running_mean grows, rewards
   are systematically shifted lower. However: the same bug exists in ppo_continuous_old.mojo, so it can't be the cause of the regression.
  2. value_loss_coef=0.5 not applied in GPU critic gradient — critic learns 2× faster than expected. Again, same in old code.




  The logs are very revealing. Let me analyze them carefully.                                                                                                                                      
                                                                                                                                                                                                   
  Critical finding — per-minibatch normalization is BROKEN in both files:                                                                                                                          
                                                                                                                                                                                                   
  ┌─────────┬──────┬───────────────┬────────────────┬──────────────┐                                                                                                                               
  │ Rollout │ File │ PRE-NORM mean │ MB gather mean │    Shift     │
  ├─────────┼──────┼───────────────┼────────────────┼──────────────┤
  │ 0       │ new  │ +0.027        │ +1.27          │ +1.24        │
  ├─────────┼──────┼───────────────┼────────────────┼──────────────┤
  │ 1       │ new  │ +0.043        │ +1.12          │ +1.08        │
  ├─────────┼──────┼───────────────┼────────────────┼──────────────┤
  │ 2       │ new  │ -0.023        │ -2.15          │ -2.13        │
  ├─────────┼──────┼───────────────┼────────────────┼──────────────┤
  │ 3       │ new  │ +0.010        │ +9.90          │ +9.89 ← HUGE │
  ├─────────┼──────┼───────────────┼────────────────┼──────────────┤
  │ 1       │ old  │ +0.017        │ -1.29          │ -1.31        │
  ├─────────┼──────┼───────────────┼────────────────┼──────────────┤
  │ 2       │ old  │ +0.061        │ -9.60          │ -9.66 ← HUGE │
  └─────────┴──────┴───────────────┴────────────────┴──────────────┘

  The PRE-NORM block confirms the global normalization works (mean≈0). But after normalize_advantages_fused_kernel runs, the mean gets a large random offset instead of staying at 0. The kernel is
   computing a wrong mean internally — the output should have mean=0 but instead gets corrupted.

  Both files have normalize_adv=True + norm_adv_per_mb=True. Global norm gives mean≈0, then per-mb norm reads that already-normalized data and produces garbage.

  Why the new file degrades but old doesn't:

  Looking at mean0 avg (actor output mean) progression:
  - New: -0.006 → -0.238 → -0.304 → -0.287 → -0.226 ← drifting negative
  - Old: -0.015 → +0.011 → +0.021 → +0.023 → +0.029 ← stable near 0

  The new file's corrupted advantages have a positive bias on average (+1.24, +1.08, −2.13, +9.89, −1.55 → avg ≈ +1.7), while the old file's have a negative bias (−1.31, −9.66, −3.52, +1.68,
  −4.14 → avg ≈ −2.8). Pure sampling variance with 5 points, but the systematic positive shift in the new file is reinforcing suboptimal negative actions.

  The real bug to fix: normalize_advantages_fused_kernel produces wrong mean.

  The simplest test — try setting norm_adv_per_minibatch=False (keep only global normalization). The PRE-NORM values are already correct (mean≈0, range≈[−2,+2]). Per-mb norm on top is actively
  corrupting them. If the new file works with only global norm, that confirms this is the sole bug.

  There's also a second suspect: in the new file, bootstrap_mean is erratic (0.40, 0.62, 3.83, −0.12, 7.87) while old is stable (1.66, 2.52, 5.04, 1.76, 2.44). Volatile bootstrap values → bad GAE
   deltas → corrupted advantages even before per-mb norm. This could be a separate bug in how the bootstrap value is stored/retrieved in the new architecture.

  Immediate action plan:
  1. Disable norm_adv_per_minibatch and rerun — does the new file stabilize?
  2. If not: add a log of bootstrap_values_host[0] in the new file's select_actions_with_meta_gpu to verify the final critic values are correct before GAE

  Good — with norm_adv_per_minibatch=False, PRE-NORM and MB gather now match perfectly, confirming global normalization is fine. The bug is deeper.

  Smoking gun: actor mean drifts -0.27 in just ONE rollout (rollout 0 → rollout 1: -0.006 → -0.276). That's catastrophically fast. Old file drifts only ~0.01 per rollout.                         
   
  Also, raw GAE adv mean swings wildly: -1.15, -8.80, -5.41, +3.30, +9.86 — the critic estimates are unstable.  