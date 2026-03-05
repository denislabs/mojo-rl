  Potential Issues Found (not definitive bugs)

  1. RunningMeanStd.normalize subtracts mean ((r - mean) / std instead of just r / std). This differs from CleanRL's reward normalization which only divides by std. As running_mean grows, rewards
   are systematically shifted lower. However: the same bug exists in ppo_continuous_old.mojo, so it can't be the cause of the regression.
  2. value_loss_coef=0.5 not applied in GPU critic gradient — critic learns 2× faster than expected. Again, same in old code.


