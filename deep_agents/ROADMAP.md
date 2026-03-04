Looking at the DDPG agent, here are the main refactoring opportunities:                                                                                                                          
                                                                                                                                                                                                   
  1. CPU State struct (mirrors GPU pattern)                                                                                                                                                        
                                                                                                                                                                                                   
  The biggest issue: the agent has ~20 _batch_* scratch buffer fields + networks + replay buffer mixed in with hyperparameters. A DDPGCPUState would clean this up:                                
                                                                                                                                                                                                   
  struct DDPGCPUState[ActorModel, ActorOpt, CriticModel, CriticOpt,                                                                                                                              
                      buffer_capacity, obs_dim, action_dim, batch_size]:
      var actor: NetworkPair[...]
      var critic: NetworkPair[...]
      var buffer: ReplayBuffer[...]
      # All _batch_*, _next_*, _q_*, _actor_*, _d_* scratch fields here

  DeepDDPGAgent would then only hold hyperparameters (gamma, tau, noise_std, etc.) + a cpu_state field. The train_step becomes cpu_state.train_step(gamma, tau).

  2. OffPolicyState trait (like GPUOffPolicyState)

  A trait parallel to GPUOffPolicyState would let the shared CPU training loop (run_offpolicy_continuous_train) work generically with any algorithm's state, rather than calling agent methods that
   just forward to internal state.

  3. Duplicate concat loops in train_step

  train_step has 3 identical manual loops concatenating obs+action (lines 450-458, 485-491, 549-557). These could be a single inline helper — the concat_obs_action_batch utility already exists in
   nn.utils but isn't used in CPU train_step (only GPU uses concat_obs_action_kernel).

  4. Shared checkpoint boilerplate

  save_checkpoint/load_checkpoint are nearly identical across DDPG/TD3/SAC. The metadata key-value parsing pattern could be a shared utility.

  Priority recommendation

  CPUState struct is the highest-value change — it eliminates the bloated __init__ (60+ lines of list.append(0) loops), makes the agent struct readable, and creates a consistent CPU/GPU symmetry.
   The pattern is already proven by DDPGGPUState.