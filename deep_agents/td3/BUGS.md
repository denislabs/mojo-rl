2. Noise decay (CPU only, MINOR):
  CPU TD3 decays noise_std from 0.1 to 0.01 rapidly. CleanRL uses constant 0.1 exploration noise throughout. This isn't critical but reduces exploration.