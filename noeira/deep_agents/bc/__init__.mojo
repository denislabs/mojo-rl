"""Behaviour cloning — a policy fitted to demonstrations by regression.

    from noeira.deep_agents.bc.policy import BcNet, BcNorm, load_bc_norm

`policy.mojo` is the network's shape and its input/output normalisation,
written once so the trainer and the driver that runs the checkpoint cannot
drift apart. Generic over the dataset: LIBERO's demos
(`examples/libero/libero_bc_train.mojo`) or a `.demo` recording.
"""
