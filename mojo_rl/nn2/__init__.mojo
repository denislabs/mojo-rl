"""Stateful NN framework redesign.

See docs/NN2_DESIGN.md for the design. Phase 1 scope:
  - Module + ParamVisitor traits
  - Linear, ReLU, Sequential
  - CrossEntropyLoss
  - Adam
  - CPU MNIST MLP end-to-end

Everything in nn2 is built alongside the existing nn/ — both coexist.
"""
