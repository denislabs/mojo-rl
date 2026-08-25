# +--------------------------------------------------------------------------+ #
# | mojo-rl robot — the REAL hardware side
# +--------------------------------------------------------------------------+ #
"""Drivers for physical robots.

The counterpart of `mojo_rl/physics3d/`, not a part of it: `physics3d` is the
simulator, `robot` is the machine on the desk, and the seam between them is an
`ObsState` / `ContAction` pair. Keeping them apart is what makes "trained in
sim, deployed on hardware, same code either side" a statement anyone can
check.

- ``feetech``: SCS/STS bus-servo protocol (packet codec + bus).
- ``so101``:   the SO-ARM101 leader/follower pair built on it.

⚠ Both need the serial shim: `pixi run build-serial`.
"""
