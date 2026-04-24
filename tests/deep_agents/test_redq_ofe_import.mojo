"""Smoke test: REDQ-OFE agent imports.

Starts out failing (agent partially wired); as we fill in the training
loop, failures narrow toward the real issues.
"""

from mojo_rl.deep_agents.redq_ofe import (
    REDQOFEConfig,
    DefaultREDQOFEConfig6,
    DefaultREDQOFEConfig8,
)
from mojo_rl.deep_agents.redq_ofe.redq_ofe import REDQOFEAgent, REDQOFEGPUState


def main() raises:
    print("REDQOFEAgent and REDQOFEGPUState resolved.")
