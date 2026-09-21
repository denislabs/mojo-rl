"""Demonstrations: the `.demo` file, and how demos reach a learner.

Shared infrastructure, not an algorithm. HIL-SERL (`deep_agents/hil_serl`)
mixes demos into SAC's replay; behaviour cloning (`deep_agents/bc`) fits
them directly; DAgger is a recorder mode here — a policy drives until a
human or an expert takes over, and the rows it hands over are flagged
INTERVENED.

    from noeira.deep_agents.demos.file import DemoSet, read_demo_file, write_demo_file

`file.mojo` is the MRLDEMO1 format (version 1). The magic predates the noeira
rename and is kept: recordings on disk carry it
(`tests/deep_agents/test_demo_golden.mojo`).
"""
