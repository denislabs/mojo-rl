"""Tree search model contract — Representation / Dynamics / Prediction.

The MCTS algorithm is agent-agnostic; everything the search needs from the
underlying model is funneled through three small traits. Concrete adapters
live next to the agent that owns the networks (MuZero / EZv2 / AlphaZero)
and translate between MCTS's plain-``Float64`` view of the world and the
agent's typed network state (param tensors, model state, categorical
encoding, hidden-state scaling).

Phase 3 of the planners refactor only ships the CPU-side surface. The GPU
counterpart (``RepresentationGPU`` etc.) lands when ``mcts_gpu.mojo`` is
promoted. Keeping the CPU contract simple now means stub world models for
tests stay trivial: ``TwoArmBandit`` and ``KnownValueTree`` get away with a
handful of lines each. See ``tests/planners/tree_search/`` for examples.

**Plain Float64 throughout** — the trait uses ``List[Float64]`` not
``Scalar[dtype]``. Real adapters convert at the boundary; for stubs the
List interface is far less ceremony than a LayoutTensor + dtype cast.
This matches the convention already in ``RolloutCallbackCPU``.

**Adapters own encoding/decoding/scaling.** The trait returns interpretable
quantities (softmaxed probabilities, decoded scalar value/reward, scaled
hidden state). Anything categorical / symlog / MinMax-scaled belongs in the
adapter, not in the MCTS loop. This keeps MCTS independent of
``ValueEncoding`` and ``HiddenScaling`` strategies — those choices live with
the agent that owns the network.

Three traits, not one, because:

* Tests want to mock one without the others (``KnownValueTree`` mocks the
  value-tree as a pure prediction stub with no learned dynamics).
* AlphaZero's "representation" is identity (obs == game state); its
  dynamics is the env's ``step``. MuZero's all three are neural nets.
  Keeping the contracts split lets each adapter only implement what it
  needs.

Future GPU variants will mirror these names with ``_gpu`` suffixes and
``LayoutTensor`` views, matching the ``RolloutCallback{CPU,GPU}`` pattern
in ``trajectory/rollout_callback.mojo``.
"""


trait Representation(ImplicitlyDestructible):
    """Encode an observation into the root hidden state.

    Called once per ``search()`` at the root. The adapter is responsible
    for any state-norm/scaling the agent needs (e.g. MuZero's MinMax
    scaling of the encoder output) — MCTS just reads what comes back.

    ``OBS_DIM`` / ``LATENT_DIM`` are comptime so the planner and adapter
    agree on shape at compile time. Lists passed must have exactly these
    lengths; implementors should not resize.
    """

    comptime OBS_DIM: Int
    comptime LATENT_DIM: Int

    def encode_cpu(
        mut self,
        obs: List[Float64],
        mut hidden_out: List[Float64],
    ) raises:
        """Write the root hidden state into ``hidden_out`` from ``obs``.

        ``obs`` has length ``OBS_DIM``; ``hidden_out`` length ``LATENT_DIM``.
        Adapter handles any hidden-state scaling internally so MCTS reads
        a fully-prepared state.
        """
        ...


trait Dynamics(ImplicitlyDestructible):
    """Apply one MCTS expansion step: (hidden, action) → (hidden', reward).

    Called once per leaf expansion. The adapter is responsible for action
    encoding (e.g. one-hot for MuZero discrete dynamics), reward decoding
    (categorical → scalar), and hidden-state scaling of the output.

    Single-action signature (not batched) matches MuZero's per-simulation
    CPU expansion. A batched variant lives on the future GPU trait.
    """

    comptime LATENT_DIM: Int
    comptime ACTION_DIM: Int

    def step_cpu(
        mut self,
        hidden_in: List[Float64],
        action: Int,
        mut hidden_out: List[Float64],
    ) raises -> Float64:
        """Advance one step. Return the scalar reward at this edge.

        ``hidden_in`` and ``hidden_out`` have length ``LATENT_DIM``;
        ``action`` is in ``[0, ACTION_DIM)``. Adapter applies any
        hidden-state scaling to ``hidden_out`` before returning, and
        decodes the categorical reward head (where applicable) to a
        scalar return value.
        """
        ...


trait Prediction(ImplicitlyDestructible):
    """Compute policy prior + scalar value at a hidden state.

    Called once per leaf expansion (after dynamics) and once at the root
    (right after representation). Adapter is responsible for softmaxing
    the policy logits and decoding the value head from whatever encoding
    the agent uses.

    Returning probabilities (sum to 1) rather than logits keeps MCTS
    independent of the encoding choice. Stubs without a real policy can
    write uniform probabilities; stubs without a real value head can
    return ``0.0``.
    """

    comptime LATENT_DIM: Int
    comptime ACTION_DIM: Int

    def predict_cpu(
        mut self,
        hidden: List[Float64],
        mut policy_out: List[Float64],
    ) raises -> Float64:
        """Write the per-action prior into ``policy_out`` and return the
        scalar value prediction at ``hidden``.

        ``hidden`` length ``LATENT_DIM``; ``policy_out`` length
        ``ACTION_DIM``. Probabilities must be non-negative and sum to 1
        (no renormalization happens inside MCTS — the root prior is
        re-normalized after legal-mask + Dirichlet noise, but child
        priors are taken verbatim, matching reference MuZero).
        """
        ...
