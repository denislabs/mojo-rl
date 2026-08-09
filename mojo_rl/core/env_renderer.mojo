"""Environment Renderer Trait using ref for safe borrowing.

This module defines a modular renderer design where:
1. Renderers are separate from environments (clean separation of concerns)
2. Renderers access env state via ref (safe, no copying, lifetime-tracked)
3. Type safety: EnvRenderer[E] ensures renderer matches env type

Usage:
    # Create renderer externally
    var renderer = HalfCheetahRenderer(width=1024, height=576)
    renderer.init()

    # In evaluation loop, pass renderer by ref
    env.render(renderer)  # renderer borrows env state via ref

    # Or call renderer directly with env ref
    renderer.render(env)

Benefits over Pointer approach:
- Type safety at compile time
- Lifetime tracking (renderer can't outlive env)
- No raw pointer dereferencing
- Cleaner API

Design Pattern:
    trait EnvRenderer[E: Env]:
        def render(mut self, ref env: E) raises -> None

    struct MyEnv:
        def render[R: EnvRenderer[Self]](self, ref renderer: R) raises -> None:
            renderer.render(self)
"""


trait EnvRenderer:
    """Base trait for all environment renderers.

    Renderers implement this trait to provide visualization for environments.
    The key method is `render_env` which takes a reference to the environment
    state and draws it.

    Using `ref` instead of Pointer provides:
    - Compile-time lifetime checking
    - No unsafe pointer dereferencing
    - Parametric mutability (can work with mut or immut refs)
    """

    def init(mut self) raises -> None:
        """Initialize the renderer (create window, load resources, etc.)."""
        ...

    def close(mut self) raises -> None:
        """Close the renderer and release resources."""
        ...

    def check_quit(mut self) -> Bool:
        """Check if user requested quit (e.g., closed window).

        Returns:
            True if quit was requested.
        """
        ...

    def is_open(self) -> Bool:
        """Check if renderer window is still open.

        Returns:
            True if renderer is initialized and window is open.
        """
        ...

    def delay(self, ms: Int) -> None:
        """Delay for specified milliseconds (for frame rate control).

        Args:
            ms: Milliseconds to delay.
        """
        ...


trait EnvRenderer3D(EnvRenderer):
    """Extended trait for 3D environment renderers.

    Adds 3D-specific methods like camera control.
    """

    def orbit_camera(
        mut self, delta_theta: Float64, delta_phi: Float64
    ) -> None:
        """Orbit camera around target.

        Args:
            delta_theta: Horizontal rotation in radians.
            delta_phi: Vertical rotation in radians.
        """
        ...

    def zoom_camera(mut self, delta: Float64) -> None:
        """Zoom camera in/out.

        Args:
            delta: Zoom amount (positive = zoom in).
        """
        ...


# =============================================================================
# Helper type for optional rendering
# =============================================================================


struct NoRenderer(EnvRenderer):
    """A no-op renderer for when rendering is disabled.

    This implements EnvRenderer but does nothing, allowing code to
    work uniformly whether rendering is enabled or not.
    """

    def __init__(out self):
        """Create a no-op renderer."""
        pass

    def init(mut self) raises -> None:
        """No-op init."""
        pass

    def close(mut self) raises -> None:
        """No-op close."""
        pass

    def check_quit(mut self) -> Bool:
        """Never quit."""
        return False

    def is_open(self) -> Bool:
        """Always reports as open (since there's nothing to close)."""
        return True

    def delay(self, ms: Int) -> None:
        """No-op delay."""
        pass
