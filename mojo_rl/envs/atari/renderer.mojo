"""SDL3-based Atari 2600 renderer with keyboard input and video recording.

Displays the emulated Atari frame in a window using SDL3 streaming textures.
Maps keyboard keys to Atari joystick actions for interactive play.
Integrates with VideoRecorder for MP4/GIF capture.

Keyboard mapping:
    Arrow keys  → Joystick directions (UP/DOWN/LEFT/RIGHT)
    Space       → FIRE button
    R           → RESET console switch
    V           → Toggle video recording
    Escape/Q    → Quit

Usage:
    var renderer = AtariRenderer()
    renderer.init()

    while not renderer.should_quit:
        env.step(renderer.current_action)
        renderer.render_frame(env.state)
"""

from std.ffi import c_float, c_int
from std.memory import alloc

from mojo_rl.render.sdl import (
    _null_ptr,
    init,
    quit as sdl_quit,
    InitFlags,
    create_window,
    destroy_window,
    Window,
    WindowFlags,
    create_renderer,
    destroy_renderer,
    Renderer as SDLRenderer,
    Texture,
    TextureAccess,
    PixelFormat,
    create_texture,
    update_texture,
    render_texture,
    destroy_texture,
    set_render_draw_color,
    render_clear,
    render_present,
    render_debug_text,
    poll_event,
    Event,
    EventType,
    KeyboardEvent,
    Keycode,
    delay,
    get_ticks,
    Ptr,
    AnyOrigin,
    FRect,
    Rect,
    Surface,
    render_read_pixels,
    destroy_surface,
    set_texture_scale_mode,
    ScaleMode,
)
from mojo_rl.render.video_recorder import VideoRecorder

from .atari_state import AtariState
from .frame_render import render_frame_bgra, FRAME_BUF_SIZE
from .flags import (
    FRAME_WIDTH,
    FRAME_HEIGHT,
    ACTION_NOOP,
    ACTION_FIRE,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UPFIRE,
    ACTION_DOWNFIRE,
    ACTION_LEFTFIRE,
    ACTION_RIGHTFIRE,
    ACTION_UPLEFTFIRE,
    ACTION_UPRIGHTFIRE,
    ACTION_DOWNLEFTFIRE,
    ACTION_DOWNRIGHTFIRE,
    ACTION_UPRIGHT,
    ACTION_UPLEFT,
    ACTION_DOWNRIGHT,
    ACTION_DOWNLEFT,
    ACTION_RESET,
)

# Default window size: 3x scale (480×630)
comptime DEFAULT_SCALE: Int = 3
comptime WINDOW_WIDTH: Int = FRAME_WIDTH * DEFAULT_SCALE  # 480
comptime WINDOW_HEIGHT: Int = FRAME_HEIGHT * DEFAULT_SCALE  # 630
# HUD bar height
comptime HUD_HEIGHT: Int = 30


struct AtariRenderer(Movable):
    """SDL3 renderer for Atari 2600 emulator output.

    Creates a streaming texture at native resolution (160×210) and
    scales it to the window size using hardware filtering.
    """

    # SDL handles
    var window: Optional[Ptr[Window, MutAnyOrigin]]
    var sdl_renderer: Optional[Ptr[SDLRenderer, MutAnyOrigin]]
    var texture: Optional[Ptr[Texture, MutAnyOrigin]]

    # Pixel buffer (BGRA, 160×210)
    var pixel_buf: UnsafePointer[UInt8, MutAnyOrigin]

    # Display settings
    var screen_width: Int
    var screen_height: Int
    var fps: Int
    var frame_delay: UInt32

    # State
    var initialized: Bool
    var should_quit: Bool
    var last_frame_time: UInt64
    var paused: Bool

    # Input state
    var key_up: Bool
    var key_down: Bool
    var key_left: Bool
    var key_right: Bool
    var key_fire: Bool
    var key_reset: Bool
    var current_action: UInt8

    # Video recording
    var recorder: VideoRecorder
    var recording_counter: Int

    def __init__(
        out self,
        width: Int = WINDOW_WIDTH,
        height: Int = WINDOW_HEIGHT + HUD_HEIGHT,
        fps: Int = 60,
    ):
        self.window = None
        self.sdl_renderer = None
        self.texture = None
        self.pixel_buf = alloc[UInt8](FRAME_BUF_SIZE)

        self.screen_width = width
        self.screen_height = height
        self.fps = fps
        self.frame_delay = UInt32(1000 // fps)

        self.initialized = False
        self.should_quit = False
        self.last_frame_time = 0
        self.paused = False

        self.key_up = False
        self.key_down = False
        self.key_left = False
        self.key_right = False
        self.key_fire = False
        self.key_reset = False
        self.current_action = ACTION_NOOP

        self.recorder = VideoRecorder()
        self.recording_counter = 0

    def __init__(out self, *, deinit take: Self):
        self.window = take.window
        self.sdl_renderer = take.sdl_renderer
        self.texture = take.texture
        self.pixel_buf = take.pixel_buf
        self.screen_width = take.screen_width
        self.screen_height = take.screen_height
        self.fps = take.fps
        self.frame_delay = take.frame_delay
        self.initialized = take.initialized
        self.should_quit = take.should_quit
        self.last_frame_time = take.last_frame_time
        self.paused = take.paused
        self.key_up = take.key_up
        self.key_down = take.key_down
        self.key_left = take.key_left
        self.key_right = take.key_right
        self.key_fire = take.key_fire
        self.key_reset = take.key_reset
        self.current_action = take.current_action
        self.recorder = take.recorder^
        self.recording_counter = take.recording_counter

    def __del__(deinit self):
        if self.initialized:
            self.close()
        self.pixel_buf.free()

    def init_display(mut self) -> Bool:
        """Initialize SDL3 window, renderer, and streaming texture."""
        if self.initialized:
            return True

        try:
            init(InitFlags.INIT_VIDEO)

            var title = String("Atari 2600")
            self.window = create_window(
                title,
                c_int(self.screen_width),
                c_int(self.screen_height),
                WindowFlags(0),
            )

            var name = String("")
            self.sdl_renderer = create_renderer(self.window.value(), name)

            # Create streaming texture at native Atari resolution
            self.texture = create_texture(self.sdl_renderer.value(),
                PixelFormat.PIXELFORMAT_BGRA8888,
                TextureAccess.TEXTUREACCESS_STREAMING,
                c_int(FRAME_WIDTH),
                c_int(FRAME_HEIGHT),
            )

            # Use nearest-neighbor scaling for crisp pixels
            set_texture_scale_mode(self.texture.value(), ScaleMode.SCALEMODE_NEAREST)

            self.initialized = True
            self.last_frame_time = get_ticks()
            return True
        except:
            print("Failed to initialize Atari renderer")
            return False

    def handle_events(mut self) -> Bool:
        """Process SDL events. Returns False if quit requested."""
        var event = Event()

        try:
            while poll_event(Ptr(to=event)):
                var event_type = event[UInt32]

                if EventType(event_type) == EventType.EVENT_QUIT:
                    self.should_quit = True
                    return False

                elif EventType(event_type) == EventType.EVENT_KEY_DOWN:
                    var key_event = event[KeyboardEvent]
                    var key_val = Int(key_event.key)
                    self._handle_key_down(key_val)

                elif EventType(event_type) == EventType.EVENT_KEY_UP:
                    var key_event = event[KeyboardEvent]
                    var key_val = Int(key_event.key)
                    self._handle_key_up(key_val)
        except:
            pass

        # Update current action from key state
        self._update_action()
        return True

    def _handle_key_down(mut self, key: Int):
        """Handle key press."""
        if key == Int(Keycode.SDLK_ESCAPE) or key == Int(Keycode.SDLK_Q):
            self.should_quit = True
        elif key == Int(Keycode.SDLK_UP):
            self.key_up = True
        elif key == Int(Keycode.SDLK_DOWN):
            self.key_down = True
        elif key == Int(Keycode.SDLK_LEFT):
            self.key_left = True
        elif key == Int(Keycode.SDLK_RIGHT):
            self.key_right = True
        elif key == Int(Keycode.SDLK_SPACE):
            self.key_fire = True
        elif key == Int(Keycode.SDLK_R):
            self.key_reset = True
        elif key == Int(Keycode.SDLK_P):
            self.paused = not self.paused
        elif key == Int(Keycode.SDLK_V):
            try:
                if self.recorder.is_recording:
                    self.recorder.stop()
                else:
                    var fname = (
                        "atari_recording_"
                        + String(self.recording_counter)
                        + ".mp4"
                    )
                    self.recorder.start(fname, self.fps)
                    self.recording_counter += 1
            except:
                pass

    def _handle_key_up(mut self, key: Int):
        """Handle key release."""
        if key == Int(Keycode.SDLK_UP):
            self.key_up = False
        elif key == Int(Keycode.SDLK_DOWN):
            self.key_down = False
        elif key == Int(Keycode.SDLK_LEFT):
            self.key_left = False
        elif key == Int(Keycode.SDLK_RIGHT):
            self.key_right = False
        elif key == Int(Keycode.SDLK_SPACE):
            self.key_fire = False
        elif key == Int(Keycode.SDLK_R):
            self.key_reset = False

    def _update_action(mut self):
        """Convert current key state to an ALE action."""
        # RESET takes priority over everything
        if self.key_reset:
            self.current_action = ACTION_RESET
            return

        var up = self.key_up
        var down = self.key_down
        var left = self.key_left
        var right = self.key_right
        var fire = self.key_fire

        # Resolve conflicting directions
        if up and down:
            up = False
            down = False
        if left and right:
            left = False
            right = False

        if fire:
            if up and right:
                self.current_action = ACTION_UPRIGHTFIRE
            elif up and left:
                self.current_action = ACTION_UPLEFTFIRE
            elif down and right:
                self.current_action = ACTION_DOWNRIGHTFIRE
            elif down and left:
                self.current_action = ACTION_DOWNLEFTFIRE
            elif up:
                self.current_action = ACTION_UPFIRE
            elif down:
                self.current_action = ACTION_DOWNFIRE
            elif left:
                self.current_action = ACTION_LEFTFIRE
            elif right:
                self.current_action = ACTION_RIGHTFIRE
            else:
                self.current_action = ACTION_FIRE
        else:
            if up and right:
                self.current_action = ACTION_UPRIGHT
            elif up and left:
                self.current_action = ACTION_UPLEFT
            elif down and right:
                self.current_action = ACTION_DOWNRIGHT
            elif down and left:
                self.current_action = ACTION_DOWNLEFT
            elif up:
                self.current_action = ACTION_UP
            elif down:
                self.current_action = ACTION_DOWN
            elif left:
                self.current_action = ACTION_LEFT
            elif right:
                self.current_action = ACTION_RIGHT
            else:
                self.current_action = ACTION_NOOP

    def get_pixel_buffer(self) -> UnsafePointer[UInt8, MutAnyOrigin]:
        """Get the pixel buffer pointer for external rendering.

        Use with run_frame_with_video() which fills the buffer
        scanline-by-scanline during CPU execution.
        """
        return self.pixel_buf

    def display_buffer(mut self):
        """Upload the pixel buffer to screen (call after buffer is filled).

        The buffer should already contain a complete 160×210 BGRA frame,
        either from run_frame_with_video() or render_frame_bgra().
        """
        if not self.initialized:
            if not self.init_display():
                return

        try:
            # Clear screen (black background for HUD area)
            set_render_draw_color(self.sdl_renderer.value(), 0, 0, 0, 255)
            render_clear(self.sdl_renderer.value())

            # Upload pixels to texture (NULL rect = entire texture).
            update_texture(self.texture.value(),
                _null_ptr[Rect, ImmutAnyOrigin](),
                rebind[Ptr[NoneType, ImmutAnyOrigin]](
                    Ptr[UInt8, ImmutAnyOrigin](self.pixel_buf)
                ),
                c_int(FRAME_WIDTH * 4),  # pitch in bytes
            )

            # Render texture scaled to window (leaving room for HUD)
            var dst = FRect(
                c_float(0),
                c_float(0),
                c_float(self.screen_width),
                c_float(self.screen_height - HUD_HEIGHT),
            )
            # NULL src rect = use entire source texture.
            render_texture(self.sdl_renderer.value(),
                self.texture.value(),
                _null_ptr[FRect, ImmutAnyOrigin](),
                rebind[Ptr[FRect, ImmutAnyOrigin]](Ptr(to=dst)),
            )
        except:
            pass

    def display_buffer_with_hud(
        mut self,
        score: Int,
        lives: Int,
        frame_num: Int,
    ):
        """Upload pixel buffer to screen with HUD overlay."""
        self.display_buffer()
        if not self.initialized:
            return

        try:
            # Draw HUD text at the bottom
            var hud_y = c_float(self.screen_height - HUD_HEIGHT + 4)

            # Score
            set_render_draw_color(self.sdl_renderer.value(), 255, 255, 255, 255)
            var score_text = "Score: " + String(score)
            render_debug_text(self.sdl_renderer.value(), c_float(8), hud_y, score_text)

            # Lives
            var lives_text = "Lives: " + String(lives)
            render_debug_text(self.sdl_renderer.value(), c_float(160), hud_y, lives_text
            )

            # Frame number
            var frame_text = "Frame: " + String(frame_num)
            render_debug_text(self.sdl_renderer.value(), c_float(300), hud_y, frame_text
            )

            # Recording indicator
            if self.recorder.is_recording:
                set_render_draw_color(self.sdl_renderer.value(), 255, 0, 0, 255)
                render_debug_text(self.sdl_renderer.value(), c_float(8), c_float(4), "REC"
                )

            # Paused indicator
            if self.paused:
                set_render_draw_color(self.sdl_renderer.value(), 255, 255, 0, 255)
                render_debug_text(self.sdl_renderer.value(),
                    c_float(self.screen_width // 2 - 24),
                    c_float(self.screen_height // 2 - 4),
                    "PAUSED",
                )
        except:
            pass

        self.flip()

    def flip(mut self):
        """Present the frame and cap framerate. Also captures for recording."""
        try:
            # Capture frame for recording before present.
            if self.recorder.is_recording:
                # NULL rect = entire viewport.
                var surf = render_read_pixels(
                    self.sdl_renderer.value(),
                    _null_ptr[Rect, ImmutAnyOrigin](),
                )
                var pixels = surf[].pixels
                self.recorder.add_frame_bgra(
                    Int(pixels), self.screen_width, self.screen_height
                )
                destroy_surface(surf)

            render_present(self.sdl_renderer.value())
        except:
            pass

        # Framerate cap
        try:
            var current_time = get_ticks()
            var elapsed = current_time - self.last_frame_time
            if elapsed < UInt64(self.frame_delay):
                delay(UInt32(UInt64(self.frame_delay) - elapsed))
            self.last_frame_time = get_ticks()
        except:
            pass

    def close(mut self):
        """Clean up SDL resources."""
        try:
            if self.recorder.is_recording:
                self.recorder.stop()
        except:
            pass

        try:
            if Bool(self.texture):
                destroy_texture(self.texture.value())
            if Bool(self.sdl_renderer):
                destroy_renderer(self.sdl_renderer.value())
            if Bool(self.window):
                destroy_window(self.window.value())
            sdl_quit()
        except:
            pass

        self.initialized = False

    def start_recording(mut self, filename: String) raises:
        """Start recording to a video file."""
        self.recorder.start(filename, self.fps)

    def stop_recording(mut self) raises:
        """Stop recording."""
        self.recorder.stop()
