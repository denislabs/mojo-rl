"""Physics3D v2 Multi-Body Renderer using the render3d wireframe renderer.

Provides visualization of multi-body physics3d_v2 simulations with:
- Multiple shaded spheres with distinct colors
- Ground plane with chessboard pattern
- Contact point indicators
- Velocity indicators for each body

Example:
    from physics3d_v2 import Model, Data, step_multi_body
    from physics3d_v2.render_multi_body import Physics3DRenderer

    # Setup physics
    var model = Model[DType.float64, 3, 10]()
    model.set_body(0, mass=1.0, radius=0.1)
    model.set_body(1, mass=1.0, radius=0.1)
    model.set_body(2, mass=1.0, radius=0.1)

    var data = Data[DType.float64, 3, 10]()
    data.set_body_position(0, 0, 0, 1.0)
    data.set_body_position(1, 0.3, 0, 1.5)
    data.set_body_position(2, -0.3, 0, 2.0)

    # Setup renderer
    var renderer = Physics3DRenderer()
    renderer.init()

    # Simulation loop
    while not renderer.check_quit():
        step_multi_body(model, data)
        renderer.render(model, data)
        renderer.delay(10)

    renderer.close()
"""

from math import cos, sin, pi

from math3d import Vec3 as Vec3Generic, Quat
from render3d import Renderer3D, Camera3D, Color3D
from render3d.shapes3d import WireframeLine
from .types import Model, Data
from .gpu.constants import GEOM_SPHERE, GEOM_CAPSULE, GEOM_BOX

comptime Vec3 = Vec3Generic[DType.float64]


# =============================================================================
# Color Palette for Multiple Bodies
# =============================================================================


struct Physics3DColors:
    """Color palette for multi-body visualization."""

    @staticmethod
    fn body_color(index: Int) -> Color3D:
        """Get color for body at given index (cycles through palette)."""
        var palette_index = index % 8
        if palette_index == 0:
            return Color3D(255, 100, 50)  # Orange
        elif palette_index == 1:
            return Color3D(50, 150, 255)  # Blue
        elif palette_index == 2:
            return Color3D(100, 255, 100)  # Green
        elif palette_index == 3:
            return Color3D(255, 255, 50)  # Yellow
        elif palette_index == 4:
            return Color3D(255, 50, 150)  # Pink
        elif palette_index == 5:
            return Color3D(150, 50, 255)  # Purple
        elif palette_index == 6:
            return Color3D(50, 255, 200)  # Cyan
        else:
            return Color3D(255, 150, 150)  # Light red

    @staticmethod
    fn ground_light() -> Color3D:
        """Ground light tile color."""
        return Color3D(140, 140, 120)

    @staticmethod
    fn ground_dark() -> Color3D:
        """Ground dark tile color."""
        return Color3D(80, 80, 70)

    @staticmethod
    fn velocity() -> Color3D:
        """Velocity indicator color - white."""
        return Color3D(255, 255, 255)

    @staticmethod
    fn shadow() -> Color3D:
        """Shadow color."""
        return Color3D(30, 30, 30)

    @staticmethod
    fn contact() -> Color3D:
        """Contact point indicator - red."""
        return Color3D(255, 50, 50)

    @staticmethod
    fn contact_sphere_sphere() -> Color3D:
        """Sphere-sphere contact indicator - yellow."""
        return Color3D(255, 255, 0)

    @staticmethod
    fn joint_link() -> Color3D:
        """Joint link color - light gray/white."""
        return Color3D(200, 200, 200)

    @staticmethod
    fn pivot() -> Color3D:
        """Pivot point color - gold."""
        return Color3D(255, 200, 50)


# =============================================================================
# Physics 3D Renderer
# =============================================================================


struct Physics3DRenderer(Movable):
    """Renderer for multi-body Physics3D v2 simulations.

    Renders multiple spheres with distinct colors, ground plane,
    contact indicators, and velocity vectors.
    """

    var renderer: Renderer3D
    var initialized: Bool
    var show_velocity: Bool
    var show_shadows: Bool
    var show_contacts: Bool

    fn __init__(
        out self,
        width: Int = 1024,
        height: Int = 768,
        show_velocity: Bool = True,
        show_shadows: Bool = True,
        show_contacts: Bool = True,
    ) raises:
        """Initialize the multi-body renderer.

        Args:
            width: Window width in pixels.
            height: Window height in pixels.
            show_velocity: Whether to show velocity indicators.
            show_shadows: Whether to show shadows.
            show_contacts: Whether to show contact points.
        """
        # Camera setup - wider view for multiple bodies
        var camera = Camera3D(
            eye=Vec3(3.0, -4.0, 3.0),  # View from side and above
            target=Vec3(0.0, 0.0, 0.5),  # Look at center above ground
            up=Vec3(0.0, 0.0, 1.0),  # Z-up
            fov=55.0,
            aspect=Float64(width) / Float64(height),
            near=0.1,
            far=100.0,
            screen_width=width,
            screen_height=height,
        )

        self.renderer = Renderer3D(
            width=width,
            height=height,
            camera=camera,
            draw_grid=False,
            draw_axes=True,
        )
        self.initialized = False
        self.show_velocity = show_velocity
        self.show_shadows = show_shadows
        self.show_contacts = show_contacts

    fn __moveinit__(out self, deinit other: Self):
        """Move constructor."""
        self.renderer = other.renderer^
        self.initialized = other.initialized
        self.show_velocity = other.show_velocity
        self.show_shadows = other.show_shadows
        self.show_contacts = other.show_contacts

    fn init(mut self) raises -> None:
        """Initialize the renderer window."""
        var title = String("Physics3D v2 - Multi-Body")
        self.renderer.init(title)
        self.initialized = True

    fn close(mut self) raises -> None:
        """Close the renderer."""
        if self.initialized:
            self.renderer.close()
            self.initialized = False

    fn check_quit(mut self) -> Bool:
        """Check if user wants to quit."""
        return self.renderer.check_quit()

    fn render[
        DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int
    ](
        mut self,
        model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS],
        data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS],
    ):
        """Render the multi-body physics state.

        Args:
            model: Multi-body model configuration.
            data: Current multi-body physics state.
        """
        if not self.initialized:
            return

        # Begin frame
        self.renderer.begin_frame()

        # Draw ground (chessboard pattern)
        self._draw_ground(Float64(model.ground_z))

        # Draw coordinate axes
        self.renderer.draw_coordinate_axes(Vec3(0.0, 0.0, 0.01), 0.5)

        # Draw shadows for all bodies
        if self.show_shadows:
            for i in range(NUM_BODIES):
                self._draw_shadow_for_geom(
                    Float64(data.positions[i * 3 + 0]),
                    Float64(data.positions[i * 3 + 1]),
                    model.geom_types[i],
                    Float64(model.radii[i]),
                    Float64(model.half_lengths[i]),
                    Float64(model.half_x[i]),
                    Float64(model.half_y[i]),
                    Float64(model.ground_z),
                )

        # Draw all bodies based on geometry type
        for i in range(NUM_BODIES):
            # Extract quaternion from data (stored as x, y, z, w)
            var qx = Float64(data.quaternions[i * 4 + 0])
            var qy = Float64(data.quaternions[i * 4 + 1])
            var qz = Float64(data.quaternions[i * 4 + 2])
            var qw = Float64(data.quaternions[i * 4 + 3])

            self._draw_body[DTYPE](
                i,
                Float64(data.positions[i * 3 + 0]),
                Float64(data.positions[i * 3 + 1]),
                Float64(data.positions[i * 3 + 2]),
                model.geom_types[i],
                Float64(model.radii[i]),
                Float64(model.half_lengths[i]),
                Float64(model.half_x[i]),
                Float64(model.half_y[i]),
                Float64(model.half_z[i]),
                Physics3DColors.body_color(i),
                qx, qy, qz, qw,
            )

        # Draw contacts
        if self.show_contacts:
            for c in range(data.num_contacts):
                var contact = data.contacts[c]
                var is_sphere_sphere = contact.body_b >= 0
                var color = (
                    Physics3DColors.contact_sphere_sphere() if is_sphere_sphere else Physics3DColors.contact()
                )
                self._draw_contact(
                    Float64(contact.pos_x),
                    Float64(contact.pos_y),
                    Float64(contact.pos_z),
                    color,
                )

        # Draw velocity indicators
        if self.show_velocity:
            for i in range(NUM_BODIES):
                self._draw_velocity(
                    Float64(data.positions[i * 3 + 0]),
                    Float64(data.positions[i * 3 + 1]),
                    Float64(data.positions[i * 3 + 2]),
                    Float64(data.velocities[i * 3 + 0]),
                    Float64(data.velocities[i * 3 + 1]),
                    Float64(data.velocities[i * 3 + 2]),
                )

        # End frame
        self.renderer.end_frame()

    fn _draw_ground(self, ground_z: Float64):
        """Draw chessboard ground pattern."""
        var tile_size = 0.5
        var num_tiles = 10

        for i in range(-num_tiles, num_tiles):
            for j in range(-num_tiles, num_tiles):
                var x0 = Float64(i) * tile_size
                var y0 = Float64(j) * tile_size
                var x1 = x0 + tile_size
                var y1 = y0 + tile_size

                var is_light = (i + j) % 2 == 0
                var color = (
                    Physics3DColors.ground_light() if is_light else Physics3DColors.ground_dark()
                )

                self.renderer.draw_filled_quad_3d(
                    Vec3(x0, y0, ground_z),
                    Vec3(x1, y0, ground_z),
                    Vec3(x1, y1, ground_z),
                    Vec3(x0, y1, ground_z),
                    color,
                )

    fn _draw_shadow(
        self, pos_x: Float64, pos_y: Float64, radius: Float64, ground_z: Float64
    ):
        """Draw shadow on ground for a sphere."""
        var shadow_z = ground_z + 0.001
        var shadow_radius = radius * 1.2

        var num_segments = 12
        from math import cos, sin, pi

        for i in range(num_segments):
            var angle0 = 2.0 * pi * Float64(i) / Float64(num_segments)
            var angle1 = 2.0 * pi * Float64(i + 1) / Float64(num_segments)

            var x0 = pos_x + shadow_radius * cos(angle0)
            var y0 = pos_y + shadow_radius * sin(angle0)
            var x1 = pos_x + shadow_radius * cos(angle1)
            var y1 = pos_y + shadow_radius * sin(angle1)

            self.renderer.draw_filled_quad_3d(
                Vec3(pos_x, pos_y, shadow_z),
                Vec3(x0, y0, shadow_z),
                Vec3(x1, y1, shadow_z),
                Vec3(pos_x, pos_y, shadow_z),
                Physics3DColors.shadow(),
            )

    fn _draw_sphere(
        self,
        pos_x: Float64,
        pos_y: Float64,
        pos_z: Float64,
        radius: Float64,
        color: Color3D,
    ):
        """Draw a shaded sphere."""
        var pos = Vec3(pos_x, pos_y, pos_z)

        var screen_pos = self.renderer.camera.project_to_screen(pos)
        if screen_pos[2]:  # Visible
            var view_pos = (
                self.renderer.camera.get_view_matrix().transform_point(pos)
            )
            var depth = -view_pos.z
            if depth > 0.1:
                var fov_scale = 1.0 / (depth * 0.7)
                var screen_radius = Int(
                    radius * Float64(self.renderer.height) * fov_scale
                )
                screen_radius = max(screen_radius, 3)

                self.renderer.draw_shaded_sphere_2d(
                    screen_pos[0],
                    screen_pos[1],
                    screen_radius,
                    color,
                )

    fn _draw_capsule(
        self,
        pos_x: Float64,
        pos_y: Float64,
        pos_z: Float64,
        radius: Float64,
        half_length: Float64,
        color: Color3D,
    ):
        """Draw a vertical capsule (cylinder + 2 hemisphere caps).

        The capsule is oriented vertically (along Z-axis).
        Total height = 2 * half_length + 2 * radius.
        """
        # Draw with identity quaternion (no rotation)
        self._draw_capsule_rotated(
            pos_x, pos_y, pos_z, radius, half_length,
            0.0, 0.0, 0.0, 1.0,  # Identity quaternion (x, y, z, w)
            color
        )

    fn _draw_capsule_rotated(
        self,
        pos_x: Float64,
        pos_y: Float64,
        pos_z: Float64,
        radius: Float64,
        half_length: Float64,
        qx: Float64,
        qy: Float64,
        qz: Float64,
        qw: Float64,
        color: Color3D,
    ):
        """Draw a capsule with arbitrary orientation using quaternion.

        Uses the Renderer3D's draw_shaded_capsule_2d for high-quality rendering
        with gradient shading that simulates lighting and 3D volume.

        The capsule's local axis is Z (pointing up in local frame).
        The quaternion rotates this local Z-axis to world frame.

        Args:
            pos_x, pos_y, pos_z: Center position of the capsule.
            radius: Capsule radius.
            half_length: Half-length of the cylindrical part.
            qx, qy, qz, qw: Quaternion components (x, y, z, w order from Data storage).
            color: Rendering color.
        """
        # Build quaternion (Quat uses w, x, y, z order)
        var q = Quat[DType.float64](qw, qx, qy, qz)
        var center = Vec3(pos_x, pos_y, pos_z)

        # Use the Renderer3D's shaded capsule drawing
        # axis=2 means local Z-axis, which is rotated by the quaternion
        self.renderer.draw_shaded_capsule_2d(
            center,
            q,
            radius,
            half_length,
            axis=2,
            color=color,
        )

    fn _draw_box(
        self,
        pos_x: Float64,
        pos_y: Float64,
        pos_z: Float64,
        half_x: Float64,
        half_y: Float64,
        half_z: Float64,
        color: Color3D,
    ):
        """Draw a wireframe box.

        Box is centered at (pos_x, pos_y, pos_z) with half-extents.
        """
        # 8 corners of the box
        var corners = List[Vec3]()
        for dx in range(2):
            for dy in range(2):
                for dz in range(2):
                    var x = pos_x + half_x * (2.0 * Float64(dx) - 1.0)
                    var y = pos_y + half_y * (2.0 * Float64(dy) - 1.0)
                    var z = pos_z + half_z * (2.0 * Float64(dz) - 1.0)
                    corners.append(Vec3(x, y, z))

        # 12 edges connecting corners
        # Corner indices: 0=(-,-,-), 1=(-,-,+), 2=(-,+,-), 3=(-,+,+)
        #                 4=(+,-,-), 5=(+,-,+), 6=(+,+,-), 7=(+,+,+)
        var lines = List[WireframeLine]()
        # Bottom face (z-)
        lines.append(WireframeLine(corners[0], corners[2]))
        lines.append(WireframeLine(corners[2], corners[6]))
        lines.append(WireframeLine(corners[6], corners[4]))
        lines.append(WireframeLine(corners[4], corners[0]))
        # Top face (z+)
        lines.append(WireframeLine(corners[1], corners[3]))
        lines.append(WireframeLine(corners[3], corners[7]))
        lines.append(WireframeLine(corners[7], corners[5]))
        lines.append(WireframeLine(corners[5], corners[1]))
        # Vertical edges
        lines.append(WireframeLine(corners[0], corners[1]))
        lines.append(WireframeLine(corners[2], corners[3]))
        lines.append(WireframeLine(corners[4], corners[5]))
        lines.append(WireframeLine(corners[6], corners[7]))

        self.renderer.draw_lines_3d(lines, color)

        # Draw filled faces for better visibility
        # Front face (y-)
        self.renderer.draw_filled_quad_3d(
            corners[0], corners[1], corners[5], corners[4], color
        )
        # Back face (y+)
        self.renderer.draw_filled_quad_3d(
            corners[2], corners[3], corners[7], corners[6], color
        )
        # Left face (x-)
        self.renderer.draw_filled_quad_3d(
            corners[0], corners[1], corners[3], corners[2], color
        )
        # Right face (x+)
        self.renderer.draw_filled_quad_3d(
            corners[4], corners[5], corners[7], corners[6], color
        )
        # Top face (z+)
        self.renderer.draw_filled_quad_3d(
            corners[1], corners[3], corners[7], corners[5], color
        )
        # Bottom face (z-)
        self.renderer.draw_filled_quad_3d(
            corners[0], corners[2], corners[6], corners[4], color
        )

    fn _draw_body[DTYPE: DType](
        self,
        index: Int,
        pos_x: Float64,
        pos_y: Float64,
        pos_z: Float64,
        geom_type: Int,
        radius: Float64,
        half_length: Float64,
        half_x: Float64,
        half_y: Float64,
        half_z: Float64,
        color: Color3D,
        qx: Float64 = 0.0,
        qy: Float64 = 0.0,
        qz: Float64 = 0.0,
        qw: Float64 = 1.0,
    ):
        """Draw a body based on its geometry type with optional rotation."""
        if geom_type == GEOM_CAPSULE:
            self._draw_capsule_rotated(pos_x, pos_y, pos_z, radius, half_length, qx, qy, qz, qw, color)
        elif geom_type == GEOM_BOX:
            self._draw_box(pos_x, pos_y, pos_z, half_x, half_y, half_z, color)
        else:  # GEOM_SPHERE or default
            self._draw_sphere(pos_x, pos_y, pos_z, radius, color)

    fn _draw_shadow_for_geom(
        self,
        pos_x: Float64,
        pos_y: Float64,
        geom_type: Int,
        radius: Float64,
        half_length: Float64,
        half_x: Float64,
        half_y: Float64,
        ground_z: Float64,
    ):
        """Draw shadow on ground based on geometry type."""
        var shadow_z = ground_z + 0.001

        if geom_type == GEOM_CAPSULE:
            # Capsule shadow: elongated ellipse
            var shadow_radius_x = radius * 1.2
            var shadow_radius_y = radius * 1.2
            var num_segments = 12

            for i in range(num_segments):
                var angle0 = 2.0 * pi * Float64(i) / Float64(num_segments)
                var angle1 = 2.0 * pi * Float64(i + 1) / Float64(num_segments)

                var x0 = pos_x + shadow_radius_x * cos(angle0)
                var y0 = pos_y + shadow_radius_y * sin(angle0)
                var x1 = pos_x + shadow_radius_x * cos(angle1)
                var y1 = pos_y + shadow_radius_y * sin(angle1)

                self.renderer.draw_filled_quad_3d(
                    Vec3(pos_x, pos_y, shadow_z),
                    Vec3(x0, y0, shadow_z),
                    Vec3(x1, y1, shadow_z),
                    Vec3(pos_x, pos_y, shadow_z),
                    Physics3DColors.shadow(),
                )

        elif geom_type == GEOM_BOX:
            # Box shadow: rectangle
            self.renderer.draw_filled_quad_3d(
                Vec3(pos_x - half_x, pos_y - half_y, shadow_z),
                Vec3(pos_x + half_x, pos_y - half_y, shadow_z),
                Vec3(pos_x + half_x, pos_y + half_y, shadow_z),
                Vec3(pos_x - half_x, pos_y + half_y, shadow_z),
                Physics3DColors.shadow(),
            )

        else:  # GEOM_SPHERE
            self._draw_shadow(pos_x, pos_y, radius, ground_z)

    fn _draw_contact(
        self, pos_x: Float64, pos_y: Float64, pos_z: Float64, color: Color3D
    ):
        """Draw contact point indicator."""
        var contact_pos = Vec3(pos_x, pos_y, pos_z + 0.02)

        var screen_pos = self.renderer.camera.project_to_screen(contact_pos)
        if screen_pos[2]:
            self.renderer.draw_filled_circle_2d(
                screen_pos[0],
                screen_pos[1],
                6,
                color,
            )

    fn _draw_velocity(
        self,
        pos_x: Float64,
        pos_y: Float64,
        pos_z: Float64,
        vel_x: Float64,
        vel_y: Float64,
        vel_z: Float64,
    ):
        """Draw velocity indicator arrow."""
        var vel_mag_sq = vel_x * vel_x + vel_y * vel_y + vel_z * vel_z
        if vel_mag_sq < 0.01:
            return  # Skip very small velocities

        var pos = Vec3(pos_x, pos_y, pos_z)
        var scale = 0.1
        var arrow_end = Vec3(
            pos_x + vel_x * scale, pos_y + vel_y * scale, pos_z + vel_z * scale
        )

        from render3d.shapes3d import WireframeLine

        var lines = List[WireframeLine]()
        lines.append(WireframeLine(pos, arrow_end))

        self.renderer.draw_lines_3d(lines, Physics3DColors.velocity())

    fn delay(self, ms: Int) -> None:
        """Delay for given milliseconds."""
        self.renderer.delay(ms)

    fn is_open(self) -> Bool:
        """Check if renderer window is still open."""
        return self.initialized

    fn render_with_joints[
        DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int
    ](
        mut self,
        model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS],
        data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS],
    ):
        """Render physics state including joint links.

        Draws lines connecting joints to visualize the pendulum links.

        Args:
            model: Model configuration with joints.
            data: Current physics state.
        """
        if not self.initialized:
            return

        from render3d.shapes3d import WireframeLine

        # Begin frame
        self.renderer.begin_frame()

        # Draw ground (chessboard pattern)
        self._draw_ground(Float64(model.ground_z))

        # Draw coordinate axes
        self.renderer.draw_coordinate_axes(Vec3(0.0, 0.0, 0.01), 0.5)

        # Draw shadows for all bodies
        if self.show_shadows:
            for i in range(NUM_BODIES):
                self._draw_shadow_for_geom(
                    Float64(data.positions[i * 3 + 0]),
                    Float64(data.positions[i * 3 + 1]),
                    model.geom_types[i],
                    Float64(model.radii[i]),
                    Float64(model.half_lengths[i]),
                    Float64(model.half_x[i]),
                    Float64(model.half_y[i]),
                    Float64(model.ground_z),
                )

        # Draw joint links (lines between pivot and bodies)
        for j_idx in range(model.num_joints):
            var joint = model.joints[j_idx]
            var parent_body = joint.parent_body
            var child_body = joint.child_body

            # Get child body position
            var child_x = Float64(data.positions[child_body * 3 + 0])
            var child_y = Float64(data.positions[child_body * 3 + 1])
            var child_z = Float64(data.positions[child_body * 3 + 2])

            # Get parent anchor position (world or body)
            var parent_x: Float64
            var parent_y: Float64
            var parent_z: Float64

            if parent_body < 0:
                # World anchor - use anchor_parent directly
                parent_x = Float64(joint.anchor_parent_x)
                parent_y = Float64(joint.anchor_parent_y)
                parent_z = Float64(joint.anchor_parent_z)
            else:
                # Body anchor - use parent body position
                parent_x = Float64(data.positions[parent_body * 3 + 0])
                parent_y = Float64(data.positions[parent_body * 3 + 1])
                parent_z = Float64(data.positions[parent_body * 3 + 2])

            # Draw link line
            var lines = List[WireframeLine]()
            lines.append(
                WireframeLine(
                    Vec3(parent_x, parent_y, parent_z),
                    Vec3(child_x, child_y, child_z),
                )
            )
            self.renderer.draw_lines_3d(lines, Physics3DColors.joint_link())

            # Draw pivot point if world-anchored
            if parent_body < 0:
                self._draw_pivot(parent_x, parent_y, parent_z)

        # Draw all bodies based on geometry type
        for i in range(NUM_BODIES):
            # Extract quaternion from data (stored as x, y, z, w)
            var qx = Float64(data.quaternions[i * 4 + 0])
            var qy = Float64(data.quaternions[i * 4 + 1])
            var qz = Float64(data.quaternions[i * 4 + 2])
            var qw = Float64(data.quaternions[i * 4 + 3])

            self._draw_body[DTYPE](
                i,
                Float64(data.positions[i * 3 + 0]),
                Float64(data.positions[i * 3 + 1]),
                Float64(data.positions[i * 3 + 2]),
                model.geom_types[i],
                Float64(model.radii[i]),
                Float64(model.half_lengths[i]),
                Float64(model.half_x[i]),
                Float64(model.half_y[i]),
                Float64(model.half_z[i]),
                Physics3DColors.body_color(i),
                qx, qy, qz, qw,
            )

        # Draw contacts
        if self.show_contacts:
            for c in range(data.num_contacts):
                var contact = data.contacts[c]
                var is_sphere_sphere = contact.body_b >= 0
                var color = (
                    Physics3DColors.contact_sphere_sphere() if is_sphere_sphere else Physics3DColors.contact()
                )
                self._draw_contact(
                    Float64(contact.pos_x),
                    Float64(contact.pos_y),
                    Float64(contact.pos_z),
                    color,
                )

        # Draw velocity indicators
        if self.show_velocity:
            for i in range(NUM_BODIES):
                self._draw_velocity(
                    Float64(data.positions[i * 3 + 0]),
                    Float64(data.positions[i * 3 + 1]),
                    Float64(data.positions[i * 3 + 2]),
                    Float64(data.velocities[i * 3 + 0]),
                    Float64(data.velocities[i * 3 + 1]),
                    Float64(data.velocities[i * 3 + 2]),
                )

        # End frame
        self.renderer.end_frame()

    fn _draw_pivot(self, pos_x: Float64, pos_y: Float64, pos_z: Float64):
        """Draw pivot point indicator (small circle)."""
        var pivot_pos = Vec3(pos_x, pos_y, pos_z)

        var screen_pos = self.renderer.camera.project_to_screen(pivot_pos)
        if screen_pos[2]:
            self.renderer.draw_filled_circle_2d(
                screen_pos[0],
                screen_pos[1],
                8,
                Physics3DColors.pivot(),
            )
