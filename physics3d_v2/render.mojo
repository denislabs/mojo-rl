"""Physics3D v2 Multi-Body Renderer using the GPU-accelerated Renderer3D.

Provides visualization of multi-body physics3d_v2 simulations with:
- GPU-rendered spheres, capsules, and boxes with Blinn-Phong lighting
- Procedural checkerboard ground plane
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

from math3d import Vec3 as Vec3Generic, Quat
from render3d import Renderer3D, Camera3D, Color3D
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
    fn velocity() -> Color3D:
        """Velocity indicator color - white."""
        return Color3D(255, 255, 255)

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

    Renders multiple spheres/capsules/boxes with distinct colors, ground plane,
    contact indicators, and velocity vectors using GPU-accelerated rendering.
    """

    var renderer: Renderer3D
    var initialized: Bool
    var show_velocity: Bool
    var show_contacts: Bool

    fn __init__(
        out self,
        width: Int = 1024,
        height: Int = 768,
        show_velocity: Bool = True,
        show_contacts: Bool = True,
    ) raises:
        """Initialize the multi-body renderer.

        Args:
            width: Window width in pixels.
            height: Window height in pixels.
            show_velocity: Whether to show velocity indicators.
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
        self.show_contacts = show_contacts

    fn __moveinit__(out self, deinit other: Self):
        """Move constructor."""
        self.renderer = other.renderer^
        self.initialized = other.initialized
        self.show_velocity = other.show_velocity
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

        # Draw ground (GPU shader checkerboard)
        self.renderer.draw_ground_grid(0.0, 10.0, Float64(model.ground_z))

        # Draw coordinate axes
        self.renderer.draw_coordinate_axes(Vec3(0.0, 0.0, 0.01), 0.5)

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
        try:
            self.renderer.end_frame()
        except:
            pass

    fn _draw_sphere(
        mut self,
        pos_x: Float64,
        pos_y: Float64,
        pos_z: Float64,
        radius: Float64,
        color: Color3D,
    ):
        """Draw a sphere in world space."""
        self.renderer.draw_sphere(
            center=Vec3(pos_x, pos_y, pos_z),
            radius=radius,
            color=color,
        )

    fn _draw_capsule_rotated(
        mut self,
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
    ) raises:
        """Draw a capsule with arbitrary orientation using quaternion.

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

        self.renderer.draw_capsule(
            center,
            q,
            radius,
            half_length,
            axis=2,
            color=color,
        )

    fn _draw_box(
        mut self,
        pos_x: Float64,
        pos_y: Float64,
        pos_z: Float64,
        half_x: Float64,
        half_y: Float64,
        half_z: Float64,
        color: Color3D,
    ):
        """Draw a solid box.

        Box is centered at (pos_x, pos_y, pos_z) with half-extents.
        """
        var center = Vec3(pos_x, pos_y, pos_z)
        var identity = Quat[DType.float64](1.0, 0.0, 0.0, 0.0)
        var half_extents = Vec3(half_x, half_y, half_z)

        self.renderer.draw_box(center, identity, half_extents, color)

    fn _draw_body[DTYPE: DType](
        mut self,
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
            try:
                self._draw_capsule_rotated(pos_x, pos_y, pos_z, radius, half_length, qx, qy, qz, qw, color)
            except:
                pass
        elif geom_type == GEOM_BOX:
            self._draw_box(pos_x, pos_y, pos_z, half_x, half_y, half_z, color)
        else:  # GEOM_SPHERE or default
            self._draw_sphere(pos_x, pos_y, pos_z, radius, color)

    fn _draw_contact(
        mut self, pos_x: Float64, pos_y: Float64, pos_z: Float64, color: Color3D
    ):
        """Draw contact point indicator as a small sphere."""
        self.renderer.draw_sphere(
            center=Vec3(pos_x, pos_y, pos_z + 0.02),
            radius=0.02,
            color=color,
        )

    fn _draw_velocity(
        mut self,
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

        self.renderer.draw_line_3d(pos, arrow_end, Physics3DColors.velocity())

    fn delay(self, ms: Int) -> None:
        """Delay for given milliseconds."""
        try:
            self.renderer.delay_ms(ms)
        except:
            pass

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

        # Begin frame
        self.renderer.begin_frame()

        # Draw ground (GPU shader checkerboard)
        self.renderer.draw_ground_grid(0.0, 10.0, Float64(model.ground_z))

        # Draw coordinate axes
        self.renderer.draw_coordinate_axes(Vec3(0.0, 0.0, 0.01), 0.5)

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
            self.renderer.draw_line_3d(
                Vec3(parent_x, parent_y, parent_z),
                Vec3(child_x, child_y, child_z),
                Physics3DColors.joint_link(),
            )

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
        try:
            self.renderer.end_frame()
        except:
            pass

    fn _draw_pivot(mut self, pos_x: Float64, pos_y: Float64, pos_z: Float64):
        """Draw pivot point indicator (small sphere)."""
        self.renderer.draw_sphere(
            center=Vec3(pos_x, pos_y, pos_z),
            radius=0.03,
            color=Physics3DColors.pivot(),
        )
