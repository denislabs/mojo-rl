"""`libero_kitchen_scene3`'s device placement table — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run gen-placement-tables
CI checks it with: pixi run gen-placement-tables --check

From `mojo_rl/tasks/families/libero_kitchen_scene3.family`,
`mojo_rl/tasks/scenes/libero_kitchen_scene3.xml` and forward kinematics on it.
2 free slots, 4 regions (0 moving, 0 followed on one slide), 1 drawable joints.
See `placement/table.mojo` for what each method means.
"""

from mojo_rl.tasks.placement.table import PlacementTable


struct LiberoKitchenScene3Placement(PlacementTable):
    comptime N_SLOTS: Int = 4
    comptime N_FREE: Int = 2
    comptime N_REGIONS: Int = 4
    comptime NQ: Int = 24
    comptime NV: Int = 22
    comptime N_JOINTS: Int = 1
    comptime NBODY: Int = 30
    comptime NSITE: Int = 12
    comptime GRIPPER_SITE: Int = 4  # robot_grip_site

    @staticmethod
    def free_slot(j: Int) -> Int:
        if j == 0:
            return 2  # chefmate_8_frypan_1
        return 3  # moka_pot_1

    @staticmethod
    def free_qadr(j: Int) -> Int:
        if j == 0:
            return 10
        return 17

    @staticmethod
    def free_dadr(j: Int) -> Int:
        if j == 0:
            return 10
        return 16

    @staticmethod
    def free_has_geom(j: Int) -> Bool:
        return True

    @staticmethod
    def free_rest[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.06)

    @staticmethod
    def free_radius[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.03535533905932738)

    @staticmethod
    def free_park_x[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        if j == 0:
            return Scalar[DTYPE](11.0)
        return Scalar[DTYPE](11.5)

    @staticmethod
    def free_park_y[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.0)

    @staticmethod
    def free_park_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](50.0)

    @staticmethod
    def free_bottom_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](-0.06)

    @staticmethod
    def free_top_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.04)

    @staticmethod
    def region_site(r: Int) -> Int:
        if r == 0:
            return 10
        if r == 1:
            return 0
        if r == 2:
            return 0
        return 0

    @staticmethod
    def region_site_x[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.05000000000000002)
        if r == 1:
            return Scalar[DTYPE](0.0)
        if r == 2:
            return Scalar[DTYPE](0.0)
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_site_y[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.2)
        if r == 1:
            return Scalar[DTYPE](0.0)
        if r == 2:
            return Scalar[DTYPE](0.0)
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_site_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.905)
        if r == 1:
            return Scalar[DTYPE](0.9)
        if r == 2:
            return Scalar[DTYPE](0.9)
        return Scalar[DTYPE](0.9)

    @staticmethod
    def region_has_rect(r: Int) -> Bool:
        if r == 0:
            return True  # flat_stove_1_cook_region
        if r == 1:
            return True  # kitchen_table_flat_stove_init_region
        if r == 2:
            return True  # kitchen_table_frypan_init_region
        return True  # kitchen_table_moka_pot_init_region

    @staticmethod
    def region_x0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.075)
        if r == 1:
            return Scalar[DTYPE](-0.21000000000000002)
        if r == 2:
            return Scalar[DTYPE](-0.07500000000000001)
        return Scalar[DTYPE](0.025)

    @staticmethod
    def region_y0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.075)
        if r == 1:
            return Scalar[DTYPE](0.19)
        if r == 2:
            return Scalar[DTYPE](-0.275)
        return Scalar[DTYPE](-0.025)

    @staticmethod
    def region_x1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.075)
        if r == 1:
            return Scalar[DTYPE](-0.19)
        if r == 2:
            return Scalar[DTYPE](-0.025)
        return Scalar[DTYPE](0.07500000000000001)

    @staticmethod
    def region_y1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.075)
        if r == 1:
            return Scalar[DTYPE](0.21000000000000002)
        if r == 2:
            return Scalar[DTYPE](-0.225)
        return Scalar[DTYPE](0.025)

    @staticmethod
    def region_anchored(r: Int) -> Bool:
        if r == 0:
            return True
        if r == 1:
            return False
        if r == 2:
            return False
        return False

    @staticmethod
    def region_contact_has_geom(r: Int) -> Bool:
        if r == 0:
            return True
        if r == 1:
            return False
        if r == 2:
            return False
        return False

    @staticmethod
    def region_contact_top_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.045)
        if r == 1:
            return Scalar[DTYPE](0.0)
        if r == 2:
            return Scalar[DTYPE](0.0)
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_move_joint(r: Int) -> Int:
        return -1

    @staticmethod
    def region_move_axis_x[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_move_axis_y[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_move_axis_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.0)

    @staticmethod
    def joint_name(k: Int) -> String:
        return String("flat_stove_1_button")

    @staticmethod
    def joint_qadr(k: Int) -> Int:
        return 9

    @staticmethod
    def joint_dadr(k: Int) -> Int:
        return 9
