"""`libero_kitchen_scene6`'s device placement table — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run gen-placement-tables
CI checks it with: pixi run gen-placement-tables --check

From `noeira/tasks/families/libero_kitchen_scene6.family`,
`noeira/tasks/scenes/libero_kitchen_scene6.xml` and forward kinematics on it.
2 free slots, 7 regions (0 moving, 0 followed on one slide), 1 drawable joints.
See `placement/table.mojo` for what each method means.
"""

from noeira.tasks.placement.table import PlacementTable


struct LiberoKitchenScene6Placement(PlacementTable):
    comptime N_SLOTS: Int = 4
    comptime N_FREE: Int = 2
    comptime N_REGIONS: Int = 7
    comptime NQ: Int = 24
    comptime NV: Int = 22
    comptime N_JOINTS: Int = 1
    comptime NBODY: Int = 27
    comptime NSITE: Int = 12
    comptime GRIPPER_SITE: Int = 4  # robot_grip_site
    comptime N_BASE_QPOS: Int = 9

    @staticmethod
    def base_qpos[DTYPE: DType](i: Int) -> Scalar[DTYPE]:
        if i == 0:
            return Scalar[DTYPE](0.0)
        if i == 1:
            return Scalar[DTYPE](-0.161037389)
        if i == 2:
            return Scalar[DTYPE](0.0)
        if i == 3:
            return Scalar[DTYPE](-2.44459747)
        if i == 4:
            return Scalar[DTYPE](0.0)
        if i == 5:
            return Scalar[DTYPE](2.2267522)
        if i == 6:
            return Scalar[DTYPE](0.7853981633974483)
        if i == 7:
            return Scalar[DTYPE](0.020833)
        return Scalar[DTYPE](-0.020833)

    @staticmethod
    def free_slot(j: Int) -> Int:
        if j == 0:
            return 2  # porcelain_mug_1
        return 3  # white_yellow_mug_1

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
            return 11
        if r == 2:
            return 1
        if r == 3:
            return 0
        if r == 4:
            return 0
        if r == 5:
            return 0
        return 0

    @staticmethod
    def region_site_x[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_site_y[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.36)
        if r == 1:
            return Scalar[DTYPE](0.36)
        if r == 2:
            return Scalar[DTYPE](0.0)
        if r == 3:
            return Scalar[DTYPE](0.0)
        if r == 4:
            return Scalar[DTYPE](0.0)
        if r == 5:
            return Scalar[DTYPE](0.0)
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_site_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](1.12)
        if r == 1:
            return Scalar[DTYPE](1.016)
        if r == 2:
            return Scalar[DTYPE](0.875)
        if r == 3:
            return Scalar[DTYPE](0.9)
        if r == 4:
            return Scalar[DTYPE](0.9)
        if r == 5:
            return Scalar[DTYPE](0.9)
        return Scalar[DTYPE](0.9)

    @staticmethod
    def region_has_rect(r: Int) -> Bool:
        if r == 0:
            return True  # microwave_1_top_side
        if r == 1:
            return True  # microwave_1_heating_region
        if r == 2:
            return True  # kitchen_table_porcelain_mug_front_region_zone
        if r == 3:
            return True  # kitchen_table_microwave_init_region
        if r == 4:
            return True  # kitchen_table_white_yellow_mug_init_region
        if r == 5:
            return True  # kitchen_table_porcelain_mug_init_region
        return True  # kitchen_table_porcelain_mug_front_region

    @staticmethod
    def region_x0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.12)
        if r == 1:
            return Scalar[DTYPE](-0.12)
        if r == 2:
            return Scalar[DTYPE](-0.05)
        if r == 3:
            return Scalar[DTYPE](-0.01)
        if r == 4:
            return Scalar[DTYPE](-0.025)
        if r == 5:
            return Scalar[DTYPE](-0.125)
        return Scalar[DTYPE](-0.05)

    @staticmethod
    def region_y0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.0835)
        if r == 1:
            return Scalar[DTYPE](-0.0835)
        if r == 2:
            return Scalar[DTYPE](-0.3)
        if r == 3:
            return Scalar[DTYPE](0.33999999999999997)
        if r == 4:
            return Scalar[DTYPE](-0.025)
        if r == 5:
            return Scalar[DTYPE](-0.275)
        return Scalar[DTYPE](-0.3)

    @staticmethod
    def region_x1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.12)
        if r == 1:
            return Scalar[DTYPE](0.12)
        if r == 2:
            return Scalar[DTYPE](0.05)
        if r == 3:
            return Scalar[DTYPE](0.01)
        if r == 4:
            return Scalar[DTYPE](0.025)
        if r == 5:
            return Scalar[DTYPE](-0.07500000000000001)
        return Scalar[DTYPE](0.05)

    @staticmethod
    def region_y1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.0835)
        if r == 1:
            return Scalar[DTYPE](0.0835)
        if r == 2:
            return Scalar[DTYPE](-0.2)
        if r == 3:
            return Scalar[DTYPE](0.36)
        if r == 4:
            return Scalar[DTYPE](0.025)
        if r == 5:
            return Scalar[DTYPE](-0.225)
        return Scalar[DTYPE](-0.2)

    @staticmethod
    def region_anchored(r: Int) -> Bool:
        if r == 0:
            return True
        if r == 1:
            return True
        if r == 2:
            return False
        if r == 3:
            return False
        if r == 4:
            return False
        if r == 5:
            return False
        return False

    @staticmethod
    def region_contact_has_geom(r: Int) -> Bool:
        if r == 0:
            return True
        if r == 1:
            return True
        if r == 2:
            return False
        if r == 3:
            return False
        if r == 4:
            return False
        if r == 5:
            return False
        return False

    @staticmethod
    def region_contact_top_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.06)
        if r == 1:
            return Scalar[DTYPE](0.06)
        if r == 2:
            return Scalar[DTYPE](0.0)
        if r == 3:
            return Scalar[DTYPE](0.0)
        if r == 4:
            return Scalar[DTYPE](0.0)
        if r == 5:
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
        return String("microwave_1_microjoint")

    @staticmethod
    def joint_qadr(k: Int) -> Int:
        return 9

    @staticmethod
    def joint_dadr(k: Int) -> Int:
        return 9
