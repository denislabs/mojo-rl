"""`libero_living_room_scene4`'s device placement table — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run gen-placement-tables
CI checks it with: pixi run gen-placement-tables --check

From `mojo_rl/tasks/families/libero_living_room_scene4.family`,
`mojo_rl/tasks/scenes/libero_living_room_scene4.xml` and forward kinematics on it.
5 free slots, 6 regions (1 moving, 0 followed on one slide), 0 drawable joints.
See `placement/table.mojo` for what each method means.
"""

from mojo_rl.tasks.placement.table import PlacementTable


struct LiberoLivingRoomScene4Placement(PlacementTable):
    comptime N_SLOTS: Int = 6
    comptime N_FREE: Int = 5
    comptime N_REGIONS: Int = 6
    comptime NQ: Int = 44
    comptime NV: Int = 39
    comptime N_JOINTS: Int = 0
    comptime NBODY: Int = 26
    comptime NSITE: Int = 10
    comptime GRIPPER_SITE: Int = 4  # robot_grip_site

    @staticmethod
    def free_slot(j: Int) -> Int:
        if j == 0:
            return 1  # akita_black_bowl_1
        if j == 1:
            return 2  # akita_black_bowl_2
        if j == 2:
            return 3  # new_salad_dressing_1
        if j == 3:
            return 4  # chocolate_pudding_1
        return 5  # wooden_tray_1

    @staticmethod
    def free_qadr(j: Int) -> Int:
        if j == 0:
            return 9
        if j == 1:
            return 16
        if j == 2:
            return 23
        if j == 3:
            return 30
        return 37

    @staticmethod
    def free_dadr(j: Int) -> Int:
        if j == 0:
            return 9
        if j == 1:
            return 15
        if j == 2:
            return 21
        if j == 3:
            return 27
        return 33

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
            return Scalar[DTYPE](10.5)
        if j == 1:
            return Scalar[DTYPE](11.0)
        if j == 2:
            return Scalar[DTYPE](11.5)
        if j == 3:
            return Scalar[DTYPE](12.0)
        return Scalar[DTYPE](12.5)

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
            return 9
        if r == 1:
            return 0
        if r == 2:
            return 0
        if r == 3:
            return 0
        if r == 4:
            return 0
        return 0

    @staticmethod
    def region_site_x[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.00863)
        if r == 1:
            return Scalar[DTYPE](0.0)
        if r == 2:
            return Scalar[DTYPE](0.0)
        if r == 3:
            return Scalar[DTYPE](0.0)
        if r == 4:
            return Scalar[DTYPE](0.0)
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_site_y[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.00471)
        if r == 1:
            return Scalar[DTYPE](0.0)
        if r == 2:
            return Scalar[DTYPE](0.0)
        if r == 3:
            return Scalar[DTYPE](0.0)
        if r == 4:
            return Scalar[DTYPE](0.0)
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_site_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.04255)
        if r == 1:
            return Scalar[DTYPE](0.41)
        if r == 2:
            return Scalar[DTYPE](0.41)
        if r == 3:
            return Scalar[DTYPE](0.41)
        if r == 4:
            return Scalar[DTYPE](0.41)
        return Scalar[DTYPE](0.41)

    @staticmethod
    def region_has_rect(r: Int) -> Bool:
        if r == 0:
            return True  # wooden_tray_1_contain_region
        if r == 1:
            return True  # living_room_table_wooden_tray_init_region
        if r == 2:
            return True  # living_room_table_chocolate_pudding_init_region
        if r == 3:
            return True  # living_room_table_akita_black_bowl_right_init_region
        if r == 4:
            return True  # living_room_table_akita_black_bowl_left_init_region
        return True  # living_room_table_salad_dressing_init_region

    @staticmethod
    def region_x0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.04038)
        if r == 1:
            return Scalar[DTYPE](-0.01)
        if r == 2:
            return Scalar[DTYPE](0.07500000000000001)
        if r == 3:
            return Scalar[DTYPE](-0.125)
        if r == 4:
            return Scalar[DTYPE](-0.125)
        return Scalar[DTYPE](-0.275)

    @staticmethod
    def region_y0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.07839)
        if r == 1:
            return Scalar[DTYPE](0.25)
        if r == 2:
            return Scalar[DTYPE](-0.225)
        if r == 3:
            return Scalar[DTYPE](0.025)
        if r == 4:
            return Scalar[DTYPE](-0.175)
        return Scalar[DTYPE](-0.125)

    @staticmethod
    def region_x1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.04038)
        if r == 1:
            return Scalar[DTYPE](0.01)
        if r == 2:
            return Scalar[DTYPE](0.125)
        if r == 3:
            return Scalar[DTYPE](-0.07500000000000001)
        if r == 4:
            return Scalar[DTYPE](-0.07500000000000001)
        return Scalar[DTYPE](-0.225)

    @staticmethod
    def region_y1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.07839)
        if r == 1:
            return Scalar[DTYPE](0.27)
        if r == 2:
            return Scalar[DTYPE](-0.17500000000000002)
        if r == 3:
            return Scalar[DTYPE](0.07500000000000001)
        if r == 4:
            return Scalar[DTYPE](-0.125)
        return Scalar[DTYPE](-0.07500000000000001)

    @staticmethod
    def region_anchored(r: Int) -> Bool:
        if r == 0:
            return True
        if r == 1:
            return False
        if r == 2:
            return False
        if r == 3:
            return False
        if r == 4:
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
        if r == 3:
            return False
        if r == 4:
            return False
        return False

    @staticmethod
    def region_contact_top_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.04)
        if r == 1:
            return Scalar[DTYPE](0.0)
        if r == 2:
            return Scalar[DTYPE](0.0)
        if r == 3:
            return Scalar[DTYPE](0.0)
        if r == 4:
            return Scalar[DTYPE](0.0)
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_move_joint(r: Int) -> Int:
        if r == 0:
            return -2
        if r == 1:
            return -1
        if r == 2:
            return -1
        if r == 3:
            return -1
        if r == 4:
            return -1
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
        return String("")

    @staticmethod
    def joint_qadr(k: Int) -> Int:
        return 0

    @staticmethod
    def joint_dadr(k: Int) -> Int:
        return 0
