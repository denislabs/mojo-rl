"""`libero_living_room_scene1`'s device placement table — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run gen-placement-tables
CI checks it with: pixi run gen-placement-tables --check

From `noeira/envs/libero/families/libero_living_room_scene1.family`,
`noeira/envs/libero/scenes/libero_living_room_scene1.xml` and forward kinematics on it.
5 free slots, 6 regions (1 moving, 0 followed on one slide), 0 drawable joints.
See `placement/table.mojo` for what each method means.
"""

from noeira.tasks.placement.table import PlacementTable


struct LiberoLivingRoomScene1Placement(PlacementTable):
    comptime N_SLOTS: Int = 6
    comptime N_FREE: Int = 5
    comptime N_REGIONS: Int = 6
    comptime NQ: Int = 44
    comptime NV: Int = 39
    comptime N_JOINTS: Int = 0
    comptime NBODY: Int = 25
    comptime NSITE: Int = 10
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
            return 1  # alphabet_soup_1
        if j == 1:
            return 2  # cream_cheese_1
        if j == 2:
            return 3  # tomato_sauce_1
        if j == 3:
            return 4  # ketchup_1
        return 5  # basket_1

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
        if j == 0:
            return Scalar[DTYPE](0.04)
        if j == 1:
            return Scalar[DTYPE](0.025)
        if j == 2:
            return Scalar[DTYPE](0.06)
        if j == 3:
            return Scalar[DTYPE](0.06)
        return Scalar[DTYPE](0.06)

    @staticmethod
    def free_radius[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        if j == 0:
            return Scalar[DTYPE](0.03535533905932738)
        if j == 1:
            return Scalar[DTYPE](0.042426406871192854)
        if j == 2:
            return Scalar[DTYPE](0.03535533905932738)
        if j == 3:
            return Scalar[DTYPE](0.03535533905932738)
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
        if j == 0:
            return Scalar[DTYPE](-0.04)
        if j == 1:
            return Scalar[DTYPE](-0.025)
        if j == 2:
            return Scalar[DTYPE](-0.06)
        if j == 3:
            return Scalar[DTYPE](-0.06)
        return Scalar[DTYPE](-0.06)

    @staticmethod
    def free_top_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        if j == 0:
            return Scalar[DTYPE](0.04)
        if j == 1:
            return Scalar[DTYPE](0.025)
        if j == 2:
            return Scalar[DTYPE](0.04)
        if j == 3:
            return Scalar[DTYPE](0.04)
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
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_site_y[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_site_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.07185)
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
            return True  # basket_1_contain_region
        if r == 1:
            return True  # living_room_table_basket_init_region
        if r == 2:
            return True  # living_room_table_alphabet_soup_init_region
        if r == 3:
            return True  # living_room_table_cream_cheese_init_region
        if r == 4:
            return True  # living_room_table_tomato_sauce_init_region
        return True  # living_room_table_ketchup_init_region

    @staticmethod
    def region_x0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.06108)
        if r == 1:
            return Scalar[DTYPE](-0.01)
        if r == 2:
            return Scalar[DTYPE](0.025)
        if r == 3:
            return Scalar[DTYPE](-0.175)
        if r == 4:
            return Scalar[DTYPE](0.07500000000000001)
        return Scalar[DTYPE](-0.225)

    @staticmethod
    def region_y0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.06108)
        if r == 1:
            return Scalar[DTYPE](0.25)
        if r == 2:
            return Scalar[DTYPE](-0.125)
        if r == 3:
            return Scalar[DTYPE](0.034999999999999996)
        if r == 4:
            return Scalar[DTYPE](-0.225)
        return Scalar[DTYPE](-0.175)

    @staticmethod
    def region_x1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.06108)
        if r == 1:
            return Scalar[DTYPE](0.01)
        if r == 2:
            return Scalar[DTYPE](0.07500000000000001)
        if r == 3:
            return Scalar[DTYPE](-0.125)
        if r == 4:
            return Scalar[DTYPE](0.125)
        return Scalar[DTYPE](-0.17500000000000002)

    @staticmethod
    def region_y1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.06108)
        if r == 1:
            return Scalar[DTYPE](0.27)
        if r == 2:
            return Scalar[DTYPE](-0.07500000000000001)
        if r == 3:
            return Scalar[DTYPE](0.08499999999999999)
        if r == 4:
            return Scalar[DTYPE](-0.17500000000000002)
        return Scalar[DTYPE](-0.125)

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
