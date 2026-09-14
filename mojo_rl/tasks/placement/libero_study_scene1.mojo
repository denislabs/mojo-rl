"""`libero_study_scene1`'s device placement table — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run gen-placement-tables
CI checks it with: pixi run gen-placement-tables --check

From `mojo_rl/tasks/families/libero_study_scene1.family`,
`mojo_rl/tasks/scenes/libero_study_scene1.xml` and forward kinematics on it.
2 free slots, 9 regions (0 moving, 0 followed on one slide), 0 drawable joints.
See `placement/table.mojo` for what each method means.
"""

from mojo_rl.tasks.placement.table import PlacementTable


struct LiberoStudyScene1Placement(PlacementTable):
    comptime N_SLOTS: Int = 4
    comptime N_FREE: Int = 2
    comptime N_REGIONS: Int = 9
    comptime NQ: Int = 23
    comptime NV: Int = 21
    comptime N_JOINTS: Int = 0
    comptime NBODY: Int = 26
    comptime NSITE: Int = 13
    comptime GRIPPER_SITE: Int = 4  # robot_grip_site

    @staticmethod
    def free_slot(j: Int) -> Int:
        if j == 0:
            return 2  # black_book_1
        return 3  # white_yellow_mug_1

    @staticmethod
    def free_qadr(j: Int) -> Int:
        if j == 0:
            return 9
        return 16

    @staticmethod
    def free_dadr(j: Int) -> Int:
        if j == 0:
            return 9
        return 15

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
            return 9
        if r == 1:
            return 10
        if r == 2:
            return 11
        if r == 3:
            return 12
        if r == 4:
            return 1
        if r == 5:
            return 0
        if r == 6:
            return 0
        if r == 7:
            return 0
        return 0

    @staticmethod
    def region_site_x[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.40631)
        if r == 1:
            return Scalar[DTYPE](-0.40631000000000006)
        if r == 2:
            return Scalar[DTYPE](-0.43364)
        if r == 3:
            return Scalar[DTYPE](-0.36898000000000003)
        if r == 4:
            return Scalar[DTYPE](-0.2)
        if r == 5:
            return Scalar[DTYPE](-0.2)
        if r == 6:
            return Scalar[DTYPE](-0.2)
        if r == 7:
            return Scalar[DTYPE](-0.2)
        return Scalar[DTYPE](-0.2)

    @staticmethod
    def region_site_y[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.0033899999999999764)
        if r == 1:
            return Scalar[DTYPE](-0.28551000000000004)
        if r == 2:
            return Scalar[DTYPE](-0.13936)
        if r == 3:
            return Scalar[DTYPE](-0.13936)
        if r == 4:
            return Scalar[DTYPE](0.0)
        if r == 5:
            return Scalar[DTYPE](0.0)
        if r == 6:
            return Scalar[DTYPE](0.0)
        if r == 7:
            return Scalar[DTYPE](0.0)
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_site_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.95258)
        if r == 1:
            return Scalar[DTYPE](0.95258)
        if r == 2:
            return Scalar[DTYPE](0.97258)
        if r == 3:
            return Scalar[DTYPE](0.92754)
        if r == 4:
            return Scalar[DTYPE](0.867)
        if r == 5:
            return Scalar[DTYPE](0.867)
        if r == 6:
            return Scalar[DTYPE](0.867)
        if r == 7:
            return Scalar[DTYPE](0.867)
        return Scalar[DTYPE](0.867)

    @staticmethod
    def region_has_rect(r: Int) -> Bool:
        if r == 0:
            return True  # desk_caddy_1_right_contain_region
        if r == 1:
            return True  # desk_caddy_1_left_contain_region
        if r == 2:
            return True  # desk_caddy_1_back_contain_region
        if r == 3:
            return True  # desk_caddy_1_front_contain_region
        if r == 4:
            return True  # study_table_desk_caddy_right_region_zone
        if r == 5:
            return True  # study_table_desk_caddy_init_region
        if r == 6:
            return True  # study_table_black_book_init_region
        if r == 7:
            return True  # study_table_white_yellow_mug_init_region
        return True  # study_table_desk_caddy_right_region

    @staticmethod
    def region_x0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.06196)
        if r == 1:
            return Scalar[DTYPE](-0.06196)
        if r == 2:
            return Scalar[DTYPE](-0.02775)
        if r == 3:
            return Scalar[DTYPE](-0.02775)
        if r == 4:
            return Scalar[DTYPE](-0.25)
        if r == 5:
            return Scalar[DTYPE](-0.21000000000000002)
        if r == 6:
            return Scalar[DTYPE](-0.025)
        if r == 7:
            return Scalar[DTYPE](0.07500000000000001)
        return Scalar[DTYPE](-0.25)

    @staticmethod
    def region_y0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.06216)
        if r == 1:
            return Scalar[DTYPE](-0.06216)
        if r == 2:
            return Scalar[DTYPE](-0.06216)
        if r == 3:
            return Scalar[DTYPE](-0.03595)
        if r == 4:
            return Scalar[DTYPE](0.09999999999999999)
        if r == 5:
            return Scalar[DTYPE](-0.15000000000000002)
        if r == 6:
            return Scalar[DTYPE](0.125)
        if r == 7:
            return Scalar[DTYPE](-0.025)
        return Scalar[DTYPE](0.09999999999999999)

    @staticmethod
    def region_x1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.06196)
        if r == 1:
            return Scalar[DTYPE](0.06196)
        if r == 2:
            return Scalar[DTYPE](0.02775)
        if r == 3:
            return Scalar[DTYPE](0.02775)
        if r == 4:
            return Scalar[DTYPE](-0.15000000000000002)
        if r == 5:
            return Scalar[DTYPE](-0.19)
        if r == 6:
            return Scalar[DTYPE](0.025)
        if r == 7:
            return Scalar[DTYPE](0.125)
        return Scalar[DTYPE](-0.15000000000000002)

    @staticmethod
    def region_y1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.06216)
        if r == 1:
            return Scalar[DTYPE](0.06216)
        if r == 2:
            return Scalar[DTYPE](0.06216)
        if r == 3:
            return Scalar[DTYPE](0.03595)
        if r == 4:
            return Scalar[DTYPE](0.2)
        if r == 5:
            return Scalar[DTYPE](-0.13)
        if r == 6:
            return Scalar[DTYPE](0.175)
        if r == 7:
            return Scalar[DTYPE](0.025)
        return Scalar[DTYPE](0.2)

    @staticmethod
    def region_anchored(r: Int) -> Bool:
        if r == 0:
            return True
        if r == 1:
            return True
        if r == 2:
            return True
        if r == 3:
            return True
        if r == 4:
            return False
        if r == 5:
            return False
        if r == 6:
            return False
        if r == 7:
            return False
        return False

    @staticmethod
    def region_contact_has_geom(r: Int) -> Bool:
        if r == 0:
            return True
        if r == 1:
            return True
        if r == 2:
            return True
        if r == 3:
            return True
        if r == 4:
            return False
        if r == 5:
            return False
        if r == 6:
            return False
        if r == 7:
            return False
        return False

    @staticmethod
    def region_contact_top_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.04)
        if r == 1:
            return Scalar[DTYPE](0.04)
        if r == 2:
            return Scalar[DTYPE](0.04)
        if r == 3:
            return Scalar[DTYPE](0.04)
        if r == 4:
            return Scalar[DTYPE](0.0)
        if r == 5:
            return Scalar[DTYPE](0.0)
        if r == 6:
            return Scalar[DTYPE](0.0)
        if r == 7:
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
        return String("")

    @staticmethod
    def joint_qadr(k: Int) -> Int:
        return 0

    @staticmethod
    def joint_dadr(k: Int) -> Int:
        return 0
