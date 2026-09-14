"""`libero_goal`'s device placement table — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run gen-placement-tables
CI checks it with: pixi run gen-placement-tables --check

From `mojo_rl/tasks/families/libero_goal.family`,
`mojo_rl/tasks/scenes/libero_goal.xml` and forward kinematics on it.
4 free slots, 15 regions (3 moving, 3 followed on one slide), 4 drawable joints.
See `placement/table.mojo` for what each method means.
"""

from mojo_rl.tasks.placement.table import PlacementTable


struct LiberoGoalPlacement(PlacementTable):
    comptime N_SLOTS: Int = 8
    comptime N_FREE: Int = 4
    comptime N_REGIONS: Int = 15
    comptime NQ: Int = 41
    comptime NV: Int = 37
    comptime N_JOINTS: Int = 4

    @staticmethod
    def free_slot(j: Int) -> Int:
        if j == 0:
            return 4  # akita_black_bowl_1
        if j == 1:
            return 5  # cream_cheese_1
        if j == 2:
            return 6  # wine_bottle_1
        return 7  # plate_1

    @staticmethod
    def free_qadr(j: Int) -> Int:
        if j == 0:
            return 13
        if j == 1:
            return 20
        if j == 2:
            return 27
        return 34

    @staticmethod
    def free_dadr(j: Int) -> Int:
        if j == 0:
            return 13
        if j == 1:
            return 19
        if j == 2:
            return 25
        return 31

    @staticmethod
    def free_has_geom(j: Int) -> Bool:
        return True

    @staticmethod
    def free_rest[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        if j == 0:
            return Scalar[DTYPE](0.06)
        if j == 1:
            return Scalar[DTYPE](0.025)
        if j == 2:
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
        return Scalar[DTYPE](0.03535533905932738)

    @staticmethod
    def free_bottom_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        if j == 0:
            return Scalar[DTYPE](-0.06)
        if j == 1:
            return Scalar[DTYPE](-0.025)
        if j == 2:
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
        return Scalar[DTYPE](0.04)

    @staticmethod
    def region_site_x[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.026719999999999997)
        if r == 1:
            return Scalar[DTYPE](0.026719999999999997)
        if r == 2:
            return Scalar[DTYPE](0.026719999999999997)
        if r == 3:
            return Scalar[DTYPE](0.026459999999999997)
        if r == 4:
            return Scalar[DTYPE](-0.26)
        if r == 5:
            return Scalar[DTYPE](-0.26)
        if r == 6:
            return Scalar[DTYPE](0.0)
        if r == 7:
            return Scalar[DTYPE](0.0)
        if r == 8:
            return Scalar[DTYPE](0.0)
        if r == 9:
            return Scalar[DTYPE](0.0)
        if r == 10:
            return Scalar[DTYPE](0.0)
        if r == 11:
            return Scalar[DTYPE](0.0)
        if r == 12:
            return Scalar[DTYPE](0.0)
        if r == 13:
            return Scalar[DTYPE](0.0)
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_site_y[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.25128)
        if r == 1:
            return Scalar[DTYPE](-0.25128)
        if r == 2:
            return Scalar[DTYPE](-0.25128)
        if r == 3:
            return Scalar[DTYPE](-0.25567)
        if r == 4:
            return Scalar[DTYPE](0.21)
        if r == 5:
            return Scalar[DTYPE](-0.17997000000000002)
        if r == 6:
            return Scalar[DTYPE](0.0)
        if r == 7:
            return Scalar[DTYPE](0.0)
        if r == 8:
            return Scalar[DTYPE](0.0)
        if r == 9:
            return Scalar[DTYPE](0.0)
        if r == 10:
            return Scalar[DTYPE](0.0)
        if r == 11:
            return Scalar[DTYPE](0.0)
        if r == 12:
            return Scalar[DTYPE](0.0)
        if r == 13:
            return Scalar[DTYPE](0.0)
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_site_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](1.09063)
        if r == 1:
            return Scalar[DTYPE](1.0219)
        if r == 2:
            return Scalar[DTYPE](0.95201)
        if r == 3:
            return Scalar[DTYPE](1.12652)
        if r == 4:
            return Scalar[DTYPE](0.905)
        if r == 5:
            return Scalar[DTYPE](1.13745)
        if r == 6:
            return Scalar[DTYPE](0.895)
        if r == 7:
            return Scalar[DTYPE](0.9)
        if r == 8:
            return Scalar[DTYPE](0.9)
        if r == 9:
            return Scalar[DTYPE](0.9)
        if r == 10:
            return Scalar[DTYPE](0.9)
        if r == 11:
            return Scalar[DTYPE](0.9)
        if r == 12:
            return Scalar[DTYPE](0.9)
        if r == 13:
            return Scalar[DTYPE](0.9)
        return Scalar[DTYPE](0.9)

    @staticmethod
    def region_has_rect(r: Int) -> Bool:
        if r == 0:
            return True  # wooden_cabinet_1_top_region
        if r == 1:
            return True  # wooden_cabinet_1_middle_region
        if r == 2:
            return True  # wooden_cabinet_1_bottom_region
        if r == 3:
            return True  # wooden_cabinet_1_top_side
        if r == 4:
            return True  # flat_stove_1_cook_region
        if r == 5:
            return True  # wine_rack_1_top_region
        if r == 6:
            return True  # main_table_stove_front_region_zone
        if r == 7:
            return True  # main_table_plate_region
        if r == 8:
            return True  # main_table_akita_black_bowl_region
        if r == 9:
            return True  # main_table_wine_bottle_region
        if r == 10:
            return True  # main_table_cream_cheese_region
        if r == 11:
            return True  # main_table_stove_front_region
        if r == 12:
            return True  # main_table_cabinet_region
        if r == 13:
            return True  # main_table_stove_region
        return True  # main_table_wine_rack_region

    @staticmethod
    def region_x0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.02993)
        if r == 1:
            return Scalar[DTYPE](-0.02993)
        if r == 2:
            return Scalar[DTYPE](-0.02993)
        if r == 3:
            return Scalar[DTYPE](-0.12534)
        if r == 4:
            return Scalar[DTYPE](-0.075)
        if r == 5:
            return Scalar[DTYPE](-0.10087)
        if r == 6:
            return Scalar[DTYPE](-0.09)
        if r == 7:
            return Scalar[DTYPE](0.04)
        if r == 8:
            return Scalar[DTYPE](-0.09999999999999999)
        if r == 9:
            return Scalar[DTYPE](-0.21000000000000002)
        if r == 10:
            return Scalar[DTYPE](-0.060000000000000005)
        if r == 11:
            return Scalar[DTYPE](-0.09)
        if r == 12:
            return Scalar[DTYPE](0.02)
        if r == 13:
            return Scalar[DTYPE](-0.42)
        return Scalar[DTYPE](-0.27)

    @staticmethod
    def region_y0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.07561)
        if r == 1:
            return Scalar[DTYPE](-0.07561)
        if r == 2:
            return Scalar[DTYPE](-0.07561)
        if r == 3:
            return Scalar[DTYPE](-0.09438)
        if r == 4:
            return Scalar[DTYPE](-0.075)
        if r == 5:
            return Scalar[DTYPE](-0.022)
        if r == 6:
            return Scalar[DTYPE](0.16999999999999998)
        if r == 7:
            return Scalar[DTYPE](-0.03)
        if r == 8:
            return Scalar[DTYPE](-0.01)
        if r == 9:
            return Scalar[DTYPE](-0.060000000000000005)
        if r == 10:
            return Scalar[DTYPE](0.12000000000000001)
        if r == 11:
            return Scalar[DTYPE](0.16999999999999998)
        if r == 12:
            return Scalar[DTYPE](-0.25)
        if r == 13:
            return Scalar[DTYPE](0.2)
        return Scalar[DTYPE](-0.27)

    @staticmethod
    def region_x1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.02993)
        if r == 1:
            return Scalar[DTYPE](0.02993)
        if r == 2:
            return Scalar[DTYPE](0.02993)
        if r == 3:
            return Scalar[DTYPE](0.12534)
        if r == 4:
            return Scalar[DTYPE](0.075)
        if r == 5:
            return Scalar[DTYPE](0.10087)
        if r == 6:
            return Scalar[DTYPE](-0.010000000000000002)
        if r == 7:
            return Scalar[DTYPE](0.060000000000000005)
        if r == 8:
            return Scalar[DTYPE](-0.08)
        if r == 9:
            return Scalar[DTYPE](-0.19)
        if r == 10:
            return Scalar[DTYPE](-0.04)
        if r == 11:
            return Scalar[DTYPE](-0.010000000000000002)
        if r == 12:
            return Scalar[DTYPE](0.04)
        if r == 13:
            return Scalar[DTYPE](-0.4)
        return Scalar[DTYPE](-0.25)

    @staticmethod
    def region_y1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.07561)
        if r == 1:
            return Scalar[DTYPE](0.07561)
        if r == 2:
            return Scalar[DTYPE](0.07561)
        if r == 3:
            return Scalar[DTYPE](0.09438)
        if r == 4:
            return Scalar[DTYPE](0.075)
        if r == 5:
            return Scalar[DTYPE](0.022)
        if r == 6:
            return Scalar[DTYPE](0.25)
        if r == 7:
            return Scalar[DTYPE](-0.01)
        if r == 8:
            return Scalar[DTYPE](0.01)
        if r == 9:
            return Scalar[DTYPE](-0.04)
        if r == 10:
            return Scalar[DTYPE](0.14)
        if r == 11:
            return Scalar[DTYPE](0.25)
        if r == 12:
            return Scalar[DTYPE](-0.23)
        if r == 13:
            return Scalar[DTYPE](0.22)
        return Scalar[DTYPE](-0.25)

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
            return True
        if r == 5:
            return True
        if r == 6:
            return False
        if r == 7:
            return False
        if r == 8:
            return False
        if r == 9:
            return False
        if r == 10:
            return False
        if r == 11:
            return False
        if r == 12:
            return False
        if r == 13:
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
            return True
        if r == 5:
            return True
        if r == 6:
            return False
        if r == 7:
            return False
        if r == 8:
            return False
        if r == 9:
            return False
        if r == 10:
            return False
        if r == 11:
            return False
        if r == 12:
            return False
        if r == 13:
            return False
        return False

    @staticmethod
    def region_contact_top_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.045)
        if r == 1:
            return Scalar[DTYPE](0.045)
        if r == 2:
            return Scalar[DTYPE](0.045)
        if r == 3:
            return Scalar[DTYPE](0.045)
        if r == 4:
            return Scalar[DTYPE](0.045)
        if r == 5:
            return Scalar[DTYPE](0.04)
        if r == 6:
            return Scalar[DTYPE](0.0)
        if r == 7:
            return Scalar[DTYPE](0.0)
        if r == 8:
            return Scalar[DTYPE](0.0)
        if r == 9:
            return Scalar[DTYPE](0.0)
        if r == 10:
            return Scalar[DTYPE](0.0)
        if r == 11:
            return Scalar[DTYPE](0.0)
        if r == 12:
            return Scalar[DTYPE](0.0)
        if r == 13:
            return Scalar[DTYPE](0.0)
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_move_joint(r: Int) -> Int:
        if r == 0:
            return 0
        if r == 1:
            return 1
        if r == 2:
            return 2
        if r == 3:
            return -1
        if r == 4:
            return -1
        if r == 5:
            return -1
        if r == 6:
            return -1
        if r == 7:
            return -1
        if r == 8:
            return -1
        if r == 9:
            return -1
        if r == 10:
            return -1
        if r == 11:
            return -1
        if r == 12:
            return -1
        if r == 13:
            return -1
        return -1

    @staticmethod
    def region_move_axis_x[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-1.214306433183765e-16)
        if r == 1:
            return Scalar[DTYPE](-1.214306433183765e-16)
        if r == 2:
            return Scalar[DTYPE](-1.214306433183765e-16)
        if r == 3:
            return Scalar[DTYPE](0.0)
        if r == 4:
            return Scalar[DTYPE](0.0)
        if r == 5:
            return Scalar[DTYPE](0.0)
        if r == 6:
            return Scalar[DTYPE](0.0)
        if r == 7:
            return Scalar[DTYPE](0.0)
        if r == 8:
            return Scalar[DTYPE](0.0)
        if r == 9:
            return Scalar[DTYPE](0.0)
        if r == 10:
            return Scalar[DTYPE](0.0)
        if r == 11:
            return Scalar[DTYPE](0.0)
        if r == 12:
            return Scalar[DTYPE](0.0)
        if r == 13:
            return Scalar[DTYPE](0.0)
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_move_axis_y[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-1.0)
        if r == 1:
            return Scalar[DTYPE](-1.0)
        if r == 2:
            return Scalar[DTYPE](-1.0)
        if r == 3:
            return Scalar[DTYPE](0.0)
        if r == 4:
            return Scalar[DTYPE](0.0)
        if r == 5:
            return Scalar[DTYPE](0.0)
        if r == 6:
            return Scalar[DTYPE](0.0)
        if r == 7:
            return Scalar[DTYPE](0.0)
        if r == 8:
            return Scalar[DTYPE](0.0)
        if r == 9:
            return Scalar[DTYPE](0.0)
        if r == 10:
            return Scalar[DTYPE](0.0)
        if r == 11:
            return Scalar[DTYPE](0.0)
        if r == 12:
            return Scalar[DTYPE](0.0)
        if r == 13:
            return Scalar[DTYPE](0.0)
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_move_axis_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.0)

    @staticmethod
    def joint_name(k: Int) -> String:
        if k == 0:
            return String("wooden_cabinet_1_top_level")
        if k == 1:
            return String("wooden_cabinet_1_middle_level")
        if k == 2:
            return String("wooden_cabinet_1_bottom_level")
        return String("flat_stove_1_button")

    @staticmethod
    def joint_qadr(k: Int) -> Int:
        if k == 0:
            return 9
        if k == 1:
            return 10
        if k == 2:
            return 11
        return 12

    @staticmethod
    def joint_dadr(k: Int) -> Int:
        if k == 0:
            return 9
        if k == 1:
            return 10
        if k == 2:
            return 11
        return 12
