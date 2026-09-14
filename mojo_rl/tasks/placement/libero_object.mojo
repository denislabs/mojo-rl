"""`libero_object`'s device placement table — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run gen-placement-tables
CI checks it with: pixi run gen-placement-tables --check

From `mojo_rl/tasks/families/libero_object.family`,
`mojo_rl/tasks/scenes/libero_object.xml` and forward kinematics on it.
11 free slots, 8 regions (1 moving, 0 followed on one slide), 0 drawable joints.
See `placement/table.mojo` for what each method means.
"""

from mojo_rl.tasks.placement.table import PlacementTable


struct LiberoObjectPlacement(PlacementTable):
    comptime N_SLOTS: Int = 12
    comptime N_FREE: Int = 11
    comptime N_REGIONS: Int = 8
    comptime NQ: Int = 86
    comptime NV: Int = 75
    comptime N_JOINTS: Int = 0
    comptime NBODY: Int = 29
    comptime NSITE: Int = 10
    comptime GRIPPER_SITE: Int = 4  # robot_grip_site

    @staticmethod
    def free_slot(j: Int) -> Int:
        if j == 0:
            return 1  # alphabet_soup_1
        if j == 1:
            return 2  # basket_1
        if j == 2:
            return 3  # salad_dressing_1
        if j == 3:
            return 4  # cream_cheese_1
        if j == 4:
            return 5  # milk_1
        if j == 5:
            return 6  # tomato_sauce_1
        if j == 6:
            return 7  # butter_1
        if j == 7:
            return 8  # bbq_sauce_1
        if j == 8:
            return 9  # chocolate_pudding_1
        if j == 9:
            return 10  # ketchup_1
        return 11  # orange_juice_1

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
        if j == 4:
            return 37
        if j == 5:
            return 44
        if j == 6:
            return 51
        if j == 7:
            return 58
        if j == 8:
            return 65
        if j == 9:
            return 72
        return 79

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
        if j == 4:
            return 33
        if j == 5:
            return 39
        if j == 6:
            return 45
        if j == 7:
            return 51
        if j == 8:
            return 57
        if j == 9:
            return 63
        return 69

    @staticmethod
    def free_has_geom(j: Int) -> Bool:
        return True

    @staticmethod
    def free_rest[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        if j == 0:
            return Scalar[DTYPE](0.04)
        if j == 1:
            return Scalar[DTYPE](0.06)
        if j == 2:
            return Scalar[DTYPE](0.06)
        if j == 3:
            return Scalar[DTYPE](0.025)
        if j == 4:
            return Scalar[DTYPE](0.06)
        if j == 5:
            return Scalar[DTYPE](0.06)
        if j == 6:
            return Scalar[DTYPE](0.02)
        if j == 7:
            return Scalar[DTYPE](0.06)
        if j == 8:
            return Scalar[DTYPE](0.06)
        if j == 9:
            return Scalar[DTYPE](0.06)
        return Scalar[DTYPE](0.06)

    @staticmethod
    def free_radius[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        if j == 0:
            return Scalar[DTYPE](0.03535533905932738)
        if j == 1:
            return Scalar[DTYPE](0.03535533905932738)
        if j == 2:
            return Scalar[DTYPE](0.03535533905932738)
        if j == 3:
            return Scalar[DTYPE](0.042426406871192854)
        if j == 4:
            return Scalar[DTYPE](0.042426406871192854)
        if j == 5:
            return Scalar[DTYPE](0.03535533905932738)
        if j == 6:
            return Scalar[DTYPE](0.0282842712474619)
        if j == 7:
            return Scalar[DTYPE](0.021213203435596427)
        if j == 8:
            return Scalar[DTYPE](0.03535533905932738)
        if j == 9:
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
        if j == 4:
            return Scalar[DTYPE](12.5)
        if j == 5:
            return Scalar[DTYPE](13.0)
        if j == 6:
            return Scalar[DTYPE](13.5)
        if j == 7:
            return Scalar[DTYPE](14.0)
        if j == 8:
            return Scalar[DTYPE](14.5)
        if j == 9:
            return Scalar[DTYPE](15.0)
        return Scalar[DTYPE](15.5)

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
            return Scalar[DTYPE](-0.06)
        if j == 2:
            return Scalar[DTYPE](-0.06)
        if j == 3:
            return Scalar[DTYPE](-0.025)
        if j == 4:
            return Scalar[DTYPE](-0.06)
        if j == 5:
            return Scalar[DTYPE](-0.06)
        if j == 6:
            return Scalar[DTYPE](-0.02)
        if j == 7:
            return Scalar[DTYPE](-0.06)
        if j == 8:
            return Scalar[DTYPE](-0.06)
        if j == 9:
            return Scalar[DTYPE](-0.06)
        return Scalar[DTYPE](-0.06)

    @staticmethod
    def free_top_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        if j == 0:
            return Scalar[DTYPE](0.04)
        if j == 1:
            return Scalar[DTYPE](0.04)
        if j == 2:
            return Scalar[DTYPE](0.04)
        if j == 3:
            return Scalar[DTYPE](0.025)
        if j == 4:
            return Scalar[DTYPE](0.04)
        if j == 5:
            return Scalar[DTYPE](0.04)
        if j == 6:
            return Scalar[DTYPE](0.02)
        if j == 7:
            return Scalar[DTYPE](0.06)
        if j == 8:
            return Scalar[DTYPE](0.04)
        if j == 9:
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
        if r == 5:
            return 0
        if r == 6:
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
            return Scalar[DTYPE](-0.035)
        if r == 2:
            return Scalar[DTYPE](-0.035)
        if r == 3:
            return Scalar[DTYPE](-0.035)
        if r == 4:
            return Scalar[DTYPE](-0.035)
        if r == 5:
            return Scalar[DTYPE](-0.035)
        if r == 6:
            return Scalar[DTYPE](-0.035)
        return Scalar[DTYPE](-0.035)

    @staticmethod
    def region_has_rect(r: Int) -> Bool:
        if r == 0:
            return True  # basket_1_contain_region
        if r == 1:
            return True  # floor_bin_region
        if r == 2:
            return True  # floor_target_object_region
        if r == 3:
            return True  # floor_other_object_region_0
        if r == 4:
            return True  # floor_other_object_region_1
        if r == 5:
            return True  # floor_other_object_region_2
        if r == 6:
            return True  # floor_other_object_region_3
        return True  # floor_other_object_region_4

    @staticmethod
    def region_x0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.06108)
        if r == 1:
            return Scalar[DTYPE](-0.01)
        if r == 2:
            return Scalar[DTYPE](-0.145)
        if r == 3:
            return Scalar[DTYPE](0.025)
        if r == 4:
            return Scalar[DTYPE](-0.175)
        if r == 5:
            return Scalar[DTYPE](0.07500000000000001)
        if r == 6:
            return Scalar[DTYPE](0.125)
        return Scalar[DTYPE](-0.225)

    @staticmethod
    def region_y0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.06108)
        if r == 1:
            return Scalar[DTYPE](0.25)
        if r == 2:
            return Scalar[DTYPE](-0.265)
        if r == 3:
            return Scalar[DTYPE](-0.125)
        if r == 4:
            return Scalar[DTYPE](0.034999999999999996)
        if r == 5:
            return Scalar[DTYPE](-0.225)
        if r == 6:
            return Scalar[DTYPE](0.0049999999999999975)
        return Scalar[DTYPE](-0.10500000000000001)

    @staticmethod
    def region_x1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.06108)
        if r == 1:
            return Scalar[DTYPE](0.01)
        if r == 2:
            return Scalar[DTYPE](-0.095)
        if r == 3:
            return Scalar[DTYPE](0.07500000000000001)
        if r == 4:
            return Scalar[DTYPE](-0.125)
        if r == 5:
            return Scalar[DTYPE](0.125)
        if r == 6:
            return Scalar[DTYPE](0.175)
        return Scalar[DTYPE](-0.17500000000000002)

    @staticmethod
    def region_y1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.06108)
        if r == 1:
            return Scalar[DTYPE](0.27)
        if r == 2:
            return Scalar[DTYPE](-0.215)
        if r == 3:
            return Scalar[DTYPE](-0.07500000000000001)
        if r == 4:
            return Scalar[DTYPE](0.08499999999999999)
        if r == 5:
            return Scalar[DTYPE](-0.17500000000000002)
        if r == 6:
            return Scalar[DTYPE](0.055)
        return Scalar[DTYPE](-0.055)

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
        if r == 5:
            return False
        if r == 6:
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
        if r == 5:
            return False
        if r == 6:
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
        if r == 5:
            return Scalar[DTYPE](0.0)
        if r == 6:
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
        if r == 5:
            return -1
        if r == 6:
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
