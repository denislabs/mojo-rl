"""`libero_spatial`'s device placement table — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run gen-placement-tables
CI checks it with: pixi run gen-placement-tables --check

From `noeira/envs/libero/families/libero_spatial.family`,
`noeira/envs/libero/scenes/libero_spatial.xml` and forward kinematics on it.
5 free slots, 16 regions (3 moving, 3 followed on one slide), 4 drawable joints.
See `placement/table.mojo` for what each method means.
"""

from noeira.tasks.placement.table import PlacementTable


struct LiberoSpatialPlacement(PlacementTable):
    comptime N_SLOTS: Int = 8
    comptime N_FREE: Int = 5
    comptime N_REGIONS: Int = 16
    comptime NQ: Int = 48
    comptime NV: Int = 43
    comptime N_JOINTS: Int = 4
    comptime NBODY: Int = 38
    comptime NSITE: Int = 16
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
            return 3  # akita_black_bowl_1
        if j == 1:
            return 4  # akita_black_bowl_2
        if j == 2:
            return 5  # cookies_1
        if j == 3:
            return 6  # glazed_rim_porcelain_ramekin_1
        return 7  # plate_1

    @staticmethod
    def free_qadr(j: Int) -> Int:
        if j == 0:
            return 13
        if j == 1:
            return 20
        if j == 2:
            return 27
        if j == 3:
            return 34
        return 41

    @staticmethod
    def free_dadr(j: Int) -> Int:
        if j == 0:
            return 13
        if j == 1:
            return 19
        if j == 2:
            return 25
        if j == 3:
            return 31
        return 37

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
            return Scalar[DTYPE](11.5)
        if j == 1:
            return Scalar[DTYPE](12.0)
        if j == 2:
            return Scalar[DTYPE](12.5)
        if j == 3:
            return Scalar[DTYPE](13.0)
        return Scalar[DTYPE](13.5)

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
            return 11
        if r == 1:
            return 12
        if r == 2:
            return 13
        if r == 3:
            return 10
        if r == 4:
            return 14
        if r == 5:
            return 0
        if r == 6:
            return 0
        if r == 7:
            return 0
        if r == 8:
            return 0
        if r == 9:
            return 0
        if r == 10:
            return 0
        if r == 11:
            return 0
        if r == 12:
            return 0
        if r == 13:
            return 0
        if r == 14:
            return 0
        return 0

    @staticmethod
    def region_site_x[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.02215061353603401)
        if r == 1:
            return Scalar[DTYPE](0.02215061353603401)
        if r == 2:
            return Scalar[DTYPE](0.02215061353603401)
        if r == 3:
            return Scalar[DTYPE](0.0200116120156533)
        if r == 4:
            return Scalar[DTYPE](-0.26)
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
        if r == 14:
            return Scalar[DTYPE](0.0)
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_site_y[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.2787397901656337)
        if r == 1:
            return Scalar[DTYPE](-0.2787397901656337)
        if r == 2:
            return Scalar[DTYPE](-0.2787397901656337)
        if r == 3:
            return Scalar[DTYPE](-0.2825822337235548)
        if r == 4:
            return Scalar[DTYPE](-0.14)
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
        if r == 14:
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
            return Scalar[DTYPE](0.9)
        if r == 6:
            return Scalar[DTYPE](0.9)
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
        if r == 14:
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
            return True  # main_table_plate_region
        if r == 6:
            return True  # main_table_next_to_plate_region
        if r == 7:
            return True  # main_table_box_region
        if r == 8:
            return True  # main_table_next_to_box_region
        if r == 9:
            return True  # main_table_between_plate_ramekin_region
        if r == 10:
            return True  # main_table_ramekin_region
        if r == 11:
            return True  # main_table_next_to_ramekin_region
        if r == 12:
            return True  # main_table_table_center
        if r == 13:
            return True  # main_table_table_front
        if r == 14:
            return True  # main_table_cabinet_region
        return True  # main_table_stove_region

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
            return Scalar[DTYPE](0.05)
        if r == 6:
            return Scalar[DTYPE](0.0)
        if r == 7:
            return Scalar[DTYPE](0.06)
        if r == 8:
            return Scalar[DTYPE](0.12)
        if r == 9:
            return Scalar[DTYPE](-0.06)
        if r == 10:
            return Scalar[DTYPE](-0.21)
        if r == 11:
            return Scalar[DTYPE](-0.19)
        if r == 12:
            return Scalar[DTYPE](-0.1)
        if r == 13:
            return Scalar[DTYPE](0.19)
        if r == 14:
            return Scalar[DTYPE](0.02)
        return Scalar[DTYPE](-0.42)

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
            return Scalar[DTYPE](0.19)
        if r == 6:
            return Scalar[DTYPE](0.3)
        if r == 7:
            return Scalar[DTYPE](0.02)
        if r == 8:
            return Scalar[DTYPE](-0.08)
        if r == 9:
            return Scalar[DTYPE](0.19)
        if r == 10:
            return Scalar[DTYPE](0.19)
        if r == 11:
            return Scalar[DTYPE](0.31)
        if r == 12:
            return Scalar[DTYPE](-0.01)
        if r == 13:
            return Scalar[DTYPE](-0.01)
        if r == 14:
            return Scalar[DTYPE](-0.28)
        return Scalar[DTYPE](-0.15)

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
            return Scalar[DTYPE](0.07)
        if r == 6:
            return Scalar[DTYPE](0.02)
        if r == 7:
            return Scalar[DTYPE](0.08)
        if r == 8:
            return Scalar[DTYPE](0.14)
        if r == 9:
            return Scalar[DTYPE](-0.04)
        if r == 10:
            return Scalar[DTYPE](-0.19)
        if r == 11:
            return Scalar[DTYPE](-0.17)
        if r == 12:
            return Scalar[DTYPE](-0.05)
        if r == 13:
            return Scalar[DTYPE](0.21)
        if r == 14:
            return Scalar[DTYPE](0.04)
        return Scalar[DTYPE](-0.4)

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
            return Scalar[DTYPE](0.21)
        if r == 6:
            return Scalar[DTYPE](0.32)
        if r == 7:
            return Scalar[DTYPE](0.04)
        if r == 8:
            return Scalar[DTYPE](-0.06)
        if r == 9:
            return Scalar[DTYPE](0.21)
        if r == 10:
            return Scalar[DTYPE](0.21)
        if r == 11:
            return Scalar[DTYPE](0.33)
        if r == 12:
            return Scalar[DTYPE](0.01)
        if r == 13:
            return Scalar[DTYPE](0.01)
        if r == 14:
            return Scalar[DTYPE](-0.26)
        return Scalar[DTYPE](-0.13)

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
            return False
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
        if r == 14:
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
            return False
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
        if r == 14:
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
        if r == 14:
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
        if r == 14:
            return -1
        return -1

    @staticmethod
    def region_move_axis_x[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.4338837391175582)
        if r == 1:
            return Scalar[DTYPE](-0.4338837391175582)
        if r == 2:
            return Scalar[DTYPE](-0.4338837391175582)
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
        if r == 14:
            return Scalar[DTYPE](0.0)
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_move_axis_y[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.9009688679024193)
        if r == 1:
            return Scalar[DTYPE](-0.9009688679024193)
        if r == 2:
            return Scalar[DTYPE](-0.9009688679024193)
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
        if r == 14:
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
