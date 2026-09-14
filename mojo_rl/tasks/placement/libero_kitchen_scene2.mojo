"""`libero_kitchen_scene2`'s device placement table — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run gen-placement-tables
CI checks it with: pixi run gen-placement-tables --check

From `mojo_rl/tasks/families/libero_kitchen_scene2.family`,
`mojo_rl/tasks/scenes/libero_kitchen_scene2.xml` and forward kinematics on it.
4 free slots, 9 regions (3 moving, 3 followed on one slide), 3 drawable joints.
See `placement/table.mojo` for what each method means.
"""

from mojo_rl.tasks.placement.table import PlacementTable


struct LiberoKitchenScene2Placement(PlacementTable):
    comptime N_SLOTS: Int = 6
    comptime N_FREE: Int = 4
    comptime N_REGIONS: Int = 9
    comptime NQ: Int = 40
    comptime NV: Int = 36
    comptime N_JOINTS: Int = 3

    @staticmethod
    def free_slot(j: Int) -> Int:
        if j == 0:
            return 2  # akita_black_bowl_1
        if j == 1:
            return 3  # akita_black_bowl_2
        if j == 2:
            return 4  # akita_black_bowl_3
        return 5  # plate_1

    @staticmethod
    def free_qadr(j: Int) -> Int:
        if j == 0:
            return 12
        if j == 1:
            return 19
        if j == 2:
            return 26
        return 33

    @staticmethod
    def free_dadr(j: Int) -> Int:
        if j == 0:
            return 12
        if j == 1:
            return 18
        if j == 2:
            return 24
        return 30

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
    def free_bottom_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](-0.06)

    @staticmethod
    def free_top_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.04)

    @staticmethod
    def region_site_x[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.003540000000000002)
        if r == 1:
            return Scalar[DTYPE](-0.0032800000000000012)
        if r == 2:
            return Scalar[DTYPE](-0.0032800000000000012)
        if r == 3:
            return Scalar[DTYPE](-0.0032800000000000012)
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
    def region_site_y[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.31567)
        if r == 1:
            return Scalar[DTYPE](-0.31128)
        if r == 2:
            return Scalar[DTYPE](-0.31128)
        if r == 3:
            return Scalar[DTYPE](-0.31128)
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
            return Scalar[DTYPE](1.12652)
        if r == 1:
            return Scalar[DTYPE](1.09063)
        if r == 2:
            return Scalar[DTYPE](1.0219)
        if r == 3:
            return Scalar[DTYPE](0.95201)
        if r == 4:
            return Scalar[DTYPE](0.9)
        if r == 5:
            return Scalar[DTYPE](0.9)
        if r == 6:
            return Scalar[DTYPE](0.9)
        if r == 7:
            return Scalar[DTYPE](0.9)
        return Scalar[DTYPE](0.9)

    @staticmethod
    def region_has_rect(r: Int) -> Bool:
        if r == 0:
            return True  # wooden_cabinet_1_top_side
        if r == 1:
            return True  # wooden_cabinet_1_top_region
        if r == 2:
            return True  # wooden_cabinet_1_middle_region
        if r == 3:
            return True  # wooden_cabinet_1_bottom_region
        if r == 4:
            return True  # kitchen_table_wooden_cabinet_init_region
        if r == 5:
            return True  # kitchen_table_akita_black_bowl_middle_init_region
        if r == 6:
            return True  # kitchen_table_akita_black_bowl_front_init_region
        if r == 7:
            return True  # kitchen_table_akita_black_bowl_back_init_region
        return True  # kitchen_table_plate_init_region

    @staticmethod
    def region_x0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.12534)
        if r == 1:
            return Scalar[DTYPE](-0.02993)
        if r == 2:
            return Scalar[DTYPE](-0.02993)
        if r == 3:
            return Scalar[DTYPE](-0.02993)
        if r == 4:
            return Scalar[DTYPE](-0.01)
        if r == 5:
            return Scalar[DTYPE](-0.07500000000000001)
        if r == 6:
            return Scalar[DTYPE](0.07500000000000001)
        if r == 7:
            return Scalar[DTYPE](-0.175)
        return Scalar[DTYPE](-0.025)

    @staticmethod
    def region_y0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.09438)
        if r == 1:
            return Scalar[DTYPE](-0.07561)
        if r == 2:
            return Scalar[DTYPE](-0.07561)
        if r == 3:
            return Scalar[DTYPE](-0.07561)
        if r == 4:
            return Scalar[DTYPE](-0.31)
        if r == 5:
            return Scalar[DTYPE](0.17500000000000002)
        if r == 6:
            return Scalar[DTYPE](0.125)
        if r == 7:
            return Scalar[DTYPE](0.025)
        return Scalar[DTYPE](-0.025)

    @staticmethod
    def region_x1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.12534)
        if r == 1:
            return Scalar[DTYPE](0.02993)
        if r == 2:
            return Scalar[DTYPE](0.02993)
        if r == 3:
            return Scalar[DTYPE](0.02993)
        if r == 4:
            return Scalar[DTYPE](0.01)
        if r == 5:
            return Scalar[DTYPE](-0.025)
        if r == 6:
            return Scalar[DTYPE](0.125)
        if r == 7:
            return Scalar[DTYPE](-0.125)
        return Scalar[DTYPE](0.025)

    @staticmethod
    def region_y1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.09438)
        if r == 1:
            return Scalar[DTYPE](0.07561)
        if r == 2:
            return Scalar[DTYPE](0.07561)
        if r == 3:
            return Scalar[DTYPE](0.07561)
        if r == 4:
            return Scalar[DTYPE](-0.29)
        if r == 5:
            return Scalar[DTYPE](0.225)
        if r == 6:
            return Scalar[DTYPE](0.175)
        if r == 7:
            return Scalar[DTYPE](0.07500000000000001)
        return Scalar[DTYPE](0.025)

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
            return Scalar[DTYPE](0.045)
        if r == 1:
            return Scalar[DTYPE](0.045)
        if r == 2:
            return Scalar[DTYPE](0.045)
        if r == 3:
            return Scalar[DTYPE](0.045)
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
        if r == 0:
            return -1
        if r == 1:
            return 0
        if r == 2:
            return 1
        if r == 3:
            return 2
        if r == 4:
            return -1
        if r == 5:
            return -1
        if r == 6:
            return -1
        if r == 7:
            return -1
        return -1

    @staticmethod
    def region_move_axis_x[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.0)
        if r == 1:
            return Scalar[DTYPE](-1.222980050563649e-16)
        if r == 2:
            return Scalar[DTYPE](-1.222980050563649e-16)
        if r == 3:
            return Scalar[DTYPE](-1.222980050563649e-16)
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
    def region_move_axis_y[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.0)
        if r == 1:
            return Scalar[DTYPE](-1.0)
        if r == 2:
            return Scalar[DTYPE](-1.0)
        if r == 3:
            return Scalar[DTYPE](-1.0)
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
    def region_move_axis_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.0)

    @staticmethod
    def joint_name(k: Int) -> String:
        if k == 0:
            return String("wooden_cabinet_1_top_level")
        if k == 1:
            return String("wooden_cabinet_1_middle_level")
        return String("wooden_cabinet_1_bottom_level")

    @staticmethod
    def joint_qadr(k: Int) -> Int:
        if k == 0:
            return 9
        if k == 1:
            return 10
        return 11

    @staticmethod
    def joint_dadr(k: Int) -> Int:
        if k == 0:
            return 9
        if k == 1:
            return 10
        return 11
