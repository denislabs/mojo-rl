"""`libero_kitchen_scene10`'s device placement table — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run gen-placement-tables
CI checks it with: pixi run gen-placement-tables --check

From `mojo_rl/tasks/families/libero_kitchen_scene10.family`,
`mojo_rl/tasks/scenes/libero_kitchen_scene10.xml` and forward kinematics on it.
4 free slots, 9 regions, 3 of them moving.
See `placement/table.mojo` for what each method means.
"""

from mojo_rl.tasks.placement.table import PlacementTable


struct LiberoKitchenScene10Placement(PlacementTable):
    comptime N_SLOTS: Int = 6
    comptime N_FREE: Int = 4
    comptime N_REGIONS: Int = 9
    comptime NQ: Int = 40
    comptime NV: Int = 36

    @staticmethod
    def free_slot(j: Int) -> Int:
        if j == 0:
            return 2  # akita_black_bowl_1
        if j == 1:
            return 3  # butter_1
        if j == 2:
            return 4  # butter_2
        return 5  # chocolate_pudding_1

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
        if j == 0:
            return Scalar[DTYPE](0.06)
        if j == 1:
            return Scalar[DTYPE](0.02)
        if j == 2:
            return Scalar[DTYPE](0.02)
        return Scalar[DTYPE](0.06)

    @staticmethod
    def free_radius[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        if j == 0:
            return Scalar[DTYPE](0.03535533905932738)
        if j == 1:
            return Scalar[DTYPE](0.0282842712474619)
        if j == 2:
            return Scalar[DTYPE](0.0282842712474619)
        return Scalar[DTYPE](0.03535533905932738)

    @staticmethod
    def free_bottom_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        if j == 0:
            return Scalar[DTYPE](-0.06)
        if j == 1:
            return Scalar[DTYPE](-0.02)
        if j == 2:
            return Scalar[DTYPE](-0.02)
        return Scalar[DTYPE](-0.06)

    @staticmethod
    def free_top_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        if j == 0:
            return Scalar[DTYPE](0.04)
        if j == 1:
            return Scalar[DTYPE](0.02)
        if j == 2:
            return Scalar[DTYPE](0.02)
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
            return True  # kitchen_table_akita_black_bowl_init_region
        if r == 6:
            return True  # kitchen_table_butter_back_init_region
        if r == 7:
            return True  # kitchen_table_butter_front_init_region
        return True  # kitchen_table_chocolate_pudding_init_region

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
            return Scalar[DTYPE](-0.125)
        if r == 6:
            return Scalar[DTYPE](-0.125)
        if r == 7:
            return Scalar[DTYPE](-0.025)
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
            return Scalar[DTYPE](-0.025)
        if r == 6:
            return Scalar[DTYPE](0.17500000000000002)
        if r == 7:
            return Scalar[DTYPE](0.17500000000000002)
        return Scalar[DTYPE](0.025)

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
            return Scalar[DTYPE](-0.07500000000000001)
        if r == 6:
            return Scalar[DTYPE](-0.07500000000000001)
        if r == 7:
            return Scalar[DTYPE](0.025)
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
            return Scalar[DTYPE](0.025)
        if r == 6:
            return Scalar[DTYPE](0.225)
        if r == 7:
            return Scalar[DTYPE](0.225)
        return Scalar[DTYPE](0.07500000000000001)

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
    def region_moves(r: Int) -> Bool:
        if r == 0:
            return False
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
