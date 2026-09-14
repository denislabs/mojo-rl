"""`libero_kitchen_scene9`'s device placement table — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run gen-placement-tables
CI checks it with: pixi run gen-placement-tables --check

From `mojo_rl/tasks/families/libero_kitchen_scene9.family`,
`mojo_rl/tasks/scenes/libero_kitchen_scene9.xml` and forward kinematics on it.
2 free slots, 8 regions (0 moving, 0 followed on one slide), 1 drawable joints.
See `placement/table.mojo` for what each method means.
"""

from mojo_rl.tasks.placement.table import PlacementTable


struct LiberoKitchenScene9Placement(PlacementTable):
    comptime N_SLOTS: Int = 5
    comptime N_FREE: Int = 2
    comptime N_REGIONS: Int = 8
    comptime NQ: Int = 24
    comptime NV: Int = 22
    comptime N_JOINTS: Int = 1

    @staticmethod
    def free_slot(j: Int) -> Int:
        if j == 0:
            return 3  # white_bowl_1
        return 4  # chefmate_8_frypan_1

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
    def free_bottom_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](-0.06)

    @staticmethod
    def free_top_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.04)

    @staticmethod
    def region_site_x[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.05000000000000002)
        if r == 1:
            return Scalar[DTYPE](-0.004960000000000002)
        if r == 2:
            return Scalar[DTYPE](-0.005380000000000002)
        if r == 3:
            return Scalar[DTYPE](-0.005550000000000002)
        if r == 4:
            return Scalar[DTYPE](0.0)
        if r == 5:
            return Scalar[DTYPE](0.0)
        if r == 6:
            return Scalar[DTYPE](0.0)
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_site_y[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.3)
        if r == 1:
            return Scalar[DTYPE](-0.26567)
        if r == 2:
            return Scalar[DTYPE](-0.26687)
        if r == 3:
            return Scalar[DTYPE](-0.26687)
        if r == 4:
            return Scalar[DTYPE](0.0)
        if r == 5:
            return Scalar[DTYPE](0.0)
        if r == 6:
            return Scalar[DTYPE](0.0)
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_site_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.905)
        if r == 1:
            return Scalar[DTYPE](1.12688)
        if r == 2:
            return Scalar[DTYPE](1.05672)
        if r == 3:
            return Scalar[DTYPE](0.9340700000000001)
        if r == 4:
            return Scalar[DTYPE](0.9)
        if r == 5:
            return Scalar[DTYPE](0.9)
        if r == 6:
            return Scalar[DTYPE](0.9)
        return Scalar[DTYPE](0.9)

    @staticmethod
    def region_has_rect(r: Int) -> Bool:
        if r == 0:
            return True  # flat_stove_1_cook_region
        if r == 1:
            return True  # wooden_two_layer_shelf_1_top_side
        if r == 2:
            return True  # wooden_two_layer_shelf_1_top_region
        if r == 3:
            return True  # wooden_two_layer_shelf_1_bottom_region
        if r == 4:
            return True  # kitchen_table_flat_stove_init_region
        if r == 5:
            return True  # kitchen_table_wooden_two_layer_shelf_init_region
        if r == 6:
            return True  # kitchen_table_frypan_init_region
        return True  # kitchen_table_white_bowl_init_region

    @staticmethod
    def region_x0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.075)
        if r == 1:
            return Scalar[DTYPE](-0.17557)
        if r == 2:
            return Scalar[DTYPE](-0.06739)
        if r == 3:
            return Scalar[DTYPE](-0.03528)
        if r == 4:
            return Scalar[DTYPE](-0.21000000000000002)
        if r == 5:
            return Scalar[DTYPE](-0.01)
        if r == 6:
            return Scalar[DTYPE](0.025)
        return Scalar[DTYPE](-0.175)

    @staticmethod
    def region_y0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.075)
        if r == 1:
            return Scalar[DTYPE](-0.09438)
        if r == 2:
            return Scalar[DTYPE](-0.08564)
        if r == 3:
            return Scalar[DTYPE](-0.08564)
        if r == 4:
            return Scalar[DTYPE](0.29)
        if r == 5:
            return Scalar[DTYPE](-0.26)
        if r == 6:
            return Scalar[DTYPE](-0.025)
        return Scalar[DTYPE](0.07500000000000001)

    @staticmethod
    def region_x1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.075)
        if r == 1:
            return Scalar[DTYPE](0.17557)
        if r == 2:
            return Scalar[DTYPE](0.06739)
        if r == 3:
            return Scalar[DTYPE](0.03528)
        if r == 4:
            return Scalar[DTYPE](-0.19)
        if r == 5:
            return Scalar[DTYPE](0.01)
        if r == 6:
            return Scalar[DTYPE](0.07500000000000001)
        return Scalar[DTYPE](-0.125)

    @staticmethod
    def region_y1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.075)
        if r == 1:
            return Scalar[DTYPE](0.09438)
        if r == 2:
            return Scalar[DTYPE](0.08564)
        if r == 3:
            return Scalar[DTYPE](0.08564)
        if r == 4:
            return Scalar[DTYPE](0.31)
        if r == 5:
            return Scalar[DTYPE](-0.24)
        if r == 6:
            return Scalar[DTYPE](0.025)
        return Scalar[DTYPE](0.125)

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
        return False

    @staticmethod
    def region_contact_top_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.045)
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
