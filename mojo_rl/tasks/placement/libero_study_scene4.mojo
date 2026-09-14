"""`libero_study_scene4`'s device placement table — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run gen-placement-tables
CI checks it with: pixi run gen-placement-tables --check

From `mojo_rl/tasks/families/libero_study_scene4.family`,
`mojo_rl/tasks/scenes/libero_study_scene4.xml` and forward kinematics on it.
3 free slots, 7 regions (0 moving, 0 followed on one slide), 0 drawable joints.
See `placement/table.mojo` for what each method means.
"""

from mojo_rl.tasks.placement.table import PlacementTable


struct LiberoStudyScene4Placement(PlacementTable):
    comptime N_SLOTS: Int = 5
    comptime N_FREE: Int = 3
    comptime N_REGIONS: Int = 7
    comptime NQ: Int = 30
    comptime NV: Int = 27
    comptime N_JOINTS: Int = 0

    @staticmethod
    def free_slot(j: Int) -> Int:
        if j == 0:
            return 2  # black_book_1
        if j == 1:
            return 3  # yellow_book_1
        return 4  # yellow_book_2

    @staticmethod
    def free_qadr(j: Int) -> Int:
        if j == 0:
            return 9
        if j == 1:
            return 16
        return 23

    @staticmethod
    def free_dadr(j: Int) -> Int:
        if j == 0:
            return 9
        if j == 1:
            return 15
        return 21

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
            return Scalar[DTYPE](-0.19504000000000002)
        if r == 1:
            return Scalar[DTYPE](-0.19462000000000002)
        if r == 2:
            return Scalar[DTYPE](-0.19445)
        if r == 3:
            return Scalar[DTYPE](-0.2)
        if r == 4:
            return Scalar[DTYPE](-0.2)
        if r == 5:
            return Scalar[DTYPE](-0.2)
        return Scalar[DTYPE](-0.2)

    @staticmethod
    def region_site_y[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.29567000000000004)
        if r == 1:
            return Scalar[DTYPE](0.29687)
        if r == 2:
            return Scalar[DTYPE](0.29687)
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
            return Scalar[DTYPE](1.09388)
        if r == 1:
            return Scalar[DTYPE](1.02372)
        if r == 2:
            return Scalar[DTYPE](0.90107)
        if r == 3:
            return Scalar[DTYPE](0.867)
        if r == 4:
            return Scalar[DTYPE](0.867)
        if r == 5:
            return Scalar[DTYPE](0.867)
        return Scalar[DTYPE](0.867)

    @staticmethod
    def region_has_rect(r: Int) -> Bool:
        if r == 0:
            return True  # wooden_two_layer_shelf_1_top_side
        if r == 1:
            return True  # wooden_two_layer_shelf_1_top_region
        if r == 2:
            return True  # wooden_two_layer_shelf_1_bottom_region
        if r == 3:
            return True  # study_table_yellow_book_right_init_region
        if r == 4:
            return True  # study_table_yellow_book_left_init_region
        if r == 5:
            return True  # study_table_black_book_init_region
        return True  # study_table_wooden_two_layer_shelf_init_region

    @staticmethod
    def region_x0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.17557)
        if r == 1:
            return Scalar[DTYPE](-0.06739)
        if r == 2:
            return Scalar[DTYPE](-0.03528)
        if r == 3:
            return Scalar[DTYPE](-0.01)
        if r == 4:
            return Scalar[DTYPE](-0.060000000000000005)
        if r == 5:
            return Scalar[DTYPE](0.04)
        return Scalar[DTYPE](-0.01)

    @staticmethod
    def region_y0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.09438)
        if r == 1:
            return Scalar[DTYPE](-0.08564)
        if r == 2:
            return Scalar[DTYPE](-0.08564)
        if r == 3:
            return Scalar[DTYPE](-0.01)
        if r == 4:
            return Scalar[DTYPE](-0.26)
        if r == 5:
            return Scalar[DTYPE](-0.16)
        return Scalar[DTYPE](0.27)

    @staticmethod
    def region_x1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.17557)
        if r == 1:
            return Scalar[DTYPE](0.06739)
        if r == 2:
            return Scalar[DTYPE](0.03528)
        if r == 3:
            return Scalar[DTYPE](0.01)
        if r == 4:
            return Scalar[DTYPE](-0.04)
        if r == 5:
            return Scalar[DTYPE](0.060000000000000005)
        return Scalar[DTYPE](0.01)

    @staticmethod
    def region_y1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.09438)
        if r == 1:
            return Scalar[DTYPE](0.08564)
        if r == 2:
            return Scalar[DTYPE](0.08564)
        if r == 3:
            return Scalar[DTYPE](0.01)
        if r == 4:
            return Scalar[DTYPE](-0.24)
        if r == 5:
            return Scalar[DTYPE](-0.13999999999999999)
        return Scalar[DTYPE](0.29000000000000004)

    @staticmethod
    def region_anchored(r: Int) -> Bool:
        if r == 0:
            return True
        if r == 1:
            return True
        if r == 2:
            return True
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
            return True
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
            return Scalar[DTYPE](0.04)
        if r == 1:
            return Scalar[DTYPE](0.04)
        if r == 2:
            return Scalar[DTYPE](0.04)
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
        return String("")

    @staticmethod
    def joint_qadr(k: Int) -> Int:
        return 0

    @staticmethod
    def joint_dadr(k: Int) -> Int:
        return 0
