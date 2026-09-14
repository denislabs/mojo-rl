"""`libero_living_room_scene5`'s device placement table — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run gen-placement-tables
CI checks it with: pixi run gen-placement-tables --check

From `mojo_rl/tasks/families/libero_living_room_scene5.family`,
`mojo_rl/tasks/scenes/libero_living_room_scene5.xml` and forward kinematics on it.
5 free slots, 5 regions, 0 of them moving.
See `placement/table.mojo` for what each method means.
"""

from mojo_rl.tasks.placement.table import PlacementTable


struct LiberoLivingRoomScene5Placement(PlacementTable):
    comptime N_SLOTS: Int = 6
    comptime N_FREE: Int = 5
    comptime N_REGIONS: Int = 5
    comptime NQ: Int = 44
    comptime NV: Int = 39

    @staticmethod
    def free_slot(j: Int) -> Int:
        if j == 0:
            return 1  # porcelain_mug_1
        if j == 1:
            return 2  # red_coffee_mug_1
        if j == 2:
            return 3  # white_yellow_mug_1
        if j == 3:
            return 4  # plate_1
        return 5  # plate_2

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
    def free_bottom_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](-0.06)

    @staticmethod
    def free_top_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.04)

    @staticmethod
    def region_site_x[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_site_y[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_site_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.41)

    @staticmethod
    def region_has_rect(r: Int) -> Bool:
        if r == 0:
            return True  # living_room_table_plate_left_region
        if r == 1:
            return True  # living_room_table_plate_right_region
        if r == 2:
            return True  # living_room_table_porcelain_mug_init_region
        if r == 3:
            return True  # living_room_table_white_yellow_mug_init_region
        return True  # living_room_table_red_coffee_mug_init_region

    @staticmethod
    def region_x0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.025)
        if r == 1:
            return Scalar[DTYPE](-0.025)
        if r == 2:
            return Scalar[DTYPE](-0.125)
        if r == 3:
            return Scalar[DTYPE](-0.07500000000000001)
        return Scalar[DTYPE](-0.225)

    @staticmethod
    def region_y0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.325)
        if r == 1:
            return Scalar[DTYPE](0.27499999999999997)
        if r == 2:
            return Scalar[DTYPE](-0.175)
        if r == 3:
            return Scalar[DTYPE](0.07500000000000001)
        return Scalar[DTYPE](-0.025)

    @staticmethod
    def region_x1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.025)
        if r == 1:
            return Scalar[DTYPE](0.025)
        if r == 2:
            return Scalar[DTYPE](-0.07500000000000001)
        if r == 3:
            return Scalar[DTYPE](-0.025)
        return Scalar[DTYPE](-0.17500000000000002)

    @staticmethod
    def region_y1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.27499999999999997)
        if r == 1:
            return Scalar[DTYPE](0.325)
        if r == 2:
            return Scalar[DTYPE](-0.125)
        if r == 3:
            return Scalar[DTYPE](0.125)
        return Scalar[DTYPE](0.025)

    @staticmethod
    def region_anchored(r: Int) -> Bool:
        return False

    @staticmethod
    def region_contact_has_geom(r: Int) -> Bool:
        return False

    @staticmethod
    def region_contact_top_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_moves(r: Int) -> Bool:
        return False
