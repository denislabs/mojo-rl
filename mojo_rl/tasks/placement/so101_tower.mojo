"""`so101_tower`'s device placement table — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run gen-placement-tables
CI checks it with: pixi run gen-placement-tables --check

From `mojo_rl/tasks/families/so101_tower.family`,
`mojo_rl/tasks/scenes/so101_tower.xml` and forward kinematics on it.
2 free slots, 3 regions (0 moving, 0 followed on one slide), 0 drawable joints.
See `placement/table.mojo` for what each method means.
"""

from mojo_rl.tasks.placement.table import PlacementTable


struct So101TowerPlacement(PlacementTable):
    comptime N_SLOTS: Int = 4
    comptime N_FREE: Int = 2
    comptime N_REGIONS: Int = 3
    comptime NQ: Int = 20
    comptime NV: Int = 18
    comptime N_JOINTS: Int = 0
    comptime NBODY: Int = 13
    comptime NSITE: Int = 4
    comptime GRIPPER_SITE: Int = 2  # robot_grasp_center
    comptime N_BASE_QPOS: Int = 0

    @staticmethod
    def base_qpos[DTYPE: DType](i: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.0)

    @staticmethod
    def free_slot(j: Int) -> Int:
        if j == 0:
            return 2  # bowl
        return 3  # brick

    @staticmethod
    def free_qadr(j: Int) -> Int:
        if j == 0:
            return 6
        return 13

    @staticmethod
    def free_dadr(j: Int) -> Int:
        if j == 0:
            return 6
        return 12

    @staticmethod
    def free_has_geom(j: Int) -> Bool:
        return True

    @staticmethod
    def free_rest[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        if j == 0:
            return Scalar[DTYPE](0.0)
        return Scalar[DTYPE](0.0125)

    @staticmethod
    def free_radius[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        if j == 0:
            return Scalar[DTYPE](0.060614)
        return Scalar[DTYPE](0.0176777)

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
        if j == 0:
            return Scalar[DTYPE](0.0)
        return Scalar[DTYPE](-0.0125)

    @staticmethod
    def free_top_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        if j == 0:
            return Scalar[DTYPE](0.045)
        return Scalar[DTYPE](0.0125)

    @staticmethod
    def region_site(r: Int) -> Int:
        return 3

    @staticmethod
    def region_site_x[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.32)

    @staticmethod
    def region_site_y[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_site_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.002)

    @staticmethod
    def region_has_rect(r: Int) -> Bool:
        if r == 0:
            return True  # desk_top
        if r == 1:
            return True  # desk_left
        return True  # desk_right

    @staticmethod
    def region_x0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](-0.14)

    @staticmethod
    def region_y0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.16)
        if r == 1:
            return Scalar[DTYPE](0.06)
        return Scalar[DTYPE](-0.16)

    @staticmethod
    def region_x1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.06)

    @staticmethod
    def region_y1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.16)
        if r == 1:
            return Scalar[DTYPE](0.16)
        return Scalar[DTYPE](-0.06)

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
