"""Every LIBERO family's batched `max_contacts` — GENERATED, DO NOT EDIT.

Regenerate with:  pixi run gen-libero-envs
CI checks it with: pixi run gen-libero-envs --check

From `mojo_rl/tasks/libero/contact_budget.kv` (`pixi run libero-contact-budget`):
the measured null-action peak + 32, rounded up to a multiple of 16.
`tools/tasks/gen_libero_envs.mojo` says why, and what it does not prove.
"""

comptime LIBERO_GOAL_MAX_CONTACTS: Int = 144
"""Measured peak 112 (libero_goal__open_the_middle_drawer_of_the_cabinet)."""

comptime LIBERO_KITCHEN_SCENE1_MAX_CONTACTS: Int = 112
"""Measured peak 72 (libero_kitchen_scene1__open_the_bottom_drawer_of_the_cabinet)."""

comptime LIBERO_KITCHEN_SCENE10_MAX_CONTACTS: Int = 96
"""Measured peak 64 (libero_kitchen_scene10__close_the_top_drawer_of_the_cabinet)."""

comptime LIBERO_KITCHEN_SCENE2_MAX_CONTACTS: Int = 208
"""Measured peak 176 (libero_kitchen_scene2__open_the_top_drawer_of_the_cabinet)."""

comptime LIBERO_KITCHEN_SCENE3_MAX_CONTACTS: Int = 128
"""Measured peak 84 (libero_kitchen_scene3__put_the_frying_pan_on_the_stove)."""

comptime LIBERO_KITCHEN_SCENE4_MAX_CONTACTS: Int = 128
"""Measured peak 88 (libero_kitchen_scene4__close_the_bottom_drawer_of_the_cabinet)."""

comptime LIBERO_KITCHEN_SCENE5_MAX_CONTACTS: Int = 128
"""Measured peak 88 (libero_kitchen_scene5__close_the_top_drawer_of_the_cabinet)."""

comptime LIBERO_KITCHEN_SCENE6_MAX_CONTACTS: Int = 112
"""Measured peak 76 (libero_kitchen_scene6__close_the_microwave)."""

comptime LIBERO_KITCHEN_SCENE7_MAX_CONTACTS: Int = 80
"""Measured peak 45 (libero_kitchen_scene7__open_the_microwave)."""

comptime LIBERO_KITCHEN_SCENE8_MAX_CONTACTS: Int = 96
"""Measured peak 56 (libero_kitchen_scene8__put_the_right_moka_pot_on_the_stove)."""

comptime LIBERO_KITCHEN_SCENE9_MAX_CONTACTS: Int = 128
"""Measured peak 92 (libero_kitchen_scene9__put_the_frying_pan_on_the_cabinet_shelf)."""

comptime LIBERO_LIVING_ROOM_SCENE1_MAX_CONTACTS: Int = 176
"""Measured peak 141 (libero_living_room_scene1__put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket)."""

comptime LIBERO_LIVING_ROOM_SCENE2_MAX_CONTACTS: Int = 80
"""Measured peak 40 (libero_living_room_scene2__pick_up_the_butter_and_put_it_in_the_basket)."""

comptime LIBERO_LIVING_ROOM_SCENE3_MAX_CONTACTS: Int = 64
"""Measured peak 28 (libero_living_room_scene3__pick_up_the_alphabet_soup_and_put_it_in_the_tray)."""

comptime LIBERO_LIVING_ROOM_SCENE4_MAX_CONTACTS: Int = 96
"""Measured peak 60 (libero_living_room_scene4__pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray)."""

comptime LIBERO_LIVING_ROOM_SCENE5_MAX_CONTACTS: Int = 128
"""Measured peak 96 (libero_living_room_scene5__put_the_red_mug_on_the_left_plate)."""

comptime LIBERO_LIVING_ROOM_SCENE6_MAX_CONTACTS: Int = 96
"""Measured peak 51 (libero_living_room_scene6__put_the_red_mug_on_the_plate)."""

comptime LIBERO_OBJECT_MAX_CONTACTS: Int = 112
"""Measured peak 68 (libero_object__pick_up_the_alphabet_soup_and_place_it_in_the_basket)."""

comptime LIBERO_SPATIAL_MAX_CONTACTS: Int = 272
"""Measured peak 228 (libero_spatial__pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate)."""

comptime LIBERO_STUDY_SCENE1_MAX_CONTACTS: Int = 64
"""Measured peak 28 (libero_study_scene1__pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy)."""

comptime LIBERO_STUDY_SCENE2_MAX_CONTACTS: Int = 64
"""Measured peak 32 (libero_study_scene2__pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy)."""

comptime LIBERO_STUDY_SCENE3_MAX_CONTACTS: Int = 96
"""Measured peak 52 (libero_study_scene3__pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy)."""

comptime LIBERO_STUDY_SCENE4_MAX_CONTACTS: Int = 48
"""Measured peak 12 (libero_study_scene4__pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf)."""
