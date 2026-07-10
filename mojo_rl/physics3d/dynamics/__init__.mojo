"""Dynamics computation on the per-field tensor path.

Fields modules (imported directly by module): mass_matrix_fields (CRBA),
rne_fields (bias forces), cdof_fields, ldl_fields, lu_fields, qderiv_fields,
subtree_com_fields, fluid_forces_fields, invweight_fields. The legacy
struct-Model/Data dynamics (`mass_matrix`, `jacobian`, `bias_forces`,
`cfrc_ext`, `fluid_forces`, `velocity_derivatives`, `lu_factorization`) were
deleted at the fields sunset (P6/G4).
"""
