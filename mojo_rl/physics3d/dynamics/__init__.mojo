"""Dynamics computation on the per-field tensor path.

Fields modules (imported directly by module): mass_matrix (CRBA),
rne (bias forces), rne_post (`mj_rnePostConstraint`, integrator-gated),
cdof, ldl, lu, qderiv,
subtree_com, fluid_forces, invweight. The legacy
struct-Model/Data dynamics (`mass_matrix`, `jacobian`, `bias_forces`,
`cfrc_ext`, `fluid_forces`, `velocity_derivatives`, `lu_factorization`) were
deleted at the fields sunset (P6/G4).
"""
