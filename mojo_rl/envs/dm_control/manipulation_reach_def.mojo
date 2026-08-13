"""`manipulation/reach_site_features` as a COMPTIME model def.

The Phase 7 gates all drove Jaco through the RUNTIME path — `parse_xml_full`
plus `build_model_fields_from_flat`, over an XML baked into a temp directory by
Python. That path proved the model, forward kinematics, site IK and the contact
set, and it cannot do any more than that: the fields `Model` carries NO
ACTUATORS at all. `apply_actions` is a method on the comptime model def, so
`Phyics3dEnv` — and therefore any stepping of this task — needs this route.

⚠ THE TWO PATHS SHARE A PARSER BUT NOT ALL OF ONE. `ModelDefFromXML.init_fields`
calls `parse_xml_full`, so the Phase 7 parser fixes (the default joint axis, the
default-class geom attributes) apply here unchanged. The COUNTS below come from
`parse_xml`, which is the separate COMPTIME counter — a fix in one is not a fix
in the other, so `test_manipulation_reach_def` asserts every count against
MuJoCo rather than trusting that they agree.

⚠ `max_contacts` is 128 against a measured worst case of 48 over 60 in-range
poses (`test_jaco_contacts_vs_mujoco`). It is NOT the 256 the gates use: that
number was chosen while plane-mesh was emitting a contact per hull vertex and a
single pair could saturate the buffer. With `maxplanemesh` in place the real
ceiling is small, and an oversized contact buffer costs memory in every env.

⚠⚠ ELLIPTIC CONE + 5 NOSLIP ITERATIONS IS A COMBINATION WE CANNOT YET
REPRODUCE, and this model def declares that rather than hiding it. The baked
XML carries `<option cone="elliptic" noslip_iterations="5">`. Our `mj_solNoSlip`
is implemented for the PYRAMIDAL cone only, so on an elliptic model the pass is
skipped — `init_fields` has a comptime guard that says exactly this and refuses
to build without `allow_missing_noslip=True`.

That flag is passed here, deliberately and with the consequence stated: THIS
MODEL WILL NOT MATCH A MUJOCO ROLLOUT under sliding friction. It is enough to
build, do kinematics and detect contacts — which is what the model probe needs
— and it is NOT enough for a step-parity gate. Switching to `cone="pyramidal"`
would silence the guard by changing the friction model the task was tuned
against, which is worse than a known gap. See task #53.
"""

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType

from .manipulation_reach_xml import reach_site_features_xml

comptime pm = parse_xml(reach_site_features_xml)

comptime ReachSiteFeaturesModel = ModelDefFromXML[
    xml=reach_site_features_xml,
    nbody=pm.NBODY,
    njoint=pm.NJOINT,
    nq=pm.NQ,
    nv=pm.NV,
    ngeom=pm.NGEOM,
    nact=pm.NACT,
    ntex=pm.NTEX,
    nmat=pm.NMAT,
    nlight=pm.NLIGHT,
    ncam=pm.NCAM,
    nsite=pm.NSITE,
    neq=pm.NEQ,
    # ⚠ EVERY ONE OF THESE IS TAKEN FROM `pm`, NOT DEFAULTED. Each has a
    # default that silently disables a feature rather than failing:
    #   nexclude  0 -> the `<contact><exclude>` fill writes past a 1-element
    #                  tensor. That is how this was found: an index assert in
    #                  `fields_build`, not a diagnostic.
    #   npair     0 -> `<contact><pair>` rows dropped (that one does raise).
    #   max_tendon 0 -> tendons clamped away.
    #   max_condim 3 -> condim 4/6 geoms lose their torsional/rolling rows.
    #   noslip_iter 0 -> the noslip pass is simply not run.
    # `max_equality` is the one that SIZES storage — `neq` alone does not, so
    # passing only `neq` makes equality constraints vanish silently.
    nexclude=pm.NEXCLUDE,
    npair=pm.NPAIR,
    max_tendon=pm.NTENDON,
    max_condim=pm.MAX_CONDIM,
    max_equality=pm.NEQ * 6,
    max_contacts=128,
    timestep=pm.TIMESTEP,
    cone_type=ConeType.ELLIPTIC,
    noslip_iter=pm.NOSLIP_ITER,
    # ⚠ See the module docstring: elliptic + noslip is not reproducible today.
    # Accepting it here keeps the TASK'S friction model and makes the gap
    # explicit; it does not make a rollout correct.
    allow_missing_noslip=True,
]
