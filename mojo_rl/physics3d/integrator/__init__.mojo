"""Physics3D Integrators.

Integrators advance the physics simulation forward in time,
orchestrating collision detection, constraint solving, and position updates.

EulerIntegrator[SOLVER]:
  - MuJoCo Euler integration: M_hat = M + arm + dt*diag(damping)
  - Solver choices: PGSSolver, NewtonSolver, CGSolver

ImplicitFastIntegrator[SOLVER]:
  - MuJoCo implicit-fast integration: M_hat = M + arm - dt*qDeriv
  - Same result as Euler for passive systems (no actuators)
  - Extensible for actuator velocity derivatives

ImplicitIntegrator[SOLVER]:
  - Full implicit integration: M_hat = M + arm - dt*qDeriv
  - qDeriv includes RNE velocity derivative (d(Coriolis)/d(qvel))
  - Non-symmetric qDeriv → uses LU factorization instead of LDL
  - CPU only (GPU deferred, falls back to ImplicitFast)
  - Better stability for systems with significant gyroscopic effects

DefaultIntegrator is an alias for ImplicitFastIntegrator[PGSSolver].

GPU Profiling:
  All GPU-capable integrators (Euler, ImplicitFast, RK4) provide:
  - register_gpu_profile_slots(timer, parent) -> base slot index
  - step_gpu_profiled[...](..., timer, base) — same as step_gpu with per-phase timing

  Euler: 4 slots (dynamics, collision, solver, finalize)
  ImplicitFast: 3 slots (dynamics, solver, finalize)
  RK4: 5 slots (stage0, stage1, stage2, stage3, combine)

  Usage:
    var timer = PerfTimer[True]()
    var phys_base = EulerIntegrator[NewtonSolver].register_gpu_profile_slots(timer)
    for _ in range(num_steps):
        EulerIntegrator[NewtonSolver].step_gpu_profiled[...](
            ctx, state_buf, model_buf, ws_buf, timer, phys_base
        )
    timer.print_report("Physics Profile")
"""

# Legacy slab integrators (Euler/RK4/Implicit/ImplicitFast + DefaultIntegrator)
# were deleted at the P6 fields sunset. The fields integrators live in
# `euler` / `rk4` / `implicit` and are imported directly.
