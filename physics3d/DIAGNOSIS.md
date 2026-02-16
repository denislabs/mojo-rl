Plan: Primal Newton/CG Solvers (MuJoCo-Matching)       

 Context

 Our Newton and CG solvers operate in dual (lambda/force) space: they build the Delassus matrix A = J*M^{-1}*J^T and solve a QP
 over constraint forces. MuJoCo's Newton and CG are primal solvers operating in qacc (acceleration) space, minimizing:

 cost = 0.5*(qacc - qacc_smooth)^T * M * (qacc - qacc_smooth)  [Gauss term]
      + sum_i penalty_i(J*qacc - aref)                         [constraint costs]

 Forces are derived from qacc: force[i] = -D[i] * (J*qacc - aref)[i]. The Hessian is H = M + J^T*D*J (naturally PD). This is
 fundamentally different from our dual approach and explains the ~5% error in contact tests.

 MuJoCo's PGS is dual, so our PGS is structurally correct.

 Source: mujoco-main/src/engine/engine_solver.c lines 750-1968 (mj_solPrimal), engine_core_constraint.c lines 2394-2587
 (mj_constraintUpdate_impl).

 Algorithm: MuJoCo Primal Solver (shared Newton/CG)

 Input: M (mass matrix), J (Jacobian), D (constraint inertia = inv_K_imp),
        R (regularizer), aref (= -bias), qacc_smooth, qfrc_smooth

 1. qacc = qacc_smooth (copy of unconstrained acceleration)
 2. Ma = M * qacc
 3. jar = J * qacc - aref
 4. constraintUpdate(jar) → force, state, cost
 5. qfrc_constraint = J^T * force
 6. Gauss_cost = 0.5*(Ma - qfrc_smooth) . (qacc - qacc_smooth)
 7. total_cost = constraint_cost + Gauss_cost

 Newton: H = M + J^T * D_active * J, factorize H via Cholesky
 8. grad = Ma - qfrc_smooth - qfrc_constraint
 Newton: search = -chol_solve(H, grad)
 CG:     search = -M^{-1}*grad, then Polak-Ribiere beta update

 9. Analytical line search → alpha
 10. qacc += alpha * search, Ma += alpha * M*search, jar += alpha * J*search
 11. constraintUpdate → new force, state, cost
 Newton: incremental Hessian update (rank-1 Cholesky update/downdate)
 12. Check convergence (improvement < tolerance OR gradient < tolerance)
 13. Go to 8

 constraintUpdate: force from jar (MuJoCo mj_constraintUpdate_impl)

 Given jar[i] = (J*qacc - aref)[i]:
 - Equality (always active): force = -D*jar, cost = 0.5*D*jar^2
 - Limit/pyramidal (jar >= 0): force = 0, cost = 0 (satisfied)
 - Limit/pyramidal (jar < 0): force = -D*jar, cost = 0.5*D*jar^2
 - Elliptic cone: 3 zones (top=satisfied, bottom=quadratic, middle=cone projection)
   - Middle zone cost: 0.5*Dm*(N-mu*T)^2 where Dm = D[0]/(mu^2*(1+mu^2))

 Analytical Line Search (MuJoCo PrimalSearch)

 NOT Armijo backtracking! Uses 1D Newton with analytical first+second derivatives:
 1. Precompute quadratic polynomials q0 + q1*alpha + q2*alpha^2 per constraint
 2. For each candidate alpha: sum all constraint costs + Gauss cost → total cost + derivatives
 3. One-sided Newton search to bracket the root of dcost/dalpha = 0
 4. Bracketed Newton refinement to convergence

 Hessian (Newton only)

 - D_active[i] = D[i] if state is QUADRATIC, else 0
 - H = M + J^T * diag(D_active) * J (dense NV×NV, ~9×9 for HalfCheetah)
 - Dense Cholesky factorize
 - When constraint state changes: rank-1 Cholesky update/downdate with sqrt(D[i]) * J[i,:]
 - Cone contacts: add cone Hessian contributions via multiple rank-1 updates

 Data Flow & Interface Changes

 What the primal solver needs (beyond current trait args)

 ┌────────────────────────────────────────────────┬──────────────────────────────┬────────────────────────┐
 │                      Data                      │            Source            │   Size (HalfCheetah)   │
 ├────────────────────────────────────────────────┼──────────────────────────────┼────────────────────────┤
 │ M (mass matrix with armature+damping)          │ Integrator local M array     │ NV*NV = 81             │
 ├────────────────────────────────────────────────┼──────────────────────────────┼────────────────────────┤
 │ f_net (qfrc_smooth = applied - bias - passive) │ Integrator local f_net array │ NV = 9                 │
 ├────────────────────────────────────────────────┼──────────────────────────────┼────────────────────────┤
 │ qacc_smooth                                    │ = initial qacc before solver │ NV = 9 (copy at entry) │
 └────────────────────────────────────────────────┴──────────────────────────────┴────────────────────────┘

 Approach: Extend ConstraintData

 Add to ConstraintData:
 var M_hat: InlineArray[Scalar[DTYPE], _max_one[NV * NV]()]      # Mass matrix
 var qfrc_smooth: InlineArray[Scalar[DTYPE], _max_one[NV]()]     # Net unconstrained force

 The integrator fills these after LDL factorization, before calling solver.solve(). Existing solvers (PGS, old Newton/CG) simply
 ignore the new fields.

 On GPU: M is already in workspace at ws_M_offset, f_net at ws_fnet_offset. No extra storage needed.

 Implementation Steps

 Step 1: Extend ConstraintData + integrator fill

 Modify: physics3d/constraints/constraint_data.mojo
 - Add M_hat[NV*NV] and qfrc_smooth[NV] fields

 Modify: physics3d/integrator/euler_integrator.mojo
 - After computing M and f_net, copy them into constraints struct before calling solve()

 Step 2: primal_common.mojo — shared primal infrastructure

 Create: physics3d/solver/primal_common.mojo

 Functions (all generic over DTYPE, NV, MAX_ROWS):

 1. constraint_update[...]() — port of mj_constraintUpdate_impl
   - Input: jar[nefc], D[nefc], R[nefc], constraint types
   - Output: force[nefc], state[nefc], cost
   - Handles equality, limits, pyramidal contacts, elliptic cone (3 zones)
 2. primal_prepare[...]() — precompute line search quadratics
   - Input: search direction, jar, D, J
   - Output: quad[nefc*3] polynomials + cone quantities
 3. primal_eval[...]() — evaluate 1D cost at alpha
   - Input: alpha, quad, Gauss quadratic, cone data
   - Output: cost, first derivative, second derivative
 4. primal_linesearch[...]() — analytical 1D Newton line search
   - Port of MuJoCo's PrimalSearch
   - Bracketing + Newton convergence

 Step 3: PrimalNewtonSolver — CPU

 Create: physics3d/solver/primal_newton_solver.mojo

 Implements ConstraintSolver trait. CPU solve():
 1. Extract M, qfrc_smooth from constraints; save qacc_smooth = copy(qacc)
 2. Compute Ma = M * qacc
 3. Compute jar = J * qacc - aref (where aref = -bias from constraint rows)
 4. Call constraint_update → force, state, cost
 5. Compute qfrc_constraint = J^T * force
 6. Build H = M + J^T * D_active * J (dense NV×NV)
 7. Dense Cholesky factorize H
 8. Main loop (up to 100 iterations):
 a. grad = Ma - qfrc_smooth - qfrc_constraint
 b. search = -chol_solve(H, grad)
 c. Analytical line search → alpha
 d. qacc += alpha * search, Ma += alpha * Msearch, jar += alpha * Jsearch
 e. constraint_update → force, state, cost
 f. qfrc_constraint = J^T * force
 g. If state changed: incremental Hessian (rank-1 Cholesky update/downdate)
 h. If cone contacts: add cone Hessian contributions
 i. Check convergence

 For solve_gpu: similar structure but reading/writing workspace buffers.

 Step 4: PrimalCGSolver — CPU

 Create: physics3d/solver/primal_cg_solver.mojo

 Same primal framework but:
 - No Hessian construction
 - Preconditioner: Mgrad = M^{-1} * grad via M_inv (already available!)
 - Polak-Ribiere-Plus beta: beta = max(0, grad.dot(Mgrad - Mgrad_old) / grad_old.dot(Mgrad_old))
 - Same analytical line search

 Step 5: Dense Cholesky utilities

 Create or add to: physics3d/solver/cholesky.mojo

 Small dense Cholesky operations for NV×NV matrices:
 - chol_factor[NV](H) — in-place Cholesky L*L^T = H
 - chol_solve[NV](L, b) → x such that H*x = b
 - chol_rank1_update[NV](L, v, sign) — rank-1 update: H ← H ± v*v^T
 (used for incremental Hessian updates)

 Step 6: Wire into integrator

 Modify: physics3d/solver/__init__.mojo — export PrimalNewtonSolver, PrimalCGSolver

 Modify: test file — use EulerIntegrator[PrimalNewtonSolver] and set MuJoCo to mjSOL_NEWTON

 Step 7: GPU implementation

 Port PrimalNewtonSolver.solve_gpu:
 - M already in workspace, jar/force/state/quad in solver workspace region
 - Dense Cholesky is cheap for NV=9 on GPU
 - Line search is sequential (runs on thread 0)

 Files Summary

 ┌────────┬────────────────────────────────────────────┬───────────────────────────────────────────────┐
 │ Action │                    File                    │                  Description                  │
 ├────────┼────────────────────────────────────────────┼───────────────────────────────────────────────┤
 │ CREATE │ physics3d/solver/primal_common.mojo        │ constraint_update, line search, prepare, eval │
 ├────────┼────────────────────────────────────────────┼───────────────────────────────────────────────┤
 │ CREATE │ physics3d/solver/primal_newton_solver.mojo │ Primal Newton solver (CPU + GPU)              │
 ├────────┼────────────────────────────────────────────┼───────────────────────────────────────────────┤
 │ CREATE │ physics3d/solver/primal_cg_solver.mojo     │ Primal CG solver (CPU + GPU)                  │
 ├────────┼────────────────────────────────────────────┼───────────────────────────────────────────────┤
 │ CREATE │ physics3d/solver/cholesky.mojo             │ Dense Cholesky factor/solve/rank1update       │
 ├────────┼────────────────────────────────────────────┼───────────────────────────────────────────────┤
 │ MODIFY │ physics3d/constraints/constraint_data.mojo │ Add M_hat, qfrc_smooth fields                 │
 ├────────┼────────────────────────────────────────────┼───────────────────────────────────────────────┤
 │ MODIFY │ physics3d/integrator/euler_integrator.mojo │ Fill M_hat, qfrc_smooth before solve()        │
 ├────────┼────────────────────────────────────────────┼───────────────────────────────────────────────┤
 │ MODIFY │ physics3d/solver/__init__.mojo             │ Export new solvers                            │
 └────────┴────────────────────────────────────────────┴───────────────────────────────────────────────┘

 Verification

 1. Unit test: test_full_step_contact_vs_mujoco.mojo — switch to PrimalNewtonSolver, set MuJoCo to Newton (mj_model.opt.solver =
 2). Expect error < 1%.
 2. Non-contact test: test_full_step_vs_mujoco.mojo — verify no regression for non-contact cases.
 3. Analytical: Free fall scenario (no contacts) — primal solver should return qacc_smooth unchanged.
 4. GPU: After CPU passes, implement GPU and run pixi run -e apple mojo run ...

 Phasing

 Phase 1 (CPU Newton — this PR):
 1. Step 1 (ConstraintData + integrator)
 2. Step 2 (primal_common)
 3. Step 5 (cholesky utils)
 4. Step 3 (PrimalNewtonSolver CPU)
 5. Step 6 (wire up + test)

 Phase 2 (CPU CG + GPU — follow-up):
 - Step 4 (PrimalCGSolver CPU)
 - Step 7 (GPU implementations)


 This is a critical finding. MuJoCo's diagApprox doesn't use sum J²/D_LDL at all — it uses precomputed body inverse weights (body_invweight0),  
  which are model constants computed once from J_body * M^{-1} * J_body^T at the body CoM. This is fundamentally different from our LDL-based    
  approach.                                                                                                                                      
                                                                                                                                                 
  Let me verify: the D_mj values printed in our solver output are actually OUR computed D values, not MuJoCo's:                                  
                                                                                                                                                 
⏺ Searched for 1 pattern (ctrl+o to expand)                                                                                                      
                                                                                                                                                 
⏺ Yes, D_mj= is printing OUR D values (which we compute via LDL diagApprox). The name is confusing. Our D=24.02 while MuJoCo's actual D=41.27.   
   
  Given the complexity of computing body_invweight0 properly, let me take a pragmatic approach: since we can't easily match MuJoCo's exact D     
  values, let me simplify to use the stored inv_K_imp and K from the constraint builder (our original primal_D formula), remove the complex    
  compute_primal_D_values function, and check if the solver logic itself is correct. The D mismatch is a known source of ~1.7x difference that we
   can fix later by adding body_invweight0.                                                                                                      

⏺ Read 1 file (ctrl+o to expand)     

⏺ Now let me understand the approach. Rather than trying to match MuJoCo's exact D values (which requires body_invweight0), let me simplify: use
  our stored primal_D (from inv_K_imp and K) and see if the solver logic is correct. The D values will differ from MuJoCo's but should still
  produce a valid solution.                                                                                                                      
   
  Actually, let me take a better approach. Let me add body_invweight0 computation to our Model and use it for diagApprox, matching MuJoCo        
  exactly. This is the right long-term fix.                                                                                                    
                                                                                                                                               
  But first, let me understand the body_invweight0 formula more concretely:

  Now I need to add body_invweight0 to the Model struct. Let me add it as a simple InlineArray field and compute it during model initialization. 
  The computation needs M_inv and body Jacobians at CoM.                                                                                         
                                                                                                                                                 
  The simplest approach: add body_invweight0 to Model (2 floats per body: translation + rotation), and compute it in a new helper function that  
  the robot definition files call. For now, let me add it to Model and compute in the constraint builder (which already has M_inv).              
                                                                                                                                                 
  Actually, the best place is in the primal solver itself — it can compute it once per solve call using the available M_inv. Let me update
  compute_primal_D_values to compute body_invweight0 from M_inv and the FK data. But I'd need body CoM positions and joint info...

  Let me take a simpler approach: store body_invweight0 in the Model struct and compute it in the robot definition files (half_cheetah_def.mojo,
  hopper_def.mojo) since they already know the body structure. I'll compute it analytically for the initial pose.

  I'll add the computation to primal_common.mojo since it's specific to the primal solver. The key insight: for a serial chain robot, the body   
  Jacobian at CoM can be computed from cdof/joint information. But since we need M_inv (which varies with qpos), let me compute it in the
  solver's solve() method, which already has M_inv.                                                                                              
                                                                                                                                               
  Actually, since body_invweight0 is a MODEL CONSTANT (MuJoCo computes it once at setConst using the initial pose), I should compute it once and 
  store it in Model. Let me add a function that robot defs call during setup.                                                                    
                                                                                                                                                 
  The simplest approach for now: compute it inside compute_primal_D_values using available data (M_inv from constraint data, J from constraint
  rows, contact body info). Instead of computing body Jacobians from scratch, I can approximate: for each body that has contacts, extract its
  diagApprox from the available M_inv and constraint Jacobian columns.

  Wait, that's circular — the constraint Jacobian is at the contact point, not the body CoM. Let me think...

  Actually, for the primal solver to work correctly, I need the EXACT same body_invweight0 values that MuJoCo computes. These come from the body
  CoM Jacobian, NOT the contact Jacobian.

  Let me add a proper computation function. It needs:
  1. cdof (joint axes/anchors from FK)
  2. xipos (body CoM in world frame)
  3. M_inv

  I'll compute it during the integrator step, right after FK and M computation, and store in Model.