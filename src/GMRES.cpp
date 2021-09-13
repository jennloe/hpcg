
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file GMRES.cpp

 HPCG routine
 */

#include <fstream>

#include <cmath>

#include "hpcg.hpp"

#include "GMRES.hpp"
#include "mytimer.hpp"
#include "ComputeSPMV.hpp"
#include "ComputeMG.hpp"
#include "ComputeDotProduct.hpp"
#include "ComputeWAXPBY.hpp"
#include "ComputeTRSM.hpp"
#include "ComputeGEMV.hpp"


// Use TICK and TOCK to time a code section in MATLAB-like fashion
#define TICK()  t0 = mytimer() //!< record current time in 't0'
#define TOCK(t) t += mytimer() - t0 //!< store time difference in 't' using time in 't0'

/*!
  Routine to compute an approximate solution to Ax = b

  @param[in]    geom The description of the problem's geometry.
  @param[inout] A    The known system matrix
  @param[inout] data The data structure with all necessary CG vectors preallocated
  @param[in]    b    The known right hand side vector
  @param[inout] x    On entry: the initial guess; on exit: the new approximate solution
  @param[in]    max_iter  The maximum number of iterations to perform, even if tolerance is not met.
  @param[in]    tolerance The stopping criterion to assert convergence: if norm of residual is <= to tolerance.
  @param[out]   niters    The number of iterations actually performed.
  @param[out]   normr     The 2-norm of the residual vector after the last iteration.
  @param[out]   normr0    The 2-norm of the residual vector before the first iteration.
  @param[out]   times     The 7-element vector of the timing information accumulated during all of the iterations.
  @param[in]    doPreconditioning The flag to indicate whether the preconditioner should be invoked at each iteration.

  @return Returns zero on success and a non-zero value otherwise.

  @see CG_ref()
*/
int GMRES(const SparseMatrix & A, CGData & data, const Vector & b, Vector & x,
          const int max_iter, const double tolerance, int & niters, double & normr, double & normr0,
          double * times, bool doPreconditioning) {

  const double one  = 1.0;
  const double zero = 0.0;
  double t_begin = mytimer();  // Start timing right away
  normr = 0.0;
  double rtz = 0.0, oldrtz = 0.0, alpha = 0.0, beta = 0.0, pAp = 0.0;
  double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0, t5 = 0.0;

  int restart_length = 30;
//#ifndef HPCG_NO_MPI
//  double t6 = 0.0;
//#endif
  local_int_t nrow = A.localNumberOfRows;
  Vector & r = data.r; // Residual vector
  Vector & z = data.z; // Preconditioned residual vector
  Vector & p = data.p; // Direction vector (in MPI mode ncol>=nrow)
  Vector & Ap = data.Ap;

  SerialDenseMatrix H;
  SerialDenseMatrix cs;
  SerialDenseMatrix ss;
  SerialDenseMatrix t;
  MultiVector Q;
  Vector Qkm1;
  Vector Qk;
  Vector Qj;
  InitializeMatrix(H,  restart_length+1, restart_length);
  InitializeMatrix(t,  restart_length+1, 1);
  InitializeMatrix(cs, restart_length+1, 1);
  InitializeMatrix(ss, restart_length+1, 1);
  InitializeMultiVector(Q, nrow, restart_length+1);

  if (!doPreconditioning && A.geom->rank==0) HPCG_fout << "WARNING: PERFORMING UNPRECONDITIONED ITERATIONS" << std::endl;

#ifdef HPCG_DEBUG
  int print_freq = 1;
  if (print_freq>50) print_freq=50;
  if (print_freq<1)  print_freq=1;
  if (A.geom->rank==0) HPCG_fout << std::endl << " Running GMRES(" << restart_length
                                 << ") with max-iters = " << max_iter
                                 << " and tol = " << tolerance
                                 << (doPreconditioning ? " with precond " : " without precond ")
                                 << ", nrow = " << nrow << std::endl;
#endif
  niters = 0;
  bool converged = false;
  while (niters <= max_iter && !converged) {
    // p is of length ncols, copy x to p for sparse MV operation
    CopyVector(x, p);
    TICK(); ComputeSPMV(A, p, Ap); TOCK(t3); // Ap = A*p
    TICK(); ComputeWAXPBY(nrow, one, b, -one, Ap, r, A.isWaxpbyOptimized);  TOCK(t2); // r = b - Ax (x stored in p)
    TICK(); ComputeDotProduct(nrow, r, r, normr, t4, A.isDotProductOptimized); TOCK(t1);
    normr = sqrt(normr);
    GetVector(Q, 0, Qj);
    CopyVector(r, Qj);
    TICK(); ComputeWAXPBY(nrow, zero, Qj, one/normr, Qj, Qj, A.isWaxpbyOptimized); TOCK(t2);

    // Record initial residual for convergence testing
    if (niters == 0) normr0 = normr;
    #ifdef HPCG_DEBUG
    if (A.geom->rank==0) HPCG_fout << "GMRES Residual at the start of restart cycle = "<< normr
                                   << ", " << normr/normr0 << std::endl;
    #endif

    if (normr/normr0 <= tolerance) {
      converged = true;
      #ifdef HPCG_DEBUG
      if (A.geom->rank==0) HPCG_fout << " > GMRES converged " << std::endl;
      #endif
    }
/*if (normr/normr0 <= tolerance || (niters > 0 && doPreconditioning)) {
  printf( " done %d iters (%s)\n",niters, (converged ? "Converged" : "Not Converged") );
  printf( " done (%s)\n",(doPreconditioning ? "Precond" : "Not Precond") );
  for (int i = 0; i < nrow; i++) printf( "x[%d] = %e\n",i,x.values[i] );
}*/

    // do forward GS instead of symmetric GS
    bool symmetric = false;

    // Start restart cycle
    int k = 1;
    SetMatrixValue(t, 0, 0, normr);
    while (k <= restart_length && normr/normr0 > tolerance) {
      GetVector(Q, k-1, Qkm1);
      GetVector(Q, k,   Qk);

      TICK();
      if (doPreconditioning)
        ComputeMG(A, Qkm1, z, symmetric); // Apply preconditioner
      else
        CopyVector(Qkm1, z);              // copy r to z (no preconditioning)
      TOCK(t5); // Preconditioner apply time

      // Qk = A*z
      TICK(); ComputeSPMV(A, z, Qk); TOCK(t3);

      // MGS to orthogonalize z against Q(:,0:k-1), using dots
      for (int j = 0; j < k; j++) {
        // get j-th column of Q
        GetVector(Q, j, Qj);

        alpha = zero;
        for (int i = 0; i < 2; i++) {
          // beta = Qk'*Qj
          TICK(); ComputeDotProduct(nrow, Qk, Qj, beta, t4, A.isDotProductOptimized); TOCK(t1);

          // Qk = Qk - beta * Qj
          TICK(); ComputeWAXPBY(nrow, one, Qk, -beta, Qj, Qk, A.isWaxpbyOptimized); TOCK(t2);
          alpha += beta;
        }
        SetMatrixValue(H, j, k-1, alpha);
      }
      // beta = norm(Qk)
      TICK(); ComputeDotProduct(nrow, Qk, Qk, beta, t4, A.isDotProductOptimized); TOCK(t1);
      beta = sqrt(beta);

      // Qk = Qk / beta
      TICK(); ComputeWAXPBY(nrow, zero, Qk, one/beta, Qk, Qk, A.isWaxpbyOptimized); TOCK(t2);
      SetMatrixValue(H, k, k-1, beta);

      // Given's rotation
      for(int j = 0; j < k-1; j++){
        double cj = GetMatrixValue(cs, j, 0);
        double sj = GetMatrixValue(ss, j, 0);
        double h1 = GetMatrixValue(H, j,   k-1);
        double h2 = GetMatrixValue(H, j+1, k-1);

        SetMatrixValue(H, j+1, k-1, -sj * h1 + cj * h2);
        SetMatrixValue(H, j,   k-1,  cj * h1 + sj * h2);
      }

      double f = GetMatrixValue(H, k-1, k-1);
      double g = GetMatrixValue(H, k,   k-1);

      double f2 = f*f;
      double g2 = g*g;
      double fg2 = f2 + g2;
      double D1 = one / sqrt(f2*fg2);
      double cj = f2*D1;
      fg2 = fg2 * D1;
      double sj = f*D1*g;
      SetMatrixValue(H, k-1, k-1, f*fg2);
      SetMatrixValue(H, k,   k-1, zero);

      double v1 = GetMatrixValue(t, k-1, 0);
      double v2 = -v1*sj;
      SetMatrixValue(t, k,   0, v2);
      SetMatrixValue(t, k-1, 0, v1*cj);

      SetMatrixValue(ss, k-1, 0, sj);
      SetMatrixValue(cs, k-1, 0, cj);

      normr = std::abs(v2);
      #ifdef HPCG_DEBUG
        if (A.geom->rank==0 && (k%print_freq == 0 || k+1 == restart_length))
          HPCG_fout << "GMRES Iteration = "<< k << " (" << niters << ")   Scaled Residual = "
                    << normr << " / " << normr0 << " = " << normr/normr0 << std::endl;
      #endif
      niters ++;
      k ++;
    } // end of restart-cycle
    // prepare to restart
    #ifdef HPCG_DEBUG
      if (A.geom->rank==0)
        HPCG_fout << "GMRES restart: k = "<< k << " (" << niters << ")" << std::endl;
    #endif
    // > update x
/*printf( "\n k = %d\n",k );
printf( "R=[\n" );
for (int i = 0; i < k; i++) {
  for (int j = 0; j < k; j++) printf("%e ",H.values[i + j * H.m] );
  printf("\n");
}
printf("];\n\n");
printf( "t=[\n" );
for (int i = 0; i < k; i++) printf( "%e\n",t.values[i]);
printf("];\n\n");

if (niters == 1) {
 printf( " nrow = %d, max_iter = %d\n",nrow,max_iter );
 printf( " Q = [\n" );
 for (int i = 0; i < nrow; i++) {
   for (int j = 0; j <= k-1; j++) printf( "%e ",Q.values[i + j * nrow] );
   printf("\n");
 }
 printf( " ];\n\n" );
}*/
    ComputeTRSM(k-1, one, H, t);
    if (doPreconditioning) {
      ComputeGEMV (nrow, k-1, one, Q, t, zero, r); // r = Q*t
      ComputeMG(A, r, z, symmetric);               // z = M*r
      TICK(); ComputeWAXPBY(nrow, one, x, one, z, x, A.isWaxpbyOptimized); TOCK(t2); // x += z
    } else {
      ComputeGEMV (nrow, k-1, one, Q, t, one, x); // x += Q*t
    }
  } // end of outer-loop


  // Store times
  times[1] += t1; // dot-product time
  times[2] += t2; // WAXPBY time
  times[3] += t3; // SPMV time
  times[4] += t4; // AllReduce time
  times[5] += t5; // preconditioner apply time
//#ifndef HPCG_NO_MPI
//  times[6] += t6; // exchange halo time
//#endif
  times[0] += mytimer() - t_begin;  // Total time. All done...
  return 0;
}
