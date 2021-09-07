
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
 @file Vector.hpp

 HPCG data structures for dense vectors
 */

#include "ComputeTRSM.hpp"

int ComputeTRSM(const local_int_t n, const double alpha, const SerialDenseMatrix & U, SerialDenseMatrix & x) {

  const double one = 1.0;
  const double zero = 0.0;

  assert(x.m >= n);
  assert(x.n == 1); // one RHS

  const local_int_t m = U.m;
  const double * const Uv = U.values;
  double * xv = x.values;

  for (local_int_t i = n-1; i >= 0; i--) {
    for (local_int_t j = i+1; j < n; j++)
      xv[i] -= Uv[i + j*m] * xv[j];
    xv[i] /= Uv[i + i*m];
  }
  return 0;
}
