
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

#include "ComputeGEMV.hpp"

int ComputeGEMV(const local_int_t m, const local_int_t n,
                const double alpha, const MultiVector & A, const SerialDenseMatrix & x,
                const double beta,  const Vector & y) {

  const double one = 1.0;
  const double zero = 0.0;

  assert(x.m >= n); // Test vector lengths
  assert(x.n == 1);
  assert(y.localLength >= m);

  const double * const Av = A.values;
  const double * const xv = x.values;
  double * yv = y.values;
  if (beta == zero) {
    for (local_int_t i = 0; i < m; i++) yv[i] = zero;
  } else if (beta != one) {
    for (local_int_t i = 0; i < m; i++) yv[i] *= beta;
  }

  if (alpha == one) {
    for (local_int_t i=0; i<m; i++) {
      for (local_int_t j=0; j<n; j++)
        yv[i] += Av[i + j*m] * xv[j];
    }
  } else {
    for (local_int_t i=0; i<m; i++) {
      for (local_int_t j=0; j<n; j++)
        yv[i] += alpha * Av[i + j*m] * xv[j];
    }
  }
  return 0;
}
