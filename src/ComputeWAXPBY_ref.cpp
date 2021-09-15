
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
 @file ComputeWAXPBY_ref.cpp

 HPCG routine
 */

#include "ComputeWAXPBY_ref.hpp"
#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif
#include <cassert>
/*!
  Routine to compute the update of a vector with the sum of two
  scaled vectors where: w = alpha*x + beta*y

  This is the reference WAXPBY impmentation.  It CANNOT be modified for the
  purposes of this benchmark.

  @param[in] n the number of vector elements (on this processor)
  @param[in] alpha, beta the scalars applied to x and y respectively.
  @param[in] x, y the input vectors
  @param[out] w the output vector.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeWAXPBY
*/
template<class Vector_type>
int ComputeWAXPBY_ref(const local_int_t n,
                      const typename Vector_type::scalar_type alpha,
                      const Vector_type & x,
                      const typename Vector_type::scalar_type beta,
                      const Vector_type & y,
                            Vector_type & w) {
  assert(x.localLength>=n); // Test vector lengths
  assert(y.localLength>=n);

  typedef typename Vector_type::scalar_type scalar_type;
  const scalar_type * const xv = x.values;
  const scalar_type * const yv = y.values;
  scalar_type * const wv = w.values;

  if (alpha==1.0) {
#ifndef HPCG_NO_OPENMP
    #pragma omp parallel for
#endif
    for (local_int_t i=0; i<n; i++) wv[i] = xv[i] + beta * yv[i];
  } else if (beta==1.0) {
#ifndef HPCG_NO_OPENMP
    #pragma omp parallel for
#endif
    for (local_int_t i=0; i<n; i++) wv[i] = alpha * xv[i] + yv[i];
  } else  {
#ifndef HPCG_NO_OPENMP
    #pragma omp parallel for
#endif
    for (local_int_t i=0; i<n; i++) wv[i] = alpha * xv[i] + beta * yv[i];
  }

  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

template
int ComputeWAXPBY_ref< Vector<double> >(int, double, Vector<double> const&, double, Vector<double> const&, Vector<double>&);

template
int ComputeWAXPBY_ref< Vector<float> >(int, float, Vector<float> const&, float, Vector<float> const&, Vector<float>&);
