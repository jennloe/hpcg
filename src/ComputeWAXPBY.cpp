
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
 @file ComputeWAXPBY.cpp

 HPCG routine
 */

#include "ComputeWAXPBY.hpp"
#include "ComputeWAXPBY_ref.hpp"

/*!
  Routine to compute the update of a vector with the sum of two
  scaled vectors where: w = alpha*x + beta*y

  This routine calls the reference WAXPBY implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in] n the number of vector elements (on this processor)
  @param[in] alpha, beta the scalars applied to x and y respectively.
  @param[in] x, y the input vectors
  @param[out] w the output vector
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeWAXPBY_ref
*/
template<class Vector_type>
int ComputeWAXPBY(const local_int_t n,
                  const typename Vector_type::scalar_type alpha,
                  const Vector_type & x,
                  const typename Vector_type::scalar_type beta,
                  const Vector_type & y,
                        Vector_type & w,
                        bool & isOptimized) {

  // This line and the next two lines should be removed and your version of ComputeWAXPBY should be used.
  isOptimized = false;
  return ComputeWAXPBY_ref(n, alpha, x, beta, y, w);
}


/* --------------- *
 * specializations *
 * --------------- */

template
int ComputeWAXPBY< Vector<double> >(int, double, Vector<double> const&, double, Vector<double> const&, Vector<double>&, bool&);

template
int ComputeWAXPBY< Vector<float> >(int, float, Vector<float> const&, float, Vector<float> const&, Vector<float>&, bool&);
