
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

#ifndef MULTIVECTOR_HPP
#define MULTIVECTOR_HPP

#include <cassert>
#include <cstdlib>
#include "Vector.hpp"

struct MultiVector_STRUCT {
  local_int_t n;            //!< number of vectors
  local_int_t localLength;  //!< length of local portion of the vector
  double * values;          //!< array of values
  /*!
   This is for storing optimized data structures created in OptimizeProblem and
   used inside optimized ComputeSPMV().
   */
  void * optimizationData;

};
typedef struct MultiVector_STRUCT MultiVector;

/*!
  Initializes input vectors.

  @param[in] V
  @param[in] localLength Length of local portion of input vector
  @param[in] n           Number of columns
 */
inline void InitializeMultiVector(MultiVector & V, local_int_t localLength, local_int_t n) {
  V.localLength = localLength;
  V.n = n;
  V.values = new double[localLength * n];
  V.optimizationData = 0;
  return;
}

/*!
  Fill the input vector with zero values.

  @param[inout] v - On entrance v is initialized, on exit all its values are zero.
 */
inline void ZeroMultiVector(MultiVector & V) {
  local_int_t n = V.n;
  local_int_t m = V.localLength;
  double * vv = V.values;
  for (int i=0; i<m*n; ++i)
    vv[i] = 0.0;
  return;
}

/*!
  @param[inout] v - On entrance v is initialized, on exit all its values are zero.
 */
inline void GetVector(MultiVector & V, local_int_t j, Vector & vj) {
  vj.localLength = V.localLength;
  vj.values = &V.values[V.localLength*j];
  return;
}

/*!
  Deallocates the members of the data structure of the known system matrix provided they are not 0.

  @param[in] A the known system matrix
 */
inline void DeleteMultiVector(MultiVector & V) {

  delete [] V.values;
  V.localLength = 0;
  V.n = 0;
  return;
}

#endif // MULTIVECTOR_HPP

