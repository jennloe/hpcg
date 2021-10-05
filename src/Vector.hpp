
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

#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <fstream>
#include <cassert>
#include <cstdlib>
#include "Geometry.hpp"

template<class SC = double>
class Vector {
public:
  typedef SC scalar_type;
  local_int_t localLength;  //!< length of local portion of the vector
  SC * values;     //!< array of values
  /*!
   This is for storing optimized data structures created in OptimizeProblem and
   used inside optimized ComputeSPMV().
   */
  void * optimizationData;
};

/*!
  Initializes input vector.

  @param[in] v
  @param[in] localLength Length of local portion of input vector
 */
template<class Vector_type>
inline void InitializeVector(Vector_type & v, local_int_t localLength) {
  typedef typename Vector_type::scalar_type scalar_type;
  v.localLength = localLength;
  v.values = new scalar_type[localLength];
  v.optimizationData = 0;
  return;
}

/*!
  Fill the input vector with zero values.

  @param[inout] v - On entrance v is initialized, on exit all its values are zero.
 */
template<class Vector_type>
inline void ZeroVector(Vector_type & v) {
  typedef typename Vector_type::scalar_type scalar_type;
  local_int_t localLength = v.localLength;
  scalar_type * vv = v.values;
  for (int i=0; i<localLength; ++i) vv[i] = 0.0;
  return;
}
/*!
  Multiply (scale) a specific vector entry by a given value.

  @param[inout] v Vector to be modified
  @param[in] index Local index of entry to scale
  @param[in] value Value to scale by
 */
template<class Vector_type>
inline void ScaleVectorValue(Vector_type & v, local_int_t index, typename Vector_type::scalar_type value) {
  typedef typename Vector_type::scalar_type scalar_type;
  assert(index>=0 && index < v.localLength);
  scalar_type * vv = v.values;
  vv[index] *= value;
  return;
}
/*!
  Fill the input vector with pseudo-random values.

  @param[in] v
 */
template<class Vector_type>
inline void FillRandomVector(Vector_type & v) {
  typedef typename Vector_type::scalar_type scalar_type;
  local_int_t localLength = v.localLength;
  scalar_type * vv = v.values;
  for (int i=0; i<localLength; ++i) vv[i] = rand() / (scalar_type)(RAND_MAX) + 1.0;
  return;
}
/*!
  Copy input vector to output vector.

  @param[in] v Input vector
  @param[in] w Output vector
 */
template<class Vector_src, class Vector_dst>
inline void CopyVector(const Vector_src & v, Vector_dst & w) {
  typedef typename Vector_src::scalar_type scalar_src;
  typedef typename Vector_dst::scalar_type scalar_dst;
  local_int_t localLength = v.localLength;
  assert(w.localLength >= localLength);
  scalar_src * vv = v.values;
  scalar_dst * wv = w.values;
  for (int i=0; i<localLength; ++i) wv[i] = vv[i];
  return;
}


/*!
  Deallocates the members of the data structure of the known system matrix provided they are not 0.

  @param[in] A the known system matrix
 */
template<class Vector_type>
inline void DeleteVector(Vector_type & v) {

  delete [] v.values;
  v.localLength = 0;
  return;
}

#endif // VECTOR_HPP
