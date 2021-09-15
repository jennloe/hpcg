
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

#ifndef SERIAL_DENSE_MATRIX_HPP
#define SERIAL_DENSE_MATRIX_HPP

#include <cassert>
#include <cstdlib>

template<class SC>
class SerialDenseMatrix {
public:
  typedef SC scalar_type;

  local_int_t m;            //!< number of rows
  local_int_t n;            //!< number of columns
  SC * values;          //!< array of values
  /*!
   This is for storing optimized data structures created in OptimizeProblem and
   used inside optimized ComputeSPMV().
   */
  void * optimizationData;
};

/*!
  Initializes input vector.

  @param[in] A
  @param[in] m   Number of rows
  @param[in] n   Number of columns
 */
template<class SerialDenseMatrix_type>
inline void InitializeMatrix(SerialDenseMatrix_type & A, local_int_t m, local_int_t n) {

  typedef typename SerialDenseMatrix_type::scalar_type scalar_type;

  A.m   = m;
  A.n   = n;
  A.values = new scalar_type[m*n];
  A.optimizationData = 0;
  return;
}


/*!
  Fill the input matrix with zero values.

  @param[inout] A - On entrance A is initialized, on exit all its values are zero.
 */
template<class SerialDenseMatrix_type>
inline void ZeroMatrix(SerialDenseMatrix_type & A) {

  typedef typename SerialDenseMatrix_type::scalar_type scalar_type;
  const scalar_type zero (0.0);

  local_int_t m = A.m;
  local_int_t n = A.n;
  scalar_type* val = A.values;

  for (int i=0; i<m*n; ++i) 
    val[i] = zero;
  return;
}

/*!
  Copy input matrix to output matrix.

  @param[in] A Input vector
  @param[in] B Output vector
 */
template<class SerialDenseMatrix_type>
inline void CopyMatrix(const SerialDenseMatrix_type & A, SerialDenseMatrix_type & B) {

  typedef typename SerialDenseMatrix_type::scalar_type scalar_type;

  local_int_t m = A.m;
  local_int_t n = A.n;
  assert(B.m >= m);
  assert(A.n >= n);
  scalar_type * val_in  = A.values;
  scalar_type * val_out = B.values;
  for (int i=0; i<m*n; ++i)
    val_out[i] = val_in[i];
  return;
}

template<class SerialDenseMatrix_type>
inline void SetMatrixValue(SerialDenseMatrix_type & A, local_int_t i, local_int_t j, typename SerialDenseMatrix_type::scalar_type value) {

  typedef typename SerialDenseMatrix_type::scalar_type scalar_type;

  assert(i>=0 && i < A.m);
  assert(j>=0 && j < A.n);
  scalar_type * vv = A.values;
  vv[i + j*A.m] = value;
  return;
}

template<class SerialDenseMatrix_type>
inline typename SerialDenseMatrix_type::scalar_type
GetMatrixValue(SerialDenseMatrix_type & A, local_int_t i, local_int_t j) {

  typedef typename SerialDenseMatrix_type::scalar_type scalar_type;

  assert(i>=0 && i < A.m);
  assert(j>=0 && j < A.n);
  scalar_type * vv = A.values;
  return vv[i + j*A.m];
}

/*!
  Deallocates the members of the data structure of the known system matrix provided they are not 0.

  @param[in] A the known system matrix
 */
template<class SerialDenseMatrix_type>
inline void DeleteSerialDenseMatrix(SerialDenseMatrix_type & A) {

  delete [] A.values;
  A.m = 0;
  A.n = 0;
  return;
}

#endif // SERIAL_DENSE_MATRIX_HPP
