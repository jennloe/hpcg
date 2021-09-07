
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

struct Matrix_STRUCT {
  local_int_t m;            //!< number of rows
  local_int_t n;            //!< number of columns
  double * values;          //!< array of values
  /*!
   This is for storing optimized data structures created in OptimizeProblem and
   used inside optimized ComputeSPMV().
   */
  void * optimizationData;

};
typedef struct Matrix_STRUCT SerialDenseMatrix;

/*!
  Initializes input vector.

  @param[in] A
  @param[in] m   Number of rows
  @param[in] n   Number of columns
 */
inline void InitializeMatrix(SerialDenseMatrix & A, local_int_t m, local_int_t n) {
  A.m   = m;
  A.n   = n;
  A.values = new double[m*n];
  A.optimizationData = 0;
  return;
}


/*!
  Fill the input matrix with zero values.

  @param[inout] A - On entrance A is initialized, on exit all its values are zero.
 */
inline void ZeroMatrix(SerialDenseMatrix & A) {

  local_int_t m = A.m;
  local_int_t n = A.n;
  double * val  = A.values;
  for (int i=0; i<m*n; ++i) 
    val[i] = 0.0;
  return;
}

/*!
  Copy input matrix to output matrix.

  @param[in] A Input vector
  @param[in] B Output vector
 */
inline void CopyMatrix(const SerialDenseMatrix & A, SerialDenseMatrix & B) {
  local_int_t m = A.m;
  local_int_t n = A.n;
  assert(B.m >= m);
  assert(A.n >= n);
  double * val_in  = A.values;
  double * val_out = B.values;
  for (int i=0; i<m*n; ++i)
    val_out[i] = val_in[i];
  return;
}

inline void SetMatrixValue(SerialDenseMatrix & A, local_int_t i, local_int_t j, double value) {
  assert(i>=0 && i < A.m);
  assert(j>=0 && j < A.n);
  double * vv = A.values;
  vv[i + j*A.m] = value;
  return;
}

inline double GetMatrixValue(SerialDenseMatrix & A, local_int_t i, local_int_t j) {
  assert(i>=0 && i < A.m);
  assert(j>=0 && j < A.n);
  double * vv = A.values;
  return vv[i + j*A.m];
}

/*!
  Deallocates the members of the data structure of the known system matrix provided they are not 0.

  @param[in] A the known system matrix
 */
inline void DeleteVector(SerialDenseMatrix & A) {

  delete [] A.values;
  A.m = 0;
  A.n = 0;
  return;
}

#endif // SERIAL_DENSE_MATRIX_HPP
