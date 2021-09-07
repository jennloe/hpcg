
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

#ifndef COMPUTE_GEMV_HPP
#define COMPUTE_GEMV_HPP

#include "Geometry.hpp"
#include "MultiVector.hpp"
#include "Vector.hpp"
#include "SerialDenseMatrix.hpp"
int ComputeGEMV(const local_int_t m, const local_int_t n,
                const double alpha, const MultiVector & A, const SerialDenseMatrix & x,
                const double beta,  const Vector & y);

#endif // COMPUTE_GEMV
