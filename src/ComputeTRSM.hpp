
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

#ifndef COMPUTE_TRSM_HPP
#define COMPUTE_TRSM_HPP

#include "Geometry.hpp"
#include "SerialDenseMatrix.hpp"
int ComputeTRSM(const local_int_t n, const double alpha, const SerialDenseMatrix & U, SerialDenseMatrix & x);

#endif // COMPUTE_TRSM_HPP
