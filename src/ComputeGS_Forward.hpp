
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

#ifndef COMPUTEGS_FORWARD_HPP
#define COMPUTEGS_FORWARD_HPP
#include "SparseMatrix.hpp"
#include "Vector.hpp"

int ComputeGS_Forward(const SparseMatrix  & A, const Vector & r, Vector & x);

#endif // COMPUTEGS_FORWARD_HPP
