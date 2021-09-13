
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

#ifndef COMPUTEGS_FORWARD_REF_HPP
#define COMPUTEGS_FORWARD_REF_HPP
#include "SparseMatrix.hpp"
#include "Vector.hpp"

int ComputeGS_Forward_ref(const SparseMatrix  & A, const Vector & r, Vector & x);

#endif // COMPUTESYMGS_FORWARD_REF_HPP
