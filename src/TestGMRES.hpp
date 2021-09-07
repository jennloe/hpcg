
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
 @file TestCG.hpp

 HPGMRES data structure
 */

#ifndef TESTGMRES_HPP
#define TESTGMRES_HPP

#include "hpcg.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "CGData.hpp"
#include "TestCG.hpp"

extern int TestGMRES(SparseMatrix & A, CGData & data, Vector & b, Vector & x, TestCGData & testcg_data);

#endif  // TESTGMRES_HPP

