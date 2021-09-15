
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

template<class SparseMatrix_type, class CGData_type, class Vector_type, class TestCGData_type>
extern int TestGMRES(SparseMatrix_type & A, CGData_type & data, Vector_type & b, Vector_type & x, TestCGData_type & testcg_data);

#endif  // TESTGMRES_HPP

