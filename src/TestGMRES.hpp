
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

#include "hpgmp.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "CGData.hpp"

template<class SparseMatrix_type, class SparseMatrix_type2, class CGData_type, class CGData_type2, class Vector_type, class TestCGData_type>
extern int TestGMRES(SparseMatrix_type & A, SparseMatrix_type2 & A_lo, CGData_type & data, CGData_type2 & data_lo, Vector_type & b, Vector_type & x, TestCGData_type & testcg_data);

template<class SparseMatrix_type, class CGData_type, class Vector_type, class TestCGData_type>
extern int TestGMRES(SparseMatrix_type & A, CGData_type & data, Vector_type & b, Vector_type & x, TestCGData_type & testcg_data);

#endif  // TESTGMRES_HPP

