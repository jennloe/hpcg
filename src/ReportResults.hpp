
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

#ifndef REPORTRESULTS_HPP
#define REPORTRESULTS_HPP
#include "SparseMatrix.hpp"
#include "TestGMRES.hpp"
#include "TestSymmetry.hpp"
#include "TestNorms.hpp"

template<class SparseMatrix_type, class TestCGData_type, class TestSymmetryData_type, class TestNormsData_type>
void ReportResults(const SparseMatrix_type & A, int numberOfMgLevels, int numberOfCgSets, int refMaxIters, int optMaxIters, double times[],
                   const TestCGData_type & testcg_data, const TestSymmetryData_type & testsymmetry_data, const TestNormsData_type & testnorms_data,
                   int global_failure, bool quickPath);

#endif // REPORTRESULTS_HPP
