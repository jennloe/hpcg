
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
 @file TestNorms.hpp

 HPCG data structure
 */

#ifndef TESTNORMS_HPP
#define TESTNORMS_HPP


template<class SC>
class TestNormsData {
public:
  typedef SC scalar_type;
  SC * values; //!< sample values
  SC   mean;   //!< mean of all sampes
  SC variance; //!< variance of mean
  int    samples;  //!< number of samples
  bool   pass;     //!< pass/fail indicator
};

template<class TestNormsData_type>
extern int TestNorms(TestNormsData_type & testnorms_data);

#endif  // TESTNORMS_HPP
