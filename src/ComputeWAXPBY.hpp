
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

#ifndef COMPUTEWAXPBY_HPP
#define COMPUTEWAXPBY_HPP
#include "Vector.hpp"

template<class Vector_type>
int ComputeWAXPBY(const local_int_t n,
                  const typename Vector_type::scalar_type alpha,
                  const Vector_type & x,
                  const typename Vector_type::scalar_type beta,
                  const Vector_type & y,
                        Vector_type & w,
                  bool & isOptimized);

#endif // COMPUTEWAXPBY_HPP
