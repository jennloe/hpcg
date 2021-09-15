
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

#ifndef COMPUTEWAXPBY_REF_HPP
#define COMPUTEWAXPBY_REF_HPP
#include "Vector.hpp"

template<class Vector_type>
int ComputeWAXPBY_ref(const local_int_t n,
                      const typename Vector_type::scalar_type alpha,
                      const Vector_type & x,
                      const typename Vector_type::scalar_type beta,
                      const Vector_type & y,
                            Vector_type & w);

#endif // COMPUTEWAXPBY_REF_HPP
