
#ifndef HPGMP_UTILS_HPP
#define HPGMP_UTILS_HPP

// MpiTypeTraits (from Teuchos)
template<class T>
class MpiTypeTraits {
public:
  static MPI_Datatype getType () {
    return MPI_DATATYPE_NULL;
  }
};

//! Specialization for T = double (from Teuchos)
template<>
class MpiTypeTraits<double> {
public:
  //! MPI_Datatype corresponding to the type T.
  static MPI_Datatype getType () {
    return MPI_DOUBLE;
  }
};

//! Specialization for T = float (from Teuchos).
template<>
class MpiTypeTraits<float> {
public:
  //! MPI_Datatype corresponding to the type T.
  static MPI_Datatype getType () {
    return MPI_FLOAT;
  }
};

#endif
