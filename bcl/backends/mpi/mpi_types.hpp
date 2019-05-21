#pragma once

#include <mpi.h>

namespace BCL {

// Helpers to get MPI datatype corresponding to a C++ type.
// Defined in accordance with MPI 3.1 report, page 26.

// XXX: these *could* all be static constexpr members, if
//      not for a bug in Spectrum MPI.

template <typename T>
struct get_mpi_type_impl_;

template <>
struct get_mpi_type_impl_<float> {
  // static constexpr MPI_Datatype mpi_type = MPI_FLOAT;
  static MPI_Datatype mpi_type() { return MPI_FLOAT; }
};

template <>
struct get_mpi_type_impl_<double> {
  // static constexpr MPI_Datatype mpi_type = MPI_DOUBLE;
  static MPI_Datatype mpi_type() { return MPI_DOUBLE; }
};

template <>
struct get_mpi_type_impl_<long double> {
  // static constexpr MPI_Datatype mpi_type = MPI_LONG_DOUBLE;
  static MPI_Datatype mpi_type() { return MPI_LONG_DOUBLE; }
};

/*
template <>
struct get_mpi_type_impl_<_Bool> {
  static constexpr MPI_Datatype mpi_type = MPI_C_BOOL;
};
*/

template <>
struct get_mpi_type_impl_<int8_t> {
  // static constexpr MPI_Datatype mpi_type = MPI_INT8_T;
  static MPI_Datatype mpi_type() { return MPI_INT8_T; }
};

template <>
struct get_mpi_type_impl_<int16_t> {
  // static constexpr MPI_Datatype mpi_type = MPI_INT16_T;
  static MPI_Datatype mpi_type() { return MPI_INT16_T; }
};

template <>
struct get_mpi_type_impl_<int32_t> {
  // static constexpr MPI_Datatype mpi_type = MPI_INT32_T;
  static MPI_Datatype mpi_type() { return MPI_INT32_T; }
};

template <>
struct get_mpi_type_impl_<int64_t> {
  // static constexpr MPI_Datatype mpi_type = MPI_INT64_T;
  static MPI_Datatype mpi_type() { return MPI_INT64_T; }
};

template <>
struct get_mpi_type_impl_<uint8_t> {
  // static constexpr MPI_Datatype mpi_type = MPI_UINT8_T;
  static MPI_Datatype mpi_type() { return MPI_UINT8_T; }
};

template <>
struct get_mpi_type_impl_<uint16_t> {
  // static constexpr MPI_Datatype mpi_type = MPI_UINT16_T;
  static MPI_Datatype mpi_type() { return MPI_UINT16_T; }
};

template <>
struct get_mpi_type_impl_<uint32_t> {
  // static constexpr MPI_Datatype mpi_type = MPI_UINT32_T;
  static MPI_Datatype mpi_type() { return MPI_UINT32_T; }
};

template <>
struct get_mpi_type_impl_<uint64_t> {
  // static constexpr MPI_Datatype mpi_type = MPI_UINT64_T;
  static MPI_Datatype mpi_type() { return MPI_UINT64_T; }
};

template <typename T>
MPI_Datatype get_mpi_type() {
  // return get_mpi_type_impl_<T>::mpi_type;
  return get_mpi_type_impl_<T>::mpi_type();
}

}
