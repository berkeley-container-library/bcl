#pragma once

#include <mpi.h>
#include "mpi_types.hpp"

namespace BCL {

extern MPI_Win win;

template <typename T>
inline T compare_and_swap(BCL::GlobalPtr<T> ptr, T old_val, T new_val) {
  static_assert(std::is_integral<T>::value, "BCL::compare_and_swap(): only integral types are supported");
  T result;
  MPI_Datatype type = get_mpi_type<T>();
  MPI_Compare_and_swap(&new_val, &old_val, &result,
                       type,
                       ptr.rank, ptr.ptr,
                       BCL::win);
  MPI_Win_flush_local(ptr.rank, BCL::win);
  return result;
}

}
