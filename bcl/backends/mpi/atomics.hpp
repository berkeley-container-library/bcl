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
  int error_code = MPI_Compare_and_swap(&new_val, &old_val, &result,
                                        type,
                                        ptr.rank, ptr.ptr,
                                        BCL::win);

  BCL_DEBUG(
          if (error_code != MPI_SUCCESS) {
            throw debug_error("BCL compare_and_swap(): MPI_Compare_and_swap return error code " + std::to_string(error_code));
          }
  )

  error_code = MPI_Win_flush_local(ptr.rank, BCL::win);

  BCL_DEBUG(
          if (error_code != MPI_SUCCESS) {
            throw debug_error("BCL compare_and_swap(): MPI_Win_flush_local return error code " + std::to_string(error_code));
          }
  )

  return result;
}

}
