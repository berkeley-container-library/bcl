#pragma once

#include <mkl.h>

namespace BCL {
namespace mkl {

sparse_status_t
mkl_sparse_create_csr_wrapper_(sparse_matrix_t* A,
                               sparse_index_base_t index,
                               MKL_INT rows, MKL_INT cols,
                               const MKL_INT* rows_start, const MKL_INT* rows_end,
                               const MKL_INT* col_indx, const float* values)
{
  MKL_INT* rows_start_ = const_cast<MKL_INT*>(rows_start);
  MKL_INT* rows_end_ = const_cast<MKL_INT*>(rows_end);
  MKL_INT* col_indx_ = const_cast<MKL_INT*>(col_indx);
  float* values_ = const_cast<float*>(values);
  return mkl_sparse_s_create_csr(A, index, rows, cols, rows_start_, rows_end_, col_indx_, values_);
}

sparse_status_t
mkl_sparse_add_wrapper_(sparse_operation_t operation, const sparse_matrix_t A,
                        float alpha, const sparse_matrix_t B, sparse_matrix_t* C) {
  return mkl_sparse_s_add(operation, A, alpha, B, C);
}

template <typename T>
struct mkl_sparse_set_value_wrapper_;

template <>
struct mkl_sparse_set_value_wrapper_<float> {
  sparse_status_t operator()(sparse_matrix_t A,
                                    MKL_INT row, MKL_INT col,
                                    const float& value) noexcept {
    return mkl_sparse_s_set_value(A, row, col, value);
  }
};

template <>
struct mkl_sparse_set_value_wrapper_<double> {
  sparse_status_t operator()(sparse_matrix_t A,
                                    MKL_INT row, MKL_INT col,
                                    const double& value) noexcept {
    return mkl_sparse_d_set_value(A, row, col, value);
  }
};

}
}
