// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <helper_cuda.h>
#include <CSR.hpp>
#include <HashSpGEMM_volta.hpp>

namespace BCL {

namespace cuda {

template <typename T, typename index_type, typename Allocator>
CSR<index_type, T> convert_to_nsparse(CudaCSRMatrix<T, index_type, Allocator>& mat) {
  CSR<index_type, T> mat_nsparse(mat.m(), mat.n(), mat.nnz(), false);
  mat_nsparse.d_rpt = mat.rowptr_data();
  mat_nsparse.d_colids = mat.colind_data();
  mat_nsparse.d_values = mat.values_data();
  return mat_nsparse;
}

template <typename T, typename index_type, typename Allocator>
CudaCSRMatrix<T, index_type, Allocator>
convert_to_csr(CSR<index_type, T> mat) {
  return CudaCSRMatrix<T, index_type, Allocator>({mat.nrow, mat.ncolumn}, mat.nnz,
                                                  mat.d_values, mat.d_rpt, mat.d_colids);
}

template <typename T, typename index_type, typename Allocator>
CudaCSRMatrix<T, index_type, Allocator>
spgemm_nsparse(CudaCSRMatrix<T, index_type, Allocator>& a,
               CudaCSRMatrix<T, index_type, Allocator>& b) {
  auto a_nsparse = convert_to_nsparse(a);
  auto b_nsparse = convert_to_nsparse(b);
  CSR<index_type, T> c_nsparse;

  SpGEMM_Hash<index_type, T, Allocator>(a_nsparse, b_nsparse, c_nsparse);

  return convert_to_csr<T, index_type, Allocator>(c_nsparse);
}

} // end cuda
} // end BCL