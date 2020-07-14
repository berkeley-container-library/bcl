
#pragma once

#include <helper_cuda.h>
#include <CSR.hpp>
#include <HashSpGEMM_volta.hpp>

namespace BCL {

template <typename T, typename index_type>
CSR<index_type, T> convert_to_nsparse(graphblas::Matrix<T>* x) {
  index_type m, n, nnz;
  x->nvals(&nnz);
  x->nrows(&m);
  x->ncols(&n);

  CSR<index_type, T> x_nsparse(m, n, nnz, false);
  x_nsparse.d_rpt = x->matrix_.sparse_.d_csrRowPtr_;
  x_nsparse.d_colids = x->matrix_.sparse_.d_csrColInd_;
  x_nsparse.d_values = x->matrix_.sparse_.d_csrVal_;

  return x_nsparse;
}

template <typename T, typename index_type>
graphblas::Matrix<T>* convert_to_graphblas(CSR<index_type, T> x) {
  // this is a test
  index_type m, n, nnz;
  m = x.nrow;
  n = x.ncolumn;
  nnz = x.nnz;

  graphblas::Matrix<T>* x_graphblas = new graphblas::Matrix<T>(m, n);

  x_graphblas->build(x.d_rpt, x.d_colids, x.d_values, nnz);
  return x_graphblas;
}

template <typename T, typename index_type, typename Allocator = BCL::cuda::cuda_allocator<T>>
graphblas::Matrix<T>* mxm_nsparse(graphblas::Matrix<T>* a, graphblas::Matrix<T>* b) {
  auto a_nsparse = convert_to_nsparse<T, index_type>(a);
  auto b_nsparse = convert_to_nsparse<T, index_type>(b);
  CSR<index_type, T> c_nsparse;

  SpGEMM_Hash<index_type, T, Allocator>(a_nsparse, b_nsparse, c_nsparse);

  return convert_to_graphblas<T, index_type>(c_nsparse);
  // return nullptr;
}

}