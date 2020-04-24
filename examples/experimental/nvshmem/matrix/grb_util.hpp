#pragma once

#include <matrix_sum.cuh>

namespace BCL {

namespace cuda {

// TODO: Actually free matrix contents.
template <typename T>
void destroy_grb(graphblas::Matrix<T>* x, const std::string lbl = "hey") {
  auto& m = x->matrix_.sparse_;
  if (m.h_csrRowPtr_) free(m.h_csrRowPtr_);
  if (m.h_csrColInd_) free(m.h_csrColInd_);
  if (m.h_csrVal_   ) free(m.h_csrVal_);
  if (BCL::cuda::__is_valid_cuda_gptr(m.d_csrRowPtr_)) {
    BCL::cuda::dealloc(BCL::cuda::__to_cuda_gptr(m.d_csrRowPtr_));
  } else {
    if (m.d_csrRowPtr_) CUDA_CALL(cudaFree(m.d_csrRowPtr_));
  }
  if (BCL::cuda::__is_valid_cuda_gptr(m.d_csrColInd_)) {
    BCL::cuda::dealloc(BCL::cuda::__to_cuda_gptr(m.d_csrColInd_));
  } else {
    if (m.d_csrColInd_) CUDA_CALL(cudaFree(m.d_csrColInd_));
  }
  if (BCL::cuda::__is_valid_cuda_gptr(m.d_csrVal_   )) {
    BCL::cuda::dealloc(BCL::cuda::__to_cuda_gptr(m.d_csrVal_));
  } else {
    if (m.d_csrVal_   ) CUDA_CALL(cudaFree(m.d_csrVal_   ));
  }

  if (m.format_ == graphblas::backend::GrB_SPARSE_MATRIX_CSRCSC) {
    throw std::runtime_error("destroy_grb: Case not handled.");
  }
}


template <typename T>
graphblas::Matrix<T>* sum_tiles(graphblas::Matrix<T>* a, graphblas::Matrix<T>* b) {
  // XXX: Do an element-wise add using cuSparse
  //      'A' here is local_c, and 'B' here is result_c
  //.     At the end, the new accumulated matrix will be put in local_c.

  // TODO: allocate handle elsewhere.

  cusparseHandle_t handle;
  cusparseStatus_t status = 
  cusparseCreate(&handle);
  BCL::cuda::throw_cusparse(status);
  status = 
  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
  BCL::cuda::throw_cusparse(status);

  graphblas::Index arows, acols;
  graphblas::Index brows, bcols;

  a->nrows(&arows);
  a->ncols(&acols);

  b->nrows(&brows);
  b->ncols(&bcols);

  assert(acols == bcols);
  assert(arows == brows);

  graphblas::Index m, n;

  m = arows;
  n = acols;

  static_assert(std::is_same<int, graphblas::Index>::value);
  cusparseMatDescr_t descr_a, descr_b, descr_c;
  status = 
  cusparseCreateMatDescr(&descr_a);
  BCL::cuda::throw_cusparse(status);
  status =
  cusparseCreateMatDescr(&descr_b);
  BCL::cuda::throw_cusparse(status);
  status =
  cusparseCreateMatDescr(&descr_c);
  BCL::cuda::throw_cusparse(status);

  status =
  cusparseSetMatType(descr_a, CUSPARSE_MATRIX_TYPE_GENERAL);
  BCL::cuda::throw_cusparse(status);
  status =
  cusparseSetMatIndexBase(descr_a, CUSPARSE_INDEX_BASE_ZERO);
  BCL::cuda::throw_cusparse(status);
  status =
  cusparseSetMatType(descr_b, CUSPARSE_MATRIX_TYPE_GENERAL);
  BCL::cuda::throw_cusparse(status);
  status =
  cusparseSetMatIndexBase(descr_b, CUSPARSE_INDEX_BASE_ZERO);
  BCL::cuda::throw_cusparse(status);
  status =
  cusparseSetMatType(descr_c, CUSPARSE_MATRIX_TYPE_GENERAL);
  BCL::cuda::throw_cusparse(status);
  status =
  cusparseSetMatIndexBase(descr_c, CUSPARSE_INDEX_BASE_ZERO);
  BCL::cuda::throw_cusparse(status);

  graphblas::Index a_nnz, b_nnz;
  graphblas::Index c_nnz;
  a->nvals(&a_nnz);
  b->nvals(&b_nnz);

  graphblas::Index* nnzTotalDevHostPtr;
  nnzTotalDevHostPtr = &c_nnz;

  graphblas::Index* row_ptr_c;
  CUDA_CALL(cudaMalloc((void**) &row_ptr_c, sizeof(graphblas::Index)*(m+1)));
  if (row_ptr_c == nullptr) {
    throw std::runtime_error("Couldn't allocate C.");
  }

  graphblas::Index* a_row_ptr = a->matrix_.sparse_.d_csrRowPtr_;
  graphblas::Index* a_col_ind = a->matrix_.sparse_.d_csrColInd_;

  graphblas::Index* b_row_ptr = b->matrix_.sparse_.d_csrRowPtr_;
  graphblas::Index* b_col_ind = b->matrix_.sparse_.d_csrColInd_;

  status = 
  cusparseXcsrgeamNnz(handle,
                      m,
                      n,
                      descr_a,
                      a_nnz,
                      a_row_ptr,
                      a_col_ind,
                      descr_b,
                      b_nnz,
                      b_row_ptr,
                      b_col_ind,
                      descr_c,
                      row_ptr_c,
                      nnzTotalDevHostPtr);
  BCL::cuda::throw_cusparse(status);

  if (nnzTotalDevHostPtr == nullptr) {
    throw std::runtime_error("Unhandled case: nnzTotalDevHostPtr is null.");
  } else {
    c_nnz = *nnzTotalDevHostPtr;
  }
  T alpha = 1.0;
  T beta = 1.0;
  graphblas::Index* col_ind_c;
  T* values_c;
  CUDA_CALL(cudaMalloc((void**) &col_ind_c, sizeof(graphblas::Index)*c_nnz));
  CUDA_CALL(cudaMalloc((void**) &values_c, sizeof(T)*c_nnz));
  if (col_ind_c == nullptr || values_c == nullptr) {
    throw std::runtime_error("sum_tiles(): out of memory.");
  }
  status = 
  cusparseScsrgeam(handle,
                   m,
                   n,
                   &alpha,
                   descr_a,
                   a_nnz,
                   a->matrix_.sparse_.d_csrVal_,
                   a->matrix_.sparse_.d_csrRowPtr_,
                   a->matrix_.sparse_.d_csrColInd_,
                   &beta,
                   descr_b,
                   b_nnz,
                   b->matrix_.sparse_.d_csrVal_,
                   b->matrix_.sparse_.d_csrRowPtr_,
                   b->matrix_.sparse_.d_csrColInd_,
                   descr_c,
                   values_c,
                   row_ptr_c,
                   col_ind_c);

  BCL::cuda::throw_cusparse(status);
  cudaDeviceSynchronize();

  graphblas::Matrix<T>* new_local_c = new graphblas::Matrix<T>(m, n);
  new_local_c->build(row_ptr_c, col_ind_c, values_c, c_nnz);
  return new_local_c;
}

template <typename T>
void free_tiles(std::vector<graphblas::Matrix<T>*> imp) {
  // TODO: free the tiles.
}

// TODO: switch to tree algorithm
template <typename T>
graphblas::Matrix<T>* sum_tiles(std::vector<graphblas::Matrix<T>*> imp) {
  if (imp.size() == 0) {
    return nullptr;
  }
  graphblas::Matrix<T>* sum = imp[0];
  for (size_t i = 1; i < imp.size(); i++) {
    graphblas::Matrix<T>* result = sum_tiles(sum, imp[i]);
    std::swap(sum, result);
    // TODO: delete result;
    // destroy_grb(result);
  }
  return sum;
}

template <typename T>
graphblas::Matrix<T>* sum_tiles_yuxin(std::vector<graphblas::Matrix<T>*> imp) {
  if (imp.size() == 0) {
    return nullptr;
  }

  graphblas::Index m, n;
  imp[0]->nrows(&m);
  imp[0]->ncols(&n);

  ::cuda::SparseSPAAccumulator<T> acc;

  for (auto mat : imp) {
    // convert mat into Yuxin's BCL::cuda::CSRMatrix
    graphblas::Index nnz;
    mat->nvals(&nnz);
    ::cuda::CSRMatrix<T, graphblas::Index> cmat(m, n, nnz,
                                                mat->matrix_.sparse_.d_csrVal_,
                                                mat->matrix_.sparse_.d_csrRowPtr_,
                                                mat->matrix_.sparse_.d_csrColInd_);
    acc.accumulate(std::move(cmat), {0, 0});
  }
  acc.sort_mats();
  acc.get_lbs();
  auto result_mat = acc.get_matrix(m, n);

  graphblas::Matrix<T>* grb_result = new graphblas::Matrix<T>(result_mat.m_, result_mat.n_);
  grb_result->build(result_mat.row_ptr_, result_mat.col_ind_, result_mat.vals_,
                    result_mat.nnz_);
  return grb_result;
}

template <typename T, typename index_type>
auto get_coo(const std::vector<T>& values,
             const std::vector<index_type>& row_indices,
             const std::vector<index_type>& col_indices)
{
    using coord_type = std::pair<index_type, index_type>;
    using tuple_type = std::pair<coord_type, T>;
    using coo_t = std::vector<tuple_type>;

    coo_t coo_values(values.size());

    for (size_t i = 0; i < values.size(); i++) {
      coo_values[i] = {{row_indices[i], col_indices[i]}, values[i]};
    }

    std::sort(coo_values.begin(), coo_values.end(),
              [](const auto& a, const auto& b) -> bool {
                if (std::get<0>(a) != std::get<0>(b)) {
                  return std::get<0>(a) < std::get<0>(b);
                } else {
                  return std::get<1>(a) < std::get<1>(b);
                }
              });

    return coo_values;
}

template <typename T>
void print_coo(const T& coo, size_t max_idx = std::numeric_limits<size_t>::max()) {
  for (size_t i = 0; i < std::min(coo.size(), max_idx); i++) {
    auto idx = std::get<0>(coo[i]);
    auto val = std::get<1>(coo[i]);
    printf("(%lu, %lu) %f\n", idx.first, idx.second, val);
  }
}

} // end cuda	

} // end BCL