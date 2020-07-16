#pragma once

#include <matrix_sum.cuh>

namespace BCL {

template <typename T>
struct max {
  T operator()(const T& a, const T& b) const {
    if (a < b) {
      return b;
    } else {
      return a;
    }
  }
};

template <typename T>
struct min {
  T operator()(const T& a, const T& b) const {
    if (a < b) {
      return a;
    } else {
      return b;
    }
  }
};

namespace cuda {

graphblas::Descriptor* grb_desc_ = nullptr;
// TODO: Actually free matrix contents.
template <typename T>
void destroy_grb(graphblas::Matrix<T>* x, const std::string lbl = "hey") {
  if (x == nullptr) {
    return;
  }
  auto& m = x->matrix_.sparse_;
  if (m.h_csrRowPtr_) free(m.h_csrRowPtr_);
  if (m.h_csrColInd_) free(m.h_csrColInd_);
  if (m.h_csrVal_   ) free(m.h_csrVal_);
  if (BCL::cuda::__is_valid_cuda_gptr(m.d_csrRowPtr_)) {
    BCL::cuda::dealloc(BCL::cuda::__to_cuda_gptr(m.d_csrRowPtr_));
  } else {
    /*
    fprintf(stderr, "0x%p is not a valid BCL pointer, checking value for %s (valid segment is 0x%p -> 0x%p\n",
            m.d_csrRowPtr_, lbl.c_str(), BCL::cuda::smem_base_ptr, BCL::cuda::smem_base_ptr + BCL::cuda::shared_segment_size);
            */
    CUDA_CALL(cudaPeekAtLastError());
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

template <typename T, typename Allocator>
graphblas::Matrix<T>* sum_tiles_yuxin(std::vector<graphblas::Matrix<T>*> imp) {
  if (imp.size() == 0) {
    return nullptr;
  }

  using index_type = graphblas::Index;

  graphblas::Index m, n;
  imp[0]->nrows(&m);
  imp[0]->ncols(&n);

  ::cuda::SparseSPAAccumulator<T, index_type, Allocator> acc;

  for (auto mat : imp) {
    // convert mat into Yuxin's BCL::cuda::CSRMatrix
    graphblas::Index nnz;
    mat->nvals(&nnz);
    ::cuda::CSRMatrix<T, index_type, Allocator> cmat(m, n, nnz,
                                                mat->matrix_.sparse_.d_csrVal_,
                                                mat->matrix_.sparse_.d_csrRowPtr_,
                                                mat->matrix_.sparse_.d_csrColInd_);
    acc.accumulate(std::move(cmat), {0, 0});
  }
  acc.sort_mats();
  acc.get_lbs();
  // Assume a 1 GB memory limit for accumualtor.
  size_t max_mem = 1*1000*1000*1000;
  size_t max_mem_row = std::min<size_t>(m, max_mem/((sizeof(graphblas::Index)+sizeof(T))*n));
  size_t block_size = 512;
  auto result_mat = acc.get_matrix(m, n, max_mem_row, block_size);

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

template <typename T, typename index_type>
auto remove_zeros(const std::vector<std::pair<std::pair<index_type, index_type>, T>>& coo_values) {
    using coord_type = std::pair<index_type, index_type>;
    using tuple_type = std::pair<coord_type, T>;
    using coo_t = std::vector<tuple_type>;

    coo_t new_coo;

    for (const auto& nz : coo_values) {
      auto val = std::get<1>(nz);
      if (val != 0.0) {
        new_coo.push_back(nz);
      }
    }
    return new_coo;
}

template <typename T>
void print_coo(const T& coo, size_t max_idx = std::numeric_limits<size_t>::max()) {
  for (size_t i = 0; i < std::min(coo.size(), max_idx); i++) {
    auto idx = std::get<0>(coo[i]);
    auto val = std::get<1>(coo[i]);
    printf("(%lu, %lu) %f\n", idx.first, idx.second, val);
  }
}

template <typename T, typename index_type, typename Allocator>
graphblas::Matrix<T>*
get_graphblast_view(CudaCSRMatrix<T, index_type, Allocator>& a) {
  graphblas::Matrix<T>* grb_matrix = new graphblas::Matrix<T>(a.m(), a.n());
  grb_matrix->build(a.rowptr_data(), a.colind_data(), a.values_data(), a.nnz());
  return grb_matrix;
}

template <typename T, typename index_type, typename Allocator>
CudaCSRMatrix<T, index_type, Allocator>
convert_to_csr(graphblas::Matrix<T>* a_grb) {
  graphblas::Index m, n, nnz;
  a_grb->nrows(&m);
  a_grb->ncols(&n);
  a_grb->nvals(&nnz);
  T* values = a_grb->matrix_.sparse_.d_csrVal_;
  index_type* rowptr = a_grb->matrix_.sparse_.d_csrRowPtr_;
  index_type* colind = a_grb->matrix_.sparse_.d_csrColInd_;
  return CudaCSRMatrix<T, index_type, Allocator>({m, n}, nnz, values, rowptr, colind);
}

template <typename T, typename index_type, typename Allocator>
CudaCSRMatrix<T, index_type, Allocator>
cusparse_spgemm(CudaCSRMatrix<T, index_type, Allocator>& a,
                CudaCSRMatrix<T, index_type, Allocator>& b)
{
  // static assert index_type is graphblas::Index
  grb_desc_->descriptor_.debug_ = false;
  if (a.nnz() == 0 || b.nnz() == 0) {
    // return empty matrix
    return CudaCSRMatrix<T, index_type, Allocator>({a.shape()[0], b.shape()[1]});
  } else {
    auto binary_op = GrB_NULL;
    auto semiring = graphblas::PlusMultipliesSemiring<T>{};
    auto a_grb = get_graphblast_view(a);
    auto b_grb = get_graphblast_view(b);

    auto* c_grb = new graphblas::Matrix<T>(a.shape()[0], b.shape()[1]);

    graphblas::mxm<T, T, T, T, decltype(binary_op), decltype(semiring),
                   Allocator>
                   (c_grb, GrB_NULL,
                    binary_op, semiring,
                    a_grb, b_grb, grb_desc_);

    auto c = convert_to_csr<T, index_type, Allocator>(c_grb);
    free(a_grb);
    free(b_grb);
    free(c_grb);
    return c;
  }
}

template <typename T, typename index_type, typename Allocator>
CudaCSRMatrix<T, index_type, Allocator>
sum_cusparse(CudaCSRMatrix<T, index_type, Allocator>& a,
             CudaCSRMatrix<T, index_type, Allocator>& b) {
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

  index_type arows = a.shape()[0];
  index_type acols = a.shape()[1];
  index_type brows = b.shape()[0];
  index_type bcols = b.shape()[1];

  assert(acols == bcols);
  assert(arows == brows);

  index_type m = arows;
  index_type n = acols;

  static_assert(std::is_same<int, index_type>::value);
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

  index_type a_nnz = a.nnz();
  index_type b_nnz = b.nnz();
  index_type c_nnz;

  index_type* nnzTotalDevHostPtr;
  nnzTotalDevHostPtr = &c_nnz;

  index_type* row_ptr_c;
  row_ptr_c = rebind_allocator_t<Allocator, index_type>{}.allocate(m+1);
  if (row_ptr_c == nullptr) {
    throw std::runtime_error("Couldn't allocate C.");
  }

  index_type* a_row_ptr = a.rowptr_data();
  index_type* a_col_ind = a.colind_data();

  index_type* b_row_ptr = b.rowptr_data();
  index_type* b_col_ind = b.colind_data();

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
  index_type* col_ind_c;
  T* values_c;
  col_ind_c = rebind_allocator_t<Allocator, index_type>{}.allocate(c_nnz);
  values_c = rebind_allocator_t<Allocator, T>{}.allocate(c_nnz);
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
                   a.values_data(),
                   a.rowptr_data(),
                   a.colind_data(),
                   &beta,
                   descr_b,
                   b_nnz,
                   b.values_data(),
                   b.rowptr_data(),
                   b.colind_data(),
                   descr_c,
                   values_c,
                   row_ptr_c,
                   col_ind_c);

  BCL::cuda::throw_cusparse(status);
  cudaDeviceSynchronize();

  return CudaCSRMatrix<T, index_type, Allocator>({m, n}, c_nnz, values_c, row_ptr_c, col_ind_c);
}

template <typename T, typename index_type, typename Allocator>
CudaCSRMatrix<T, index_type, Allocator>
sum_tiles_cusparse(std::vector<CudaCSRMatrix<T, index_type, Allocator>>& imp) {
  using csr_type = CudaCSRMatrix<T, index_type, Allocator>;
  if (imp.size() == 0) {
    return csr_type({0, 0}, 0);
  }
  csr_type sum = std::move(imp[0]);
  for (size_t i = 1; i < imp.size(); i++) {
    csr_type comp = std::move(imp[i]);
    fprintf(stderr, "RANK(%lu) computing sum %lu (%d + %d)...\n",
            BCL::rank(), i,
            sum.nnz(), comp.nnz());
    csr_type result = sum_cusparse<T, index_type, Allocator>(sum, comp);
    fprintf(stderr, "RANK(%lu) computed...\n", BCL::rank());
    std::swap(sum, result);
  }
  fprintf(stderr, "RANK(%lu) returning sum...\n", BCL::rank());
  return sum;
}

template <typename T, typename index_type, typename Allocator>
bool is_shared_seg(CudaCSRMatrix<T, index_type, Allocator>& mat) {
  if (!__is_valid_cuda_gptr(mat.values_data())) {
    return false;
  } else if (!__is_valid_cuda_gptr(mat.rowptr_data())) {
    return false;
  } else if (!__is_valid_cuda_gptr(mat.colind_data())) {
    return false;
  } else {
    return true;
  }
}

} // end cuda	

} // end BCL
