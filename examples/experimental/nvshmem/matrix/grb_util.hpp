// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

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

template <typename MatrixType>
CSRMatrix<typename MatrixType::value_type, typename MatrixType::index_type> to_cpu_generic(MatrixType& mat) {
  using T = typename MatrixType::value_type;
  using index_type = typename MatrixType::index_type;
  std::vector<T> values(mat.nnz());
  std::vector<index_type> rowptr(mat.m()+1);
  std::vector<index_type> colind(mat.nnz());

  cudaMemcpy(values.data(), mat.values_data(), sizeof(T)*mat.nnz(), cudaMemcpyDeviceToHost);
  cudaMemcpy(rowptr.data(), mat.rowptr_data(), sizeof(index_type)*(mat.m()+1), cudaMemcpyDeviceToHost);
  cudaMemcpy(colind.data(), mat.colind_data(), sizeof(index_type)*mat.nnz(), cudaMemcpyDeviceToHost);

  return CSRMatrix<T, index_type>(mat.m(), mat.n(), mat.nnz(),
                                  std::move(values), std::move(rowptr),
                                  std::move(colind));
}

template <typename T>
struct cusparse_type_t;

template <>
struct cusparse_type_t<int32_t> {
  using type = int32_t;
  static auto cusparse_type() {
    return CUSPARSE_INDEX_32I;
  }
};

template <>
struct cusparse_type_t<int64_t> {
  using type = int64_t;
  static auto cusparse_type() {
    return CUSPARSE_INDEX_64I;
  }
};

cusparseHandle_t bcl_cusparse_handle_;

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

/*
struct alloc_t {
  char* ptr_;
  size_t size_;

  template <typename T>
  alloc_t(T* ptr, size_t size) : ptr_((char *) ptr), size_(size*sizeof(T)) {}

  char* begin() {
    return ptr_;
  }

  char* end() {
    return ptr_ + size_;
  }
};

bool check_overlap(alloc_t a, alloc_t b) {
  return a.begin() <= b.end() && a.end() >= b.begin();
}

bool check_overlap(const std::vector<alloc_t>& allocations) {
  for (size_t i = 0; i < allocations.size(); i++) {
    for (size_t j = i+1; j < allocations.size(); j++) {
      if (check_overlap(allocations[i], allocations[j])) {
        fprintf(stderr, "%p overlaps with %p\n", allocations[i].ptr_, allocations[j].ptr_);
        return true;
      }
    }
  }
  return false;
}
*/

template <typename T, typename index_type, typename Allocator>
CudaCSRMatrix<T, index_type, Allocator>
sum_cusparse(CudaCSRMatrix<T, index_type, Allocator>& a,
             CudaCSRMatrix<T, index_type, Allocator>& b) {
  // XXX: Do an element-wise add using cuSparse
  //      'A' here is local_c, and 'B' here is result_c
  //.     At the end, the new accumulated matrix will be put in local_c.

  cusparseHandle_t& handle = bcl_cusparse_handle_;

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
  cusparseStatus_t status = 
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
  
  T alpha = 1.0;
  T beta = 1.0;

  size_t pBufferSizeInBytes;

  // TODO: what am I supposed to pass for csrValC and csrColIndC???
  //       we don't know nnz yet.
  status =
  cusparseScsrgeam2_bufferSizeExt(handle,
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
                                  nullptr, row_ptr_c, nullptr,
                                  &pBufferSizeInBytes);
  BCL::cuda::throw_cusparse(status);

  char* buffer = rebind_allocator_t<Allocator, char>{}.allocate(pBufferSizeInBytes);

  status = 
  cusparseXcsrgeam2Nnz(handle,
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
                       nnzTotalDevHostPtr,
                       buffer);
  BCL::cuda::throw_cusparse(status);

  if (nnzTotalDevHostPtr == nullptr) {
    throw std::runtime_error("Unhandled case: nnzTotalDevHostPtr is null.");
  } else {
    c_nnz = *nnzTotalDevHostPtr;
  }
  index_type* col_ind_c;
  T* values_c;
  col_ind_c = rebind_allocator_t<Allocator, index_type>{}.allocate(c_nnz);
  values_c = rebind_allocator_t<Allocator, T>{}.allocate(c_nnz);
  if (col_ind_c == nullptr || values_c == nullptr) {
    throw std::runtime_error("sum_tiles(): out of memory.");
  }
  status = 
  cusparseScsrgeam2(handle,
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
                    col_ind_c,
                    buffer);

  BCL::cuda::throw_cusparse(status);
  cudaDeviceSynchronize();

  cusparseDestroyMatDescr(descr_a);
  cusparseDestroyMatDescr(descr_b);
  cusparseDestroyMatDescr(descr_c);

  deallocate_with<char, Allocator>(buffer);

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
    csr_type result = sum_cusparse<T, index_type, Allocator>(sum, comp);
    std::swap(sum, result);
  }
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

template <typename MatrixType>
void check_matrix(MatrixType& x) {
  size_t m = x.m();
  size_t n = x.n();
  size_t nnz = x.nnz();

  using T = typename MatrixType::value_type;
  using index_type = typename MatrixType::index_type;

  std::vector<T> values(nnz);
  std::vector<index_type> rowptr(m+1);
  std::vector<index_type> colind(nnz);

  cudaMemcpy(values.data(), x.values_data(), sizeof(T)*nnz, cudaMemcpyDeviceToHost);
  cudaMemcpy(rowptr.data(), x.rowptr_data(), sizeof(index_type)*(m+1), cudaMemcpyDeviceToHost);
  cudaMemcpy(colind.data(), x.colind_data(), sizeof(index_type)*nnz, cudaMemcpyDeviceToHost);

  size_t counted_nnz = 0;
  for (size_t i = 0; i < m; i++) {
    index_type last_colidx = -1;
    for (index_type j_ptr = rowptr[i]; j_ptr < rowptr[i+1]; j_ptr++) {
      index_type j = colind[j_ptr];
      T value = values[j_ptr];

      assert(j > last_colidx);
      last_colidx = j;

      assert(i >= 0 && i < m);
      assert(j >= 0 && j < n);
      counted_nnz++;
    }
  }
  fprintf(stderr, "Counted %lu / %lu nnz, all within bounds (%lu, %lu)\n",
          counted_nnz, nnz, m, n);
  // assert(counted_nnz == nnz);
}

template <typename AMatrixType, typename BMatrixType>
auto spgemm_cusparse(AMatrixType& a,
                     BMatrixType& b)
{
    // fprintf(stderr, "FIRST PRINT a values_ptr: %p, b values_ptr: %p\n",
    //         a.values_data(), b.values_data());

/*
    std::vector<alloc_t> allocations = {alloc_t(a.values_data(), a.nnz()),
                                        alloc_t(a.rowptr_data(), a.shape()[0]+1),
                                        alloc_t(a.colind_data(), a.nnz()),
                                        alloc_t(b.values_data(), b.nnz()),
                                        alloc_t(b.rowptr_data(), b.shape()[0]+1),
                                        alloc_t(b.colind_data(), b.nnz())};
    if (check_overlap(allocations)) {
      fprintf(stderr, "Overlapping!\n");
    } else {
      fprintf(stderr, "NOT overlapping.\n");
    }
    */
  using T = typename AMatrixType::value_type;
  using index_type = typename AMatrixType::index_type;
  using Allocator = BCL::cuda::bcl_allocator<T>;
  // using Allocator = typename AMatrixType::allocator_type;
  // static assert index_type is graphblas::Index
  assert(a.n() == b.m());
  if (a.nnz() == 0 || b.nnz() == 0) {
    // return empty matrix
    return CudaCSRMatrix<T, index_type, Allocator>({a.shape()[0], b.shape()[1]}, 0);
  } else {
    size_t m = a.m();
    size_t n = b.n();
    size_t k = a.n();
    assert(b.m() == k);

    // fprintf(stderr, "First time A:\n");
    // check_matrix(a);
    // fprintf(stderr, "First time B:\n");
    // check_matrix(b);


/*
    fprintf(stderr, "(%lu) Multiplying A (%lu x %lu), %lu nnz by B (%lu x %lu), %lu nnz -> C(%lu x %lu), ? nnz\n",
            BCL::rank(), a.m(), a.n(), a.nnz(), b.m(), b.n(), b.nnz(),
            m, n);
            */

    // check_matrix(a);
    // check_matrix(b);
    // fprintf(stderr, "Matrices okay.\n");

    cusparseHandle_t& handle = bcl_cusparse_handle_;

    int baseC, nnzC;
    csrgemm2Info_t info = nullptr;
    size_t bufferSize;
    char* buffer = nullptr;
    // nnzTotalDevHostPtr points to host memory
    int* nnzTotalDevHostPtr = &nnzC;
    T alpha = 1;
    T beta = 0;

    cusparseMatDescr_t descr;
    cusparseStatus_t status = 
    cusparseCreateMatDescr(&descr);
    BCL::cuda::throw_cusparse(status);

    status = 
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    BCL::cuda::throw_cusparse(status);
    status = 
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    BCL::cuda::throw_cusparse(status);

    status = cusparseCreateCsrgemm2Info(&info);
    BCL::cuda::throw_cusparse(status);

    T* values_d = allocate_with<T, Allocator>(1);
    index_type* colind_d = allocate_with<index_type, Allocator>(1);
    index_type* rowptr_d = allocate_with<index_type, Allocator>(m+1);
    cudaMemset(rowptr_d, 0, sizeof(index_type)*(m+1));
    cudaMemset(values_d, 0, sizeof(T));
    cudaMemset(colind_d, 0, sizeof(index_type));

    /*
    fprintf(stderr, "Calling cusparseScsrgemm2_bufferSizeExt(handle, %lu, %lu, %lu, %f,\n"
                    "                                        descr, %lu, %p, %p,\n"
                    "                                        descr, %lu, %p, %p,\n"
                    "                                        %f,\n"
                    "                                        descr, 0, 0, 0,\n"
                    "                                        info,\n"
                    "                                        &bufferSize);\n\n",
                    m, n, k, alpha,
                    a.nnz(), a.rowptr_data(), a.colind_data(),
                    b.nnz(), b.rowptr_data(), b.colind_data(),
                    beta);
    fprintf(stderr, "Check A:\n");
    check_matrix(a);
    fprintf(stderr, "Check B:\n");
    check_matrix(b);

    fprintf(stderr, "a values_ptr: %p, b values_ptr: %p\n",
            a.values_data(), b.values_data());

    std::vector<alloc_t> allocations = {alloc_t(a.values_data(), a.nnz()),
                                        alloc_t(a.rowptr_data(), m+1),
                                        alloc_t(a.colind_data(), a.nnz()),
                                        alloc_t(b.values_data(), b.nnz()),
                                        alloc_t(b.rowptr_data(), k+1),
                                        alloc_t(b.colind_data(), b.nnz())};
    if (check_overlap(allocations)) {
      fprintf(stderr, "Overlapping!\n");
    } else {
      fprintf(stderr, "NOT overlapping.\n");
    }
    */

    status = 
    cusparseScsrgemm2_bufferSizeExt(handle, m, n, k, &alpha,
        descr, a.nnz(), a.rowptr_data(), a.colind_data(),
        descr, b.nnz(), b.rowptr_data(), b.colind_data(),
        &beta,
        descr, 0, rowptr_d, colind_d,
        info,
        &bufferSize);
    cudaDeviceSynchronize();
    BCL::cuda::throw_cusparse(status);
    // fprintf(stderr, "(%lu): cusparseScsrgemm2_bufferSizeExt\n", BCL::rank());
    // cudaDeviceSynchronize();


    buffer = allocate_with<char, Allocator>(bufferSize);
    // fprintf(stderr, "buffer: %p (%lu bytes)\n", buffer, bufferSize);

    // step 3: compute csrRowPtrC

/*
      {
      fprintf(stderr, "First print, before:\n");
      fprintf(stderr, "Tried to mutiply A (%lu, %lu), %lu nnz by B(%lu, %lu) %lu nnz\n",
              a.shape()[0], a.shape()[1], b.nnz(), b.shape()[0], b.shape()[1], b.nnz());
      auto local_a = BCL::cuda::to_cpu_generic(a);
      auto local_a_coo = local_a.get_coo();
      fprintf(stderr, "BEGIN Printing matrix:\n");
      fprintf(stderr, "%lu %lu %lu\n", local_a.shape()[0], local_a.shape()[1], a.nnz());
      for (size_t i = 0; i < local_a.shape()[0]; i++) {
        // fprintf(stderr, "printing row %lu (%lu -> %lu)\n", i, local_a.rowptr_data()[i], local_a.rowptr_data()[i+1]);
        for (index_type j_ptr = local_a.rowptr_data()[i]; j_ptr < local_a.rowptr_data()[i+1]; j_ptr++) {
          index_type j = local_a.colind_data()[j_ptr];
          T value = local_a.values_data()[j_ptr];
          fprintf(stderr, "%lu %d %f\n", i+1, j+1, value);
        }
      }
      /*
      for (size_t i = 0; i < local_a.vals_.size(); i++) {
        std::cout << local_a.vals_[i] << std::endl;
        printf("%f\n", local_a.vals_[i]);
      }
      for (const auto& tuple : local_a_coo) {
        std::cout << std::get<0>(std::get<0>(tuple)) << " " << std::get<1>(std::get<0>(tuple)) << " " << std::get<1>(tuple) << std::endl;
      }
      */
      // fprintf(stderr, "END print\n");
      // fflush(stdout);
      // sleep(1);
      // }

    index_type* csrRowPtrC = allocate_with<index_type, Allocator>(m+1);
    status = 
    cusparseXcsrgemm2Nnz(handle, m, n, k,
                         descr, a.nnz(), a.rowptr_data(), a.colind_data(),
                         descr, b.nnz(), b.rowptr_data(), b.colind_data(),
                         descr, 0, rowptr_d, colind_d,
                         descr, csrRowPtrC, nnzTotalDevHostPtr, info, buffer);
    cudaDeviceSynchronize();
    /*
    if (status == CUSPARSE_STATUS_EXECUTION_FAILED) {
      {
      fprintf(stderr, "Failed. I should probably print the matrices.\n");
      fprintf(stderr, "Tried to mutiply A (%lu, %lu), %lu nnz by B(%lu, %lu) %lu nnz\n",
              a.shape()[0], a.shape()[1], b.nnz(), b.shape()[0], b.shape()[1], b.nnz());
      auto local_a = BCL::cuda::to_cpu_generic(a);
      auto local_a_coo = local_a.get_coo();
      fprintf(stderr, "BEGIN Printing matrix:\n");
      for (size_t i = 0; i < local_a.shape()[0]; i++) {
        fprintf(stderr, "printing row %lu (%lu -> %lu)\n", i, local_a.rowptr_data()[i], local_a.rowptr_data()[i+1]);
        for (index_type j_ptr = local_a.rowptr_data()[i]; j_ptr < local_a.rowptr_data()[i+1]; j_ptr++) {
          index_type j = local_a.colind_data()[j_ptr];
          T value = local_a.values_data()[j_ptr];
          fprintf(stderr, "%lu %d %f\n", i, j, value);
        }
      }
      for (size_t i = 0; i < local_a.vals_.size(); i++) {
        std::cout << local_a.vals_[i] << std::endl;
        printf("%f\n", local_a.vals_[i]);
      }
      for (const auto& tuple : local_a_coo) {
        std::cout << std::get<0>(std::get<0>(tuple)) << " " << std::get<1>(std::get<0>(tuple)) << " " << std::get<1>(tuple) << std::endl;
      }
      fprintf(stderr, "END print\n");
      fflush(stdout);
      sleep(1);
      }
    }
    */
    // fprintf(stderr, "(%lu): cusparseXcsrgemm2Nnz\n", BCL::rank());
    BCL::cuda::throw_cusparse(status);
    // fprintf(stderr, "(%lu): after cusparseXcsrgemm2Nnz\n", BCL::rank());
    // cudaDeviceSynchronize();

    if (nnzTotalDevHostPtr != nullptr) {
      // fprintf(stderr, "RegCopy...\n");
      nnzC = *nnzTotalDevHostPtr;
    } else {
      // fprintf(stderr, "Mmecpying...\n");
      cudaMemcpy(&nnzC, csrRowPtrC+m, sizeof(index_type), cudaMemcpyDeviceToHost);
      cudaMemcpy(&baseC, csrRowPtrC, sizeof(index_type), cudaMemcpyDeviceToHost);
      nnzC -= baseC;
    }

    // step 4: finish sparsity pattern and value of C
    index_type* csrColIndC = allocate_with<index_type, Allocator>(nnzC);
    T* csrValC = allocate_with<T, Allocator>(nnzC);
    // Remark: set csrValC to null if only sparsity pattern is required.
    status = 
    cusparseScsrgemm2(handle, m, n, k, &alpha,
            descr, a.nnz(), a.values_data(), a.rowptr_data(), a.colind_data(),
            descr, b.nnz(), b.values_data(), b.rowptr_data(), b.colind_data(),
            &beta,
            descr, 0, values_d, rowptr_d, colind_d,
            descr, csrValC, csrRowPtrC, csrColIndC,
            info, buffer);
    BCL::cuda::throw_cusparse(status);
    cudaDeviceSynchronize();

    // step 5: destroy the opaque structure
    cusparseDestroyCsrgemm2Info(info);
    cusparseDestroyMatDescr(descr);
    deallocate_with<char, Allocator>(buffer);
    deallocate_with<index_type, Allocator>(rowptr_d);
    deallocate_with<index_type, Allocator>(colind_d);
    deallocate_with<T, Allocator>(values_d);

    return CudaCSRMatrix<T, index_type, Allocator>({m, n}, nnzC, csrValC, csrRowPtrC, csrColIndC);
  }
}

// New CuSPARSE SpGEMM API, `cusparseSpGEMM`
// Currently no advantage, since it does not
// actually support any extra types.
// TODO: concepts
template <typename AMatrixType, typename BMatrixType>
auto
spgemm_cusparse_newapi(AMatrixType& a,
                       BMatrixType& b)
{
  // static assert index_type is graphblas::Index
  using T = typename AMatrixType::value_type;
  using index_type = typename AMatrixType::index_type;
  using Allocator = BCL::cuda::bcl_allocator<T>;
  if (a.nnz() == 0 || b.nnz() == 0) {
    // return empty matrix
    return CudaCSRMatrix<T, index_type, Allocator>({a.shape()[0], b.shape()[1]}, 0);
  } else {
    size_t m = a.m();
    size_t n = b.n();
    size_t k = a.n();

    cusparseHandle_t& handle = bcl_cusparse_handle_;

    T alpha = 1;
    T beta = 0;

    fprintf(stderr, "(%lu) Multiplying A (%lu x %lu), %lu nnz by B (%lu x %lu), %lu nnz -> C(%lu x %lu), ? nnz\n",
            BCL::rank(), a.m(), a.n(), a.nnz(), b.m(), b.n(), b.nnz(),
            m, n);

    cusparseSpMatDescr_t descr_a, descr_b, descr_c;

    fprintf(stderr, "(%lu) Create CSRs...\n", BCL::rank());
    cusparseStatus_t status =
    cusparseCreateCsr(&descr_a, a.m(), a.n(), a.nnz(),
                      a.rowptr_data(), a.colind_data(), a.values_data(),
                      cusparse_type_t<index_type>::cusparse_type(),
                      cusparse_type_t<index_type>::cusparse_type(),
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    BCL::cuda::throw_cusparse(status);
    cudaDeviceSynchronize();

    status =
    cusparseCreateCsr(&descr_b, b.m(), b.n(), b.nnz(),
                      b.rowptr_data(), b.colind_data(), b.values_data(),
                      cusparse_type_t<index_type>::cusparse_type(),
                      cusparse_type_t<index_type>::cusparse_type(),
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    BCL::cuda::throw_cusparse(status);
    cudaDeviceSynchronize();

    status =
    cusparseCreateCsr(&descr_c, m, n, 0,
                      NULL, NULL, NULL,
                      cusparse_type_t<index_type>::cusparse_type(),
                      cusparse_type_t<index_type>::cusparse_type(),
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    BCL::cuda::throw_cusparse(status);
    cudaDeviceSynchronize();

    cusparseSpGEMMDescr_t descr_spgemm;
    status =
    cusparseSpGEMM_createDescr(&descr_spgemm);
    BCL::cuda::throw_cusparse(status);

    size_t bufferSize1;

    fprintf(stderr, "(%lu) Estimate work...\n", BCL::rank());
    status =
    cusparseSpGEMM_workEstimation(handle,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &alpha, descr_a, descr_b, &beta, descr_c,
                                  CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                  descr_spgemm, &bufferSize1, NULL);
    BCL::cuda::throw_cusparse(status);
    cudaDeviceSynchronize();

    fprintf(stderr, "(%lu) first buffer %lu bytes\n", BCL::rank(), bufferSize1);

    char* buffer_1 = allocate_with<char, Allocator>(bufferSize1);

    status =
    cusparseSpGEMM_workEstimation(handle,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &alpha, descr_a, descr_b, &beta, descr_c,
                                  CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                  descr_spgemm, &bufferSize1, buffer_1);
    BCL::cuda::throw_cusparse(status);
    cudaDeviceSynchronize();

    fprintf(stderr, "(%lu) Compute 1...\n", BCL::rank());
    size_t bufferSize2;
    status =
    cusparseSpGEMM_compute(handle,
                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                           &alpha, descr_a, descr_b, &beta, descr_c,
                           CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                           descr_spgemm, &bufferSize2, NULL);
    BCL::cuda::throw_cusparse(status);
    cudaDeviceSynchronize();

    fprintf(stderr, "(%lu) second buffer %lu bytes\n", BCL::rank(), bufferSize2);

    char* buffer_2 = allocate_with<char, Allocator>(bufferSize2);

    fprintf(stderr, "(%lu) Compute 2...\n", BCL::rank());
    status =
    cusparseSpGEMM_compute(handle,
                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                           &alpha, descr_a, descr_b, &beta, descr_c,
                           CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                           descr_spgemm, &bufferSize2, buffer_2);
    BCL::cuda::throw_cusparse(status);
    cudaDeviceSynchronize();

    fprintf(stderr, "(%lu) Extracting...\n", BCL::rank());

    int64_t nrows, ncols, nnz;
    status =
    cusparseSpMatGetSize(descr_c, &nrows, &ncols, &nnz);
    BCL::cuda::throw_cusparse(status);
    cudaDeviceSynchronize();

    index_type* c_colind = allocate_with<index_type, Allocator>(nnz);
    T* c_values = allocate_with<T, Allocator>(nnz);
    index_type* c_rowptr = allocate_with<index_type, Allocator>(m+1);

    status =
    cusparseCsrSetPointers(descr_c, c_rowptr, c_colind, c_values);
    BCL::cuda::throw_cusparse(status);
    cudaDeviceSynchronize();

    status =
    cusparseSpGEMM_copy(handle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, descr_a, descr_b, &beta, descr_c,
                        CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, descr_spgemm);
    BCL::cuda::throw_cusparse(status);
    cudaDeviceSynchronize();

    deallocate_with<char, Allocator>(buffer_2);
    deallocate_with<char, Allocator>(buffer_1);

    cusparseSpGEMM_destroyDescr(descr_spgemm);
    cusparseDestroySpMat(descr_a);
    cusparseDestroySpMat(descr_b);
    cusparseDestroySpMat(descr_c);

    return CudaCSRMatrix<T, index_type, Allocator>({m, n}, nnz, c_values, c_rowptr, c_colind);
  }
}


template <typename AMatrixType, typename BMatrixType, typename CMatrixType>
void spmm_cusparse(AMatrixType& a,
                   BMatrixType& b,
                   CMatrixType& c,
                   typename AMatrixType::value_type alpha = 1,
                   typename AMatrixType::value_type beta = 1)
{
  if (a.nnz() == 0) {
    return;
  }
  using T = typename AMatrixType::value_type;
  using Allocator = BCL::cuda::bcl_allocator<T>;
  // using Allocator = typename AMatrix::allocator_type;
  static_assert(std::is_same<typename AMatrixType::value_type, float>::value);
  static_assert(std::is_same<typename BMatrixType::value_type, float>::value);
  static_assert(std::is_same<typename CMatrixType::value_type, float>::value);
  using index_type = typename AMatrixType::index_type;
  // static_assert(std::is_same<typename AMatrixType::index_type, int32_t>::value);
  cusparseHandle_t& handle = bcl_cusparse_handle_;

  cusparseOrder_t order;
  cusparseSpMMAlg_t algorithm;

  using bmatrix_indexing = typename BMatrixType::indexing_type;
  using cmatrix_indexing = typename CMatrixType::indexing_type;
  static_assert(std::is_same<bmatrix_indexing, cmatrix_indexing>::value);
  constexpr bool row_major = std::is_same<bmatrix_indexing, RowMajorIndexing>::value;
  constexpr bool column_major = std::is_same<bmatrix_indexing, ColumnMajorIndexing>::value;

  static_assert(row_major || column_major);

  if (std::is_same<bmatrix_indexing, RowMajorIndexing>::value) {
    order = CUSPARSE_ORDER_ROW;
    algorithm = CUSPARSE_SPMM_CSR_ALG2;
  } else if (std::is_same<bmatrix_indexing, ColumnMajorIndexing>::value) {
    order = CUSPARSE_ORDER_COL;
    algorithm = CUSPARSE_MM_ALG_DEFAULT;
  } 

  cusparseSpMatDescr_t a_cusparse;
  cusparseStatus_t status = 
  cusparseCreateCsr(&a_cusparse, a.m(), a.n(), a.nnz(),
                    a.rowptr_data(), a.colind_data(), a.values_data(),
                    cusparse_type_t<index_type>::cusparse_type(),
                    cusparse_type_t<index_type>::cusparse_type(),
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  BCL::cuda::throw_cusparse(status);
  cusparseDnMatDescr_t b_cusparse;

  status = 
  cusparseCreateDnMat(&b_cusparse, b.m(), b.n(), b.ld(),
                      b.data(), CUDA_R_32F, order);
  BCL::cuda::throw_cusparse(status);

  cusparseDnMatDescr_t c_cusparse;
  status = 
  cusparseCreateDnMat(&c_cusparse, c.m(), c.n(), c.ld(),
                      c.data(), CUDA_R_32F, order);
  BCL::cuda::throw_cusparse(status);

  size_t bufferSize;
  status = 
  cusparseSpMM_bufferSize(handle,
                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                          &alpha,
                          a_cusparse,
                          b_cusparse,
                          &beta,
                          c_cusparse,
                          CUDA_R_32F,
                          algorithm,
                          &bufferSize);
  BCL::cuda::throw_cusparse(status);

  char* externalBuffer = allocate_with<char, Allocator>(bufferSize);

  status = 
  cusparseSpMM(handle,
               CUSPARSE_OPERATION_NON_TRANSPOSE,
               CUSPARSE_OPERATION_NON_TRANSPOSE,
               &alpha,
               a_cusparse,
               b_cusparse,
               &beta,
               c_cusparse,
               CUDA_R_32F,
               algorithm,
               externalBuffer);
  BCL::cuda::throw_cusparse(status);
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    assert(error == cudaSuccess);
  }

  deallocate_with<char, Allocator>(externalBuffer);
  cusparseDestroySpMat(a_cusparse);
  cusparseDestroyDnMat(b_cusparse);
  cusparseDestroyDnMat(c_cusparse);
}

// TODO: Put this in another file

template <typename T, typename index_type, typename Allocator = BCL::bcl_allocator<T>>
CudaCSRMatrix<T, index_type, Allocator> to_gpu(CSRMatrix<T, index_type>& mat) {
  CudaCSRMatrix<T, index_type, Allocator> mat_gpu({mat.m(), mat.n()}, mat.nnz());
  cudaMemcpy(mat_gpu.values_data(), mat.values_data(), sizeof(T)*mat.nnz(), cudaMemcpyHostToDevice);
  cudaMemcpy(mat_gpu.rowptr_data(), mat.rowptr_data(), sizeof(index_type)*(mat.m()+1), cudaMemcpyHostToDevice);
  cudaMemcpy(mat_gpu.colind_data(), mat.colind_data(), sizeof(index_type)*mat.nnz(), cudaMemcpyHostToDevice);
  return mat_gpu;
}

template <typename T, typename index_type, typename Allocator>
CSRMatrix<T, index_type> to_cpu(CudaCSRMatrix<T, index_type, Allocator>& mat) {
  std::vector<T> values(mat.nnz());
  std::vector<index_type> rowptr(mat.m()+1);
  std::vector<index_type> colind(mat.nnz());

  cudaMemcpy(values.data(), mat.values_data(), sizeof(T)*mat.nnz(), cudaMemcpyDeviceToHost);
  cudaMemcpy(rowptr.data(), mat.rowptr_data(), sizeof(index_type)*(mat.m()+1), cudaMemcpyDeviceToHost);
  cudaMemcpy(colind.data(), mat.colind_data(), sizeof(index_type)*mat.nnz(), cudaMemcpyDeviceToHost);

  return CSRMatrix<T, index_type>(mat.m(), mat.n(), mat.nnz(),
                                  std::move(values), std::move(rowptr),
                                  std::move(colind));
}

} // end cuda	

} // end BCL
